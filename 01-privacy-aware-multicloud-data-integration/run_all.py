#!/usr/bin/env python3
"""A1P3 - Policy-constrained multi-cloud placement with auditable (blockchain-anchored) lineage.

Runs everything for the paper:
  1. preprocessing (imputation, outlier winsorisation) of the synthetic P2P energy-trading integration log
  2. planner models (logistic violation model, logit-linear re-identification model, anchoring latency/gas model)
     and an independent auditor (gradient boosting) trained on a disjoint split
  3. proposed MILP placement (HiGHS via scipy.optimize.milp) and four baselines
     (greedy cheapest-compliant, single-cloud, random feasible, unconstrained) on 30 windows (5 seeds x 6 windows)
  4. ablation, sensitivity (tau/budget Pareto, capacity x gas heatmap), scalability
  5. Merkle-tree lineage anchoring + tamper-detection audit
  6. figures (PNG 300 dpi -> ../figures, JPG copies -> ../results), tables (CSV) and metrics.json
Usage: python3 run_all.py
"""
import os, json, time, hashlib
import numpy as np
import pandas as pd
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import csr_matrix, coo_matrix
from scipy.stats import wilcoxon
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, r2_score, mean_absolute_error
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data', 'A1P3_privacy_lineage_ops.csv')
RES = os.path.join(HERE, '..', 'results')
FIG = os.path.join(HERE, '..', 'figures')
os.makedirs(RES, exist_ok=True); os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({'font.family': 'serif', 'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8,
                     'legend.fontsize': 7, 'xtick.labelsize': 7, 'ytick.labelsize': 7, 'figure.dpi': 100,
                     'axes.spines.top': False, 'axes.spines.right': False})
ACC = '#c0392b'; GREYS = ['#222222', '#555555', '#888888', '#bbbbbb']

# ------------------------------------------------------------------ configuration (assumed cost model)
CFG = dict(
    tau=0.20,            # per-operation max predicted violation probability
    v_bar=0.04,          # window violation budget (expected violations / N)
    rho=0.35,            # per-operation max predicted re-identification risk
    kappa=0.80,          # cloud capacity = kappa * window volume / 3
    lat_bar=2600.0,      # mean anchoring latency budget per window (ms)
    gas_frac=0.30,       # gas budget = gas_frac * gas if every public op anchored with batch 1
    N=150, seeds=[0, 1, 2, 3, 4], windows_per_seed=6,
    reject_cost=5.0,     # USD penalty for deferring an operation (no compliant option)
    lambda_u=0.6,        # USD per GB per unit information loss
    gas_usd=2e-6,        # USD per gas unit on the public L2 rollup
    perm_tx_usd=0.004,   # USD per permissioned-ledger anchoring transaction
    repair_usd=0.03,     # USD per GB per lineage hop for lineage repair
)
CLOUD_PRICE = {'cloud_a': 0.12, 'cloud_b': 0.10, 'cloud_c': 0.08}            # USD/GB storage+compute
REGION_MULT = {'EU': 1.10, 'US': 1.00, 'IN': 0.80, 'UK': 1.05, 'SG': 0.95}
SITES = [('cloud_a', 'EU'), ('cloud_a', 'US'), ('cloud_a', 'IN'), ('cloud_b', 'EU'), ('cloud_b', 'UK'),
         ('cloud_b', 'US'), ('cloud_c', 'SG'), ('cloud_c', 'IN'), ('cloud_c', 'UK')]
REC_KB = {'trade_orders': 4, 'smart_meter_readings': 2, 'prosumer_profiles': 20, 'settlement_records': 8,
          'grid_constraint_maps': 40, 'wallet_kyc_refs': 30}                   # KB per record
# privacy transforms: (method, k, eps, info_loss, cost USD/GB)
TRANSFORMS = [('none', 0, np.nan, 0.00, 0.000), ('pseudonymization', 0, np.nan, 0.05, 0.010),
              ('k_anonymity', 10, np.nan, 0.30, 0.040), ('differential_privacy', 0, 2.0, 0.25, 0.030),
              ('differential_privacy', 0, 0.5, 0.55, 0.030)]
TR_NAMES = ['none', 'pseudo', 'k=10', 'DP e=2', 'DP e=0.5']
ENCS = [('tls_only', 0.0, 0.0), ('tls_plus_field_level', 0.02, 0.0), ('confidential_compute', 0.0, 0.30)]
ANCH_PERM = [('permissioned_ledger', 1), ('permissioned_ledger', 64)]
ANCH_PUB = [('public_l2_rollup', 1), ('public_l2_rollup', 16), ('public_l2_rollup', 64), ('public_l2_rollup', 256)]
PUBLIC_TYPES = {'settlement_records', 'trade_orders'}   # need publicly verifiable anchoring
CONSENT_OK = {'full': {'settlement', 'load_forecasting', 'market_analytics', 'regulatory_audit', 'marketing'},
              'research': {'settlement', 'load_forecasting', 'market_analytics', 'regulatory_audit'},
              'contract_only': {'settlement', 'regulatory_audit'}, 'none': set()}

LOG = []
def log(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); LOG.append(s)

# ------------------------------------------------------------------ 1. preprocessing
def preprocess(df):
    df = df.copy()
    rep = {}
    rep['missing_before'] = {c: int(v) for c, v in df.isna().sum().items() if v > 0 and c != 'dp_epsilon'}
    df['quasi_identifier_count'] = df.groupby('contains_location')['quasi_identifier_count'] \
        .transform(lambda s: s.fillna(s.median()))
    df['lineage_completeness'] = df.groupby('lineage_hops')['lineage_completeness'] \
        .transform(lambda s: s.fillna(s.median()))
    df['anchor_latency_ms'] = df.groupby(['ledger_type', 'anchor_batch_size'])['anchor_latency_ms'] \
        .transform(lambda s: s.fillna(s.median()))
    out = {}
    for c in ['records_transferred', 'anchor_latency_ms']:
        lg = np.log(df[c].clip(lower=1))
        q1, q3 = np.percentile(lg, [25, 75]); hi = q3 + 3 * (q3 - q1); lo = q1 - 3 * (q3 - q1)
        if c == 'anchor_latency_ms':   # per ledger/batch group (latency is multimodal)
            g = df.groupby(['ledger_type', 'anchor_batch_size'])[c]
            med = g.transform('median'); mad = g.transform(lambda s: np.median(np.abs(np.log(s) - np.log(s.median()))))
            z = np.abs(np.log(df[c]) - np.log(med)) / (1.4826 * mad)
            mask = z > 4
            df.loc[mask, c] = med[mask]
        else:
            mask = (lg > hi) | (lg < lo)
            df.loc[mask, c] = np.exp(np.clip(lg[mask], lo, hi))
        out[c] = int(mask.sum())
    rep['outliers_fixed'] = out
    df['dp_eps_f'] = df['dp_epsilon'].fillna(0.0)
    return df, rep

def protection(method, k, eps):
    method = np.asarray(method); k = np.asarray(k, float); eps = np.asarray(eps, float)
    return np.select([method == 'none', method == 'pseudonymization', method == 'k_anonymity'],
                     [0.0, 0.35, 1 - 1 / np.sqrt(np.maximum(k, 1))],
                     default=1 - np.minimum(np.nan_to_num(eps, nan=1) / 8, 0.9))

CATS = {'dataset_type': list(REC_KB), 'processing_purpose': ['settlement', 'load_forecasting', 'market_analytics',
        'regulatory_audit', 'marketing'], 'consent_scope': ['none', 'contract_only', 'research', 'full'],
        'anonymization_method': ['none', 'pseudonymization', 'k_anonymity', 'differential_privacy'],
        'encryption_mode': [e[0] for e in ENCS]}

def planner_features(d):
    """Policy-aware features for the planner (logistic) violation model. No target / post-event columns."""
    X = {}
    prot = protection(d['anonymization_method'].values, d['k_anonymity_k'].values, d['dp_epsilon'].values)
    pii = d['pii_field_count'].values.astype(float)
    mism = np.array([p not in CONSENT_OK[c] for c, p in zip(d['consent_scope'], d['processing_purpose'])], float)
    noadq = d['cross_border'].values * (1 - d['adequacy_agreement'].values)
    X['pii'] = pii; X['quasi'] = d['quasi_identifier_count'].values; X['loc'] = d['contains_location'].values
    X['logrec'] = np.log10(d['records_transferred'].values.clip(1)); X['prot'] = prot
    X['mism_pii2'] = mism * (pii > 2); X['mism'] = mism
    X['noadq_pii1'] = noadq * (pii > 1); X['noadq'] = noadq
    X['ret_pii3'] = (d['retention_days'].values > 365) * (pii > 3)
    X['lin_inc'] = 1 - d['lineage_completeness'].values; X['hops'] = d['lineage_hops'].values
    X['cc'] = (d['encryption_mode'].values == 'confidential_compute').astype(float)
    X['fle'] = (d['encryption_mode'].values == 'tls_plus_field_level').astype(float)
    X['mkt'] = (d['processing_purpose'].values == 'marketing').astype(float)
    X['pii_x_prot'] = pii * prot; X['quasi_x_prot'] = X['quasi'] * prot
    return pd.DataFrame(X)

def raw_features(d):
    """Raw features for the independent auditor (gradient boosting) - no engineered policy rules."""
    X = pd.DataFrame({c: d[c].values for c in ['cross_border', 'adequacy_agreement', 'pii_field_count',
                      'quasi_identifier_count', 'contains_location', 'k_anonymity_k', 'retention_days',
                      'lineage_hops', 'lineage_completeness', 'policy_rules_evaluated']})
    X['logrec'] = np.log10(d['records_transferred'].values.clip(1)); X['eps'] = d['dp_epsilon'].values
    for c, levels in CATS.items():
        X[c] = pd.Categorical(d[c].values, categories=levels).codes
    return X
CAT_IDX = list(range(12, 12 + len(CATS)))

def reid_features(d):
    prot = protection(d['anonymization_method'].values, d['k_anonymity_k'].values, d['dp_epsilon'].values)
    return np.column_stack([d['pii_field_count'].values, d['quasi_identifier_count'].values,
                            np.log10(d['records_transferred'].values.clip(1)), prot])

def logit(p): p = np.clip(p, 1e-4, 1 - 1e-4); return np.log(p / (1 - p))
def sigm(z): return 1 / (1 + np.exp(-z))

class Models:
    def fit(self, tr, au):
        # planner violation model
        Xp = planner_features(tr); self.sc = StandardScaler().fit(Xp)
        self.lr = LogisticRegression(C=1.0, max_iter=2000).fit(self.sc.transform(Xp), tr['policy_violation'])
        # planner re-identification model (logit-linear)
        self.rr = LinearRegression().fit(reid_features(tr), logit(tr['reidentification_risk'].values))
        # auditor models (disjoint split, different family)
        self.au = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                                 categorical_features=CAT_IDX, random_state=0) \
            .fit(raw_features(au), au['policy_violation'])
        self.aur = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, categorical_features=CAT_IDX,
                                                 random_state=0).fit(raw_features(au), au['reidentification_risk'])
        # anchoring latency model: log(lat) ~ ledger + log2(batch) + ledger*log2(batch) + hops
        A = self._anch_X(tr['ledger_type'].values, tr['anchor_batch_size'].values, tr['lineage_hops'].values)
        self.lat = LinearRegression().fit(A, np.log(tr['anchor_latency_ms'].values))
        self.lat_sigma = float(np.std(np.log(tr['anchor_latency_ms'].values) - self.lat.predict(A)))
        pub = tr[tr['ledger_type'] == 'public_l2_rollup']
        self.gas = LinearRegression().fit(pub[['lineage_hops']].values, (pub['anchor_gas_units'] * pub['anchor_batch_size']).values)
        return self
    @staticmethod
    def _anch_X(ledger, b, hops):
        pubf = (np.asarray(ledger) == 'public_l2_rollup').astype(float); lb = np.log2(np.asarray(b, float))
        return np.column_stack([pubf, lb, pubf * lb, np.asarray(hops, float)])
    def p_plan(self, d): return self.lr.predict_proba(self.sc.transform(planner_features(d)))[:, 1]
    def p_audit(self, d): return self.au.predict_proba(raw_features(d))[:, 1]
    def reid_plan(self, d): return sigm(self.rr.predict(reid_features(d)))
    def reid_audit(self, d): return np.clip(self.aur.predict(raw_features(d)), 0, 1)
    def latency(self, ledger, b, hops): return np.exp(self.lat.predict(self._anch_X(ledger, b, hops)) + self.lat_sigma ** 2 / 2)
    def gas_units(self, ledger, b, hops):
        g = self.gas.predict(np.asarray(hops, float).reshape(-1, 1)) / np.asarray(b, float)
        return np.where(np.asarray(ledger) == 'public_l2_rollup', g, 0.0)

# ------------------------------------------------------------------ 2. option enumeration
def adequacy_pairs(df):
    s = df[(df.cross_border == 1) & (df.adequacy_agreement == 1)]
    return set(zip(s.source_jurisdiction, s.target_jurisdiction))

def build_options(W, M, ADQ, allow_repair=True, policy=True, sites=None):
    """Cross-join window ops with processing options; returns options frame with cost/p/reid/feasibility."""
    sites = sites or SITES
    opts = []
    for si, (c, r) in enumerate(SITES):
        if (c, r) not in sites: continue
        for ti, t in enumerate(TRANSFORMS):
            for ei, e in enumerate(ENCS):
                for rep in ([0, 1] if allow_repair else [0]):
                    opts.append((si, c, r, ti, ei, rep))
    O = pd.DataFrame(opts, columns=['site', 'cloud', 'region', 'tr', 'enc', 'repair'])
    n, m = len(W), len(O)
    D = W.loc[W.index.repeat(m)].reset_index(drop=True)
    D['op'] = np.repeat(np.arange(n), m)
    for c in O.columns: D[c] = np.tile(O[c].values, n)
    # counterfactual policy context
    D['target_cloud'] = D['cloud']; D['target_jurisdiction'] = D['region']
    D['cross_border'] = (D['source_jurisdiction'] != D['region']).astype(int)
    D['adequacy_agreement'] = [int(a == b or (a, b) in ADQ) for a, b in zip(D['source_jurisdiction'], D['region'])]
    tr = np.array(TRANSFORMS, dtype=object)[D['tr'].values]
    D['anonymization_method'] = tr[:, 0]; D['k_anonymity_k'] = tr[:, 1].astype(int)
    D['dp_epsilon'] = tr[:, 2].astype(float)
    D['encryption_mode'] = np.array([e[0] for e in ENCS])[D['enc'].values]
    D['lineage_completeness'] = np.where(D['repair'] == 1, 1.0, D['lineage_completeness'])
    # predictions
    D['p'] = M.p_plan(D); D['reid'] = M.reid_plan(D)
    # cost model (USD)
    gb = D['records_transferred'].values * D['dataset_type'].map(REC_KB).values / 1e6
    D['gb'] = gb
    price = D['cloud'].map(CLOUD_PRICE).values * D['region'].map(REGION_MULT).values
    egress = 0.09 * (D['cloud'] != D['source_cloud']) + 0.02 * (D['region'] != D['source_jurisdiction'])
    trc = np.array([t[4] for t in TRANSFORMS])[D['tr'].values]; loss = np.array([t[3] for t in TRANSFORMS])[D['tr'].values]
    encc = np.array([e[1] for e in ENCS])[D['enc'].values]; encm = np.array([e[2] for e in ENCS])[D['enc'].values]
    D['cost_infra'] = gb * (price * (1 + encm) + egress + trc + encc)
    D['cost_util'] = gb * CFG['lambda_u'] * loss
    D['cost_repair'] = D['repair'] * gb * CFG['repair_usd'] * D['lineage_hops']
    D['cost'] = D['cost_infra'] + D['cost_util'] + D['cost_repair']
    resid = (D['cross_border'] == 1) & (D['adequacy_agreement'] == 0) & (D['pii_field_count'] > 1) & (D['tr'] <= 1)
    D['residency_ok'] = ~resid
    if policy:
        D['feas'] = (D['p'] <= CFG['tau']) & (D['reid'] <= CFG['rho']) & D['residency_ok']
    else:
        D['feas'] = True
    return D

def prune(D):
    """Keep feasible options that are Pareto-optimal in (cost, p) within each (op, cloud). Exact for the MILP."""
    F = D[D['feas']].sort_values(['op', 'cloud', 'cost', 'p'])
    F = F[F.groupby(['op', 'cloud'])['p'].cummin().eq(F['p']) &
          (F.groupby(['op', 'cloud'])['p'].transform(lambda s: s.cummin().shift(fill_value=np.inf)) > F['p'])]
    return F

def anchor_options(W, M):
    rows = []
    for i, (_, r) in enumerate(W.iterrows()):
        lst = ANCH_PUB if r['dataset_type'] in PUBLIC_TYPES else ANCH_PERM
        for (l, b) in lst: rows.append((i, l, b, r['lineage_hops']))
    A = pd.DataFrame(rows, columns=['op', 'ledger', 'batch', 'hops'])
    A['lat'] = M.latency(A['ledger'].values, A['batch'].values, A['hops'].values)
    A['gas'] = M.gas_units(A['ledger'].values, A['batch'].values, A['hops'].values)
    A['cost'] = A['gas'] * CFG['gas_usd'] + np.where(A['ledger'] == 'permissioned_ledger', CFG['perm_tx_usd'] / A['batch'], 0)
    return A

def budgets(W, A, M):
    N = len(W)
    vol = (W['records_transferred'] * W['dataset_type'].map(REC_KB) / 1e6).sum()
    cap = {c: CFG['kappa'] * vol / 3 for c in CLOUD_PRICE}
    pub = A[(A['ledger'] == 'public_l2_rollup') & (A['batch'] == 1)]
    G = CFG['gas_frac'] * pub['gas'].sum()
    return dict(cap=cap, V=CFG['v_bar'] * N, G=G, L=CFG['lat_bar'] * N, N=N)

# ------------------------------------------------------------------ 3. solvers
def solve_milp(F, A, B, N, use_V=True, use_cap=True, use_anchor=True, time_limit=60):
    """x_io (processing options incl. reject), y_ia (anchoring). Returns chosen rows and stats."""
    F = F.reset_index(drop=True)
    nx, ny = len(F), len(A)
    nv = nx + N + ny            # + reject var per op
    c = np.concatenate([F['cost'].values, np.full(N, CFG['reject_cost']), A['cost'].values])
    rows, cols, vals, lo, hi = [], [], [], [], []
    r = 0
    # assignment
    for i, grp in F.groupby('op').groups.items():
        pass
    opx = F['op'].values
    rows += list(opx); cols += list(range(nx)); vals += [1] * nx
    rows += list(range(N)); cols += list(range(nx, nx + N)); vals += [1] * N
    lo += [1] * N; hi += [1] * N; r = N
    oya = A['op'].values
    rows += list(r + oya); cols += list(range(nx + N, nv)); vals += [1] * ny
    lo += [1] * N; hi += [1] * N; r += N
    if use_cap:
        for cl, capv in B['cap'].items():
            idx = np.where(F['cloud'].values == cl)[0]
            rows += [r] * len(idx); cols += list(idx); vals += list(F['gb'].values[idx]); lo.append(-np.inf); hi.append(capv); r += 1
    if use_V:
        rows += [r] * nx; cols += list(range(nx)); vals += list(F['p'].values); lo.append(-np.inf); hi.append(B['V']); r += 1
    if use_anchor:
        rows += [r] * ny; cols += list(range(nx + N, nv)); vals += list(A['gas'].values); lo.append(-np.inf); hi.append(B['G']); r += 1
        rows += [r] * ny; cols += list(range(nx + N, nv)); vals += list(A['lat'].values); lo.append(-np.inf); hi.append(B['L']); r += 1
    Acon = coo_matrix((vals, (rows, cols)), shape=(r, nv)).tocsr()
    t0 = time.time()
    res = milp(c, constraints=LinearConstraint(Acon, lo, hi), integrality=np.ones(nv), bounds=Bounds(0, 1),
               options=dict(time_limit=time_limit, mip_rel_gap=1e-4, disp=False))
    dt = time.time() - t0
    if res.x is None: raise RuntimeError('MILP failed: ' + res.message)
    x = res.x > 0.5
    selF = F[x[:nx]]; rej = np.where(x[nx:nx + N])[0]; selA = A[x[nx + N:]]
    gap = float(getattr(res, 'mip_gap', 0) or 0)
    return selF, rej, selA, dict(time=dt, nvars=nv, ncons=r, gap=gap, obj=float(res.fun))

def greedy(F, A, B, N, order=None, respect_global=True):
    order = np.arange(N) if order is None else order
    capl = dict(B['cap']); V = B['V']; G = B['G']; L = B['L']
    groups = {i: g.sort_values('cost') for i, g in F.groupby('op')}
    agroups = {i: g.sort_values('cost') for i, g in A.groupby('op')}
    selF, rej, selA = [], [], []
    for k, i in enumerate(order):
        chosen = None
        for _, o in (groups[i].iterrows() if i in groups else []):
            if o['gb'] > capl[o['cloud']] + 1e-12: continue
            if respect_global and o['p'] > V + 1e-12: continue
            chosen = o; break
        if chosen is None: rej.append(i)
        else:
            capl[chosen['cloud']] -= chosen['gb']; V -= chosen['p']; selF.append(chosen)
        ag = agroups[i]; remaining = N - k - 1
        minlat_rest = 0.0
        ok = None
        for _, a in ag.iterrows():
            if respect_global and (a['gas'] > G + 1e-9 or a['lat'] > L - 300 * remaining): continue
            ok = a; break
        if ok is None:   # no option respects the reservation: fall back to fastest within gas, else least gas
            within = ag[ag['gas'] <= G + 1e-9]
            ok = within.sort_values('lat').iloc[0] if len(within) else ag.sort_values('gas').iloc[0]
        G -= ok['gas']; L -= ok['lat']; selA.append(ok)
    return pd.DataFrame(selF), np.array(rej), pd.DataFrame(selA)

def random_feasible(F, A, B, N, rng):
    capl = dict(B['cap']); selF, rej, selA = [], [], []
    for i in range(N):
        g = F[F['op'] == i]
        g = g[g['gb'] <= g['cloud'].map(capl)]
        if len(g) == 0: rej.append(i)
        else:
            o = g.iloc[rng.integers(len(g))]; capl[o['cloud']] -= o['gb']; selF.append(o)
        ag = A[A['op'] == i]; selA.append(ag.iloc[rng.integers(len(ag))])
    return pd.DataFrame(selF), np.array(rej), pd.DataFrame(selA)

# ------------------------------------------------------------------ 4. evaluation
def evaluate(selF, rej, selA, W, M, B, D_all):
    """Auditor-based evaluation of the decisions on the full option frame (incl. infeasible ones)."""
    N = B['N']
    out = {}
    cost = (selF['cost'].sum() if len(selF) else 0) + len(rej) * CFG['reject_cost'] + selA['cost'].sum()
    out['cost'] = float(cost)
    out['cost_proc'] = float(selF['cost'].sum() if len(selF) else 0)
    out['cost_util'] = float(selF['cost_util'].sum() if len(selF) else 0)
    out['cost_anchor'] = float(selA['cost'].sum())
    out['rejected'] = int(len(rej)); out['served_pct'] = 100 * (N - len(rej)) / N
    if len(selF):
        pa = M.p_audit(selF); ra = M.reid_audit(selF)
        out['viol_audit_pct'] = float(100 * pa.sum() / N)        # rejected ops contribute 0 violations
        out['viol_plan_pct'] = float(100 * selF['p'].sum() / N)
        out['reid_mean'] = float(ra.mean()); out['reid_hi_pct'] = float(100 * (ra > CFG['rho']).mean())
        out['residency_breach'] = int((~selF['residency_ok']).sum())
        out['util_loss_gb'] = float((selF['cost_util'] / CFG['lambda_u']).sum() / selF['gb'].sum())
        out['repair_pct'] = float(100 * selF['repair'].mean())
        caps = selF.groupby('cloud')['gb'].sum()
        out['cap_ok'] = bool(all(caps.get(c, 0) <= B['cap'][c] * (1 + 1e-6) for c in B['cap']))
    else:
        for k in ['viol_audit_pct', 'viol_plan_pct', 'reid_mean', 'reid_hi_pct', 'util_loss_gb', 'repair_pct']: out[k] = 0.0
        out['residency_breach'] = 0; out['cap_ok'] = True
    out['lat_mean'] = float(selA['lat'].mean()); out['gas'] = float(selA['gas'].sum())
    out['gas_ok'] = bool(selA['gas'].sum() <= B['G'] * (1 + 1e-6)); out['lat_ok'] = bool(selA['lat'].sum() <= B['L'] * (1 + 1e-6))
    out['budget_ok'] = bool(selF['p'].sum() <= B['V'] * (1 + 1e-6)) if len(selF) else True
    out['all_ok'] = bool(out['cap_ok'] and out['gas_ok'] and out['lat_ok'] and out['budget_ok'] and out['residency_breach'] == 0)
    return out

# ------------------------------------------------------------------ 5. Merkle lineage anchoring
def sha(b): return hashlib.sha256(b).digest()
def merkle_root(leaves):
    lvl = list(leaves)
    while len(lvl) > 1:
        if len(lvl) % 2: lvl.append(lvl[-1])
        lvl = [sha(lvl[i] + lvl[i + 1]) for i in range(0, len(lvl), 2)]
    return lvl[0]
def merkle_proof(leaves, idx):
    lvl = list(leaves); proof = []
    while len(lvl) > 1:
        if len(lvl) % 2: lvl.append(lvl[-1])
        sib = idx ^ 1; proof.append((lvl[sib], sib < idx))
        lvl = [sha(lvl[i] + lvl[i + 1]) for i in range(0, len(lvl), 2)]; idx //= 2
    return proof
def verify(leaf, proof, root):
    h = leaf
    for s, left in proof: h = sha(s + h) if left else sha(h + s)
    return h == root

def lineage_audit(selF, selA, W, rng, tamper_frac=0.05):
    recs = []
    A = selA.set_index('op')
    for _, o in selF.iterrows():
        i = int(o['op']); a = A.loc[i]
        rec = dict(op_id=W.iloc[i]['op_id'], src=f"{W.iloc[i]['source_cloud']}/{W.iloc[i]['source_jurisdiction']}",
                   dst=f"{o['cloud']}/{o['region']}", transform=TR_NAMES[int(o['tr'])], enc=ENCS[int(o['enc'])][0],
                   repair=int(o['repair']), p_viol=round(float(o['p']), 4), reid=round(float(o['reid']), 4),
                   ledger=a['ledger'], batch=int(a['batch']), purpose=W.iloc[i]['processing_purpose'],
                   consent=W.iloc[i]['consent_scope'])
        recs.append(rec)
    t0 = time.time()
    leaves = [sha(json.dumps(r, sort_keys=True).encode()) for r in recs]
    # group by anchoring batch class
    root = merkle_root(leaves); t_build = time.time() - t0
    proofs = [merkle_proof(leaves, i) for i in range(len(leaves))]
    t0 = time.time(); ok = all(verify(leaves[i], proofs[i], root) for i in range(len(leaves))); t_ver = (time.time() - t0) / len(leaves)
    nt = max(1, int(tamper_frac * len(recs))); idx = rng.choice(len(recs), nt, replace=False)
    det = 0
    for i in idx:
        r = dict(recs[i]); r['dst'] = 'cloud_x/ZZ' if r['dst'] != 'cloud_x/ZZ' else 'cloud_y/YY'
        det += int(not verify(sha(json.dumps(r, sort_keys=True).encode()), proofs[i], root))
    return dict(n=len(recs), honest_ok=ok, tampered=nt, detected=det, t_build_ms=1000 * t_build,
                t_verify_us=1e6 * t_ver, proof_hashes=len(proofs[0]) if proofs else 0, root=root.hex()[:16])

# ------------------------------------------------------------------ main experiment
def savefig(fig, name):
    fig.savefig(os.path.join(FIG, name + '.png'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(RES, name + '.jpg'), dpi=300, bbox_inches='tight', pil_kwargs={'quality': 92})
    plt.close(fig)

def fig1_flowchart():
    """Methodology flowchart (Graphviz dot) -> figures/fig1_flowchart.png + results/fig1_flowchart.jpg."""
    import subprocess
    dot = r'''digraph G { rankdir=TB; nodesep=0.15; ranksep=0.16; dpi=300;
    node [shape=box, style="rounded,filled", fillcolor="#f2f2f2", color="#444444", fontname="Times-Roman", fontsize=10, width=2.9, height=0.32, margin="0.06,0.03"];
    edge [color="#444444", arrowsize=0.6];
    a [label="P2P energy-trading integration log (10,000 ops)\nimpute, winsorise; split 40/30/30 %"];
    b [label="Planner models: logistic violation p(i,o),\nlogit-OLS re-id risk r(i,o), anchoring latency/gas"];
    c [label="Option enumeration per op: 9 cloud-regions x 5 transforms\nx 3 encryption modes x lineage repair (270)"];
    d [label="Policy filter: p <= tau, r <= rho, residency rule\n+ exact Pareto pruning (cost, p)"];
    e [label="MILP (HiGHS): min cost + deferral + anchoring\ns.t. assignment, capacity, violation, gas, latency", fillcolor="#f6d5d1", color="#c0392b", penwidth=1.4];
    f [label="Merkle tree over lineage records;\nroot anchored on permissioned ledger / L2 rollup"];
    g [label="Independent auditor (gradient boosting):\nviolation, re-id, cost; tamper-detection check"];
    a->b->c->d->e->f->g; }'''
    png = os.path.join(FIG, 'fig1_flowchart.png')
    subprocess.run(['dot', '-Tpng', '-o', png], input=dot.encode(), check=True)
    from PIL import Image
    Image.open(png).convert('RGB').save(os.path.join(RES, 'fig1_flowchart.jpg'), quality=92)

def main():
    T0 = time.time()
    fig1_flowchart()
    raw = pd.read_csv(DATA)
    df, prep = preprocess(raw); log('preprocess', prep)
    ADQ = adequacy_pairs(df); log('adequacy pairs', sorted(ADQ))
    rng = np.random.default_rng(42)
    perm = rng.permutation(len(df))
    tr = df.iloc[perm[:4000]].reset_index(drop=True)
    au = df.iloc[perm[4000:7000]].reset_index(drop=True)
    pool = df.iloc[perm[7000:]].reset_index(drop=True)
    M = Models().fit(tr, au)
    metrics = {'preprocessing': prep, 'config': CFG}

    # model quality on the held-out pool (true labels)
    y = pool['policy_violation'].values
    pp, pa = M.p_plan(pool), M.p_audit(pool)
    mq = []
    for name, p in [('Planner LR', pp), ('Auditor GBM', pa)]:
        mq.append(dict(model=name, roc_auc=roc_auc_score(y, p), pr_auc=average_precision_score(y, p),
                       f1_at_0_5=f1_score(y, p > 0.5), brier=float(np.mean((p - y) ** 2))))
    yr = pool['reidentification_risk'].values
    for name, p in [('Planner logit-OLS', M.reid_plan(pool)), ('Auditor GBM', M.reid_audit(pool))]:
        mq.append(dict(model=name + ' (re-id)', r2=r2_score(yr, p), mae=mean_absolute_error(yr, p)))
    al = M.latency(pool['ledger_type'], pool['anchor_batch_size'], pool['lineage_hops'])
    mq.append(dict(model='Latency log-OLS', r2=r2_score(np.log(pool['anchor_latency_ms']), np.log(al)),
                   mape=float(np.mean(np.abs(al - pool['anchor_latency_ms']) / pool['anchor_latency_ms']))))
    mqd = pd.DataFrame(mq); mqd.to_csv(os.path.join(RES, 'table_model_quality.csv'), index=False)
    log(mqd.to_string()); metrics['model_quality'] = mqd.to_dict('records')

    # dataset summary table
    summ = pd.DataFrame([
        ('Operations (rows)', len(raw)), ('Columns', raw.shape[1]), ('Violation rate (%)', round(100 * raw.policy_violation.mean(), 2)),
        ('Mean re-id risk', round(raw.reidentification_risk.mean(), 3)), ('Cross-border ops (%)', round(100 * raw.cross_border.mean(), 1)),
        ('Cross-border w/o adequacy (%)', round(100 * (raw.cross_border * (1 - raw.adequacy_agreement)).mean(), 1)),
        ('Median records per op', int(raw.records_transferred.median())), ('Public L2 anchored (%)', round(100 * (raw.ledger_type == 'public_l2_rollup').mean(), 1)),
        ('Missing cells (3 cols)', int(sum(prep['missing_before'].values()))), ('Outliers winsorised', int(sum(prep['outliers_fixed'].values()))),
        ('Planner / auditor / scenario split', '4000 / 3000 / 3000')], columns=['Item', 'Value'])
    summ.to_csv(os.path.join(RES, 'table_dataset_summary.csv'), index=False)

    # ---------------- main comparison
    methods = ['MILP (proposed)', 'Greedy cheapest-compliant', 'Single-cloud', 'Random feasible', 'Unconstrained']
    rows, audits, milp_stats = [], [], []
    keep = {}
    for seed in CFG['seeds']:
        srng = np.random.default_rng(1000 + seed)
        for w in range(CFG['windows_per_seed']):
            W = pool.iloc[srng.choice(len(pool), CFG['N'], replace=False)].reset_index(drop=True)
            D = build_options(W, M, ADQ); F = prune(D); A = anchor_options(W, M); B = budgets(W, A, M)
            N = len(W)
            # proposed
            sF, rj, sA, st = solve_milp(F, A, B, N); milp_stats.append(st)
            rows.append(dict(seed=seed, win=w, method=methods[0], **evaluate(sF, rj, sA, W, M, B, D), solve_s=st['time']))
            audits.append(lineage_audit(sF, sA, W, srng))
            if seed == 0 and w == 0: keep = dict(W=W, D=D, F=F, A=A, B=B, sF=sF, sA=sA)
            # greedy (arrival order)
            t0 = time.time(); gF, grj, gA = greedy(F, A, B, N); tg = time.time() - t0
            rows.append(dict(seed=seed, win=w, method=methods[1], **evaluate(gF, grj, gA, W, M, B, D), solve_s=tg))
            if seed == 0 and w == 0: keep.update(gF=gF, gA=gA)
            # single cloud: best of the three providers, all its regions, no capacity limit
            best = None
            for cl in CLOUD_PRICE:
                Fc = F[F['cloud'] == cl]
                t0 = time.time(); r_ = solve_milp(Fc, A, B, N, use_cap=False); tc = time.time() - t0
                ev = evaluate(r_[0], r_[1], r_[2], W, M, B, D)
                if best is None or ev['cost'] < best[0]['cost']: best = (ev, tc, cl)
            rows.append(dict(seed=seed, win=w, method=methods[2], **best[0], solve_s=best[1], cloud=best[2]))
            # random feasible
            t0 = time.time(); rF, rrj, rA = random_feasible(F, A, B, N, srng); trd = time.time() - t0
            rows.append(dict(seed=seed, win=w, method=methods[3], **evaluate(rF, rrj, rA, W, M, B, D), solve_s=trd))
            # unconstrained (cost-only, capacity and anchoring budgets kept, no privacy policy)
            Du = D.copy(); Du['feas'] = True; Fu = prune(Du)
            uF, urj, uA, ust = solve_milp(Fu, A, B, N, use_V=False)
            rows.append(dict(seed=seed, win=w, method=methods[4], **evaluate(uF, urj, uA, W, M, B, D), solve_s=ust['time']))
            log(f'seed {seed} win {w} done  t={time.time() - T0:.0f}s  milp vars={st["nvars"]} cons={st["ncons"]} {st["time"]:.2f}s')
    R = pd.DataFrame(rows); R.to_csv(os.path.join(RES, 'per_window_results.csv'), index=False)
    cols = ['cost', 'cost_proc', 'cost_anchor', 'rejected', 'viol_audit_pct', 'viol_plan_pct', 'reid_mean', 'reid_hi_pct', 'served_pct', 'residency_breach',
            'util_loss_gb', 'lat_mean', 'gas', 'solve_s', 'all_ok']
    R['all_ok'] = R['all_ok'].astype(float) * 100
    agg = R.groupby('method', sort=False)[cols].agg(['mean', 'std'])
    main_tab = pd.DataFrame({'Method': methods})
    for c in cols:
        main_tab[c + '_mean'] = [agg.loc[m, (c, 'mean')] for m in methods]
        main_tab[c + '_std'] = [agg.loc[m, (c, 'std')] for m in methods]
    main_tab.to_csv(os.path.join(RES, 'table_main_comparison.csv'), index=False)
    log(main_tab.round(3).to_string())
    metrics['main'] = main_tab.round(5).to_dict('records')
    # statistics: paired Wilcoxon MILP vs each baseline (cost and auditor violation)
    stats = []
    mil = R[R.method == methods[0]].sort_values(['seed', 'win'])
    for m in methods[1:]:
        o = R[R.method == m].sort_values(['seed', 'win'])
        for c in ['cost', 'viol_audit_pct']:
            d = o[c].values - mil[c].values
            p = wilcoxon(o[c].values, mil[c].values).pvalue if np.any(d != 0) else 1.0
            stats.append(dict(baseline=m, metric=c, mean_diff=float(d.mean()), wins=int((d > 0).sum()), n=len(d), p_value=float(p)))
    S = pd.DataFrame(stats); S.to_csv(os.path.join(RES, 'table_wilcoxon.csv'), index=False); log(S.to_string())
    metrics['wilcoxon'] = S.to_dict('records')
    MS = pd.DataFrame(milp_stats); metrics['milp_solver'] = dict(time_mean=MS.time.mean(), time_max=MS.time.max(),
                                                                 nvars_mean=MS.nvars.mean(), ncons=int(MS.ncons.mean()), gap_max=MS.gap.max())
    LA = pd.DataFrame(audits); LA.to_csv(os.path.join(RES, 'lineage_audit.csv'), index=False)
    metrics['lineage_audit'] = dict(records=int(LA.n.sum()), honest_verified=bool(LA.honest_ok.all()), tampered=int(LA.tampered.sum()),
                                    detected=int(LA.detected.sum()), build_ms_mean=LA.t_build_ms.mean(),
                                    verify_us_mean=LA.t_verify_us.mean(), proof_hashes=int(LA.proof_hashes.median()))
    log('lineage', metrics['lineage_audit'])
    # pruning statistics
    D0, F0 = keep['D'], keep['F']
    metrics['pruning'] = dict(options_per_op=int(len(D0) / CFG['N']), feasible_frac=float(D0['feas'].mean()),
                              kept_per_op=float(len(F0) / CFG['N']), ops_without_feasible=int(CFG['N'] - F0['op'].nunique()))

    # ---------------- ablation (on all 30 windows)
    abl_cfg = [('Full MILP', {}), ('w/o violation budget V', dict(use_V=False)), ('w/o lineage repair', dict(norepair=True)),
               ('w/o residency rule', dict(noresid=True)), ('w/o re-id cap rho', dict(norho=True)),
               ('w/o anchoring budgets', dict(use_anchor=False))]
    abl = []
    for seed in CFG['seeds']:
        srng = np.random.default_rng(1000 + seed)
        for w in range(CFG['windows_per_seed']):
            W = pool.iloc[srng.choice(len(pool), CFG['N'], replace=False)].reset_index(drop=True)
            A = anchor_options(W, M)
            for name, a in abl_cfg:
                D = build_options(W, M, ADQ, allow_repair=not a.get('norepair'))
                if a.get('noresid'): D['feas'] = (D['p'] <= CFG['tau']) & (D['reid'] <= CFG['rho'])
                if a.get('norho'): D['feas'] = (D['p'] <= CFG['tau']) & D['residency_ok']
                F = prune(D); B = budgets(W, A, M)
                sF, rj, sA, st = solve_milp(F, A, B, len(W), use_V=a.get('use_V', True), use_anchor=a.get('use_anchor', True))
                ev = evaluate(sF, rj, sA, W, M, B, D); ev['all_ok'] = float(ev['all_ok']) * 100
                abl.append(dict(seed=seed, win=w, variant=name, **ev))
    AB = pd.DataFrame(abl)
    abt = AB.groupby('variant', sort=False)[['cost', 'viol_audit_pct', 'reid_mean', 'served_pct', 'residency_breach', 'lat_mean', 'gas', 'all_ok']].agg(['mean', 'std'])
    abt.columns = ['_'.join(c) for c in abt.columns]; abt = abt.reset_index()
    abt.to_csv(os.path.join(RES, 'table_ablation.csv'), index=False); log(abt.round(3).to_string())
    metrics['ablation'] = abt.round(5).to_dict('records')
    log(f'ablation done t={time.time() - T0:.0f}s')

    # ---------------- sensitivity 1: violation budget sweep (Pareto front), 6 windows of seed 0
    sens_w = []
    srng = np.random.default_rng(1000)
    for w in range(CFG['windows_per_seed']):
        sens_w.append(pool.iloc[srng.choice(len(pool), CFG['N'], replace=False)].reset_index(drop=True))
    pareto = []
    base_vbar = CFG['v_bar']
    for vb in [0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.14, 0.20]:
        CFG['v_bar'] = vb
        for w, W in enumerate(sens_w):
            D = build_options(W, M, ADQ); F = prune(D); A = anchor_options(W, M); B = budgets(W, A, M)
            sF, rj, sA, st = solve_milp(F, A, B, len(W))
            ev = evaluate(sF, rj, sA, W, M, B, D); pareto.append(dict(v_bar=vb, win=w, **ev))
    CFG['v_bar'] = base_vbar
    P = pd.DataFrame(pareto); P.to_csv(os.path.join(RES, 'sensitivity_violation_budget.csv'), index=False)
    Pm = P.groupby('v_bar')[['cost', 'viol_audit_pct', 'viol_plan_pct', 'served_pct', 'util_loss_gb', 'repair_pct']].mean().reset_index()
    metrics['sens_vbar'] = Pm.round(4).to_dict('records'); log(Pm.round(3).to_string())

    # ---------------- sensitivity 2: capacity kappa x anchoring-latency budget heatmap (cost), 3 windows
    kap = [0.5, 0.6, 0.7, 0.8, 1.0]; lbs = [1800.0, 2200.0, 2600.0, 3200.0, 4000.0]
    H = np.full((len(kap), len(lbs)), np.nan); Hrej = np.full_like(H, np.nan)
    bk, bl = CFG['kappa'], CFG['lat_bar']
    for a_, k in enumerate(kap):
        for b_, lb in enumerate(lbs):
            CFG['kappa'], CFG['lat_bar'] = k, lb
            cs, rs = [], []
            for W in sens_w[:3]:
                D = build_options(W, M, ADQ); F = prune(D); A = anchor_options(W, M); B = budgets(W, A, M)
                try:
                    sF, rj, sA, st = solve_milp(F, A, B, len(W))
                    ev = evaluate(sF, rj, sA, W, M, B, D); cs.append(ev['cost']); rs.append(ev['rejected'])
                except RuntimeError:
                    cs.append(np.nan); rs.append(np.nan)
            if not np.any(np.isnan(cs)): H[a_, b_] = np.mean(cs); Hrej[a_, b_] = np.mean(rs)
    CFG['kappa'], CFG['lat_bar'] = bk, bl
    pd.DataFrame(H, index=[f'kappa={k}' for k in kap], columns=[f'Lbar={l:.0f}' for l in lbs]).to_csv(os.path.join(RES, 'sensitivity_capacity_latency_cost.csv'))
    metrics['sens_heat'] = dict(kappa=kap, lat_bar=lbs, cost=np.round(H, 3).tolist(), rejected=np.round(Hrej, 2).tolist())
    log('heat', np.round(H, 2))

    # ---------------- scalability
    scal = []
    for n in [50, 100, 150, 300, 600]:
        for rep in range(3):
            srng = np.random.default_rng(500 + rep)
            W = pool.iloc[srng.choice(len(pool), n, replace=False)].reset_index(drop=True)
            t0 = time.time(); D = build_options(W, M, ADQ); F = prune(D); A = anchor_options(W, M); B = budgets(W, A, M); tb = time.time() - t0
            sF, rj, sA, st = solve_milp(F, A, B, n, time_limit=120)
            t0 = time.time(); greedy(F, A, B, n); tg = time.time() - t0
            scal.append(dict(N=n, rep=rep, build_s=tb, milp_s=st['time'], greedy_s=tg, nvars=st['nvars'], gap=st['gap']))
    SC = pd.DataFrame(scal); SC.to_csv(os.path.join(RES, 'scalability.csv'), index=False)
    SCm = SC.groupby('N').mean().reset_index(); metrics['scalability'] = SCm.round(4).to_dict('records'); log(SCm.round(3).to_string())

    # ---------------- anchoring trade-off from the data
    anc = df.groupby(['ledger_type', 'anchor_batch_size']).agg(lat_med=('anchor_latency_ms', 'median'),
                                                               lat_q25=('anchor_latency_ms', lambda s: s.quantile(.25)),
                                                               lat_q75=('anchor_latency_ms', lambda s: s.quantile(.75)),
                                                               gas_mean=('anchor_gas_units', 'mean')).reset_index()
    anc.to_csv(os.path.join(RES, 'anchoring_tradeoff.csv'), index=False); metrics['anchoring'] = anc.round(2).to_dict('records')

    # ---------------- hyper-parameter / setup table
    hp = pd.DataFrame([
        ('Window size N / windows', f"{CFG['N']} ops / 5 seeds x 6"), ('Per-op violation cap tau', CFG['tau']),
        ('Window violation budget v_bar', CFG['v_bar']), ('Re-id cap rho', CFG['rho']),
        ('Capacity factor kappa', CFG['kappa']), ('Mean anchoring latency L_bar (ms)', CFG['lat_bar']),
        ('Gas budget fraction', CFG['gas_frac']), ('Deferral penalty (USD/op)', CFG['reject_cost']),
        ('Utility-loss weight lambda_u (USD/GB)', CFG['lambda_u']), ('Gas price (USD/gas)', CFG['gas_usd']),
        ('Options per op', '9 sites x 5 transforms x 3 enc x 2 repair'), ('Anchoring options', 'perm {1,64}; L2 {1,16,64,256}'),
        ('MILP solver', 'HiGHS (SciPy milp), rel. gap 1e-4, 60 s'), ('Planner model', 'L2-logistic, C=1'),
        ('Auditor model', 'HistGB, 300 trees, lr 0.05')], columns=['Parameter', 'Value'])
    hp.to_csv(os.path.join(RES, 'table_setup.csv'), index=False)

    # ================= figures
    # Fig 2: violation-budget sweep: planner/auditor violation and lineage-repair share vs v_bar
    fig, ax = plt.subplots(figsize=(3.3, 2.3))
    Ps = P.groupby('v_bar')[['viol_audit_pct', 'viol_plan_pct', 'repair_pct', 'cost']].agg(['mean', 'std'])
    xv = np.array(Ps.index) * 100
    ax.errorbar(xv, Ps[('viol_audit_pct', 'mean')], yerr=Ps[('viol_audit_pct', 'std')], color=ACC, marker='o', ms=3, lw=1.2, capsize=1.5, label='Auditor violation')
    ax.plot(xv, Ps[('viol_plan_pct', 'mean')], 's--', color=GREYS[0], ms=3, lw=1, label='Planner violation')
    ax.plot(xv, xv, ':', color=GREYS[2], lw=0.8, label='Budget $\\bar v$')
    ax.set_xlabel('Window violation budget $\\bar v$ (%)'); ax.set_ylabel('Expected violation rate (%)')
    ax.set_ylim(0, 11)
    ax2 = ax.twinx(); ax2.plot(xv, Ps[('repair_pct', 'mean')], '^-.', color=GREYS[1], ms=3, lw=1, label='Lineage repair (right)')
    ax2.set_ylabel('Ops with lineage repair (%)'); ax2.spines['right'].set_visible(True); ax2.set_ylim(0, 16.5)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, loc='upper right', ncol=2, fontsize=6.3, columnspacing=0.8); ax.grid(alpha=0.25, lw=0.5)
    savefig(fig, 'fig3_budget_sweep')

    # Fig 3: bars of cost and violation with error bars
    fig, axs = plt.subplots(1, 2, figsize=(3.3, 2.1))
    short = ['MILP', 'Greedy', 'Single', 'Random', 'Uncon.']
    for ax, c, lab in [(axs[0], 'cost', 'Window cost (USD)'), (axs[1], 'viol_audit_pct', 'Violation rate (%)')]:
        mu = [R[R.method == m][c].mean() for m in methods]; sd = [R[R.method == m][c].std() for m in methods]
        ax.bar(range(5), mu, yerr=sd, color=[ACC] + GREYS, capsize=2, width=0.7, error_kw=dict(lw=0.7))
        ax.set_xticks(range(5)); ax.set_xticklabels(short, rotation=45, ha='right'); ax.set_ylabel(lab)
        ax.grid(axis='y', alpha=0.25, lw=0.5)
    fig.tight_layout(w_pad=1.0); savefig(fig, 'fig2_bars')

    # Fig 4: heatmap (capacity x latency budget)
    fig, ax = plt.subplots(figsize=(3.1, 2.3))
    Hm = np.ma.masked_invalid(H)
    cmap = plt.get_cmap('Greys').copy(); cmap.set_bad('#f6d5d1')
    im = ax.imshow(Hm, cmap=cmap, aspect='auto', origin='lower')
    thr = np.nanmean(H)
    for a_ in range(len(kap)):
        for b_ in range(len(lbs)):
            v = H[a_, b_]
            ax.text(b_, a_, 'infeas.' if np.isnan(v) else f'{v:.1f}', ha='center', va='center', fontsize=6.5,
                    color=ACC if np.isnan(v) else ('white' if v > thr else 'black'))
    ax.set_xticks(range(len(lbs))); ax.set_xticklabels([f'{l / 1000:.1f}' for l in lbs]); ax.set_yticks(range(len(kap))); ax.set_yticklabels(kap)
    ax.set_xlabel('Mean anchoring-latency budget $\\bar L$ (s)'); ax.set_ylabel('Capacity factor $\\kappa$')
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03); cb.set_label('Window cost (USD)')
    savefig(fig, 'fig4_heatmap')

    # Fig 5: anchoring trade-off (data) + MILP batch choice
    fig, ax = plt.subplots(figsize=(3.3, 2.3))
    for led, mkr, col, lab in [('permissioned_ledger', 'o', GREYS[0], 'Permissioned'), ('public_l2_rollup', 's', ACC, 'Public L2 rollup')]:
        s = anc[anc.ledger_type == led]
        ax.errorbar(s.anchor_batch_size, s.lat_med / 1000, yerr=[(s.lat_med - s.lat_q25) / 1000, (s.lat_q75 - s.lat_med) / 1000],
                    marker=mkr, color=col, ms=3.5, capsize=2, lw=1.1, label=f'{lab} latency')
    ax.set_xscale('log', base=2); ax.set_xlabel('Anchor batch size (records/hash)'); ax.set_ylabel('Anchoring latency (s)')
    ax2 = ax.twinx(); s = anc[anc.ledger_type == 'public_l2_rollup']
    ax2.plot(s.anchor_batch_size, s.gas_mean / 1000, '--', color=ACC, marker='x', ms=4, lw=1, label='L2 gas per op')
    ax2.set_ylabel('Amortised gas (k units/op)'); ax2.spines['right'].set_visible(True)
    sA = pd.concat([r for r in [keep['sA']]])
    cnt = sA[sA.ledger == 'public_l2_rollup'].batch.value_counts().reindex([1, 16, 64, 256]).fillna(0)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, loc='upper center', fontsize=6.5); ax.grid(alpha=0.25, lw=0.5)
    savefig(fig, 'fig5_anchoring')
    metrics['fig5_milp_batch_counts_win0'] = {int(k): int(v) for k, v in cnt.items()}

    # Fig 6: decision composition MILP vs greedy (transform mix + cloud load) for window 0
    fig, axs = plt.subplots(1, 2, figsize=(3.3, 2.0))
    sF, gF, B0 = keep['sF'], keep['gF'], keep['B']
    trm = pd.DataFrame({'MILP': sF.tr.value_counts().reindex(range(5)).fillna(0).values,
                        'Greedy': gF.tr.value_counts().reindex(range(5)).fillna(0).values}, index=TR_NAMES)
    trm.plot.barh(ax=axs[0], color=[ACC, GREYS[2]], width=0.75, legend=True)
    axs[0].set_xlabel('Operations'); axs[0].legend(frameon=False, fontsize=6); axs[0].invert_yaxis()
    cl = list(CLOUD_PRICE)
    um = [sF[sF.cloud == c].gb.sum() / B0['cap'][c] * 100 for c in cl]; ug = [gF[gF.cloud == c].gb.sum() / B0['cap'][c] * 100 for c in cl]
    xx = np.arange(3); axs[1].bar(xx - 0.18, um, 0.36, color=ACC, label='MILP'); axs[1].bar(xx + 0.18, ug, 0.36, color=GREYS[2], label='Greedy')
    axs[1].axhline(100, color='k', lw=0.7, ls=':'); axs[1].set_xticks(xx); axs[1].set_xticklabels(['A', 'B', 'C'])
    axs[1].set_xlabel('Cloud'); axs[1].set_ylabel('Capacity used (%)')
    fig.tight_layout(w_pad=0.8); savefig(fig, 'fig6_decisions')
    metrics['fig6'] = dict(transform_mix=trm.to_dict(), cap_used_milp=um, cap_used_greedy=ug)

    # Fig 7: scalability
    fig, ax = plt.subplots(figsize=(3.3, 2.0))
    ax.plot(SCm.N, SCm.milp_s, 'o-', color=ACC, ms=3, label='MILP solve')
    ax.plot(SCm.N, SCm.build_s, 's--', color=GREYS[1], ms=3, label='Option build + pruning')
    ax.plot(SCm.N, SCm.greedy_s, '^:', color=GREYS[2], ms=3, label='Greedy')
    ax.set_xlabel('Operations per window N'); ax.set_ylabel('Wall-clock time (s)'); ax.legend(frameon=False, loc='upper left'); ax.grid(alpha=0.25, lw=0.5)
    savefig(fig, 'fig7_scalability')

    metrics['runtime_s'] = time.time() - T0
    json.dump(metrics, open(os.path.join(RES, 'metrics.json'), 'w'), indent=1, default=lambda o: o if not isinstance(o, (np.generic,)) else o.item())
    open(os.path.join(RES, 'run_log.txt'), 'w').write('\n'.join(LOG))
    log(f'TOTAL {time.time() - T0:.0f}s')

if __name__ == '__main__':
    main()
