#!/usr/bin/env python3
"""A1P1 - Workload-aware cost/latency optimisation of hybrid batch-streaming pipelines (smart-grid telemetry).

Runs everything end-to-end:
  1. preprocessing (median imputation, robust outlier repair, feature engineering)
  2. ML surrogates (LightGBM vs Ridge vs kNN) for log p95 latency and log cost, train 2025 / test 2026 H1
  3. proposed method: surrogate-assisted NSGA-II with chance-constrained SLA risk (P[p95 > SLA] <= alpha)
  4. baselines: static batch, static streaming, weighted-sum random search, greedy; plus as-operated logs
  5. ablation (no SLA constraint, no uncertainty margin, no load feature, linear surrogate) and sensitivity (alpha)
  6. 5 seeds, mean +- std, Wilcoxon signed-rank test
  7. all CSV tables, metrics.json and figures (PNG 300 dpi -> ../figures, JPG -> ../results)
All decisions are scored by a counterfactual oracle (oracle.py) that the optimisers never see.
"""
import os, json, time, warnings, subprocess
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score, roc_auc_score, average_precision_score
from scipy.stats import norm, wilcoxon
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import oracle

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data", "A1P1_batch_stream_costlat.csv")
RES = os.path.join(HERE, "..", "results")
FIG = os.path.join(HERE, "..", "figures")
os.makedirs(RES, exist_ok=True); os.makedirs(FIG, exist_ok=True)
SEEDS = [0, 1, 2, 3, 4]
N_WL = 200            # test workloads optimised
POP, GEN = 40, 50     # NSGA-II budget = POP*(GEN+1) surrogate evaluations per workload
ALPHA = 0.05          # SLA risk budget
T_START = time.time()

plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
                     "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7, "figure.dpi": 100,
                     "axes.spines.top": False, "axes.spines.right": False})
ACC = "#c0392b"; GREYS = ["#222222", "#555555", "#888888", "#aaaaaa", "#cccccc"]


def savefig(fig, name):
    fig.savefig(os.path.join(FIG, name + ".png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(RES, name + ".jpg"), dpi=200, bbox_inches="tight", pil_kwargs={"quality": 92})
    plt.close(fig)


# ------------------------------------------------------------------ 1. preprocessing
df = pd.read_csv(DATA, parse_dates=["window_start"])
raw_missing = df[["avg_event_size_kb", "burstiness_cv", "late_event_pct"]].isna().sum().to_dict()
train_mask = df.window_start < "2026-01-01"
# robust outlier repair on the two injected columns: |log x - median_src| > 4.5 * MAD_src -> NaN
n_out = {}
for c in ["ingest_events_per_sec", "avg_event_size_kb"]:
    lx = np.log(df[c])
    med = lx[train_mask].groupby(df.data_source[train_mask]).median()
    mad = (lx[train_mask] - df.data_source[train_mask].map(med)).abs().groupby(df.data_source[train_mask]).median()
    z = (lx - df.data_source.map(med)).abs() / (1.4826 * df.data_source.map(mad))
    thr = 4.5 if c == "avg_event_size_kb" else 6.0  # rate is legitimately heavy-tailed (storms x6)
    bad = z > thr
    n_out[c] = int(bad.sum()); df.loc[bad, c] = np.nan
# median imputation per data source (fitted on training period only)
for c in ["ingest_events_per_sec", "avg_event_size_kb", "burstiness_cv", "late_event_pct"]:
    med = df[train_mask].groupby("data_source")[c].median()
    df[c] = df[c].fillna(df.data_source.map(med))

CAT = ["data_source", "processing_mode", "engine", "cloud_region", "storage_tier"]
for c in CAT:
    df[c] = df[c].astype("category")
CATS = {c: list(df[c].cat.categories) for c in CAT}


def features(d, load_feat=True):
    """Workload + configuration features (no targets, no sla_violation)."""
    X = pd.DataFrame({
        "log_rate": np.log(d["ingest_events_per_sec"].astype(float)),
        "event_kb": d["avg_event_size_kb"].astype(float), "burst": d["burstiness_cv"].astype(float),
        "late": d["late_event_pct"].astype(float), "peak": d["grid_peak_hour"].astype(float),
        "storm": d["storm_alert_active"].astype(float),
        "window": d["window_size_sec"].astype(float), "ckpt": d["checkpoint_interval_sec"].astype(float),
        "execs": d["num_executors"].astype(float), "vcpu": d["executor_vcpu"].astype(float),
        "mem": d["executor_mem_gb"].astype(float), "spot": d["spot_fraction"].astype(float)})
    for c in CAT:
        X[c] = pd.Categorical(d[c].astype(str), categories=CATS[c])
    if load_feat:  # workload-aware load ratio: offered load per unit of provisioned compute
        X["log_load"] = X.log_rate + np.log1p(0.5 * X.burst) + np.log(0.6 + X.event_kb) - np.log(X.execs * X.vcpu)
        X["log_cores"] = np.log(X.execs * X.vcpu)
    return X


def onehot(X):
    return pd.get_dummies(X, columns=CAT, dtype=float)


tr, te = df[train_mask].reset_index(drop=True), df[~train_mask].reset_index(drop=True)
y_lat_tr, y_lat_te = np.log(tr.p95_latency_ms.values), np.log(te.p95_latency_ms.values)
y_cost_tr, y_cost_te = np.log(tr.cost_usd_per_hour.values), np.log(te.cost_usd_per_hour.values)
ds_summary = pd.DataFrame({
    "item": ["rows total", "train rows (2025)", "test rows (2026 H1)", "pipelines", "data sources",
             "SLA violation rate (%)", "median p95 latency (s)", "median cost (USD/h)", "storm windows (%)",
             "missing values imputed", "outliers repaired"],
    "value": [len(df), len(tr), len(te), df.pipeline_id.nunique(), df.data_source.nunique(),
              round(100 * df.sla_violation.mean(), 2), round(df.p95_latency_ms.median() / 1000, 2),
              round(df.cost_usd_per_hour.median(), 2), round(100 * df.storm_alert_active.mean(), 2),
              int(sum(raw_missing.values())), int(sum(n_out.values()))]})
ds_summary.to_csv(os.path.join(RES, "table_dataset_summary.csv"), index=False)
per_src = df.groupby("data_source", observed=True).agg(rows=("p95_latency_ms", "size"),
                                                       rate_med=("ingest_events_per_sec", "median"),
                                                       sla_med_s=("sla_latency_ms", lambda s: s.median() / 1000),
                                                       viol_pct=("sla_violation", lambda s: 100 * s.mean()),
                                                       cost_med=("cost_usd_per_hour", "median")).round(2)
per_src.to_csv(os.path.join(RES, "table_dataset_per_source.csv"))
print("preprocessing done", raw_missing, n_out)

# ------------------------------------------------------------------ 2. surrogates
LGB_P = dict(n_estimators=600, learning_rate=0.04, num_leaves=31, min_child_samples=20, subsample=0.8,
             subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, verbose=-1, n_jobs=1)


class Surrogate:
    """Latency + cost surrogate with a Gaussian residual model for SLA-risk estimation."""

    def __init__(self, kind="lgb", seed=0, load_feat=True):
        self.kind, self.seed, self.load_feat = kind, seed, load_feat

    def _model(self):
        if self.kind == "lgb":
            return lgb.LGBMRegressor(random_state=self.seed, **LGB_P)
        if self.kind == "ridge":
            return make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        return make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=15, weights="distance"))

    def _X(self, d):
        X = features(d, self.load_feat)
        return X if self.kind == "lgb" else onehot(X)

    def fit(self, d, ylat, ycost):
        X = self._X(d)
        self.cols = list(X.columns)
        # out-of-fold residual std (time-ordered 4 folds) for the latency risk model
        idx = np.arange(len(d)); folds = np.array_split(idx, 4); res = []
        for k in range(1, 4):
            trn = np.concatenate(folds[:k]); m = self._model().fit(X.iloc[trn], ylat[trn])
            res.append(ylat[folds[k]] - m.predict(X.iloc[folds[k]]))
        res = np.concatenate(res); self.sig_lat = float(np.std(res))
        self.m_lat = self._model().fit(X, ylat)
        self.m_cost = self._model().fit(X, ycost)
        self.sig_cost = float(np.std(ycost - self.m_cost.predict(X)))
        return self

    def predict(self, d):
        X = self._X(d)
        if self.kind != "lgb":
            X = X.reindex(columns=self.cols, fill_value=0.0)
        mu_l = self.m_lat.predict(X); mu_c = self.m_cost.predict(X)
        return mu_l, np.exp(mu_c + 0.5 * self.sig_cost ** 2)

    def p_viol(self, mu_l, sla, use_unc=True):
        if use_unc:
            return 1 - norm.cdf((np.log(sla) - mu_l) / self.sig_lat)
        return (mu_l > np.log(sla)).astype(float)


def mape(y, p):
    return float(100 * np.mean(np.abs(np.exp(p) - np.exp(y)) / np.exp(y)))


sur_rows = []
storm_te = te.storm_alert_active.values == 1
for kind in ["lgb", "ridge", "knn"]:
    s = Surrogate(kind, 0).fit(tr, y_lat_tr, y_cost_tr)
    mu_l, ec = s.predict(te)
    pv = s.p_viol(mu_l, te.sla_latency_ms.values)
    yv = te.sla_violation.values
    pred_pos = pv > 0.5
    tp = (pred_pos & (yv == 1)).sum(); f1 = 2 * tp / (pred_pos.sum() + yv.sum())
    sur_rows.append({"surrogate": kind, "lat_R2_log": r2_score(y_lat_te, mu_l), "lat_MAPE_pct": mape(y_lat_te, mu_l),
                     "lat_R2_log_storm": r2_score(y_lat_te[storm_te], mu_l[storm_te]),
                     "cost_R2_log": r2_score(y_cost_te, np.log(ec) - 0.5 * s.sig_cost ** 2),
                     "cost_MAPE_pct": mape(y_cost_te, np.log(ec) - 0.5 * s.sig_cost ** 2),
                     "viol_ROC_AUC": roc_auc_score(yv, pv), "viol_PR_AUC": average_precision_score(yv, pv),
                     "viol_F1": f1, "sigma_lat": s.sig_lat})
    if kind == "lgb":
        parity = (y_lat_te, mu_l)
        imp = pd.Series(s.m_lat.booster_.feature_importance("gain"), index=s.cols).sort_values(ascending=False)
sur_df = pd.DataFrame(sur_rows).round(4)
sur_df.to_csv(os.path.join(RES, "table_surrogate_accuracy.csv"), index=False)
imp.to_csv(os.path.join(RES, "latency_surrogate_importance.csv"), header=["gain"])
print(sur_df.to_string())

# ------------------------------------------------------------------ 3. decision space
MODES = [("batch", "spark_batch"), ("batch", "dbt_on_warehouse"), ("micro_batch", "spark_structured_streaming"),
         ("streaming", "flink"), ("streaming", "spark_structured_streaming")]
WIN = {"batch": [300, 900, 1800, 3600], "micro_batch": [5, 10, 30, 60, 120], "streaming": [0]}
CKPT = [10, 30, 60]; VCPU = [2, 4, 8]; MEMR = [4, 8]; TIERS = ["hot_object", "standard_object", "infrequent_object"]
NG = 8  # genes: mode/engine, window, ckpt, execs, vcpu, mem ratio, spot, tier  (all in [0,1])


def decode(G):
    """G (..., 8) in [0,1] -> dict of flat config arrays."""
    G = np.clip(G.reshape(-1, NG), 0, 1 - 1e-9)
    mi = (G[:, 0] * 5).astype(int)
    mode = np.array([MODES[i][0] for i in mi], dtype=object); eng = np.array([MODES[i][1] for i in mi], dtype=object)
    win = np.array([WIN[m][int(g * len(WIN[m]))] for m, g in zip(mode, G[:, 1])], dtype=float)
    ck = np.where(mode == "batch", 0, np.array(CKPT)[(G[:, 2] * 3).astype(int)]).astype(float)
    execs = 2 + np.round(G[:, 3] * 38).astype(int)
    vcpu = np.array(VCPU)[(G[:, 4] * 3).astype(int)]
    mem = vcpu * np.array(MEMR)[(G[:, 5] * 2).astype(int)]
    spot = np.round(G[:, 6] * 0.99, 2)
    tier = np.array(TIERS, dtype=object)[(G[:, 7] * 3).astype(int)]
    return dict(mode=mode, engine=eng, window=win, ckpt=ck, execs=execs, vcpu=vcpu, mem=mem, spot=spot, tier=tier)


def cfg_frame(wl, cfg):
    """Build a dataset-schema frame: workload rows (repeated) + config columns."""
    d = wl.copy()
    d["processing_mode"] = cfg["mode"]; d["engine"] = cfg["engine"]; d["window_size_sec"] = cfg["window"]
    d["checkpoint_interval_sec"] = cfg["ckpt"]; d["num_executors"] = cfg["execs"]; d["executor_vcpu"] = cfg["vcpu"]
    d["executor_mem_gb"] = cfg["mem"]; d["spot_fraction"] = cfg["spot"]; d["storage_tier"] = cfg["tier"]
    return d


def wl_oracle(wl):
    return dict(rate=wl.ingest_events_per_sec.values.astype(float), kb=wl.avg_event_size_kb.values.astype(float),
                burst=wl.burstiness_cv.values.astype(float), late=wl.late_event_pct.values.astype(float),
                peak=wl.grid_peak_hour.values.astype(float), storm=wl.storm_alert_active.values.astype(float),
                region=wl.cloud_region.astype(str).values, sla=wl.sla_latency_ms.values.astype(float))


rs = np.random.default_rng(2026)
WL = te.iloc[np.sort(rs.choice(len(te), N_WL, replace=False))].reset_index(drop=True)
WCOLS = ["data_source", "cloud_region", "ingest_events_per_sec", "avg_event_size_kb", "burstiness_cv",
         "late_event_pct", "grid_peak_hour", "storm_alert_active", "sla_latency_ms"]
WLW = WL[WCOLS]
SLA = WL.sla_latency_ms.values.astype(float)
LOGGED_CFG = dict(mode=WL.processing_mode.astype(str).values, engine=WL.engine.astype(str).values,
                  window=WL.window_size_sec.values.astype(float), ckpt=WL.checkpoint_interval_sec.values.astype(float),
                  execs=WL.num_executors.values, vcpu=WL.executor_vcpu.values, mem=WL.executor_mem_gb.values,
                  spot=WL.spot_fraction.values, tier=WL.storage_tier.astype(str).values)


def sur_eval(sur, G, use_unc=True):
    """Evaluate genomes G (W,P,8) with the surrogate -> cost (W,P), lat ratio log (W,P), pviol (W,P)."""
    W, P, _ = G.shape
    cfg = decode(G)
    wl = WLW.loc[np.repeat(np.arange(W), P)].reset_index(drop=True)
    mu_l, ec = sur.predict(cfg_frame(wl, cfg))
    sla = np.repeat(SLA, P)
    pv = sur.p_viol(mu_l, sla, use_unc)
    return ec.reshape(W, P), (mu_l - np.log(sla)).reshape(W, P), pv.reshape(W, P)


def ora_eval(cfg, idx=None):
    w = wl_oracle(WL if idx is None else WL.iloc[idx])
    return oracle.evaluate(w, cfg)


# ------------------------------------------------------------------ NSGA-II machinery
def nd_sort(F, cv):
    """Constrained non-dominated sorting (Deb). F (n,2), cv (n,) >=0 violation. Returns rank array."""
    n = len(F)
    feas = cv <= 0
    better = np.zeros((n, n), bool)
    le = (F[:, None, :] <= F[None, :, :]).all(2) & (F[:, None, :] < F[None, :, :]).any(2)
    better = (feas[:, None] & feas[None, :] & le) | (feas[:, None] & ~feas[None, :]) | \
             (~feas[:, None] & ~feas[None, :] & (cv[:, None] < cv[None, :]))
    dom_cnt = better.sum(0); rank = np.full(n, -1); r = 0; cur = np.where(dom_cnt == 0)[0]
    while len(cur):
        rank[cur] = r
        dom_cnt = dom_cnt - better[cur].sum(0); dom_cnt[rank >= 0] = -1
        cur = np.where(dom_cnt == 0)[0]; r += 1
    return rank


def crowding(F, rank):
    cd = np.zeros(len(F))
    for r in np.unique(rank):
        ids = np.where(rank == r)[0]
        if len(ids) <= 2:
            cd[ids] = np.inf; continue
        for k in range(F.shape[1]):
            o = ids[np.argsort(F[ids, k])]; span = F[o[-1], k] - F[o[0], k] + 1e-12
            cd[o[0]] = cd[o[-1]] = np.inf
            cd[o[1:-1]] += (F[o[2:], k] - F[o[:-2], k]) / span
    return cd


def sbx_pm(rng, P1, P2, eta_c=15, eta_m=20, pc=0.9):
    u = rng.random(P1.shape)
    beta = np.where(u <= 0.5, (2 * u) ** (1 / (eta_c + 1)), (1 / (2 * (1 - u))) ** (1 / (eta_c + 1)))
    do = (rng.random(P1.shape[:-1] + (1,)) < pc) & (rng.random(P1.shape) < 0.5)
    C1 = np.where(do, 0.5 * ((1 + beta) * P1 + (1 - beta) * P2), P1)
    C2 = np.where(do, 0.5 * ((1 - beta) * P1 + (1 + beta) * P2), P2)
    C = np.concatenate([C1, C2], 1)
    u = rng.random(C.shape); m = rng.random(C.shape) < 1.0 / NG
    dq = np.where(u < 0.5, (2 * u) ** (1 / (eta_m + 1)) - 1, 1 - (2 * (1 - u)) ** (1 / (eta_m + 1)))
    return np.clip(np.where(m, C + dq, C), 0, 1 - 1e-9)


C_REF, L_LO = 40.0, -8.0  # HV normalisation: cost / 40 USD/h, log(p95/SLA) mapped from [-8,0] to [0,1]


def norm_obj(cost, lr):
    return np.stack([np.clip(cost / C_REF, 0, 1), np.clip((lr - L_LO) / -L_LO, 0, 1)], -1)


def hv2d(F):
    """Exact 2-D hypervolume of a minimisation set w.r.t. reference (1,1)."""
    if len(F) == 0:
        return 0.0
    F = F[np.argsort(F[:, 0])]; hv, best = 0.0, 1.0
    for f0, f1 in F:
        if f1 < best:
            hv += (1 - f0) * (best - f1); best = f1
    return hv


def front_hv(cost, lr, feas):
    F = norm_obj(cost[feas], lr[feas])
    return hv2d(F)


def nsga2(sur, seed, alpha=ALPHA, constrained=True, use_unc=True, pop=POP, gen=GEN, track=False):
    rng = np.random.default_rng(seed)
    W = N_WL
    G = rng.random((W, pop, NG))
    c, l, pv = sur_eval(sur, G, use_unc)
    archC, archL, archP, archG = [c], [l], [pv], [G]
    hv_curve = []

    def cvf(pv_):
        return np.maximum(0, pv_ - alpha) if constrained else np.zeros_like(pv_)

    rank = np.zeros((W, pop), int); cd = np.zeros((W, pop))
    for w in range(W):
        F = np.stack([c[w], l[w]], 1); rank[w] = nd_sort(F, cvf(pv[w])); cd[w] = crowding(F, rank[w])
    for g in range(gen):
        if track:
            hv_curve.append(np.mean([front_hv(c[w], l[w], pv[w] <= alpha) for w in range(W)]))
        # binary tournament on (rank, crowding)
        a, b = rng.integers(0, pop, (2, W, pop))
        wi = np.arange(W)[:, None]
        win = np.where((rank[wi, a] < rank[wi, b]) | ((rank[wi, a] == rank[wi, b]) & (cd[wi, a] > cd[wi, b])), a, b)
        par = G[wi, win]
        Q = sbx_pm(rng, par[:, 0::2], par[:, 1::2])
        qc, ql, qp = sur_eval(sur, Q, use_unc)
        archC.append(qc); archL.append(ql); archP.append(qp); archG.append(Q)
        RG = np.concatenate([G, Q], 1); Rc = np.concatenate([c, qc], 1); Rl = np.concatenate([l, ql], 1)
        Rp = np.concatenate([pv, qp], 1)
        nG, nc, nl, npv = np.empty_like(G), np.empty_like(c), np.empty_like(l), np.empty_like(pv)
        for w in range(W):
            F = np.stack([Rc[w], Rl[w]], 1); rk = nd_sort(F, cvf(Rp[w])); cdw = crowding(F, rk)
            sel = np.lexsort((-cdw, rk))[:pop]
            nG[w], nc[w], nl[w], npv[w] = RG[w, sel], Rc[w, sel], Rl[w, sel], Rp[w, sel]
            rank[w], cd[w] = rk[sel], cdw[sel]
        G, c, l, pv = nG, nc, nl, npv
    if track:
        hv_curve.append(np.mean([front_hv(c[w], l[w], pv[w] <= alpha) for w in range(W)]))
    return dict(G=G, c=c, l=l, pv=pv, hv_curve=hv_curve)


def pick_min_cost(G, c, l, pv, alpha=ALPHA):
    """Deployment rule: cheapest predicted-feasible configuration, else the lowest-risk one."""
    feas = pv <= alpha
    score = np.where(feas, c, 1e6 + pv)
    j = score.argmin(1)
    return G[np.arange(len(G)), j]


def oracle_metrics(cfg):
    o = ora_eval(cfg)
    return o, {"cost_usd_h": float(o["exp_cost"].mean()), "viol_pct": float(100 * o["p_viol"].mean()),
               "attain_pct": float(100 * (o["p_viol"] <= ALPHA).mean()),
               "lat_ratio_med": float(np.median(o["p95_nominal"] / SLA))}


def oracle_front_hv(G, c, l, pv, alpha=ALPHA):
    """Oracle-verified HV of the method's predicted-feasible set (points re-scored by the oracle)."""
    W, P, _ = G.shape
    cfg = decode(G)
    o = oracle.evaluate({k: np.repeat(v, P) for k, v in wl_oracle(WL).items()}, cfg)
    oc = o["exp_cost"].reshape(W, P); ol = np.log(o["p95_nominal"].reshape(W, P) / SLA[:, None])
    op = o["p_viol"].reshape(W, P)
    return float(np.mean([front_hv(oc[w], ol[w], (pv[w] <= alpha) & (op[w] <= alpha)) for w in range(W)]))


# ------------------------------------------------------------------ baselines
BUDGET = POP * (GEN + 1)


def static_cfg(mode):
    n = N_WL
    base = dict(execs=np.full(n, 21), vcpu=np.full(n, 4), mem=np.full(n, 16), spot=np.full(n, 0.35),
                tier=np.array(["standard_object"] * n, dtype=object))
    if mode == "batch":
        base.update(mode=np.array(["batch"] * n, dtype=object), engine=np.array(["spark_batch"] * n, dtype=object),
                    window=np.full(n, 300.0), ckpt=np.zeros(n))
    else:
        base.update(mode=np.array(["streaming"] * n, dtype=object), engine=np.array(["flink"] * n, dtype=object),
                    window=np.zeros(n), ckpt=np.full(n, 30.0))
    return base


def random_search(sur, seed, track=False):
    rng = np.random.default_rng(1000 + seed)
    G = rng.random((N_WL, BUDGET, NG))
    c, l, pv = [], [], []
    for k in range(0, BUDGET, 340):
        cc, ll, pp = sur_eval(sur, G[:, k:k + 340]); c.append(cc); l.append(ll); pv.append(pp)
    c, l, pv = np.concatenate(c, 1), np.concatenate(l, 1), np.concatenate(pv, 1)
    # weighted-sum scalarisation (w=0.5) with a penalty on SLA risk above alpha
    cn = c / c.max(1, keepdims=True); ln = (l - l.min(1, keepdims=True)) / (np.ptp(l, 1)[:, None] + 1e-9)
    J = 0.5 * cn + 0.5 * ln + 10 * np.maximum(0, pv - ALPHA)
    pick = G[np.arange(N_WL), J.argmin(1)]
    curve = []
    if track:
        for e in [POP * (g + 1) for g in range(GEN + 1)]:
            curve.append(np.mean([front_hv(c[w, :e], l[w, :e], pv[w, :e] <= ALPHA) for w in range(N_WL)]))
    return dict(G=G, c=c, l=l, pv=pv, pick=pick, hv_curve=curve)


def greedy(sur):
    """Start from the cheapest configuration; repeatedly apply the single-gene move with the best
    risk reduction per added predicted cost until predicted SLA risk <= alpha or budget is used."""
    steps = np.linspace(0, 1 - 1e-9, 9)
    G = np.tile(np.array([0.0, 0.99, 0.99, 0.0, 0.0, 0.0, 0.99, 0.99]), (N_WL, 1))  # batch, long window, min size
    c, l, pv = sur_eval(sur, G[:, None])
    c, l, pv = c[:, 0], l[:, 0], pv[:, 0]; used = 1
    while used < BUDGET:
        active = pv > ALPHA
        if not active.any():
            break
        cand = np.repeat(G[:, None], NG * len(steps), 1)
        for gi in range(NG):
            cand[:, gi * len(steps):(gi + 1) * len(steps), gi] = steps
        cc, cl, cp = sur_eval(sur, cand); used += cand.shape[1]
        gain = (pv[:, None] - cp) / np.maximum(cc - c[:, None], 1e-3)
        gain = np.where(cp < pv[:, None] - 1e-6, gain, -np.inf)
        j = gain.argmax(1); ok = active & np.isfinite(gain.max(1))
        if not ok.any():
            break
        G[ok] = cand[ok, j[ok]]; c[ok] = cc[ok, j[ok]]; l[ok] = cl[ok, j[ok]]; pv[ok] = cp[ok, j[ok]]
    return G, used


# ------------------------------------------------------------------ 4. main comparison over 5 seeds
rows, per_wl_cost, per_wl_viol, curves, store = [], {}, {}, {"NSGA-II": [], "Random search": []}, {}
o_log, m_log = oracle_metrics(LOGGED_CFG)
for name in ["Static batch", "Static streaming"]:
    o, m = oracle_metrics(static_cfg("batch" if "batch" in name else "streaming"))
    for s in SEEDS:
        rows.append(dict(method=name, seed=s, hv_sur=np.nan, hv_oracle=np.nan, evals=0, runtime_s=0.0, **m))
    per_wl_cost[name] = o["exp_cost"]; per_wl_viol[name] = o["p_viol"]
for s in SEEDS:
    print(f"seed {s}  t={time.time() - T_START:.0f}s")
    sur = Surrogate("lgb", s).fit(tr, y_lat_tr, y_cost_tr)
    t0 = time.time(); R = random_search(sur, s, track=True); t_rs = time.time() - t0
    cfg = decode(R["pick"]); o, m = oracle_metrics(cfg)
    hv_s = np.mean([front_hv(R["c"][w], R["l"][w], R["pv"][w] <= ALPHA) for w in range(N_WL)])
    rows.append(dict(method="Weighted-sum random search", seed=s, hv_sur=hv_s, hv_oracle=np.nan, evals=BUDGET,
                     runtime_s=t_rs, **m))
    per_wl_cost.setdefault("Weighted-sum random search", []).append(o["exp_cost"])
    per_wl_viol.setdefault("Weighted-sum random search", []).append(o["p_viol"])
    curves["Random search"].append(R["hv_curve"])
    # random search front verified by oracle (non-dominated feasible subset only, to keep cost low)
    rs_hv_o = []
    for w in range(N_WL):
        f = np.where(R["pv"][w] <= ALPHA)[0]
        if len(f) == 0:
            rs_hv_o.append(0.0); continue
        F = norm_obj(R["c"][w, f], R["l"][w, f]); rk = nd_sort(F, np.zeros(len(f))); f = f[rk == 0]
        cfg_w = decode(R["G"][w, f]); ow = oracle.evaluate({k: np.repeat(v[w:w + 1], len(f)) for k, v in wl_oracle(WL).items()}, cfg_w)
        okk = ow["p_viol"] <= ALPHA
        rs_hv_o.append(hv2d(norm_obj(ow["exp_cost"][okk], np.log(ow["p95_nominal"][okk] / SLA[w]))))
    rows[-1]["hv_oracle"] = float(np.mean(rs_hv_o))
    del R
    t0 = time.time(); Gg, used = greedy(sur); t_g = time.time() - t0
    o, m = oracle_metrics(decode(Gg))
    rows.append(dict(method="Greedy", seed=s, hv_sur=np.nan, hv_oracle=np.nan, evals=used, runtime_s=t_g, **m))
    per_wl_cost.setdefault("Greedy", []).append(o["exp_cost"]); per_wl_viol.setdefault("Greedy", []).append(o["p_viol"])
    t0 = time.time(); N = nsga2(sur, s, track=True); t_n = time.time() - t0
    pick = pick_min_cost(N["G"], N["c"], N["l"], N["pv"]); cfg = decode(pick); o, m = oracle_metrics(cfg)
    rows.append(dict(method="NSGA-II + surrogate (proposed)", seed=s,
                     hv_sur=np.mean([front_hv(N["c"][w], N["l"][w], N["pv"][w] <= ALPHA) for w in range(N_WL)]),
                     hv_oracle=oracle_front_hv(N["G"], N["c"], N["l"], N["pv"]), evals=BUDGET, runtime_s=t_n, **m))
    per_wl_cost.setdefault("NSGA-II + surrogate (proposed)", []).append(o["exp_cost"])
    per_wl_viol.setdefault("NSGA-II + surrogate (proposed)", []).append(o["p_viol"])
    curves["NSGA-II"].append(N["hv_curve"])
    if s == 0:
        store = dict(N=N, sur=sur, cfg=cfg)
for s in SEEDS:
    rows.append(dict(method="As-operated (logged)", seed=s, hv_sur=np.nan, hv_oracle=np.nan, evals=0, runtime_s=0.0, **m_log))
per_wl_cost["As-operated (logged)"] = o_log["exp_cost"]; per_wl_viol["As-operated (logged)"] = o_log["p_viol"]

main = pd.DataFrame(rows)
main.to_csv(os.path.join(RES, "main_comparison_per_seed.csv"), index=False)
ORDER = ["As-operated (logged)", "Static batch", "Static streaming", "Weighted-sum random search", "Greedy",
         "NSGA-II + surrogate (proposed)"]
agg = main.groupby("method").agg(["mean", "std"]).drop(columns=["seed"], level=0).reindex(ORDER)
agg.columns = [f"{a}_{b}" for a, b in agg.columns]
agg = agg.round(4); agg.to_csv(os.path.join(RES, "table_main_comparison.csv"))
print(agg[["cost_usd_h_mean", "cost_usd_h_std", "viol_pct_mean", "attain_pct_mean", "hv_sur_mean", "hv_oracle_mean"]])

# per-workload mean over seeds -> Wilcoxon (proposed vs each adaptive baseline)
pw_c = {k: (np.mean(v, 0) if isinstance(v, list) else v) for k, v in per_wl_cost.items()}
pw_v = {k: (np.mean(v, 0) if isinstance(v, list) else v) for k, v in per_wl_viol.items()}
stats = {}
P = "NSGA-II + surrogate (proposed)"
for b in ["Weighted-sum random search", "Greedy", "Static streaming", "Static batch", "As-operated (logged)"]:
    stats[b] = {"cost_wilcoxon_p": float(wilcoxon(pw_c[P], pw_c[b]).pvalue),
                "viol_wilcoxon_p": float(wilcoxon(pw_v[P], pw_v[b]).pvalue),
                "median_cost_ratio": float(np.median(pw_c[P] / pw_c[b])),
                "cost_saving_pct": float(100 * (1 - pw_c[P].sum() / pw_c[b].sum()))}
pd.DataFrame(stats).T.to_csv(os.path.join(RES, "table_significance.csv"))
pd.DataFrame({k: v for k, v in pw_c.items()}).assign(sla_ms=SLA, data_source=WL.data_source.astype(str).values) \
    .to_csv(os.path.join(RES, "per_workload_cost.csv"), index=False)

# ------------------------------------------------------------------ 5. ablation + sensitivity
abl_rows = []
ABL_SEEDS = [0, 1, 2]
for s in ABL_SEEDS:
    sur = Surrogate("lgb", s).fit(tr, y_lat_tr, y_cost_tr)
    sur_nl = Surrogate("lgb", s, load_feat=False).fit(tr, y_lat_tr, y_cost_tr)
    sur_lin = Surrogate("ridge", s).fit(tr, y_lat_tr, y_cost_tr)
    variants = {"Full (proposed)": (sur, True, True), "No SLA-risk constraint": (sur, False, True),
                "No uncertainty margin": (sur, True, False), "No load-ratio feature": (sur_nl, True, True),
                "Linear (ridge) surrogate": (sur_lin, True, True)}
    for name, (su, con, unc) in variants.items():
        if name == "Full (proposed)":
            r = main[(main.method == P) & (main.seed == s)].iloc[0]
            abl_rows.append(dict(variant=name, seed=s, cost_usd_h=r.cost_usd_h, viol_pct=r.viol_pct,
                                 attain_pct=r.attain_pct, hv_oracle=r.hv_oracle)); continue
        N = nsga2(su, s, constrained=con, use_unc=unc)
        if con:
            pick = pick_min_cost(N["G"], N["c"], N["l"], N["pv"], ALPHA if unc else 0.5)
        else:  # unconstrained front: knee point (closest to utopia in normalised space)
            Fn = norm_obj(N["c"], N["l"])
            d = np.linalg.norm(Fn - Fn.min(1, keepdims=True), axis=2); pick = N["G"][np.arange(N_WL), d.argmin(1)]
        o, m = oracle_metrics(decode(pick))
        hvo = oracle_front_hv(N["G"], N["c"], N["l"], N["pv"], ALPHA if unc else 0.5)
        abl_rows.append(dict(variant=name, seed=s, cost_usd_h=m["cost_usd_h"], viol_pct=m["viol_pct"],
                             attain_pct=m["attain_pct"], hv_oracle=hvo))
    print(f"ablation seed {s} t={time.time() - T_START:.0f}s")
abl = pd.DataFrame(abl_rows)
abl.to_csv(os.path.join(RES, "ablation_per_seed.csv"), index=False)
abl_agg = abl.groupby("variant", sort=False).agg(["mean", "std"]).drop(columns=["seed"], level=0)
abl_agg.columns = [f"{a}_{b}" for a, b in abl_agg.columns]; abl_agg = abl_agg.round(4)
abl_agg.to_csv(os.path.join(RES, "table_ablation.csv"))
print(abl_agg)

sens_rows = []
sur0 = store["sur"]
for a in [0.01, 0.02, 0.05, 0.10, 0.20]:
    for s in [0, 1, 2]:
        N = nsga2(sur0, s, alpha=a)
        pick = pick_min_cost(N["G"], N["c"], N["l"], N["pv"], a)
        o, m = oracle_metrics(decode(pick))
        sens_rows.append(dict(alpha=a, seed=s, cost_usd_h=m["cost_usd_h"], viol_pct=m["viol_pct"],
                              met_alpha_pct=float(100 * (o["p_viol"] <= a).mean())))
sens = pd.DataFrame(sens_rows); sens.to_csv(os.path.join(RES, "sensitivity_alpha_per_seed.csv"), index=False)
sens_agg = sens.groupby("alpha").agg(["mean", "std"]).drop(columns=["seed"], level=0)
sens_agg.columns = [f"{a}_{b}" for a, b in sens_agg.columns]; sens_agg = sens_agg.round(4)
sens_agg.to_csv(os.path.join(RES, "table_sensitivity_alpha.csv"))
print(sens_agg)

# budget sensitivity: population size
bud_rows = []
for pop in [20, 40, 80]:
    for s in [0, 1, 2]:
        N = nsga2(sur0, s, pop=pop, gen=int(2000 / pop))
        pick = pick_min_cost(N["G"], N["c"], N["l"], N["pv"]); o, m = oracle_metrics(decode(pick))
        bud_rows.append(dict(pop=pop, gen=int(2000 / pop), seed=s, cost_usd_h=m["cost_usd_h"], viol_pct=m["viol_pct"],
                             hv_sur=np.mean([front_hv(N["c"][w], N["l"][w], N["pv"][w] <= ALPHA) for w in range(N_WL)])))
bud = pd.DataFrame(bud_rows); bud.to_csv(os.path.join(RES, "sensitivity_population.csv"), index=False)
print(bud.groupby("pop").mean())

# ------------------------------------------------------------------ 6. figures
# Fig 2: surrogate parity
y, p = parity
fig, ax = plt.subplots(figsize=(3.3, 2.5))
ax.hexbin(y / np.log(10), p / np.log(10), gridsize=45, cmap="Greys", bins="log", mincnt=1, linewidths=0)
lo, hi = 2.2, 8.2
ax.plot([lo, hi], [lo, hi], color=ACC, lw=1)
ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
ax.set_xlabel(r"Observed $\log_{10}$ p95 latency (ms)"); ax.set_ylabel(r"Predicted $\log_{10}$ p95 latency (ms)")
r = sur_df.set_index("surrogate").loc["lgb"]
ax.text(0.04, 0.93, f"LightGBM, test 2026 H1\n$R^2$={r.lat_R2_log:.3f}, MAPE={r.lat_MAPE_pct:.1f}%",
        transform=ax.transAxes, va="top", fontsize=7)
savefig(fig, "fig2_surrogate_parity")

# Fig 4: HV convergence
fig, ax = plt.subplots(figsize=(3.3, 2.3))
ev = np.array([POP * (g + 1) for g in range(GEN + 1)])
for name, col, ls in [("NSGA-II", ACC, "-"), ("Random search", GREYS[1], "--")]:
    A = np.array(curves[name]); mu, sd = A.mean(0), A.std(0)
    ax.plot(ev, mu, color=col, ls=ls, lw=1.3, label=name + " (surrogate)")
    ax.fill_between(ev, mu - sd, mu + sd, color=col, alpha=0.2, lw=0)
ax.set_xlabel("Surrogate evaluations per workload"); ax.set_ylabel("Mean hypervolume (feasible set)")
ax.legend(frameon=False, loc="lower right"); ax.grid(alpha=0.3, lw=0.5)
savefig(fig, "fig4_hv_convergence")

# Fig 5: example Pareto front for one workload (high-rate, tight SLA)
N0 = store["N"]; W0 = int(np.argmax(np.where(SLA <= 20000, WL.ingest_events_per_sec.values, -1)))
cfg_w = decode(N0["G"][W0]); feas = N0["pv"][W0] <= ALPHA
ow = oracle.evaluate({k: np.repeat(v[W0:W0 + 1], POP) for k, v in wl_oracle(WL).items()}, cfg_w)
fig, ax = plt.subplots(figsize=(3.3, 2.5))
ax.scatter(N0["c"][W0][feas], np.exp(N0["l"][W0][feas]) * SLA[W0] / 1000, s=14, facecolors="none",
           edgecolors=ACC, lw=0.9, label="NSGA-II front (surrogate)")
ax.scatter(ow["exp_cost"][feas], ow["p95_nominal"][feas] / 1000, s=8, c=GREYS[0], marker="x", lw=0.7,
           label="Same configs (oracle)")
mk = {"Static batch": "s", "Static streaming": "^", "As-operated (logged)": "D"}
for nm in mk:
    cfg_b = LOGGED_CFG if nm.startswith("As") else static_cfg("batch" if "batch" in nm else "streaming")
    ob = oracle.evaluate({k: v[W0:W0 + 1] for k, v in wl_oracle(WL).items()}, {k: v[W0:W0 + 1] for k, v in cfg_b.items()})
    ax.scatter(ob["exp_cost"], ob["p95_nominal"] / 1000, s=22, marker=mk[nm], c=GREYS[2], edgecolors="k", lw=0.5, label=nm)
ax.axhline(SLA[W0] / 1000, color="k", ls=":", lw=0.8); ax.text(ax.get_xlim()[0], SLA[W0] / 1000 * 1.1, " SLA", fontsize=7)
ax.set_yscale("log"); ax.set_xlabel("Expected cost (USD/h)"); ax.set_ylabel("p95 latency (s, log scale)")
ax.legend(frameon=False, fontsize=6, loc="upper right")
savefig(fig, "fig5_pareto_example")
ex_info = dict(workload_index=W0, source=str(WL.data_source[W0]), rate=float(WL.ingest_events_per_sec[W0]),
               sla_ms=float(SLA[W0]), n_front=int(feas.sum()),
               front_cost_min=float(N0["c"][W0][feas].min()), front_cost_max=float(N0["c"][W0][feas].max()))

# Fig 3: cost and violation per method (mean +- std over seeds)
labels = ["As-operated", "Static batch", "Static stream.", "WS random", "Greedy", "NSGA-II (prop.)"]
fig, axs = plt.subplots(1, 2, figsize=(3.4, 2.1), sharey=True)
yy = np.arange(len(ORDER))
cols = [GREYS[2]] * 5 + [ACC]
axs[0].barh(yy, agg.cost_usd_h_mean, xerr=agg.cost_usd_h_std.fillna(0), color=cols, capsize=2, error_kw={"lw": 0.7})
axs[1].barh(yy, agg.viol_pct_mean, xerr=agg.viol_pct_std.fillna(0), color=cols, capsize=2, error_kw={"lw": 0.7})
axs[1].axvline(100 * ALPHA, color="k", ls=":", lw=0.8)
axs[0].set_yticks(yy); axs[0].set_yticklabels(labels); axs[0].invert_yaxis()
axs[0].set_xlabel("Exp. cost (USD/h)"); axs[1].set_xlabel("Exp. SLA viol. (%)")
for a_ in axs: a_.grid(axis="x", alpha=0.3, lw=0.5)
fig.tight_layout(w_pad=0.6)
savefig(fig, "fig3_method_comparison")

# Fig 6: alpha sensitivity
fig, ax = plt.subplots(figsize=(3.3, 2.2))
al = sens_agg.index.values * 100
ax.errorbar(al, sens_agg.cost_usd_h_mean, yerr=sens_agg.cost_usd_h_std, color=ACC, marker="o", ms=3, lw=1.2, capsize=2)
ax.set_xscale("log"); ax.set_xticks(al); ax.set_xticklabels([f"{a:g}" for a in al])
ax.set_xlabel(r"SLA risk budget $\alpha$ (%)"); ax.set_ylabel("Expected cost (USD/h)", color=ACC)
ax2 = ax.twinx(); ax2.spines["right"].set_visible(True)
ax2.errorbar(al, sens_agg.viol_pct_mean, yerr=sens_agg.viol_pct_std, color=GREYS[0], marker="s", ms=3, lw=1, ls="--", capsize=2)
ax2.plot(al, al, color=GREYS[3], lw=0.8, ls=":")
ax2.set_ylabel("Expected SLA violation (%)")
ax.grid(alpha=0.3, lw=0.5)
savefig(fig, "fig6_alpha_sensitivity")

# Fig 7: mode mix chosen by NSGA-II per SLA class vs logged
cls = np.where(SLA <= 20000, "<=20 s", np.where(SLA <= 300000, "60-300 s", ">=900 s"))
mix = pd.crosstab(cls, store["cfg"]["mode"], normalize="index").reindex(columns=["batch", "micro_batch", "streaming"], fill_value=0)
mixl = pd.crosstab(cls, LOGGED_CFG["mode"], normalize="index").reindex(columns=["batch", "micro_batch", "streaming"], fill_value=0)
order_c = ["<=20 s", "60-300 s", ">=900 s"]; mix = mix.reindex(order_c); mixl = mixl.reindex(order_c)
pd.concat({"proposed": mix, "logged": mixl}).round(4).to_csv(os.path.join(RES, "mode_mix_by_sla.csv"))
fig, ax = plt.subplots(figsize=(3.3, 2.1))
x = np.arange(3); wbar = 0.38; hatch = {"batch": "", "micro_batch": "///", "streaming": ""}
fc = {"batch": GREYS[4], "micro_batch": "white", "streaming": GREYS[1]}
for off, M, lab in [(-wbar / 2 - 0.01, mixl, "Logged"), (wbar / 2 + 0.01, mix, "NSGA-II")]:
    bot = np.zeros(3)
    for md in ["batch", "micro_batch", "streaming"]:
        ax.bar(x + off, 100 * M[md].values, wbar, bottom=bot, color=fc[md], hatch=hatch[md], edgecolor="k", lw=0.5,
               label=md.replace("_", "-") if off > 0 else None)
        bot += 100 * M[md].values
    for xi in x:
        ax.text(xi + off, 101, lab[0], ha="center", fontsize=6)
ax.set_xticks(x); ax.set_xticklabels([f"SLA {c}" for c in order_c]); ax.set_ylabel("Share of workloads (%)")
ax.set_ylim(0, 108); ax.legend(frameon=False, fontsize=6, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.17))
savefig(fig, "fig7_mode_mix")

# Fig 1: methodology flowchart (graphviz)
dot = r'''digraph G { rankdir=TB; nodesep=0.15; ranksep=0.16; bgcolor="white";
node [shape=box, style="rounded,filled", fillcolor="#f2f2f2", color="#444444", fontname="Times-Roman", fontsize=12, margin="0.08,0.03", width=2.9];
edge [color="#444444", arrowsize=0.6];
a [label="Hourly telemetry logs (AMI, PMU, SCADA, DER, outage, alarm)\nworkload + configuration → p95 latency, cost"];
b [label="Preprocessing: robust-MAD outlier repair, per-source median\nimputation, workload-aware load-ratio feature"];
c [label="LightGBM surrogates: log p95 latency, log cost\n+ out-of-fold residual model → SLA risk P[p95 > SLA]"];
d [label="NSGA-II per workload: min {cost, log(p95/SLA)}\ns.t. P[p95 > SLA] ≤ α (constrained domination)", fillcolor="#f6dcd8", color="#c0392b"];
e [label="Pareto front → deployment rule: cheapest SLA-safe config\n(mode, engine, window, executors, vCPU, spot, tier)"];
f [label="Counterfactual oracle evaluation vs. baselines\n(cost, SLA violation, hypervolume, 5 seeds)"];
a->b->c->d->e->f; d->c [style=dashed, label=" query", fontsize=8, fontname="Times-Roman"]; }'''
with open(os.path.join(RES, "fig1_flowchart.dot"), "w") as fh:
    fh.write(dot)
subprocess.run(["dot", "-Tpng", "-Gdpi=300", "-o", os.path.join(FIG, "fig1_flowchart.png")], input=dot.encode(), check=True)
from PIL import Image
Image.open(os.path.join(FIG, "fig1_flowchart.png")).convert("RGB").save(os.path.join(RES, "fig1_flowchart.jpg"), quality=92)

# ------------------------------------------------------------------ 7. metrics.json
best_base = agg.drop(index=[P, "As-operated (logged)"])
best_base = best_base[best_base.attain_pct_mean >= agg.loc[P, "attain_pct_mean"] - 5]
metrics = dict(
    dataset=dict(rows=len(df), train=len(tr), test=len(te), missing_imputed=raw_missing, outliers_repaired=n_out,
                 n_workloads_optimised=N_WL),
    setup=dict(pop=POP, gen=GEN, budget=BUDGET, alpha=ALPHA, seeds=SEEDS, lgb=LGB_P),
    surrogates=sur_df.set_index("surrogate").to_dict(orient="index"),
    latency_importance_top5=imp.head(5).round(1).to_dict(),
    main=agg.to_dict(orient="index"), significance=stats,
    ablation=abl_agg.to_dict(orient="index"), alpha_sensitivity=sens_agg.to_dict(orient="index"),
    population_sensitivity=bud.groupby("pop").agg(["mean", "std"]).drop(columns=["seed"], level=0)
        .pipe(lambda d: d.set_axis([f"{a}_{b}" for a, b in d.columns], axis=1)).round(4).to_dict(orient="index"),
    pareto_example=ex_info, mode_mix_proposed=mix.round(4).to_dict(orient="index"),
    mode_mix_logged=mixl.round(4).to_dict(orient="index"),
    runtime_total_s=round(time.time() - T_START, 1))
with open(os.path.join(RES, "metrics.json"), "w") as fh:
    json.dump(metrics, fh, indent=2, default=float)
print("done in", round(time.time() - T_START, 1), "s")
