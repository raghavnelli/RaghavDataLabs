"""Counterfactual evaluation oracle (digital twin) for A1P1.

The logged dataset only contains the outcome of the configuration that the operator actually ran.
To score a *new* configuration proposed by any optimiser we re-use the ground-truth equations of the
synthetic generator (datasets/A1_Rahul/gen_author1.py, function paper1): queueing-based p95 latency and
the cloud cost model. The optimisers NEVER see this module; they only see ML surrogates trained on the
logged CSV. The oracle returns expectations over the generator's hidden noise:
  * expected cost (lognormal(0,0.3) multiplier and the unobserved egress share Beta(1,4) in expectation)
  * nominal p95 latency (hidden contention noise at its median, 1.0)
  * probability that the realised p95 exceeds the SLA, integrating the hidden lognormal(0,0.4) noise and the
    4 % heavy-tail episodes (x U(1.5,4)).
"""
import numpy as np
from scipy.stats import norm

ENGINE_CAP = {"flink": 2600.0, "dbt_on_warehouse": 1500.0, "spark_structured_streaming": 2000.0, "spark_batch": 2000.0}
TIER_PEN = {"hot_object": 1.0, "standard_object": 1.15, "infrequent_object": 1.6}
TIER_PRICE = {"hot_object": 0.023, "standard_object": 0.0125, "infrequent_object": 0.004}
REGION_PRICE = {"region-west": 0.048, "region-central": 0.040, "region-east": 0.044}
DUTY = {"streaming": 1.0, "batch": 0.35, "micro_batch": 0.7}
_U = np.linspace(1.5, 4.0, 51)  # grid for the heavy-tail multiplier


def _map(arr, d):
    return np.array([d[a] for a in arr], dtype=float)


def evaluate(w, c):
    """w: dict of workload arrays (rate, kb, burst, late, peak, storm, region, sla);
    c: dict of config arrays (mode, engine, window, ckpt, execs, vcpu, mem, spot, tier). All length n.
    Returns dict with exp_cost, p95_nominal, p_viol."""
    rate, kb, burst, late = w["rate"], w["kb"], w["burst"], w["late"]
    execs, vcpu, mem, spot = c["execs"].astype(float), c["vcpu"].astype(float), c["mem"].astype(float), c["spot"]
    mode, engine, tier = c["mode"], c["engine"], c["tier"]
    cap = execs * vcpu * _map(engine, ENGINE_CAP) / (0.6 + kb)
    util = rate * (1 + 0.5 * burst) / cap
    queue = np.where(util < 0.95, util / (1 - np.minimum(util, 0.95)), 20 + 30 * (util - 0.95))
    win, ck = c["window"].astype(float), c["ckpt"].astype(float)
    base = np.where(mode == "streaming", 150 + 40 * ck,
                    np.where(mode == "micro_batch", win * 1000 * 0.6 + 400, win * 1000 * 0.55 + 20000))
    p95 = base * _map(tier, TIER_PEN) * (1 + 0.6 * np.minimum(queue, 40)) * (1 + 0.02 * late) \
        * (1 + 0.25 * spot * burst)
    price = _map(w["region"], REGION_PRICE) * (1 + 0.25 * w["peak"])
    compute = execs * vcpu * price * (1 - 0.65 * spot) * _map(mode, DUTY) + execs * mem * 0.005
    vol = rate * kb * 3600 / 1e6
    storage = vol * _map(tier, TIER_PRICE) * 4
    retry = 0.02 * w["storm"] * execs
    egress = vol * 0.09 * 0.2
    exp_cost = (compute + storage + retry + egress) * np.exp(0.3 ** 2 / 2)
    lr = np.log(w["sla"] / p95)
    p_norm = 1 - norm.cdf(lr / 0.4)
    p_tail = (1 - norm.cdf((lr[:, None] - np.log(_U)[None, :]) / 0.4)).mean(1)
    p_viol = 0.96 * p_norm + 0.04 * p_tail
    return {"exp_cost": exp_cost, "p95_nominal": p95, "p_viol": p_viol}
