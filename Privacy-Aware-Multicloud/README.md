# A1P3 code: MILP policy-constrained multi-cloud placement with auditable lineage

## Run
```
pip install -r requirements.txt      # plus Graphviz `dot`
python3 run_all.py                    # reads ../data/A1P3_privacy_lineage_ops.csv
```
Everything uses fixed seeds. Tested with Python 3.13, numpy 2.5, scipy 1.16 (HiGHS), scikit-learn 1.9 on a 2-vCPU Xeon.
Runtime is a few minutes; `results/metrics.json` stores the exact runtime.

## Pipeline (run_all.py)
1. Preprocessing: group-median imputation of `quasi_identifier_count`, `lineage_completeness`, `anchor_latency_ms`;
   outlier handling with a log-IQR fence (records) and a group-wise robust z-score (latency).
2. Split 40/30/30 %: planner models (logistic violation, logit-OLS re-identification, log-linear anchoring latency, gas),
   independent auditor (HistGradientBoosting) and the scenario pool used to draw the windows.
   `reidentification_risk` is never used as a feature for `policy_violation`.
3. For each of 30 windows (5 seeds x 6 windows, 150 operations each), enumerate 270 processing options per operation,
   filter by the policy, prune exactly, and solve the MILP (scipy.optimize.milp/HiGHS). Baselines: greedy
   cheapest-compliant, single-cloud, random feasible, unconstrained (cost only).
4. Ablation (6 variants), violation-budget sweep, capacity x anchoring-latency heat map, scalability (N = 50 to 600).
5. Merkle-tree lineage anchoring with tamper-detection test.

## Outputs
| File | Content |
|---|---|
| `results/table_dataset_summary.csv` | Table II, dataset summary |
| `results/table_setup.csv` | Table III, hyper-parameters / setup |
| `results/table_main_comparison.csv` | Table IV, mean and std over 30 windows per method |
| `results/table_ablation.csv` | Table V, ablation |
| `results/table_model_quality.csv` | planner vs auditor model quality on the held-out pool |
| `results/table_wilcoxon.csv` | paired Wilcoxon tests, MILP vs each baseline |
| `results/per_window_results.csv` | raw per-window results for all methods |
| `results/sensitivity_violation_budget.csv`, `sensitivity_capacity_latency_cost.csv`, `scalability.csv`, `anchoring_tradeoff.csv`, `lineage_audit.csv` | sensitivity and audit raw data |
| `results/metrics.json` | all headline numbers, config, solver statistics |
| `results/run_log.txt` | console log |
| `figures/fig*.png` (300 dpi) and `results/fig*.jpg` | Figs. 1 to 7 |
