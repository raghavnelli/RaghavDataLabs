# A1P1 code - surrogate-assisted NSGA-II for hybrid batch/streaming configuration (smart-grid telemetry)

## Run
```
pip install -r requirements.txt      # plus graphviz (`dot`) on the PATH
python3 run_all.py                    # reads ../data/A1P1_batch_stream_costlat.csv
```
All seeds are fixed (numpy seeds 0-4, workload sample seed 2026). Runtime is about 10-20 min on 2 CPU cores
(LightGBM is pinned to `n_jobs=1` because thread oversubscription on small machines slows fitting by about 60x).

## Files
- `run_all.py`: the whole pipeline. It does preprocessing (robust-MAD outlier repair, per-source median imputation fitted on 2025),
  fits the LightGBM / ridge / kNN surrogates (train 2025, test 2026 H1), runs the proposed chance-constrained NSGA-II and the
  baselines (static batch, static streaming, weighted-sum random search, greedy, plus as-operated logs) over 200 test
  workloads x 5 seeds, then the ablation (3 seeds), the alpha and population sensitivity studies, the Wilcoxon tests, and the figures.
- `oracle.py`: the counterfactual evaluator. It re-implements the ground-truth latency and cost equations of the synthetic
  generator so that a configuration never seen in the logs can be scored. Only the evaluation uses it; the optimisers only see
  the ML surrogates trained on the CSV.

## Outputs (../results, figures also as 300-dpi PNG in ../figures)
| file | content |
|---|---|
| table_dataset_summary.csv, table_dataset_per_source.csv | dataset statistics (Table I) |
| table_surrogate_accuracy.csv, latency_surrogate_importance.csv | surrogate R2/MAPE/ROC-AUC/PR-AUC, LightGBM gain importance |
| main_comparison_per_seed.csv, table_main_comparison.csv | cost, expected SLA violation, attainment, hypervolume per method (mean, std over 5 seeds) |
| table_significance.csv, per_workload_cost.csv | Wilcoxon signed-rank p-values and cost savings, per-workload costs |
| ablation_per_seed.csv, table_ablation.csv | ablation variants |
| sensitivity_alpha_per_seed.csv, table_sensitivity_alpha.csv, sensitivity_population.csv | sensitivity studies |
| mode_mix_by_sla.csv | processing-mode mix chosen per SLA class |
| metrics.json | all headline numbers used in the paper |
| fig1_flowchart.jpg (.dot), fig2_surrogate_parity, fig3_method_comparison, fig4_hv_convergence, fig5_pareto_example, fig6_alpha_sensitivity, fig7_mode_mix (.jpg) | figures 1-7 of the paper |
