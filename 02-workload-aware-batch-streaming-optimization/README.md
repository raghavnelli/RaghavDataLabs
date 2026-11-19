# A1P1 - Workload-Aware Cost and Latency Optimization for Hybrid Batch-Streaming Data Architectures in Cloud Environments

Author: Rahul (A1, paper 1), ICE2CPT 2026, Track 1. Application: smart-meter and grid telemetry pipelines in a utility smart grid.
Method: chance-constrained NSGA-II with LightGBM surrogates (cost vs latency vs SLA risk). Baselines: static batch, static streaming, weighted-sum random search, greedy.

Contents:
- paper/   IEEE conference paper (.docx and .pdf, 6 pages)
- data/    synthetic dataset A1P1_batch_stream_costlat.csv (12,000 pipeline-hours) and its README (data dictionary). The data is SYNTHETIC.
- code/    run_all.py (full pipeline), oracle.py (counterfactual digital twin for evaluation), requirements.txt, README.md
- results/ all figures (JPG), all tables (CSV), metrics.json

Reproduce (expects data/ next to code/, as in this zip):
    pip install -r code/requirements.txt   # plus graphviz `dot`
    cd code && python3 run_all.py          # about 10-30 min on 2 CPU cores; writes ../results and ../figures

