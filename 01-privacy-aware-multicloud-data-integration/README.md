# A1P3 - Privacy-Aware Data Integration Across Multi-Cloud Platforms Through Policy-Constrained Processing and Auditable Lineage

Application: blockchain-anchored lineage for peer-to-peer energy-trading data (Track 6), ICE2CPT 2026.
Proposed method: Mixed-Integer Linear Programming (MILP) for policy-constrained placement with lineage audit.
Baselines: greedy cheapest-compliant, single-cloud, random feasible, unconstrained. The data is synthetic.

## Contents
- `paper/` IEEE conference paper (.docx and .pdf, 6 pages)
- `data/`  synthetic dataset `A1P3_privacy_lineage_ops.csv` (10,000 operations) and its README
- `code/`  `run_all.py` (whole pipeline), `requirements.txt`, `README.md`
- `results/` all figures (JPG), all tables (CSV), `metrics.json`, `run_log.txt`

## Reproduce
    pip install -r code/requirements.txt   # plus Graphviz `dot`
    cd code && python3 run_all.py          # about 4-8 min on 2 vCPUs, writes ../results and ../figures
