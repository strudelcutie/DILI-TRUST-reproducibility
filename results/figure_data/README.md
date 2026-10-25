# Figure source data

Frozen tables from which Figures 2–5 of the manuscript were plotted. The plotting
scripts are not distributed with this repository; these tables are provided so
that every value shown in the figures can be traced and re-plotted independently.

## Contents

| Path | Used for |
|---|---|
| `Figure_5_source.csv` | Assembled per-setting matrix plotted in Figure 5: AUROC, Brier skill score, calibration slope, and low-score enrichment for the 12 evaluation settings, grouped by domain |
| `dili/audit/` | DILI baseline metrics and recomputed conformal outputs |
| `dili/canonical/` | Frozen HistGB predictions and triage assignments for the 388-compound external cohort and the three internal settings |
| `dili/supplementary_tables/` | Bootstrap confidence intervals and conformal coverage tables |
| `dili/secondary_analyses/`, `dili/stage2/` | High-score metrics and chemical-distance stratified metrics |
| `tox21/` | NR-AR and NR-ER task tables (ranking, probability, triage) for the cross-domain panels |
| `nasa/` | KC1, PC1, JM1, and KC2 task tables (ranking, probability, triage) for the cross-domain panels |

## Figure-to-data map

| Figure | Source tables |
|---|---|
| Figure 2 | `dili/audit/baseline_metrics.csv`, `dili/supplementary_tables/S8_bootstrap_ci_full.csv`, `dili/canonical/predictions/histgb_predictions.csv` |
| Figure 3 | `dili/canonical/predictions/histgb_predictions.csv`, `dili/supplementary_tables/S7_conformal_coverage.csv`, `dili/audit/conformal_recomputed.csv` |
| Figure 4 | `dili/canonical/triage/histgb_triage.csv`, `dili/secondary_analyses/high_score/high_score_metrics_by_split.csv` |
| Figure 5 | `Figure_5_source.csv`, with the cross-domain rows built from `tox21/*/tables/` and `nasa/tables/` |

Values in `Figure_5_source.csv` are rounded only by the plotting step; the CSV
carries full precision.
