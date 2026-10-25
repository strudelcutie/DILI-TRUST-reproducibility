# Results

`Data S1.xlsx` contains the final supplementary tables referenced in the
manuscript:

Content
literature-model adapted probes 
metric glossary, analysis governance, and method positioning 
confidence intervals and permutation tests 
plit sensitivity and attribution 
chemical-space distance and ranking 
calibration selection and prevalence shift 
conformal coverage and prediction-set composition 
operational triage and review burden 
low-risk exclusion sensitivity 
uncertainty, validation reuse, and applicability-domain sensitivity 
decision-curve and prevalence sensitivity 
Tox21 endpoint generalization 
NASA MDP cross-project generalization 
label definition and sensitivity 
data sources, cohort construction, and preprocessing 
model eligibility, negative controls, and leakage checks 

Figure source data:

- `figure_data/` — frozen tables behind Figures 2–5, with a figure-to-data map
  in `figure_data/README.md`. The plotting scripts are not distributed.

Cross-domain summary files:

- `tox21_nrar_triage_summary.csv` — NR-AR triage counts and low-score tier
  (scaffold and cluster-out splits, HistGB and ExtraTrees).
- `tox21_nrer_triage_summary.csv` — NR-ER equivalents.
- `task1_ranking.csv`, `task2_probability_reliability.csv`,
  `task3_decision_utility.csv` — NASA MDP cross-project results.

Frozen predictions:

- `predictions/prediction_scores_388.csv` — calibrated scores, labels, locked
  thresholds, triage categories, and uncertainty fields for the 388-compound
  external cohort and the scaffold, temporal, and cluster-out settings.
  `code/reproduce_main_tables.py` recomputes Tables 1-4 from this file.
