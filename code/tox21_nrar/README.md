# Tox21 NR-AR independent validation of the DILI-TRUST reliability framework

Supplementary validation for the claim:

> Generalization to an independent non-hepatic toxicity endpoint

The pipeline reuses the DILI-TRUST feature block, models, calibration protocol,
conformal protocol, uncertainty score and decision policy unchanged. It does not
modify any script under `../scripts/` and introduces no new algorithm.

## Run

```bash
bash run_all.sh "data/tox21/tox21.csv"
```

## Stages

| stage | script | output |
|---|---|---|
| 01 | `01_prepare_nrar_data.py` | `data/nrar_curated.csv` (`compound_id`, `smiles`, `label`) |
| 02 | `02_build_splits.py` | `splits/scaffold_split.csv`, `splits/cluster_out_split.csv` |
| 03 | `03_features.py` | `features/rdkit_2d.csv`, `features/morgan.csv`, `features/maccs.csv` |
| 04 | `04_model_ready.py` | `matrices/<split>/model_ready_<split>_<partition>.csv` |
| 05 | `05_train_predict.py` | `predictions/<split>_<model>_*_predictions.csv` |
| 06 | `06_calibrate_and_lock.py` | `calibrators/`, `tables/threshold_lock_table.csv` |
| 07 | `07_conformal.py` | `tables/conformal_coverage.csv` |
| 08 | `08_applicability_domain.py` | `tables/ad_*.csv` |
| 09 | `09_metrics_and_triage.py` | `tables/task1..task3*.csv`, `tables/triage_*.csv` |
| 10 | `10_figures.py` | `figures/Figure_S*.png/.pdf`, `tables/Table_S*.csv` |
| 12 | `12_calibration_collapse_check.py` | calibration-collapse verification (raw vs calibrated, by chemical distance, prior-shift stress test) |
| 11 | `11_final_report.py` | `reports/FINAL_REPORT_tox21_nrar_validation.md` |

## Design decisions

- **No temporal split.** Tox21 carries no time annotation; none was fabricated.
- **External-like cohort = cluster-out held-out partition** (chemically disjoint
  from the development set). The scaffold held-out partition is the internal
  held-out cohort.
- **Butina cutoff.** DILI derives the cutoff from the 10th percentile of pairwise
  Tanimoto distances; on the far more diverse Tox21 library that rule collapses
  to a single cluster, so a fixed distance cutoff of 0.70 (similarity >= 0.30) is
  used. See `reports/stage02_split_qc.md`.
- **Parameter locking.** The calibrator, q-hat, uncertainty threshold and decision
  thresholds are all fixed on the validation partition and applied unchanged to
  the held-out cohorts.
- **Low-score threshold.** The DILI release stores a conservative
  `near_zero_locked_threshold` that cannot be recomputed from the released
  scripts; the validation-locked 95%-sensitivity threshold is used instead.

## Parameter values reused from DILI-TRUST

| parameter | value | source |
|---|---|---|
| Morgan radius / bits | 2 / 2048 | `configs/feature_config.yaml` |
| HistGB | `lr=0.05, max_iter=200, max_leaf_nodes=31` | `fast_train.py` |
| ExtraTrees | `n_estimators=200, max_depth=15, min_samples_leaf=3, class_weight=balanced` | `fast_train.py` |
| Seeds | 20260521-20260525 | `fast_train.py`, `external_evaluation.py` |
| Calibration candidates | identity, Platt, isotonic, temperature | `s8final.py` |
| Calibration selection | OOF ranking `brier + 0.5*ECE(q5) + 0.1*NLL` | `s8final.py` |
| Conformal alphas | 0.05, 0.10 (primary), 0.20 | `configs/conformal_config.yaml` |
| Conformal q-hat | `quantile(ceil((n+1)(1-alpha))/n)` | `configs/conformal_config.yaml` |
| Uncertainty | `U(p) = H(p) + 0.5*M(p)`, validation 80th percentile | `configs/abstention_policy.yaml` |
| High-risk cut | validation 95%-specificity threshold | `configs/decision_policy.yaml` |
| OOD cut | max Tanimoto < 0.15 | `configs/decision_policy.yaml` |
| Bootstrap | 2000 resamples | `external_evaluation.py` |
