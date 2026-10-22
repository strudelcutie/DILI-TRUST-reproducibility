#!/usr/bin/env bash
# Full Tox21 NR-AR generalization validation.
#
#   bash run_all.sh [/path/to/tox21.csv]
#
# Runs from the tox21_nrar_validation root. Each stage is independent and reads
# only the artefacts of the previous stage; `-P` keeps the interpreter from
# prepending the script directory when reading the externally supplied dataset.
set -euo pipefail

SRC="${1:-F:/Tox21 NR-AR/tox21.csv}"
PY="${PYTHON:-python}"

"$PY" -P scripts/01_prepare_nrar_data.py "$SRC"
"$PY" -P scripts/02_build_splits.py
"$PY" -P scripts/03_features.py
"$PY" -P scripts/04_model_ready.py
"$PY" -P scripts/05_train_predict.py
"$PY" -P scripts/06_calibrate_and_lock.py
"$PY" -P scripts/07_conformal.py
"$PY" -P scripts/08_applicability_domain.py
"$PY" -P scripts/09_metrics_and_triage.py
"$PY" -P scripts/10_figures.py
"$PY" -P scripts/12_calibration_collapse_check.py
"$PY" -P scripts/11_final_report.py

echo "done - see reports/FINAL_REPORT_tox21_nrar_validation.md"
