#!/usr/bin/env bash
# Full cross-domain tabular generalization validation.
#
#   bash run_all.sh
#
# Runs from this validation root. Stage 00 replaces the molecular stages 01-04
# and 08; stages 05-09 are the DILI-TRUST scripts unchanged apart from the split
# registry, which is read from common.py.
set -euo pipefail

PY="${PYTHON:-python}"

"$PY" -P scripts/00_prepare_tabular.py
"$PY" -P scripts/05_train_predict.py
"$PY" -P scripts/06_calibrate_and_lock.py
"$PY" -P scripts/07_conformal.py
"$PY" -P scripts/09_metrics_and_triage.py
"$PY" -P scripts/13_table_s42.py

echo "done - see reports/Table_S42_cross_project_generalization.md"
