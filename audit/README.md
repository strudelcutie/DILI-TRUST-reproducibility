gates.py audits two abstention rules built on one calibrated probability.

Inputs: p (array), qhat, U80, th_spec95, th_sens95.
Outputs: p_star, conformal and uncertainty edges, gate label, tier_untouched_sufficient, and optional counts.
Assumes r = 1 - p and a common scale for qhat and the thresholds.
U80 < 0 is rejected; U80 = 0 is evaluated per sample as U(p) > U80.

Example:
audit_gates(None, 0.7328, 1.082826, 0.828, 0.4273)["gate"] -> "conformal binds"
audit_gates(None, 0.055, 0.174631, 0.0654, 0.0063)["gate"] -> "uncertainty binds"

verify.py regenerates the Table S17 counts for all 12 settings.
DILI per-sample predictions are included; Tox21 and NASA predictions are produced by their run_all.sh pipelines.
python audit/verify.py --tox21-root PATH --nasa-root PATH
Writes audit/verification.csv with observed and expected counts, and prints 12/12 PASS.
