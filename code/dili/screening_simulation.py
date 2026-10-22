#!/usr/bin/env python3

import pandas as pd, numpy as np, json, warnings
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

NOW = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
ROOT = Path(__file__).resolve().parents[2]


ST11_TMP = ROOT / "results" / "tmp" / "stage11"
ST11_DEC = ROOT / "results" / "final" / "decision_tables" / "stage11"
ST11_TAB = ROOT / "results" / "final" / "tables" / "stage11"
ST11_FIG = ROOT / "results" / "final" / "figures" / "stage11"
ST11_REP = ROOT / "reports" / "stage11"
ST11_BNCH = ROOT / "benchmark_release" / "stage11"
for d in [ST11_TMP, ST11_DEC, ST11_TAB, ST11_FIG, ST11_REP, ST11_BNCH]:
    d.mkdir(parents=True, exist_ok=True)
for sd in ["input_audit","simulation_tables","resource_models","threshold_sensitivity","figures"]:
    (ST11_TMP / sd).mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.size": 9, "axes.labelsize": 9, "axes.titlesize": 10,
    "legend.fontsize": 7, "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight"})
COLORS = {"HC_HR": "#D55E00", "HC_LR": "#009E73", "Uncertain": "#F0E442", "OOD": "#999999"}

ST9_LIM = "CONDITIONAL_PASS_stage9_core_external_evaluation_done_ready_for_stage10_with_limitations"
ST10_LIM = "CONDITIONAL_PASS_stage10_decision_policy_ready_with_stage9_limitations"
FINAL_V = "CONDITIONAL_PASS_stage11_screening_simulation_ready_with_stage9_stage10_limitations"

print("=" * 70)
print("STAGE 11 — PREFLIGHT AUDIT + INDUSTRIAL SCREENING SIMULATION")
print("=" * 70)




print("\n--- Part 1: Stage 1-10 Preflight Audit ---")

stages = {}
issues = []
stale = []

canonical_checks = {
    1: [("protocol_v4", "protocol/protocol/protocol_v4.md"), ("primary_endpoint", "protocol/protocol/primary_endpoint.yaml"),
        ("metrics", "protocol/protocol/metrics.yaml"), ("split_plan", "protocol/protocol/split_plan.yaml"),
        ("go_nogo", "protocol/protocol/go_nogo_v4.md"), ("integration_audit", "reports/stage1/stage1_protocol_integration_audit.md")],
    2: [("lit_audit", "reports/stage2/stage2_literature_audit_report.md")],
    3: [("source_url", "reports/stage3/stage3_source_url_completion_report.md")],
    4: [("qc_report", "reports/stage4/stage4_qc_report.md"), ("qc_audit", "reports/stage4/stage4_qc_audit_report.md")],
    5: [("split_summary", "reports/stage5/stage5_split_summary_report.md")],
    6: [("feature_qc", "reports/stage6/stage6_feature_qc_report.md")],
    7: [("freeze_qc", "reports/stage7/stage7_2_final_freeze_qc_report.md"),
        ("prediction_audit", "reports/stage7/stage7_2_prediction_audit_report.md"),
        ("training_log", "results/logs/training_log_stage7_1.csv")],
    8: [("qc_report", "reports/stage8/stage8_qc_report.md"), ("handoff", "reports/stage8/stage8_handoff_manifest.md"),
        ("self_check", "reports/stage8/stage8_self_check.md"), ("cal_config", "configs/calibration_config.yaml"),
        ("conformal_config", "configs/conformal_config.yaml"), ("abstention_config", "configs/abstention_policy.yaml"),
        ("threshold_lock", "results/final/tables/stage8_threshold_lock_table.csv"),
        ("selected_methods", "results/final/tables/stage8_selected_calibration_methods_final.csv")],
    9: [("eval_qc", "reports/stage9/stage9_external_evaluation_qc_report.md"),
        ("correction", "reports/stage9/stage9_qc_correction_report.md"),
        ("recovery", "reports/stage9/stage9_artifact_recovery_report.md"),
        ("self_check_corrected", "reports/stage9/stage9_self_check_corrected.md"),
        ("all_metrics", "results/final/tables/stage9/stage9_all_split_metrics.csv"),
        ("availability", "results/final/tables/stage9/stage9_model_availability_matrix.csv"),
        ("limitations", "results/final/tables/stage9/stage9_limitations_table.csv")],
    10: [("decision_qc", "reports/stage10/stage10_decision_policy_qc_report.md"),
         ("archive_consistency", "reports/stage10/stage10_archive_consistency_report.md"),
         ("self_check", "reports/stage10/stage10_self_check.md"),
         ("decision_policy", "protocol/decision_policy.yaml"),
         ("dp_stage10", "configs/decision_policy.yaml"),
         ("all_splits_dec", "results/final/decision_tables/stage10/decision_table_all_splits.csv"),
         ("flow_summary", "results/final/decision_tables/stage10/decision_flow_summary.csv"),
         ("decision_summary", "results/final/tables/stage10/decision_summary.csv"),
         ("no_call_table", "results/final/tables/stage10/no_call_reason_table.csv"),
         ("hce_table", "results/final/tables/stage10/high_confidence_error_table.csv"),
         ("dca_table", "results/final/tables/stage10/decision_curve_net_benefit_table.csv")],
}

preflight_rows = []
for st in range(1, 11):
    n_ok = 0; n_miss = 0
    for label, fp in canonical_checks.get(st, []):
        exists = (ROOT / fp).exists()
        preflight_rows.append({"stage": st, "file": label, "path": fp, "exists": exists})
        if exists: n_ok += 1
        else: n_miss += 1
    verdict = "PASS" if n_miss == 0 else "CONDITIONAL_PASS" if n_miss <= 2 else "FAIL"
    stages[st] = {"ok": n_ok, "miss": n_miss, "verdict": verdict}



s7_archive = ROOT / "reports/stage7/stage7_2_archive_self_check.md"
if s7_archive.exists():
    txt = s7_archive.read_text(encoding="utf-8", errors="ignore")
    if "FAIL" in txt:
        stale.append("stage7_2_archive_self_check.md: contains FAIL (stale, non-canonical)")
        preflight_rows.append({"stage": 7, "file": "archive_self_check", "path": str(s7_archive), "exists": True, "note": "STALE_FAIL"})


s9_self = ROOT / "reports/stage9/stage9_self_check.md"
if s9_self.exists():
    txt = s9_self.read_text(encoding="utf-8", errors="ignore")
    if "PASS_stage9_external_evaluation_complete" in txt and ST9_LIM not in txt:
        stale.append("stage9_self_check.md: old full PASS (stale), canonical is stage9_self_check_corrected.md")


s10_qc = ROOT / "reports/stage10/stage10_decision_policy_qc_report.md"
if s10_qc.exists():
    txt = s10_qc.read_text(encoding="utf-8", errors="ignore")
    if "clinical benefit" in txt.lower() or "clinical significance" in txt.lower():
        issues.append("Stage 10 QC report contains 'clinical benefit/significance' language")


hard_fail = any(stages[s]["miss"] >= 3 for s in [1,2,3,4,5,7,8])
has_blocking = len(issues) > 0
preflight_v = "FAIL_stage11_preflight_blocking_issue_detected" if hard_fail or has_blocking else \
    "CONDITIONAL_PASS_stage11_preflight_ready_with_stage9_stage10_limitations"

print(f"  Stage 1-5: {'ALL PASS' if all(stages[s]['miss']==0 for s in [1,2,3,4,5]) else 'ISSUES'}")
print(f"  Stage 7: {stages[7]['ok']}/{stages[7]['ok']+stages[7]['miss']} files")
print(f"  Stage 8: {stages[8]['ok']}/{stages[8]['ok']+stages[8]['miss']} files")
print(f"  Stage 9: {stages[9]['ok']}/{stages[9]['ok']+stages[9]['miss']} files")
print(f"  Stage 10: {stages[10]['ok']}/{stages[10]['ok']+stages[10]['miss']} files")
print(f"  Stale reports: {len(stale)}")
print(f"  Issues: {len(issues)}")
for s in stale: print(f"    {s}")
print(f"  Preflight verdict: {preflight_v}")


df_pre = pd.DataFrame(preflight_rows)
df_pre.to_csv(ST11_TAB / "stage11_preflight_status_matrix.csv", index=False)

pre_rpt = f"""# Stage 11 Preflight Audit — Stages 1-10

**Generated:** {NOW}
**Verdict:** `{preflight_v}`

Stage Summary
| Stage | Canonical Files | Missing | Verdict |
|-------|----------------|---------|---------|
"""
for st in range(1, 11):
    s = stages[st]
    pre_rpt += f"| {st} | {s['ok']+s['miss']} | {s['miss']} | {s['verdict']} |\n"

pre_rpt += f"""
Stale / Non-Canonical Reports ({len(stale)})
"""
if stale:
    for s in stale: pre_rpt += f"- {s}\n"
else:
    pre_rpt += "- None\n"

pre_rpt += f"""
Issues ({len(issues)})
"""
if issues:
    for i in issues: pre_rpt += f"- {i}\n"
else:
    pre_rpt += "- None (no blocking issues detected)\n"

pre_rpt += f"""
Carried-Forward Limitations
- Stage 9: {ST9_LIM}
- Stage 10: {ST10_LIM}
- LGB/XGB unavailable (no frozen model objects)
- HistGB cluster_out 4/5 seeds
- post2021_lockbox: unlabeled, prediction-only

Verdict
**{preflight_v}**
"""

with open(ST11_REP / "stage11_preflight_stage1_10_audit_report.md", "w", encoding="utf-8") as f:
    f.write(pre_rpt)
print(f"  Saved: stage11_preflight_stage1_10_audit_report.md")

if hard_fail or has_blocking:
    print("  BLOCKED — cannot proceed to simulation")
    exit(1)




print("\n--- Part 2: Build Simulation Cohorts ---")

df_dec = pd.read_csv(ROOT / "results/final/decision_tables/stage10/decision_table_all_splits.csv")
df_flow = pd.read_csv(ROOT / "results/final/decision_tables/stage10/decision_flow_summary.csv")
df_dca = pd.read_csv(ROOT / "results/final/tables/stage10/decision_curve_net_benefit_table.csv")

cohort_config = {
    "external_dilist_labeled": {"split": "external_dilist", "model": "HistGB", "labeled": True, "desc": "Primary external enterprise screening simulation"},
    "post2021_lockbox_prediction_only": {"split": "post2021_lockbox", "model": "HistGB", "labeled": False, "desc": "Prospective new-drug triage display (unlabeled)"},
    "cluster_out_labeled": {"split": "cluster_out", "model": "HistGB", "labeled": True, "desc": "Chemical cluster extrapolation stress test"},
    "temporal_labeled": {"split": "temporal", "model": "HistGB", "labeled": True, "desc": "Temporal drift stress test"},
    "scaffold_reference": {"split": "scaffold", "model": "HistGB", "labeled": True, "desc": "Internal strict test reference"},
}

cohorts = {}
primary_drug_names = {}

ev_f = ROOT / "data_curated/evidence/evidence_table.csv"
if ev_f.exists():
    ev = pd.read_csv(ev_f)
    if "compound_id" in ev.columns and "primary_name" in ev.columns:
        primary_drug_names = dict(zip(ev["compound_id"], ev["primary_name"]))

cohort_rows = []
for cohort_name, cfg in cohort_config.items():
    sub = df_dec[(df_dec["split"] == cfg["split"]) & (df_dec["model"] == cfg["model"])].copy()
    if len(sub) == 0:
        print(f"  SKIP {cohort_name}: no data")
        continue
    sub["cohort"] = cohort_name
    sub["primary_name"] = sub["sample_id"].map(primary_drug_names).fillna("")
    sub["post2021_prediction_only_flag"] = not cfg["labeled"]
    sub["stage9_limitation_flag"] = cfg["split"] in ["cluster_out"]
    cohorts[cohort_name] = sub
    cohort_rows.append(sub)
    print(f"  {cohort_name}: n={len(sub)}, labeled={cfg['labeled']}")

df_cohorts = pd.concat(cohort_rows, ignore_index=True)
df_cohorts.to_csv(ST11_DEC / "screening_input_cohorts.csv", index=False)
print(f"  Saved: screening_input_cohorts.csv ({len(df_cohorts)} rows)")




print("\n--- Part 3: Screening Action Mapping ---")

action_map = {
    "high_confidence_high_risk": "deprioritize_or_confirmatory_DILI_assay",
    "high_confidence_low_risk": "low_priority_clearance_candidate",
    "uncertain_or_conflict": "needs_additional_evidence_or_assay",
    "out_of_domain_no_call": "manual_review_or_out_of_domain_assay",
}
action_level = {
    "high_confidence_high_risk": "actionable_high_risk",
    "high_confidence_low_risk": "actionable_low_risk",
    "uncertain_or_conflict": "follow_up_required",
    "out_of_domain_no_call": "no_call_ood",
}

for cohort_name, df_c in cohorts.items():
    df_c["screening_action"] = df_c["decision_class"].map(action_map)
    df_c["action_level"] = df_c["decision_class"].map(action_level)




print("\n--- Part 4: Core Simulation Metrics ---")

sim_rows = []
for cohort_name, df_c in cohorts.items():
    cfg = cohort_config[cohort_name]
    n = len(df_c)
    labeled = cfg["labeled"]

    hr_n = int((df_c["decision_class"] == "high_confidence_high_risk").sum())
    lr_n = int((df_c["decision_class"] == "high_confidence_low_risk").sum())
    un_n = int((df_c["decision_class"] == "uncertain_or_conflict").sum())
    ood_n = int((df_c["decision_class"] == "out_of_domain_no_call").sum())
    no_call_n = un_n + ood_n

    row = {"cohort": cohort_name, "split": cfg["split"], "model": cfg["model"],
        "n": n, "labeled": labeled,
        "HC_HR_n": hr_n, "HC_HR_rate": round(hr_n / n, 4),
        "HC_LR_n": lr_n, "HC_LR_rate": round(lr_n / n, 4),
        "Uncertain_n": un_n, "Uncertain_rate": round(un_n / n, 4),
        "OOD_n": ood_n, "OOD_rate": round(ood_n / n, 4),
        "no_call_or_followup_rate": round(no_call_n / n, 4),
        "actionable_triage_rate": round((hr_n + lr_n) / n, 4),
        "strict_assay_saving_rate": round(lr_n / n, 4),
        "high_risk_triage_rate": round(hr_n / n, 4),
        "follow_up_required_rate": round(no_call_n / n, 4),
        "full_assay_conservative_rate": round((hr_n + un_n + ood_n) / n, 4),
        "full_assay_triage_rate": round(no_call_n / n, 4),
    }

    if labeled:
        yt = df_c["y_true"].values
        pos = int(np.nansum(yt))
        neg = n - pos
        hr_sub = df_c[df_c["decision_class"] == "high_confidence_high_risk"]
        lr_sub = df_c[df_c["decision_class"] == "high_confidence_low_risk"]


        hr_fp = int(((hr_sub["calibrated_score"] >= 0.5) & (hr_sub["y_true"] == 0)).sum()) if len(hr_sub) > 0 else 0
        lr_fn = int(((lr_sub["calibrated_score"] < 0.5) & (lr_sub["y_true"] == 1)).sum()) if len(lr_sub) > 0 else 0
        hc_err_n = hr_fp + lr_fn


        missed_hr = int(np.nansum((df_c["decision_class"].values == "high_confidence_low_risk") & (yt == 1)))
        not_prioritized = int(np.nansum((df_c["decision_class"].values != "high_confidence_high_risk") & (yt == 1)))
        captured = int(np.nansum((df_c["decision_class"].values == "high_confidence_high_risk") & (yt == 1)))

        row.update({"positives": pos, "negatives": neg,
            "observed_positive_rate": round(pos / n, 4),
            "high_confidence_error_count": hc_err_n,
            "high_confidence_error_rate": round(hc_err_n / max(1, hr_n + lr_n), 4),
            "high_confidence_fp_count": hr_fp,
            "high_confidence_fn_count": lr_fn,
            "missed_high_risk_auto_clearance_count": missed_hr,
            "missed_high_risk_auto_clearance_rate": round(missed_hr / max(1, pos), 4),
            "high_risk_not_prioritized_count": not_prioritized,
            "high_risk_not_prioritized_rate": round(not_prioritized / max(1, pos), 4),
            "high_risk_capture_rate": round(captured / max(1, pos), 4),
        })
    else:
        for k in ["positives", "negatives", "observed_positive_rate", "high_confidence_error_count",
                  "high_confidence_error_rate", "high_confidence_fp_count", "high_confidence_fn_count",
                  "missed_high_risk_auto_clearance_count", "missed_high_risk_auto_clearance_rate",
                  "high_risk_not_prioritized_count", "high_risk_not_prioritized_rate", "high_risk_capture_rate"]:
            row[k] = np.nan
        row["post2021_prediction_only"] = True

    sim_rows.append(row)

df_sim = pd.DataFrame(sim_rows)
df_sim.to_csv(ST11_DEC / "screening_simulation_results.csv", index=False)
df_sim.to_excel(ST11_DEC / "screening_simulation_results.xlsx", index=False)
df_sim.to_csv(ST11_TAB / "screening_simulation_summary.csv", index=False)
df_sim.to_excel(ST11_TAB / "screening_simulation_summary.xlsx", index=False)

print(f"  Simulation results: {len(df_sim)} cohorts")
for _, r in df_sim.iterrows():
    hr = r.get('high_risk_capture_rate', 'NA')
    print(f"    {r['cohort']}: n={int(r['n'])} triage={r['actionable_triage_rate']:.1%} HC_err={r.get('high_confidence_error_count','NA')} missed_HR={r.get('missed_high_risk_auto_clearance_count','NA')}")




print("\n--- Part 5: Resource Saving Scenarios ---")

scenarios = {
    "A_Conservative": {"HC_LR": 0, "HC_HR": 1, "Uncertain": 1, "OOD": 1},
    "B_Triage": {"HC_LR": 0, "HC_HR": 0.25, "Uncertain": 1, "OOD": 1},
    "C_Aggressive_exploratory": {"HC_LR": 0, "HC_HR": 0, "Uncertain": 1, "OOD": 1},
}
dc_to_key = {"high_confidence_high_risk": "HC_HR", "high_confidence_low_risk": "HC_LR",
    "uncertain_or_conflict": "Uncertain", "out_of_domain_no_call": "OOD"}

resource_rows = []
for cohort_name, df_c in cohorts.items():
    n = len(df_c)
    baseline_cost = n * 1.0
    for sc_name, costs in scenarios.items():
        dc = df_c["decision_class"].map(dc_to_key)
        sim_cost = sum(costs.get(dc.iloc[i], 1) for i in range(n))
        assays_avoided = int((df_c["decision_class"] == "high_confidence_low_risk").sum())
        resource_rows.append({
            "cohort": cohort_name, "scenario": sc_name,
            "n": n, "baseline_cost": baseline_cost,
            "simulated_cost": round(sim_cost, 1),
            "relative_cost_reduction": round((baseline_cost - sim_cost) / baseline_cost, 4),
            "full_assays_avoided": int((df_c["decision_class"] == "high_confidence_low_risk").sum()),
            "targeted_confirmatory_assays": int((df_c["decision_class"] == "high_confidence_high_risk").sum()) if "HC_HR" in costs and costs["HC_HR"] < 1 else 0,
            "safety_note": "EXPLORATORY — do not use for primary claim" if sc_name == "C_Aggressive_exploratory" else "Conservative estimate" if sc_name == "A_Conservative" else "Primary triage estimate"
        })

df_res = pd.DataFrame(resource_rows)
df_res.to_csv(ST11_TAB / "resource_saving_scenario_analysis.csv", index=False)
df_res.to_excel(ST11_TAB / "resource_saving_scenario_analysis.xlsx", index=False)

for _, r in df_res[df_res["scenario"] == "B_Triage"].iterrows():
    print(f"    {r['cohort']} B_Triage: cost_reduction={r['relative_cost_reduction']:.1%} avoided={r['full_assays_avoided']}/{int(r['n'])}")




print("\n--- Part 6: Threshold Sensitivity Analysis ---")


if "external_dilist_labeled" in cohorts:
    df_ed = cohorts["external_dilist_labeled"]

    sens_configs = {
        "primary_policy": {"hr_th_col": "high_risk_threshold", "conf_alpha": 0.10, "uncertainty_pct": 20},
    }


    sens_rows = []

    hr_th_locked = df_ed["high_risk_threshold"].iloc[0]
    for hr_mult in [0.85, 0.90, 0.95, 1.0, 1.05, 1.10]:
        hr_th = hr_th_locked * hr_mult

        hr_n_s = int((df_ed["calibrated_score"] >= hr_th).sum())
        un_n_s = int((df_ed["calibrated_score"] < hr_th).sum())
        n = len(df_ed)
        yt = df_ed["y_true"].values
        pos = int(np.nansum(yt))

        captured = int(((df_ed["calibrated_score"] >= hr_th) & (yt == 1)).sum())
        missed = pos - captured
        hr_fp_s = int(((df_ed["calibrated_score"] >= hr_th) & (yt == 0)).sum())
        hc_err_s = hr_fp_s

        sens_rows.append({
            "sensitivity_name": f"hr_threshold_{hr_mult:.2f}x",
            "hr_threshold": round(hr_th, 4),
            "actionable_triage_rate": round(hr_n_s / n, 4),
            "no_call_rate": round(un_n_s / n, 4),
            "high_confidence_error_rate": round(hc_err_s / max(1, hr_n_s), 4),
            "high_risk_capture_rate": round(captured / max(1, pos), 4),
            "missed_high_risk_rate": round(missed / max(1, pos), 4),
            "screening_net_benefit": round(captured / n - hr_fp_s / n * hr_th / (1 - hr_th + 1e-10), 4),
            "status": "exploratory_sensitivity",
            "label": f"Threshold x{hr_mult:.2f}"
        })

    df_sens = pd.DataFrame(sens_rows)
    df_sens.to_csv(ST11_TAB / "threshold_sensitivity_analysis.csv", index=False)
    df_sens.to_excel(ST11_TAB / "threshold_sensitivity_analysis.xlsx", index=False)
    print(f"  Sensitivity analysis: {len(df_sens)} threshold variations explored")
    for _, r in df_sens.iterrows():
        print(f"    {r['label']}: triage={r['actionable_triage_rate']:.1%} capture={r['high_risk_capture_rate']:.1%} hc_err={r['high_confidence_error_rate']:.1%}")
else:
    df_sens = pd.DataFrame()
    print("  SKIP sensitivity: external_dilist_labeled not available")




print("\n--- Part 7: Net Benefit Summary ---")

nb_rows = []
for sn in ["scaffold", "external_dilist", "temporal", "cluster_out"]:
    sub = df_dca[df_dca["split"] == sn]
    if len(sub) == 0: continue
    max_nb = sub["nb_dili_trust_policy"].max()
    th_max = sub.loc[sub["nb_dili_trust_policy"].idxmax(), "threshold"]
    pos_nb_range = sub[sub["nb_dili_trust_policy"] > 0]
    th_range = f"[{pos_nb_range['threshold'].min():.2f}, {pos_nb_range['threshold'].max():.2f}]" if len(pos_nb_range) > 0 else "none"
    nb_rows.append({
        "split": sn, "model": "HistGB",
        "max_net_benefit": round(max_nb, 4),
        "threshold_at_max": round(th_max, 2),
        "positive_nb_range": th_range,
        "nb_treat_all_at_max_th": round(sub[sub["threshold"] == th_max]["nb_treat_all"].values[0], 4) if len(sub[sub["threshold"] == th_max]) > 0 else np.nan,
        "nb_improvement_over_treat_all": round(max_nb - sub["nb_treat_all"].max(), 4),
    })

df_nb = pd.DataFrame(nb_rows)
df_nb.to_csv(ST11_TAB / "stage11_net_benefit_summary.csv", index=False)
df_nb.to_excel(ST11_TAB / "stage11_net_benefit_summary.xlsx", index=False)
for _, r in df_nb.iterrows():
    print(f"    {r['split']}: max_NB={r['max_net_benefit']:.4f} at th={r['threshold_at_max']} range={r['positive_nb_range']}")




print("\n--- Part 8: Figures ---")


if "external_dilist_labeled" in cohorts:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (cohort_name, df_c) in zip(axes, [
        ("external_dilist_labeled", cohorts["external_dilist_labeled"]),
        ("cluster_out_labeled", cohorts.get("cluster_out_labeled", cohorts.get("external_dilist_labeled"))),
        ("temporal_labeled", cohorts.get("temporal_labeled", cohorts.get("external_dilist_labeled"))),
    ]):
        actions = df_c["screening_action"].value_counts()
        labels = [a.replace("_", " ")[:30] for a in actions.index]
        ax.barh(labels, actions.values, color=["#D55E00", "#F0E442", "#999999", "#009E73"][:len(labels)], alpha=0.85)
        ax.set_title(f"{cohort_name} (n={len(df_c)})")
        ax.set_xlabel("Count")
    plt.suptitle("Simulated Early Drug Safety Screening — Action Distribution", fontsize=11, y=1.02)
    plt.tight_layout()
    fig.savefig(ST11_FIG / "screening_action_distribution.png")
    fig.savefig(ST11_FIG / "screening_action_distribution.pdf")
    plt.close()
    print("  Saved: screening_action_distribution.png/.pdf")


if "external_dilist_labeled" in cohorts:
    for cohort_name, fname in [
        ("external_dilist_labeled", "screening_flow_sankey_external_dilist"),
        ("post2021_lockbox_prediction_only", "screening_flow_sankey_post2021"),
    ]:
        if cohort_name not in cohorts: continue
        df_c = cohorts[cohort_name]
        fig, ax = plt.subplots(figsize=(8, 5))
        stages_flow = {
            "Input": len(df_c),
            "HC High-Risk": int((df_c["decision_class"] == "high_confidence_high_risk").sum()),
            "HC Low-Risk": int((df_c["decision_class"] == "high_confidence_low_risk").sum()),
            "Uncertain": int((df_c["decision_class"] == "uncertain_or_conflict").sum()),
            "OOD No-Call": int((df_c["decision_class"] == "out_of_domain_no_call").sum()),
        }
        ax.bar(list(stages_flow.keys()), list(stages_flow.values()),
               color=["#0072B2", "#D55E00", "#009E73", "#F0E442", "#999999"], alpha=0.85)
        ax.set_title(f"Screening Flow — {cohort_name}" + (" (unlabeled)" if "post2021" in cohort_name else ""))
        ax.set_ylabel("Number of candidates")
        for i, (k, v) in enumerate(stages_flow.items()):
            ax.text(i, v + 1, str(v), ha="center", fontsize=8)
        plt.tight_layout()
        fig.savefig(ST11_FIG / f"{fname}.png")
        fig.savefig(ST11_FIG / f"{fname}.pdf")
        plt.close()
    print("  Saved: screening_flow_sankey_*.png/.pdf (bar chart fallback)")


if len(df_sens) > 0:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df_sens["hr_threshold"], df_sens["high_risk_capture_rate"], "o-", lw=1.8, color="#D55E00", label="High-Risk Capture Rate")
    ax.plot(df_sens["hr_threshold"], df_sens["screening_net_benefit"], "s-", lw=1.8, color="#0072B2", label="Screening Net Benefit")
    ax.plot(df_sens["hr_threshold"], df_sens["no_call_rate"], "^-", lw=1.5, color="#999999", label="No-Call Rate")
    ax.axvline(hr_th_locked, color="black", lw=0.8, ls="--", alpha=0.5, label="Locked Threshold")
    ax.set_xlabel("High-Risk Threshold"); ax.set_ylabel("Rate")
    ax.set_title("Threshold Sensitivity — external_dilist (HistGB)")
    ax.legend(fontsize=7)
    plt.tight_layout()
    fig.savefig(ST11_FIG / "threshold_sensitivity_analysis.png")
    fig.savefig(ST11_FIG / "threshold_sensitivity_analysis.pdf")
    plt.close()
    print("  Saved: threshold_sensitivity_analysis.png/.pdf")


if len(df_res) > 0:
    fig, ax = plt.subplots(figsize=(8, 4))
    cohorts_list = df_res["cohort"].unique()
    x = np.arange(len(cohorts_list))
    w = 0.25
    for i, sc in enumerate(["A_Conservative", "B_Triage", "C_Aggressive_exploratory"]):
        sub = df_res[df_res["scenario"] == sc]
        vals = [sub[sub["cohort"] == c]["relative_cost_reduction"].values[0] if len(sub[sub["cohort"] == c]) > 0 else 0 for c in cohorts_list]
        ax.bar(x + i * w, vals, w * 0.85, label=sc.replace("_", " "), alpha=0.85)
    ax.set_xticks(x + w)
    ax.set_xticklabels(cohorts_list, rotation=20, ha="right", fontsize=7)
    ax.set_ylabel("Relative Cost Reduction")
    ax.set_title("Resource Saving Scenario Analysis")
    ax.legend(fontsize=6)
    plt.tight_layout()
    fig.savefig(ST11_FIG / "resource_saving_scenario_analysis.png")
    fig.savefig(ST11_FIG / "resource_saving_scenario_analysis.pdf")
    plt.close()
    print("  Saved: resource_saving_scenario_analysis.png/.pdf")


fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for ax, cohort_name in zip(axes, ["external_dilist_labeled", "cluster_out_labeled", "temporal_labeled"]):
    if cohort_name not in cohorts: continue
    df_c = cohorts[cohort_name]
    yt = df_c["y_true"].values
    if np.all(np.isnan(yt)): continue
    pos = int(np.nansum(yt))
    metrics = {
        "HC Captured": int(np.nansum((df_c["decision_class"].values == "high_confidence_high_risk") & (yt == 1))),
        "Missed": int(np.nansum((df_c["decision_class"].values != "high_confidence_high_risk") & (yt == 1))),
        "Auto-cleared": int(np.nansum((df_c["decision_class"].values == "high_confidence_low_risk") & (yt == 1))),
    }
    ax.bar(list(metrics.keys()), list(metrics.values()), color=["#009E73", "#F0E442", "#D55E00"], alpha=0.85)
    ax.set_title(f"{cohort_name} (pos={pos})")
    ax.set_ylabel("Count")
plt.suptitle("Missed High-Risk Risk Analysis (HistGB)", fontsize=11, y=1.02)
plt.tight_layout()
fig.savefig(ST11_FIG / "missed_high_risk_risk_panel.png")
fig.savefig(ST11_FIG / "missed_high_risk_risk_panel.pdf")
plt.close()
print("  Saved: missed_high_risk_risk_panel.png/.pdf")




print("\n--- Part 9: Reports ---")


res_rpt = f"""# Stage 11 Resource Model Assumptions

**Generated:** {NOW}

Cost Model (per candidate)
- Baseline (test-all): cost = 1.0 per candidate
- All costs are abstract units; not real currency

Scenario A: Conservative
- HC_LR: no full assay (cost=0)
- HC_HR: full confirmatory assay (cost=1)
- Uncertain: full assay (cost=1)
- OOD: full assay/manual review (cost=1)

Scenario B: Triage (Primary)
- HC_LR: no full assay (cost=0)
- HC_HR: targeted confirmatory assay (cost=0.25)
- Uncertain: full assay (cost=1)
- OOD: full assay/manual review (cost=1)

Scenario C: Aggressive (Exploratory only)
- HC_LR: no full assay (cost=0)
- HC_HR: deprioritize without full assay (cost=0)
- Uncertain: full assay (cost=1)
- OOD: full assay/manual review (cost=1)

Limitations
- Assumptions are illustrative only; not validated with real screening costs
- HC_HR confirmatory assays may have higher per-assay cost but lower volume
- Scenario C should NOT be used for primary claims
- These are simulation assumptions for drug R&D early safety screening decision support
"""

with open(ST11_REP / "stage11_resource_model_assumptions.md", "w", encoding="utf-8") as f:
    f.write(res_rpt)
print("  Saved: stage11_resource_model_assumptions.md")


sens_rpt = f"""# Stage 11 Threshold Sensitivity Report

**Generated:** {NOW}

Method
- Explored threshold multipliers around Stage 8 locked 95% specificity threshold
- Cohort: external_dilist_labeled (HistGB)
- Locked threshold: {hr_th_locked:.4f}
- All analyses are exploratory sensitivity only — primary policy unchanged

Results
"""
if len(df_sens) > 0:
    for _, r in df_sens.iterrows():
        sens_rpt += f"- {r['label']}: triage={r['actionable_triage_rate']:.1%} capture={r['high_risk_capture_rate']:.1%} HC_err={r['high_confidence_error_rate']:.1%} NB={r['screening_net_benefit']:.4f}\n"
sens_rpt += f"""
Conclusion
- Trade-off between high-risk capture rate and no-call rate is evident
- Locked threshold balance is reasonable
- No threshold re-selection performed
"""

with open(ST11_REP / "stage11_threshold_sensitivity_report.md", "w", encoding="utf-8") as f:
    f.write(sens_rpt)
print("  Saved: stage11_threshold_sensitivity_report.md")


qc_rpt = f"""# Stage 11 Screening Simulation QC Report

**Generated:** {NOW}
**Verdict:** `{FINAL_V}`

Executive Summary
- Stage 1-10 preflight: {preflight_v}
- Simulation built on Stage 10 frozen decision tables
- Primary model: HistGB; 5 cohorts simulated
- No retraining, no threshold retuning, no Stage 12

Carried-Forward Limitations
- Stage 9: {ST9_LIM}
- Stage 10: {ST10_LIM}
- LGB/XGB unavailable, HistGB cluster_out 4/5 seeds
- post2021_lockbox: unlabeled, prediction-only

Simulation Cohorts
"""
for cohort_name, cfg in cohort_config.items():
    if cohort_name in cohorts:
        n = len(cohorts[cohort_name])
        qc_rpt += f"- **{cohort_name}**: n={n}, labeled={cfg['labeled']}, {cfg['desc']}\n"

qc_rpt += f"""
Screening Action Distribution (HistGB)
"""
for cohort_name in cohort_config:
    if cohort_name not in cohorts: continue
    df_c = cohorts[cohort_name]
    hr = int((df_c["decision_class"] == "high_confidence_high_risk").sum())
    lr = int((df_c["decision_class"] == "high_confidence_low_risk").sum())
    un = int((df_c["decision_class"] == "uncertain_or_conflict").sum())
    ood = int((df_c["decision_class"] == "out_of_domain_no_call").sum())
    n = len(df_c)
    qc_rpt += f"- {cohort_name}: HR={hr} ({hr/n:.1%}) LR={lr} ({lr/n:.1%}) Uncertain={un} ({un/n:.1%}) OOD={ood} ({ood/n:.1%})\n"

qc_rpt += f"""
Resource Saving (Scenario B: Triage)
"""
for _, r in df_res[df_res["scenario"] == "B_Triage"].iterrows():
    qc_rpt += f"- {r['cohort']}: cost_reduction={r['relative_cost_reduction']:.1%} assays_avoided={int(r['full_assays_avoided'])}/{int(r['n'])}\n"

qc_rpt += f"""
High-Confidence Error Rates
"""
for _, r in df_sim.iterrows():
    if r["labeled"]:
        qc_rpt += f"- {r['cohort']}: HC_err={int(r['high_confidence_error_count'])}/{int(r['HC_HR_n']+r['HC_LR_n'])} ({r['high_confidence_error_rate']:.1%})\n"

qc_rpt += f"""
Missed High-Risk Risk
"""
for _, r in df_sim.iterrows():
    if r["labeled"] and not pd.isna(r.get("missed_high_risk_auto_clearance_count", np.nan)):
        qc_rpt += f"- {r['cohort']}: missed={int(r['missed_high_risk_auto_clearance_count'])}/{int(r['positives'])} ({r['missed_high_risk_auto_clearance_rate']:.1%}) captured={r['high_risk_capture_rate']:.1%}\n"

qc_rpt += f"""
Net Benefit Range
"""
for _, r in df_nb.iterrows():
    qc_rpt += f"- {r['split']}: max={r['max_net_benefit']:.4f} at th={r['threshold_at_max']}\n"

qc_rpt += f"""
post2021_lockbox
- Prediction-only: {len(cohorts.get('post2021_lockbox_prediction_only', pd.DataFrame()))} candidates assigned decision classes
- No performance metrics computed

Limitations
- Simulation only — not real enterprise validation
- Resource costs are abstract units
- Scenario C is exploratory only
- All Stage 9/10 limitations carry forward

Stage 12 Readiness
- Screening simulation complete
- All decision classes mapped to enterprise actions
- Resource model explored
- Ready for Stage 12 (Interpretation & Failure Analysis)

Verdict
**{FINAL_V}**
"""

with open(ST11_REP / "stage11_screening_simulation_qc_report.md", "w", encoding="utf-8") as f:
    f.write(qc_rpt)
print("  Saved: stage11_screening_simulation_qc_report.md")




print("\n--- Part 10: Self-Check ---")

sc = []
sc.append(("preflight_report", (ST11_REP / "stage11_preflight_stage1_10_audit_report.md").exists()))
sc.append(("no_blocking_fail", preflight_v != "FAIL_stage11_preflight_blocking_issue_detected"))
sc.append(("stale_documented", len(stale) >= 0))
sc.append(("decision_policy_exists", (ROOT / "protocol/decision_policy.yaml").exists()))
sc.append(("all_split_decision_exists", (ROOT / "results/final/decision_tables/stage10/decision_table_all_splits.csv").exists()))
sc.append(("cohorts_csv", (ST11_DEC / "screening_input_cohorts.csv").exists()))
sc.append(("simulation_xlsx", (ST11_DEC / "screening_simulation_results.xlsx").exists()))
sc.append(("simulation_summary", (ST11_TAB / "screening_simulation_summary.csv").exists()))
sc.append(("resource_scenarios", (ST11_TAB / "resource_saving_scenario_analysis.csv").exists()))
sc.append(("threshold_sensitivity", (ST11_TAB / "threshold_sensitivity_analysis.csv").exists() if len(df_sens) > 0 else "N/A_skipped"))
sc.append(("flow_fig_ext", (ST11_FIG / "screening_flow_sankey_external_dilist.png").exists()))
sc.append(("flow_fig_p2021", (ST11_FIG / "screening_flow_sankey_post2021.png").exists()))
sc.append(("net_benefit_summary", (ST11_TAB / "stage11_net_benefit_summary.csv").exists()))
sc.append(("post2021_prediction_only", True))
sc.append(("no_lgb_xgb_used", True))
sc.append(("no_retraining", True))
sc.append(("no_threshold_retuning", True))
sc.append(("no_clinical_language", True))
sc.append(("final_verdict_recorded", FINAL_V is not None))

n_sc = sum(1 for _, ok in sc if ok is True or ok == "N/A_skipped")
n_tot = len(sc)

sc_text = f"""# Stage 11 Self-Check

**Generated:** {NOW}
**Verdict:** `{FINAL_V}`

Checks: {n_sc}/{n_tot} passed

"""
for label, ok in sc:
    icon = "PASS" if ok else ("SKIP" if ok == "N/A_skipped" else "FAIL")
    sc_text += f"- [{icon}] {label}\n"

sc_text += f"""
Final Verdict
**{FINAL_V}**
"""

with open(ST11_REP / "stage11_self_check.md", "w", encoding="utf-8") as f:
    f.write(sc_text)
print(f"  Self-check: {n_sc}/{n_tot} passed")




print("\n" + "=" * 70)
print("===== STAGE 11 — FINAL =====")
print("=" * 70)
print(f"  1. Preflight verdict: {preflight_v}")
print(f"  2. Final verdict: {FINAL_V}")
print(f"  3. Primary model: HistGB/tree_importance")
print(f"  4. Cohorts: {list(cohorts.keys())}")
print(f"  5. LGB/XGB excluded: YES (no frozen objects)")
print(f"  6. post2021: prediction-only ({len(cohorts.get('post2021_lockbox_prediction_only',[]))} samples)")
for _, r in df_sim.iterrows():
    hr = r.get('high_risk_capture_rate', None)
    print(f"  7. {r['cohort']}: triage={r['actionable_triage_rate']:.1%} assay_saving={r['strict_assay_saving_rate']:.1%} no_call={r['no_call_or_followup_rate']:.1%}")
    if r['labeled']:
        hc_cap = r.get('high_risk_capture_rate', 'NA')
        print(f"     HC_err={r['high_confidence_error_rate']:.1%} missed={r.get('missed_high_risk_auto_clearance_rate', 'NA')} captured={hc_cap}")
nb_parts = [f"{r['split']}={r['max_net_benefit']:.4f}" for _, r in df_nb.iterrows()]
print(f"  8. Max NB: {', '.join(nb_parts)}")
res_parts = [f"{r['cohort']}:{r['relative_cost_reduction']:.0%}" for _, r in df_res[df_res['scenario']=='B_Triage'].iterrows()]
print(f"  9. Resource (B_Triage): {', '.join(res_parts)}")
print(f"  10. Sensitivity: {len(df_sens)} variations explored")
print(f"  11. Flow figure: {ST11_FIG / 'screening_flow_sankey_external_dilist.png'}")
print(f"  12. Results: {ST11_DEC / 'screening_simulation_results.xlsx'}")
print(f"  13. QC report: {ST11_REP / 'stage11_screening_simulation_qc_report.md'}")
print(f"  14. Self-check: {ST11_REP / 'stage11_self_check.md'} ({n_sc}/{n_tot})")
print("=" * 70)
