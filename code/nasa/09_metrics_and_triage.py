
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (PREDICTIONS, PRIMARY_ALPHA, REPORTS, SPLIT_NAMES, TABLES,
                    ensure_dirs, metric_set, reliability_curve, uncertainty_score)

from sklearn.metrics import roc_auc_score

SPLITS = SPLIT_NAMES
MODELS = ["HistGB", "ET"]
ABSTENTION_RATES = [0.0, 0.05, 0.10, 0.20, 0.30]
OOD_EXTREME = 0.15


def conformal_sets_from(scores, q_hat):
    sets = []
    for s in scores:
        if s >= 1 - q_hat:
            sets.append("both" if (1 - s) >= 1 - q_hat else "positive")
        elif (1 - s) >= 1 - q_hat:
            sets.append("negative")
        else:
            sets.append("empty")
    return np.array(sets)


def main():
    ensure_dirs()
    thresholds = pd.read_csv(TABLES / "threshold_lock_table.csv")
    conformal_table = pd.read_csv(TABLES / "conformal_coverage.csv")
    ad_features = {s: pd.read_csv(PREDICTIONS / f"ad_features_{s}.csv") for s in SPLITS}

    task1, task2, curve_rows = [], [], []
    utility, abst_rows, unc_rows = [], [], []
    triage_rows, triage_summary = [], []

    for split in SPLITS:
        for model_name in MODELS:
            thr = thresholds[(thresholds["split"] == split) &
                             (thresholds["model"] == model_name)].iloc[0]

            for partition in ["valid", "test"]:
                d = pd.read_csv(PREDICTIONS /
                                f"{split}_{model_name}_locked_calibrated_{partition}.csv")
                y = d["y_true"].values.astype(int)
                raw = d["raw_score"].values.astype(float)
                cal = d["calibrated_score"].values.astype(float)

                for score_name, score in [("raw", raw), ("calibrated", cal)]:
                    m = metric_set(y, score)
                    task1.append({
                        "split": split, "model": model_name, "partition": partition,
                        "score": score_name, "n": m["n"], "positives": m["positives"],
                        "prevalence": m["prevalence"],
                        "auroc": m["auroc"], "auroc_ci_lower": m["auroc_ci_lower"],
                        "auroc_ci_upper": m["auroc_ci_upper"],
                        "auprc": m["auprc"], "auprc_ci_lower": m["auprc_ci_lower"],
                        "auprc_ci_upper": m["auprc_ci_upper"],
                        "prevalence_baseline_auprc": m["prevalence"],
                    })

                m = metric_set(y, cal)
                task2.append({
                    "split": split, "model": model_name, "partition": partition,
                    "n": m["n"], "positives": m["positives"], "prevalence": m["prevalence"],
                    "brier": m["brier"], "brier_ci_lower": m["brier_ci_lower"],
                    "brier_ci_upper": m["brier_ci_upper"],
                    "brier_skill_score": m["brier_skill_score"],
                    "brier_skill_score_ci_lower": m["brier_skill_score_ci_lower"],
                    "brier_skill_score_ci_upper": m["brier_skill_score_ci_upper"],
                    "calibration_bias": m["calibration_bias"],
                    "calibration_intercept": m["calibration_intercept"],
                    "calibration_slope": m["calibration_slope"],
                    "ece_q5": m["ece_q5"], "ece_u10": m["ece_u10"], "nll": m["nll"],
                })

                for rc in reliability_curve(y, cal, 10):
                    curve_rows.append({"split": split, "model": model_name,
                                       "partition": partition, "score": "calibrated", **rc})
                for rc in reliability_curve(y, raw, 10):
                    curve_rows.append({"split": split, "model": model_name,
                                       "partition": partition, "score": "raw", **rc})

                u = uncertainty_score(cal)
                u_thr = float(thr["u80_threshold"])
                ad = ad_features[split].set_index("sample_id")
                tanimoto = ad.loc[d["sample_id"].values, "max_tanimoto_to_train"].values

                conf_row = conformal_table[
                    (conformal_table["split"] == split) &
                    (conformal_table["model"] == model_name) &
                    (conformal_table["mode"] == "validation_locked") &
                    (conformal_table["alpha"] == PRIMARY_ALPHA) &
                    (conformal_table["evaluation_partition"] == partition)].iloc[0]
                conformal_set = conformal_sets_from(cal, float(conf_row["q_hat"]))
                utility.append({
                    "split": split, "model": model_name, "partition": partition,
                    "n": len(y),
                    "no_call_rate_u80": round(float((u > u_thr).mean()), 4),
                    "conformal_both_rate": round(float((conformal_set == "both").mean()), 4),
                    "conformal_empty_rate": round(float((conformal_set == "empty").mean()), 4),
                    "ood_extreme_rate": round(float((tanimoto < OOD_EXTREME).mean()), 4),
                    "ad_warning_rate_lt0.3": round(float((tanimoto < 0.3).mean()), 4),
                    "median_max_tanimoto": round(float(np.median(tanimoto)), 4),
                })


                yp = d["y_pred_0p5"].values.astype(int)
                err = (yp != y).astype(int)
                if 0 < err.sum() < len(err):
                    unc_rows.append({
                        "split": split, "model": model_name, "partition": partition,
                        "n": len(y), "n_errors": int(err.sum()),
                        "error_rate": round(float(err.mean()), 4),
                        "auroc_u_for_error": round(float(roc_auc_score(err, u)), 4),
                        "auroc_u_for_error_raw": round(float(roc_auc_score(err, raw)), 4),
                        "u80_threshold": round(u_thr, 6),
                        "error_rate_retained_u80": round(float(err[u <= u_thr].mean()), 4),
                        "error_rate_abstained_u80": round(float(err[u > u_thr].mean()), 4)
                        if (u > u_thr).sum() else np.nan,
                        "n_abstained_u80": int((u > u_thr).sum()),
                        "error_rate_retained_conformal_singleton": round(
                            float(err[conformal_set != "both"].mean()), 4)
                        if (conformal_set != "both").sum() else np.nan,
                    })

                for rate in ABSTENTION_RATES:
                    k = int(len(y) * rate)
                    order = np.argsort(-u)
                    keep = order[k:] if k > 0 else order
                    yk, ck = y[keep], cal[keep]
                    mm = metric_set(yk, ck, nboot=0) if len(set(yk)) > 1 else None
                    abst_rows.append({
                        "split": split, "model": model_name, "partition": partition,
                        "abstention_rate": rate,
                        "retained_n": len(keep), "abstained_n": k,
                        "retained_fraction": round(len(keep) / len(y), 4),
                        "auroc": mm["auroc"] if mm else np.nan,
                        "auprc": mm["auprc"] if mm else np.nan,
                        "brier": mm["brier"] if mm else np.nan,
                        "error_rate": round(float(((ck >= 0.5).astype(int) != yk).mean()), 4),
                        "abstained_positive_share": round(float(y[order[:k]].mean()), 4) if k else 0.0,
                    })


                if partition == "test":
                    th_spec = float(thr["th_spec95"])
                    th_sens = float(thr["th_sens95"])
                    ood = tanimoto < OOD_EXTREME
                    no_call = ood | (conformal_set == "both") | (u > u_thr)
                    high = (cal >= th_spec) & ~no_call
                    low = (cal <= th_sens) & ~no_call
                    mid = ~high & ~low & ~no_call



                    high_dom = (cal >= th_spec) & ~ood
                    low_dom = (cal <= th_sens) & ~ood

                    def rate(mask):
                        return round(float(mask.mean()), 4)

                    def precision(mask):
                        return round(float(y[mask].mean()), 4) if mask.sum() else np.nan

                    triage_summary.append({
                        "split": split, "model": model_name, "n": len(y),
                        "positives": int(y.sum()), "prevalence": round(float(y.mean()), 4),
                        "th_spec95": th_spec, "th_sens95": th_sens, "u80_threshold": round(u_thr, 6),
                        "high_risk_n": int(high.sum()), "high_risk_rate": rate(high),
                        "high_risk_precision": precision(high),
                        "high_risk_recall": round(float(y[high].sum() / max(y.sum(), 1)), 4),
                        "low_score_n": int(low.sum()), "low_score_rate": rate(low),
                        "low_score_negative_share": round(float((1 - y[low]).mean()), 4) if low.sum() else np.nan,
                        "low_score_recall_negatives": round(float((1 - y[low]).sum() / max((1 - y).sum(), 1)), 4),
                        "intermediate_n": int(mid.sum()), "intermediate_rate": rate(mid),
                        "no_call_n": int(no_call.sum()), "no_call_rate": rate(no_call),
                        "no_call_from_u80": int(((u > u_thr) & ~ood & (conformal_set != "both")).sum()),
                        "no_call_from_conformal_both": int(((conformal_set == "both") & ~ood & (u <= u_thr)).sum()),
                        "no_call_from_ood": int(ood.sum()),
                        "sens_high_risk_n_domain_guard_only": int(high_dom.sum()),
                        "sens_high_risk_rate_domain_guard_only": rate(high_dom),
                        "sens_high_risk_precision_domain_guard_only": precision(high_dom),
                        "sens_high_risk_recall_domain_guard_only": round(
                            float(y[high_dom].sum() / max(y.sum(), 1)), 4),
                        "sens_low_score_n_domain_guard_only": int(low_dom.sum()),
                        "sens_low_score_negative_share_domain_guard_only": round(
                            float((1 - y[low_dom]).mean()), 4) if low_dom.sum() else np.nan,
                    })

                    for i, sid in enumerate(d["sample_id"].values):
                        if high[i]:
                            decision = "high_score_high_risk_priority"
                        elif low[i]:
                            decision = "low_score_candidate_group"
                        elif no_call[i]:
                            decision = "out_of_domain_no_call" if ood[i] else "uncertain_or_conflict"
                        else:
                            decision = "intermediate"
                        triage_rows.append({
                            "sample_id": sid, "split": split, "model": model_name,
                            "y_true": int(y[i]), "calibrated_score": round(float(cal[i]), 6),
                            "uncertainty_U": round(float(u[i]), 6),
                            "max_tanimoto_to_train": round(float(tanimoto[i]), 4),
                            "conformal_prediction_set": conformal_set[i],
                            "decision_class": decision,
                            "no_call": bool(no_call[i]),
                            "correct_at_0p5": bool((cal[i] >= 0.5) == (y[i] == 1)),
                        })

    pd.DataFrame(task1).to_csv(TABLES / "task1_ranking.csv", index=False)
    pd.DataFrame(task2).to_csv(TABLES / "task2_probability_reliability.csv", index=False)
    pd.DataFrame(curve_rows).to_csv(TABLES / "task2_reliability_curve.csv", index=False)
    pd.DataFrame(utility).to_csv(TABLES / "task3_decision_utility.csv", index=False)
    pd.DataFrame(abst_rows).to_csv(TABLES / "abstention_curve.csv", index=False)
    pd.DataFrame(unc_rows).to_csv(TABLES / "uncertainty_error_discrimination.csv", index=False)
    pd.DataFrame(triage_rows).to_csv(TABLES / "triage_table.csv", index=False)
    pd.DataFrame(triage_summary).to_csv(TABLES / "triage_summary.csv", index=False)

    t1 = pd.DataFrame(task1)
    t2 = pd.DataFrame(task2)
    t3 = pd.DataFrame(triage_summary)
    unc = pd.DataFrame(unc_rows)

    lines = [
        "# Stage 09 - three-task reliability assessment",
        "",
        "## Task I - Ranking (held-out test partitions)",
        "",
        "| split | model | score | n | AUROC (95% CI) | AUPRC (95% CI) | prevalence |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, r in t1[t1["partition"] == "test"].iterrows():
        lines.append(
            f"| {r['split']} | {r['model']} | {r['score']} | {r['n']} |"
            f" {r['auroc']} ({r['auroc_ci_lower']}-{r['auroc_ci_upper']}) |"
            f" {r['auprc']} ({r['auprc_ci_lower']}-{r['auprc_ci_upper']}) | {r['prevalence']} |")

    lines += [
        "",
        "## Task II - Probability reliability (held-out test partitions)",
        "",
        "| split | model | Brier | BSS | cal. bias | cal. intercept | cal. slope | ECE(q5) | NLL |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r in t2[t2["partition"] == "test"].iterrows():
        lines.append(
            f"| {r['split']} | {r['model']} | {r['brier']} | {r['brier_skill_score']} |"
            f" {r['calibration_bias']} | {r['calibration_intercept']} | {r['calibration_slope']} |"
            f" {r['ece_q5']} | {r['nll']} |")

    lines += [
        "",
        "## Task III - Decision utility (held-out test partitions)",
        "",
        "| split | model | no-call (U80) | conf. both | conf. empty | OOD <0.15 | AD warn <0.3 |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, r in pd.DataFrame(utility)[pd.DataFrame(utility)["partition"] == "test"].iterrows():
        lines.append(
            f"| {r['split']} | {r['model']} | {r['no_call_rate_u80']} | {r['conformal_both_rate']} |"
            f" {r['conformal_empty_rate']} | {r['ood_extreme_rate']} | {r['ad_warning_rate_lt0.3']} |")

    lines += [
        "",
        "### Fixed-threshold triage",
        "",
        "| split | model | high-risk n (%) | high-risk precision | high-risk recall | low-score n (%) | no-call n (%) |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, r in t3.iterrows():
        lines.append(
            f"| {r['split']} | {r['model']} | {r['high_risk_n']} ({r['high_risk_rate']}) |"
            f" {r['high_risk_precision']} | {r['high_risk_recall']} |"
            f" {r['low_score_n']} ({r['low_score_rate']}) |"
            f" {r['no_call_n']} ({r['no_call_rate']}) |")

    lines += [
        "",
        "### Fixed-threshold triage, domain guard only (sensitivity)",
        "",
        "The locked policy places the uncertainty no-call band above the 95%",
        "specificity threshold whenever the calibrated score distribution is",
        "concentrated near zero; the high-risk tier then empties out. Removing the",
        "uncertainty and conformal no-call rules (keeping only the out-of-domain",
        "guard) isolates the raw threshold behaviour.",
        "",
        "| split | model | high-risk n (%) | precision | recall | low-score n (%) | low-score negative share |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, r in t3.iterrows():
        lines.append(
            f"| {r['split']} | {r['model']} |"
            f" {r['sens_high_risk_n_domain_guard_only']} ({r['sens_high_risk_rate_domain_guard_only']}) |"
            f" {r['sens_high_risk_precision_domain_guard_only']} |"
            f" {r['sens_high_risk_recall_domain_guard_only']} |"
            f" {r['sens_low_score_n_domain_guard_only']} |"
            f" {r['sens_low_score_negative_share_domain_guard_only']} |")

    lines += [
        "",
        "### Can uncertainty identify incorrect predictions?",
        "",
        "| split | model | partition | error rate | AUROC(U -> error) | error rate retained | error rate abstained |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, r in unc.iterrows():
        lines.append(
            f"| {r['split']} | {r['model']} | {r['partition']} | {r['error_rate']} |"
            f" {r['auroc_u_for_error']} | {r['error_rate_retained_u80']} |"
            f" {r['error_rate_abstained_u80']} |")

    lines += [
        "",
        "Definitions: high-risk group = calibrated score >= validation-locked 95%",
        "specificity threshold, not OOD, not no-call. Low-score group = calibrated",
        "score <= validation-locked 95% sensitivity threshold, not OOD, not no-call",
        "(the DILI release also stores a more conservative near-zero threshold that",
        "is not reproducible from the released scripts; th_sens95 is used here).",
        "No-call = max Tanimoto < 0.15, or conformal set = both at alpha = 0.10, or",
        "U(p) above the validation 80th percentile.",
        "",
    ]
    (REPORTS / "stage09_three_task_report.md").write_text("\n".join(lines), encoding="utf-8")

    print(t3.to_string())
    print(unc.to_string())
    print("saved task tables")


if __name__ == "__main__":
    main()
