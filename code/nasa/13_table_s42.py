
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import PRIMARY_ALPHA, TABLES, TASKS, PREDICTIONS, REPORTS

MODEL = "HistGB"


def binom_sf(k, n, p):
    if n == 0:
        return float("nan")
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))


def collect():
    rows = []
    for split in TASKS:
        t1 = pd.read_csv(TABLES / "task1_ranking.csv")
        t2 = pd.read_csv(TABLES / "task2_probability_reliability.csv")
        conf = pd.read_csv(TABLES / "conformal_coverage.csv")
        tri = pd.read_csv(TABLES / "triage_summary.csv")

        r1 = t1[(t1.split == split) & (t1.model == MODEL) & (t1.partition == "test")].iloc[0]
        r2 = t2[(t2.split == split) & (t2.model == MODEL) & (t2.partition == "test")].iloc[0]
        rc = conf[(conf.split == split) & (conf.model == MODEL) & (conf.alpha == PRIMARY_ALPHA)
                  & (conf["mode"] == "validation_locked")
                  & (conf.evaluation_partition == "test")]
        rt = tri[(tri.split == split) & (tri.model == MODEL)].iloc[0]

        n = int(r1["n"])
        pos = int(r1["positives"])
        base_neg = 1.0 - pos / n

        low_n = int(rt["low_score_n"])
        if low_n > 0 and pd.notna(rt["low_score_negative_share"]):
            share = float(rt["low_score_negative_share"])
            k = int(round(share * low_n))
            p = binom_sf(k, low_n, base_neg)
        else:
            share, p = float("nan"), float("nan")

        rows.append({
            "dataset": TASKS[split]["label"],
            "split": split,
            "shift": TASKS[split]["shift"],
            "test_n": n,
            "pos": pos,
            "neg": n - pos,
            "prevalence": round(pos / n, 4),
            "auroc": r1["auroc"],
            "auroc_lo": r1["auroc_ci_lower"],
            "auroc_hi": r1["auroc_ci_upper"],
            "auprc": r1["auprc"],
            "prevalence_baseline_auprc": r1["prevalence_baseline_auprc"],
            "bss": r2["brier_skill_score"],
            "calibration_slope": r2["calibration_slope"],
            "calibration_bias": r2["calibration_bias"],
            "ece_q5": r2["ece_q5"],
            "conformal_coverage": float(rc.iloc[0]["observed_coverage"]) if len(rc) else np.nan,
            "conformal_neg_coverage": float(rc.iloc[0]["negative_coverage"]) if len(rc) else np.nan,
            "low_score_n": low_n,
            "low_score_negative_share": share,
            "low_score_base_neg_rate": round(base_neg, 4),
            "low_score_enrichment": round(share / base_neg, 4) if low_n else np.nan,
            "exclusion_binom_p": p,
            "high_risk_n": int(rt["high_risk_n"]),
            "high_risk_precision": (round(float(rt["high_risk_precision"]), 4)
                                    if pd.notna(rt["high_risk_precision"]) else np.nan),
            "no_call_rate": float(rt["no_call_rate"]),
        })
    return pd.DataFrame(rows)


def markdown(df):
    def f(v, d=3):
        return "n/a" if pd.isna(v) else f"{v:.{d}f}"

    L = [
        "# Supplementary Table S42 - cross-project generalization",
        "",
        "Third tier of the generalization ladder. One development project (KC1) is",
        "split into train/valid/test; three further projects (PC1, JM1, KC2) serve as",
        "external held-out cohorts. All four tasks share the same development",
        "partition, so the locked calibrator, conformal quantile and decision",
        "thresholds are applied unchanged. Primary model HistGB; conformal figures at",
        "alpha = 0.10.",
        "",
        "| dataset | shift | test n | pos | prev | AUROC (95% CI) | AUPRC | base AUPRC | BSS | cal slope | ECE | conformal cov | neg cov | low-score n | low-score neg share | base neg rate | enrichment |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r in df.iterrows():
        L.append(
            f"| {r['dataset']} | {r['shift']} | {r['test_n']} | {r['pos']} | {f(r['prevalence'])} |"
            f" {f(r['auroc'])} ({f(r['auroc_lo'])}-{f(r['auroc_hi'])}) | {f(r['auprc'])} |"
            f" {f(r['prevalence_baseline_auprc'])} | {f(r['bss'])} | {f(r['calibration_slope'])} |"
            f" {f(r['ece_q5'])} | {f(r['conformal_coverage'])} | {f(r['conformal_neg_coverage'])} |"
            f" {r['low_score_n']} | {f(r['low_score_negative_share'])} |"
            f" {f(r['low_score_base_neg_rate'])} | {f(r['low_score_enrichment'], 2)} |")
    L += [
        "",
        "## Columns",
        "",
        "- **BSS** - Brier skill score against the base-rate reference; negative means",
        "  the calibrated probabilities are worse than predicting the cohort prevalence.",
        "- **cal slope** - Cox logistic recalibration slope; 1.0 is ideal.",
        "- **conformal cov / neg cov** - marginal and negative-class empirical coverage",
        "  of the validation-locked conformal sets at alpha = 0.10 (target 0.90).",
        "- **low-score n** - modules below the validation-locked low-score cut, after",
        "  removing out-of-domain and no-call cases.",
        "- **enrichment** - negative share of the low-score group divided by the cohort",
        "  base negative rate. Values near 1.0 mean the low-score group is no more",
        "  negative than the cohort at large, so it carries no exclusion signal.",
        "",
        "No binary exclusion verdict is reported. The diagnostic quantity is the",
        "enrichment magnitude: a low-score group that is barely more negative than the",
        "cohort base rate cannot support automated deprioritisation regardless of the",
        "significance of that difference.",
        "",
    ]
    return "\n".join(L)


if __name__ == "__main__":
    df = collect()
    out_csv = TABLES / "Table_S42_cross_project_generalization.csv"
    out_md = REPORTS / "Table_S42_cross_project_generalization.md"
    df.to_csv(out_csv, index=False)
    out_md.write_text(markdown(df), encoding="utf-8")
    for stale in (TABLES / "Table_S42_cross_domain_tabular.csv",
                  REPORTS / "Table_S42_cross_domain_tabular.md"):
        stale.unlink(missing_ok=True)
    print(f"saved {out_csv}")
    print(f"saved {out_md}")
    print()
    print(markdown(df))
