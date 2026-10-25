#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
PRED = ROOT / "results" / "predictions" / "prediction_scores_388.csv"

SPLITS = ["scaffold", "external_dilist", "temporal", "cluster_out"]
LABELS = {
    "scaffold": "Scaffold test",
    "external_dilist": "External DILIst",
    "temporal": "Temporal proxy test",
    "cluster_out": "Cluster-out test",
}


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (centre - half, centre + half)


def calibration_slope(y, p):
    eps = 1e-6
    z = np.log(np.clip(p, eps, 1 - eps) / (1 - np.clip(p, eps, 1 - eps)))
    model = LogisticRegression(C=1e6, max_iter=1000)
    model.fit(z.reshape(-1, 1), y)
    return model.coef_[0][0]


def calibration_intercept(y, p):
    eps = 1e-6
    z = np.log(np.clip(p, eps, 1 - eps) / (1 - np.clip(p, eps, 1 - eps)))
    a = 0.0
    for _ in range(100):
        q = 1 / (1 + np.exp(-(a + z)))
        a -= (q - y).sum() / (q * (1 - q)).sum()
    return a


def ece(y, p, bins=5):
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi) if hi < 1 else (p >= lo) & (p <= hi)
        if m.sum():
            total += m.sum() / len(p) * abs(p[m].mean() - y[m].mean())
    return total


def main():
    df = pd.read_csv(PRED)

    rows_rank, rows_prob, rows_conf, rows_triage = [], [], [], []
    for split in SPLITS:
        for model in ["HistGB", "ET"]:
            d = df[(df.split_name == split) & (df.model == model)]
            if d.empty:
                continue
            y = d.y_true.to_numpy(dtype=float)
            p = d.calibrated_score.to_numpy(dtype=float)
            brier = brier_score_loss(y, p)
            null = brier_score_loss(y, np.full_like(p, y.mean()))
            slope = calibration_slope(y, p)
            intercept = calibration_intercept(y, p)

            rows_rank.append(
                dict(split=LABELS[split], model=model, n=len(d),
                     auroc=roc_auc_score(y, p), auprc=average_precision_score(y, p),
                     brier=brier, bss=1 - brier / null)
            )
            rows_prob.append(
                dict(split=LABELS[split], model=model, n=len(d),
                     positive_rate=y.mean(), mean_p=p.mean(), bias=p.mean() - y.mean(),
                     brier=brier, null_brier=null, bss=1 - brier / null,
                     ece=ece(y, p), intercept=intercept, slope=slope)
            )

            sets = d.conformal_set_alpha_0_10.to_numpy()
            if not (sets == "no_conformal").all():
                in_set = (((y == 1) & np.isin(sets, ["positive", "both"]))
                          | ((y == 0) & np.isin(sets, ["negative", "both"])))
                size = np.where(sets == "both", 2, 1)
                rows_conf.append(
                    dict(split=LABELS[split], model=model, n=len(d),
                         target=0.90, observed=in_set.mean(),
                         difference=in_set.mean() - 0.90, mean_set_size=size.mean(),
                         single_label_rate=(size == 1).mean(),
                         both_label_rate=(size == 2).mean(),
                         empty_rate=(size == 0).mean(),
                         positive_coverage=in_set[y == 1].mean(),
                         negative_coverage=in_set[y == 0].mean())
                )

            thr = d.high_risk_threshold.iloc[0]
            selected = p >= thr
            tp = int((y[selected] == 1).sum())
            fp = int((y[selected] == 0).sum())
            k = tp + fp
            lo, hi = wilson_ci(tp, k) if k else (np.nan, np.nan)
            rows_triage.append(
                dict(split=LABELS[split], model=model, n=len(d),
                     threshold=thr, selected=k, tp=tp, fp=fp,
                     precision=tp / k if k else np.nan, precision_ci_lower=lo,
                     precision_ci_upper=hi, recall=tp / y.sum(),
                     enrichment=(tp / k) / y.mean() if k else np.nan)
            )

    for name, rows in [("Table 1. Ranking", rows_rank),
                       ("Table 2. Probability", rows_prob),
                       ("Table 3. Conformal coverage", rows_conf),
                       ("Table 4. High-score screening", rows_triage)]:
        out = pd.DataFrame(rows)
        print("\n" + name)
        print(out.round(3).to_string(index=False))

    out = pd.DataFrame(rows_rank)
    out.to_csv(ROOT / "results" / "recomputed_table1.csv", index=False)
    pd.DataFrame(rows_prob).to_csv(ROOT / "results" / "recomputed_table2.csv", index=False)
    pd.DataFrame(rows_conf).to_csv(ROOT / "results" / "recomputed_table3.csv", index=False)
    pd.DataFrame(rows_triage).to_csv(ROOT / "results" / "recomputed_table4.csv", index=False)



    thr = pd.read_csv(ROOT / "tables_archive" / "low_risk_threshold.csv", skiprows=1)
    thr = thr[(thr.Model == "HistGB") & (thr["Threshold rule"] == "th_sens95")]
    cut = {k: float(v) for k, v in zip(thr.Split, thr.Cutoff)}

    rows_low = []
    for split in SPLITS:
        d = df[(df.split_name == split) & (df.model == "HistGB")]
        low = d[(d.conformal_set_alpha_0_10 == "negative")
                & (d.calibrated_score < cut[split])]
        rows_low.append(
            dict(split=LABELS[split], threshold=cut[split], n=len(low),
                 positives=int(low.y_true.sum()),
                 positive_share=low.y_true.mean(),
                 npv=1 - low.y_true.mean())
        )
    total_n = sum(r["n"] for r in rows_low)
    total_p = sum(r["positives"] for r in rows_low)
    rows_low.append(dict(split="All four DILI settings", threshold=np.nan,
                         n=total_n, positives=total_p,
                         positive_share=total_p / total_n,
                         npv=1 - total_p / total_n))
    print("\nLow-score tier (conformal-negative, score < th_sens95)")
    print(pd.DataFrame(rows_low).round(3).to_string(index=False))
    pd.DataFrame(rows_low).to_csv(ROOT / "results" / "recomputed_triage.csv", index=False)

    print("\nWrote results/recomputed_table{1,2,3,4}.csv and recomputed_triage.csv")


if __name__ == "__main__":
    main()
