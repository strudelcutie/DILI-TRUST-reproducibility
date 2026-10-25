
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (PREDICTIONS, REPORTS, TABLES, COLORS, ensure_dirs, ece)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

plt.rcParams.update({"font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9.5,
                     "legend.fontsize": 6.5, "lines.linewidth": 1.4,
                     "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
                     "axes.grid": True, "grid.alpha": 0.25})

SPLITS = ["scaffold", "cluster_out"]
MODELS = ["HistGB", "ET"]
FIGURES = Path(__file__).resolve().parents[1] / "figures"


EXTERNAL_PREVALENCE = 286 / 388
PRIOR_TARGETS = [None, 0.10, 0.25, 0.50, round(EXTERNAL_PREVALENCE, 4)]

BIN_EDGES = [0.0, 0.3, 0.5, 0.7, 1.0001]
BIN_LABELS = ["<0.3", "0.3-0.5", "0.5-0.7", ">=0.7"]


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def wmean(x, w):
    return float(np.sum(w * x) / np.sum(w))


def weighted_metrics(y, p, w=None):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    w = np.ones_like(y) if w is None else np.asarray(w, dtype=float)
    prev = wmean(y, w)
    bs = wmean((p - y) ** 2, w)
    bs_null = wmean((prev - y) ** 2, w)
    bss = 1.0 - bs / bs_null if bs_null > 0 else np.nan
    bias = wmean(p, w) - prev
    out = {
        "n": int(len(y)),
        "n_weighted": round(float(w.sum()), 2),
        "prevalence": round(prev, 4),
        "brier": round(bs, 4),
        "brier_null": round(bs_null, 4),
        "brier_skill_score": round(bss, 4) if not np.isnan(bss) else np.nan,
        "calibration_bias": round(bias, 4),
        "mean_predicted": round(wmean(p, w), 4),
    }
    try:
        m = LogisticRegression(C=1e6, max_iter=5000)
        m.fit(logit(p).reshape(-1, 1), y, sample_weight=w)
        out["calibration_intercept"] = round(float(m.intercept_[0]), 4)
        out["calibration_slope"] = round(float(m.coef_[0][0]), 4)
    except Exception:
        out["calibration_intercept"] = np.nan
        out["calibration_slope"] = np.nan
    bins = np.linspace(0, 1, 6)
    e = 0.0
    for i in range(5):
        msk = (p >= bins[i]) & (p < bins[i + 1])
        if msk.sum() > 0:
            e += (w[msk].sum() / w.sum()) * abs(wmean(y[msk], w[msk]) - wmean(p[msk], w[msk]))
    out["ece_q5"] = round(float(e), 4)
    n_pos = int(y.sum())
    out["positives"] = n_pos
    out["negatives"] = int(len(y) - n_pos)
    return out


def prior_weights(y, target_prev):
    y = np.asarray(y, dtype=int)
    p0 = y.mean()
    if target_prev is None or not (0 < target_prev < 1) or p0 in (0.0, 1.0):
        return np.ones_like(y, dtype=float)
    w_pos = target_prev / p0
    w_neg = (1 - target_prev) / (1 - p0)
    return np.where(y == 1, w_pos, w_neg)


def main():
    ensure_dirs()
    raw_vs_cal, by_distance, prior_rows = [], [], []

    for split in SPLITS:
        ad = pd.read_csv(PREDICTIONS / f"ad_features_{split}.csv").set_index("sample_id")
        for model_name in MODELS:
            test = pd.read_csv(PREDICTIONS / f"{split}_{model_name}_locked_calibrated_test.csv")
            y = test["y_true"].values.astype(int)
            raw = test["raw_score"].values.astype(float)
            cal = test["calibrated_score"].values.astype(float)
            tan = ad.loc[test["sample_id"].values, "max_tanimoto_to_train"].values


            for score_name, p in [("raw", raw), ("calibrated", cal)]:
                row = {"split": split, "model": model_name, "score": score_name}
                row.update(weighted_metrics(y, p))
                raw_vs_cal.append(row)


            for i in range(len(BIN_EDGES) - 1):
                lo, hi = BIN_EDGES[i], BIN_EDGES[i + 1]
                msk = (tan >= lo) & (tan < hi)
                if msk.sum() < 10:
                    continue
                row = {"split": split, "model": model_name, "tanimoto_bin": BIN_LABELS[i]}
                row.update(weighted_metrics(y[msk], cal[msk]))
                by_distance.append(row)
            row = {"split": split, "model": model_name, "tanimoto_bin": "overall"}
            row.update(weighted_metrics(y, cal))
            by_distance.append(row)


            for target in PRIOR_TARGETS:
                w = prior_weights(y, target)
                row = {"split": split, "model": model_name,
                       "target_prevalence": "as observed" if target is None else round(target, 4)}
                row.update(weighted_metrics(y, cal, w))
                prior_rows.append(row)

    df_a = pd.DataFrame(raw_vs_cal)
    df_b = pd.DataFrame(by_distance)
    df_c = pd.DataFrame(prior_rows)
    df_a.to_csv(TABLES / "calibration_raw_vs_calibrated.csv", index=False)
    df_b.to_csv(TABLES / "calibration_by_chemical_distance.csv", index=False)
    df_c.to_csv(TABLES / "calibration_prior_shift.csv", index=False)


    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.3))
    combos = [(s, m) for s in SPLITS for m in MODELS]
    x = np.arange(len(combos))
    width = 0.38

    for k, (model_name, col) in enumerate([("HistGB", "#0072B2"), ("ET", "#009E73")]):
        vals = [df_a[(df_a["split"] == s) & (df_a["model"] == model_name) &
                     (df_a["score"] == "calibrated")]["brier_skill_score"].iloc[0] for s, _ in combos]
        axes[0].bar(x + (k - 0.5) * width, vals, width, color=col, label=model_name)
    axes[0].axhline(0, ls="--", color="black", lw=1)
    axes[0].axhline(-0.162, ls=":", color="crimson", lw=1.2, label="DILI external BSS = -0.162")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"{m}\n{s}" for s, m in combos], fontsize=6.5)
    axes[0].set_ylabel("Brier skill score (held-out)")
    axes[0].set_title("A. Calibration skill on held-out cohorts")
    axes[0].legend(fontsize=6)

    sub = df_b[df_b["tanimoto_bin"] != "overall"]
    for model_name, col in [("HistGB", "#0072B2"), ("ET", "#009E73")]:
        for split, ls in [("scaffold", "-"), ("cluster_out", "--")]:
            s2 = sub[(sub["split"] == split) & (sub["model"] == model_name)]
            axes[1].plot(s2["tanimoto_bin"], s2["brier_skill_score"], ls, marker="o", ms=3.5,
                         color=col, alpha=0.85, label=f"{model_name}/{split}")
    axes[1].axhline(0, ls="--", color="black", lw=1)
    axes[1].set_xlabel("Max Tanimoto to development set")
    axes[1].set_ylabel("Brier skill score")
    axes[1].set_title("B. Calibration skill by chemical distance")
    axes[1].legend(fontsize=6)

    for model_name, col in [("HistGB", "#0072B2"), ("ET", "#009E73")]:
        for split, ls in [("scaffold", "-"), ("cluster_out", "--")]:
            s2 = df_c[(df_c["split"] == split) & (df_c["model"] == model_name) &
                      (df_c["target_prevalence"] != "as observed")]
            axes[2].plot(s2["prevalence"], s2["brier_skill_score"], ls, marker="o", ms=3.5,
                         color=col, alpha=0.85, label=f"{model_name}/{split}")
    axes[2].axhline(0, ls="--", color="black", lw=1)
    axes[2].axhline(-0.162, ls=":", color="crimson", lw=1.2)
    axes[2].set_xlabel("Evaluation-cohort positive rate after reweighting")
    axes[2].set_ylabel("Brier skill score")
    axes[2].set_title("C. Prior-shift stress test")
    axes[2].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(FIGURES / "Figure_SG_calibration_collapse.png")
    fig.savefig(FIGURES / "Figure_SG_calibration_collapse.pdf")
    plt.close(fig)


    L = ["# Stage 12 - calibration-collapse verification", "",
         "## A. Raw vs locked calibrated (held-out cohorts)", "",
         "| split | model | score | prevalence | Brier | null Brier | BSS | bias | intercept | slope | ECE(q5) |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in df_a.iterrows():
        L.append(f"| {r['split']} | {r['model']} | {r['score']} | {r['prevalence']} | {r['brier']} |"
                 f" {r['brier_null']} | {r['brier_skill_score']} | {r['calibration_bias']} |"
                 f" {r['calibration_intercept']} | {r['calibration_slope']} | {r['ece_q5']} |")

    L += ["", "## B. Calibration by chemical distance to the development set", "",
          "| split | model | Tanimoto bin | n | positives | Brier | BSS | bias | intercept | slope |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in df_b.iterrows():
        L.append(f"| {r['split']} | {r['model']} | {r['tanimoto_bin']} | {r['n']} | {r['positives']} |"
                 f" {r['brier']} | {r['brier_skill_score']} | {r['calibration_bias']} |"
                 f" {r['calibration_intercept']} | {r['calibration_slope']} |")

    L += ["", "## C. Prior-shift stress test", "",
          f"Weights are applied to the evaluation cohort only, shifting its positive rate to",
          f"the target; the locked calibrator is unchanged. The external DILIst positive rate",
          f"was {EXTERNAL_PREVALENCE:.4f} (286/388, manuscript Methods).", "",
          "| split | model | target prevalence | achieved prevalence | Brier | BSS | bias | intercept | slope |",
          "|---|---|---|---|---|---|---|---|---|"]
    for _, r in df_c.iterrows():
        L.append(f"| {r['split']} | {r['model']} | {r['target_prevalence']} | {r['prevalence']} |"
                 f" {r['brier']} | {r['brier_skill_score']} | {r['calibration_bias']} |"
                 f" {r['calibration_intercept']} | {r['calibration_slope']} |")

    (REPORTS / "stage12_calibration_collapse_report.md").write_text("\n".join(L), encoding="utf-8")

    print(df_a.to_string(index=False))
    print()
    print(df_b.to_string(index=False))
    print()
    print(df_c.to_string(index=False))
    print("\nsaved stage12 outputs")


if __name__ == "__main__":
    main()
