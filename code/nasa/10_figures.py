
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (COLORS, CURATED, FIGURES, MATRICES, PREDICTIONS, TABLES,
                    ensure_dirs)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9.5,
    "legend.fontsize": 7, "lines.linewidth": 1.4, "figure.dpi": 300,
    "savefig.dpi": 300, "savefig.bbox": "tight", "axes.grid": True,
    "grid.alpha": 0.25, "grid.linewidth": 0.5,
})

SPLITS = ["scaffold", "cluster_out"]
MODELS = ["HistGB", "ET"]
MARKERS = {"scaffold": "-", "cluster_out": "--"}


def load(split, model, partition):
    return pd.read_csv(PREDICTIONS / f"{split}_{model}_locked_calibrated_{partition}.csv")


def save(fig, name):
    fig.savefig(FIGURES / f"{name}.png")
    fig.savefig(FIGURES / f"{name}.pdf")
    plt.close(fig)
    print(f"  saved {name}.png/.pdf")


def fig_ranking():
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4))
    for split in SPLITS:
        for model in MODELS:
            d = load(split, model, "test")
            y, c = d["y_true"].values, d["calibrated_score"].values
            fpr, tpr, _ = roc_curve(y, c)
            axes[0].plot(fpr, tpr, MARKERS[split], color=COLORS[model], alpha=0.85,
                         label=f"{model} / {split} (AUROC={roc_auc_score(y, c):.3f})")
        d = load(split, "HistGB", "test")
        axes[0].plot([0, 1], [0, 1], ":", color="gray", lw=1)
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[0].set_title("A. ROC (held-out cohorts)")
    axes[0].legend(loc="lower right", fontsize=6.5)

    for split in SPLITS:
        for model in MODELS:
            d = load(split, model, "test")
            y, c = d["y_true"].values, d["calibrated_score"].values
            from sklearn.metrics import precision_recall_curve
            prec, rec, _ = precision_recall_curve(y, c)
            axes[1].plot(rec, prec, MARKERS[split], color=COLORS[model], alpha=0.85,
                         label=f"{model} / {split} (AUPRC={average_precision_score(y, c):.3f})")
    base = load("scaffold", "HistGB", "test")["y_true"].mean()
    axes[1].axhline(base, ls=":", color="gray", lw=1, label=f"prevalence={base:.3f}")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("B. Precision-recall (held-out cohorts)")
    axes[1].legend(loc="upper right", fontsize=6.5)
    fig.tight_layout()
    save(fig, "Figure_SA_ranking")


def fig_reliability():
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.3))
    curve = pd.read_csv(TABLES / "task2_reliability_curve.csv")

    for ax, part, title in [(axes[0], "valid", "A. Reliability - validation"),
                            (axes[1], "test", "B. Reliability - held-out")]:
        for split in SPLITS:
            for model in MODELS:
                sub = curve[(curve["split"] == split) & (curve["model"] == model) &
                            (curve["partition"] == part) & (curve["score"] == "calibrated") &
                            curve["mean_predicted"].notna()]
                ax.plot(sub["mean_predicted"], sub["observed_frequency"], MARKERS[split],
                        marker="o", ms=3.5, color=COLORS[model], alpha=0.85,
                        label=f"{model} / {split}")
        ax.plot([0, 1], [0, 1], ":", color="gray", lw=1)
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Observed frequency")
        ax.set_title(title)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=6, loc="upper left")

    t2 = pd.read_csv(TABLES / "task2_probability_reliability.csv")
    t2 = t2[t2["partition"] == "test"]
    labels, slopes, intercepts = [], [], []
    for split in SPLITS:
        for model in MODELS:
            r = t2[(t2["split"] == split) & (t2["model"] == model)].iloc[0]
            labels.append(f"{model}\n{split}")
            slopes.append(r["calibration_slope"])
            intercepts.append(r["calibration_intercept"])
    x = np.arange(len(labels))
    axes[2].bar(x - 0.2, slopes, 0.4, color="#4C72B0", label="calibration slope")
    axes[2].bar(x + 0.2, intercepts, 0.4, color="#DD8452", label="calibration intercept")
    axes[2].axhline(1.0, ls=":", color="gray", lw=1)
    axes[2].axhline(0.0, ls=":", color="gray", lw=1)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, fontsize=6.5)
    axes[2].set_title("C. Cox calibration on held-out")
    axes[2].legend(fontsize=6.5)
    fig.tight_layout()
    save(fig, "Figure_SB_reliability")


def fig_conformal():
    conf = pd.read_csv(TABLES / "conformal_coverage.csv")
    conf = conf[conf["mode"] == "validation_locked"]
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.3))

    for split in SPLITS:
        for model in MODELS:
            sub = conf[(conf["split"] == split) & (conf["model"] == model) &
                       (conf["evaluation_partition"] == "test")].sort_values("alpha")
            axes[0].plot(sub["target_coverage"], sub["observed_coverage"], MARKERS[split],
                         marker="o", ms=4, color=COLORS[model],
                         label=f"{model} / {split}")
    axes[0].plot([0.7, 1.0], [0.7, 1.0], ":", color="gray", lw=1, label="target = observed")
    axes[0].set_xlabel("Target coverage (1 - alpha)")
    axes[0].set_ylabel("Observed coverage")
    axes[0].set_title("A. Conformal coverage - held-out cohorts")
    axes[0].legend(fontsize=6)

    a = 0.10
    sub = conf[(conf["alpha"] == a) & (conf["evaluation_partition"] == "test")]
    labels = [f"{r['model']}\n{r['split']}" for _, r in sub.iterrows()]
    x = np.arange(len(labels))
    axes[1].bar(x - 0.22, sub["positive_coverage"], 0.44, color="#C44E52", label="positive class")
    axes[1].bar(x + 0.22, sub["negative_coverage"], 0.44, color="#55A868", label="negative class")
    axes[1].axhline(1 - a, ls=":", color="gray", lw=1)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, fontsize=6.5)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title(f"B. Class-conditional coverage (alpha={a})")
    axes[1].legend(fontsize=6.5)
    fig.tight_layout()
    save(fig, "Figure_SC_conformal")


def fig_uncertainty():
    abst = pd.read_csv(TABLES / "abstention_curve.csv")
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.3))
    for split in SPLITS:
        for model in MODELS:
            sub = abst[(abst["split"] == split) & (abst["model"] == model) &
                       (abst["partition"] == "test")].sort_values("abstention_rate")
            axes[0].plot(sub["abstention_rate"], sub["auroc"], MARKERS[split], marker="o",
                         ms=3.5, color=COLORS[model], label=f"{model} / {split}")
            axes[1].plot(sub["abstention_rate"], sub["error_rate"], MARKERS[split], marker="o",
                         ms=3.5, color=COLORS[model], label=f"{model} / {split}")
    axes[0].set_xlabel("Abstention rate (U(p) descending)")
    axes[0].set_ylabel("AUROC on retained samples")
    axes[0].set_title("A. Ranking after abstention")
    axes[1].set_xlabel("Abstention rate (U(p) descending)")
    axes[1].set_ylabel("Error rate at 0.5 on retained samples")
    axes[1].set_title("B. Error rate after abstention")
    for ax in axes:
        ax.legend(fontsize=6)
    fig.tight_layout()
    save(fig, "Figure_SD_uncertainty_abstention")


def fig_ad():
    strat = pd.read_csv(TABLES / "ad_stratified_performance.csv")
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.3))
    for split in SPLITS:
        ad = pd.read_csv(PREDICTIONS / f"ad_features_{split}.csv")
        ad = ad[ad["split"] == "test"]
        axes[0].hist(ad["max_tanimoto_to_train"], bins=30, alpha=0.55,
                     color="#0072B2" if split == "scaffold" else "#D55E00",
                     label=f"{split} (median={ad['max_tanimoto_to_train'].median():.3f})")
    axes[0].axvline(0.3, ls="--", color="gray", lw=1, label="AD warning < 0.3")
    axes[0].axvline(0.15, ls=":", color="black", lw=1, label="OOD < 0.15")
    axes[0].set_xlabel("Max Morgan-Tanimoto similarity to training set")
    axes[0].set_ylabel("Compounds (held-out)")
    axes[0].set_title("A. Distance to development set")
    axes[0].legend(fontsize=6)

    sub = strat[strat["tanimoto_bin"] != "overall"]
    labels = [f"{r['split']}\n{r['tanimoto_bin']}" for _, r in sub.iterrows()]
    x = np.arange(len(labels))
    axes[1].bar(x, sub["auroc"].astype(float), 0.6,
                color=["#0072B2" if s == "scaffold" else "#D55E00" for s in sub["split"]])
    axes[1].axhline(0.5, ls=":", color="gray", lw=1)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, fontsize=6)
    axes[1].set_ylabel("AUROC")
    axes[1].set_title("B. AUROC by similarity bin")
    fig.tight_layout()
    save(fig, "Figure_SE_applicability_domain")


def fig_triage():
    tri = pd.read_csv(TABLES / "triage_table.csv")
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.3))
    classes = ["high_score_high_risk_priority", "low_score_candidate_group",
               "intermediate", "uncertain_or_conflict", "out_of_domain_no_call"]
    colors = ["#D55E00", "#0072B2", "#BBBBBB", "#F0E442", "#666666"]
    combos = [(s, m) for s in SPLITS for m in MODELS]
    x = np.arange(len(combos))
    bottom = np.zeros(len(combos))
    for cls, col in zip(classes, colors):
        vals = []
        for split, model in combos:
            sub = tri[(tri["split"] == split) & (tri["model"] == model)]
            vals.append((sub["decision_class"] == cls).mean() if len(sub) else 0.0)
        axes[0].bar(x, vals, 0.6, bottom=bottom, color=col, label=cls.replace("_", " "))
        bottom += np.array(vals)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"{m}\n{s}" for s, m in combos], fontsize=6.5)
    axes[0].set_ylabel("Fraction of held-out cohort")
    axes[0].set_title("A. Triage composition")
    axes[0].legend(fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)

    ts = pd.read_csv(TABLES / "triage_summary.csv")
    x = np.arange(len(ts))
    axes[1].bar(x - 0.2, ts["high_risk_precision"], 0.4, color="#D55E00", label="high-risk precision")
    axes[1].bar(x + 0.2, ts["high_risk_recall"], 0.4, color="#0072B2", label="high-risk recall (positives captured)")
    axes[1].axhline(ts["prevalence"].mean(), ls=":", color="gray", lw=1, label="prevalence")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"{r['model']}\n{r['split']}" for _, r in ts.iterrows()], fontsize=6.5)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("B. High-risk group performance")
    axes[1].legend(fontsize=6.5)
    fig.tight_layout()
    save(fig, "Figure_SF_triage")


def export_tables():
    manifest = pd.read_csv(MATRICES / "matrix_manifest.csv")
    curated = pd.read_csv(CURATED)
    rows = [{"entity": "curated_compounds", "value": len(curated)},
            {"entity": "actives", "value": int(curated["label"].sum())},
            {"entity": "inactives", "value": int((1 - curated["label"]).sum())},
            {"entity": "prevalence", "value": round(float(curated["label"].mean()), 4)}]
    for _, r in manifest.iterrows():
        rows.append({"entity": f"{r['split']}_{r['partition']}",
                     "value": f"n={r['n_samples']}, pos={r['n_positives']}, features={r['n_features']}"})
    pd.DataFrame(rows).to_csv(TABLES / "Table_S1_data_and_splits.csv", index=False)

    pd.read_csv(TABLES / "task1_ranking.csv").to_csv(TABLES / "Table_S2_ranking.csv", index=False)
    pd.read_csv(TABLES / "task2_probability_reliability.csv").to_csv(
        TABLES / "Table_S3_probability_reliability.csv", index=False)
    pd.read_csv(TABLES / "conformal_coverage.csv").to_csv(TABLES / "Table_S4_conformal.csv", index=False)
    pd.read_csv(TABLES / "triage_summary.csv").to_csv(TABLES / "Table_S5_triage.csv", index=False)

    t1 = pd.read_csv(TABLES / "task1_ranking.csv")
    t2 = pd.read_csv(TABLES / "task2_probability_reliability.csv")
    bs = []
    for _, r in t1.iterrows():
        for met in ["auroc", "auprc"]:
            bs.append({"split": r["split"], "model": r["model"], "partition": r["partition"],
                       "score": r["score"], "metric": met, "estimate": r[met],
                       "ci_lower": r[f"{met}_ci_lower"], "ci_upper": r[f"{met}_ci_upper"],
                       "n": r["n"], "n_bootstrap": 2000})
    for _, r in t2.iterrows():
        for met in ["brier", "brier_skill_score"]:
            bs.append({"split": r["split"], "model": r["model"], "partition": r["partition"],
                       "score": "calibrated", "metric": met, "estimate": r[met],
                       "ci_lower": r[f"{met}_ci_lower"], "ci_upper": r[f"{met}_ci_upper"],
                       "n": r["n"], "n_bootstrap": 2000})
    pd.DataFrame(bs).to_csv(TABLES / "Table_S6_bootstrap_ci.csv", index=False)

    pd.read_csv(TABLES / "ad_stratified_performance.csv").to_csv(
        TABLES / "Table_S7_tanimoto_stratified_performance.csv", index=False)
    pd.read_csv(TABLES / "ad_tanimoto_summary.csv").to_csv(
        TABLES / "Table_S8_tanimoto_distance_summary.csv", index=False)
    pd.read_csv(TABLES / "threshold_lock_table.csv").to_csv(
        TABLES / "Table_S9_locked_thresholds.csv", index=False)
    print("  saved Table_S1..S9")


def main():
    ensure_dirs()
    fig_ranking()
    fig_reliability()
    fig_conformal()
    fig_uncertainty()
    fig_ad()
    fig_triage()
    export_tables()


if __name__ == "__main__":
    main()
