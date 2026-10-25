#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
SPLITS = ["scaffold", "temporal", "cluster_out"]
SEEDS = [20260521, 20260522, 20260523, 20260524, 20260525]


def load_matrix(matrix_root, split, part):
    if split == "temporal" and part == "test":
        name = "model_ready_temporal_temporal_test.csv"
    else:
        name = f"model_ready_{split}_{part}.csv"
    return pd.read_csv(Path(matrix_root) / split / name)


def filter_labels(df, label_map):
    df = df.copy()
    df["dilirank_label"] = df["sample_id"].map(label_map)
    df = df[df["dilirank_label"].isin(
        ["most_dili_concern", "vmost_dili_concern", "no_dili_concern", "vno_dili_concern"])]
    df["y"] = df["dilirank_label"].isin(
        ["most_dili_concern", "vmost_dili_concern"]).astype(int)
    return df


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def ece(y, p, bins=5):
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi) if hi < 1 else (p >= lo) & (p <= hi)
        if m.sum():
            total += m.sum() / len(p) * abs(p[m].mean() - y[m].mean())
    return total


def run_split(matrix_root, split, label_map):
    train = filter_labels(load_matrix(matrix_root, split, "train"), label_map)
    valid = filter_labels(load_matrix(matrix_root, split, "valid"), label_map)
    test = filter_labels(load_matrix(matrix_root, split, "test"), label_map)

    features = [c for c in train.columns
                if c not in ("sample_id", "label", "dilirank_label", "y")]

    p_valid = np.zeros(len(valid))
    p_test = np.zeros(len(test))
    for seed in SEEDS:
        model = HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=200, max_leaf_nodes=31, random_state=seed)
        model.fit(train[features], train["y"])
        p_valid += model.predict_proba(valid[features])[:, 1] / len(SEEDS)
        p_test += model.predict_proba(test[features])[:, 1] / len(SEEDS)

    calibrator = LogisticRegression(C=1e6)
    calibrator.fit(logit(p_valid).reshape(-1, 1), valid["y"])
    p_test = calibrator.predict_proba(logit(p_test).reshape(-1, 1))[:, 1]

    y = test["y"].to_numpy()
    brier = brier_score_loss(y, p_test)
    null = brier_score_loss(y, np.full_like(p_test, y.mean()))
    return {
        "split": split,
        "n": len(test),
        "positives": int(y.sum()),
        "negatives": int((1 - y).sum()),
        "prevalence": round(float(y.mean()), 4),
        "AUROC": roc_auc_score(y, p_test),
        "AUPRC": average_precision_score(y, p_test),
        "Brier": brier,
        "BSS": 1 - brier / null,
        "avg_prob_bias": p_test.mean() - y.mean(),
        "ECE_q5": ece(y, p_test),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrices", default=str(ROOT / "data" / "matrices"))
    parser.add_argument("--labels", default=str(ROOT / "data" / "dili" / "modeling_base_table.csv"))
    parser.add_argument("--out", default=str(ROOT / "results" / "label_sensitivity.csv"))
    args = parser.parse_args()

    labels = pd.read_csv(args.labels, usecols=["parent_sample_id", "dilirank_label"])
    label_map = dict(zip(labels["parent_sample_id"], labels["dilirank_label"]))

    rows = [run_split(args.matrices, split, label_map) for split in SPLITS]
    out = pd.DataFrame(rows)
    print(out.round(4).to_string(index=False))
    out.to_csv(args.out, index=False)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
