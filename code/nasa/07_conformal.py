
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (ALPHAS, PREDICTIONS, PRIMARY_ALPHA, REPORTS, SPLIT_NAMES,
                    TABLES, ensure_dirs)

from sklearn.model_selection import StratifiedKFold

SPLITS = SPLIT_NAMES
MODELS = ["HistGB", "ET"]


def nonconformity(scores, y):
    return np.where(y == 1, 1.0 - scores, scores)


def q_hat_from(scores, y, alpha):
    ncs = np.sort(nonconformity(scores, y))
    n = len(ncs)
    q = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    return float(np.quantile(ncs, q))


def conformal_sets(scores, qh):
    sets = []
    for s in scores:
        st = set()
        if s >= 1 - qh:
            st.add(1)
        if (1 - s) >= 1 - qh:
            st.add(0)
        sets.append(st)
    return sets


def set_metrics(y, sets):
    sets = np.array(sets, dtype=object)
    sizes = np.array([0 if len(s) == 0 else (2 if len(s) == 2 else 1) for s in sets])
    covered = np.array([int(y[i]) in sets[i] for i in range(len(y))])
    pos = y == 1
    neg = y == 0
    sing_pos = np.array([len(sets[i]) == 1 and 1 in sets[i] for i in range(len(y))])
    sing_neg = np.array([len(sets[i]) == 1 and 0 in sets[i] for i in range(len(y))])
    sing_pos_n = int(sing_pos.sum())
    sing_neg_n = int(sing_neg.sum())
    return {
        "observed_coverage": round(float(covered.mean()), 4),
        "avg_set_size": round(float(sizes.mean()), 4),
        "single_rate": round(float((sizes == 1).mean()), 4),
        "both_rate": round(float((sizes == 2).mean()), 4),
        "empty_rate": round(float((sizes == 0).mean()), 4),
        "positive_coverage": round(float(covered[pos].mean()), 4) if pos.sum() else np.nan,
        "negative_coverage": round(float(covered[neg].mean()), 4) if neg.sum() else np.nan,
        "positive_miss_count": int((~covered & pos).sum()),
        "negative_miss_count": int((~covered & neg).sum()),
        "singleton_positive_accuracy": round(float(y[sing_pos].mean()), 4) if sing_pos_n else np.nan,
        "singleton_negative_accuracy": round(float((1 - y[sing_neg]).mean()), 4) if sing_neg_n else np.nan,
        "singleton_positive_n": sing_pos_n,
        "singleton_negative_n": sing_neg_n,
    }


def main():
    ensure_dirs()
    rows = []

    for split in SPLITS:
        for model_name in MODELS:
            valid = pd.read_csv(PREDICTIONS / f"{split}_{model_name}_locked_calibrated_valid.csv")
            test = pd.read_csv(PREDICTIONS / f"{split}_{model_name}_locked_calibrated_test.csv")
            yv = valid["y_true"].values.astype(int)
            cv = valid["calibrated_score"].values.astype(float)
            yt = test["y_true"].values.astype(int)
            ct = test["calibrated_score"].values.astype(float)

            for alpha in ALPHAS:
                qh = q_hat_from(cv, yv, alpha)


                base = {
                    "split": split, "model": model_name, "alpha": alpha,
                    "target_coverage": round(1 - alpha, 2),
                    "n_cal": len(yv), "q_hat": round(qh, 4),
                    "mode": "validation_locked",
                }
                rows.append({**base, "evaluation_partition": "valid",
                             **set_metrics(yv, conformal_sets(cv, qh))})
                rows.append({**base, "evaluation_partition": "test",
                             **set_metrics(yt, conformal_sets(ct, qh))})

                if alpha == PRIMARY_ALPHA:
                    pd.DataFrame({
                        "sample_id": test["sample_id"].values,
                        "y_true": yt,
                        "calibrated_score": ct,
                        "conformal_prediction_set": ["both" if len(s) == 2 else
                                                     ("positive" if 1 in s else
                                                      ("negative" if 0 in s else "empty"))
                                                     for s in conformal_sets(ct, qh)],
                        "conformal_set_size": [len(s) for s in conformal_sets(ct, qh)],
                        "q_hat": round(qh, 4),
                        "alpha": alpha,
                    }).to_csv(PREDICTIONS / f"conformal_sets_{split}_{model_name}_alpha{alpha}.csv",
                              index=False)


            for alpha in ALPHAS:
                skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=20260521)
                covs, qhs = [], []
                for tr, vl in skf.split(np.zeros(len(yv)), yv):
                    qh = q_hat_from(cv[tr], yv[tr], alpha)
                    qhs.append(qh)
                    covs.append(np.mean(nonconformity(cv[vl], yv[vl]) <= qh))
                rows.append({
                    "split": split, "model": model_name, "alpha": alpha,
                    "target_coverage": round(1 - alpha, 2),
                    "n_cal": len(yv), "q_hat": round(float(np.mean(qhs)), 4),
                    "mode": "cross_conformal_sensitivity",
                    "evaluation_partition": "valid",
                    "observed_coverage": round(float(np.mean(covs)), 4),
                    "avg_set_size": np.nan, "single_rate": np.nan,
                    "both_rate": np.nan, "empty_rate": np.nan,
                    "positive_coverage": np.nan, "negative_coverage": np.nan,
                    "positive_miss_count": np.nan, "negative_miss_count": np.nan,
                    "singleton_positive_accuracy": np.nan,
                    "singleton_negative_accuracy": np.nan,
                    "singleton_positive_n": np.nan, "singleton_negative_n": np.nan,
                })

    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "conformal_coverage.csv", index=False)

    lines = [
        "# Stage 07 - conformal prediction",
        "",
        "q-hat is derived from the validation partition only. The held-out column",
        "below is the compound-disjoint test partition of each split; for the",
        "cluster-out split that cohort is the external-like, chemically disjoint",
        "cohort.",
        "",
        "| split | model | alpha | partition | n_cal | q-hat | target | observed | set size | single | both | empty | pos cov | neg cov |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r in df[df["mode"] == "validation_locked"].iterrows():
        lines.append(
            f"| {r['split']} | {r['model']} | {r['alpha']} | {r['evaluation_partition']} |"
            f" {r['n_cal']} | {r['q_hat']} | {r['target_coverage']} | {r['observed_coverage']} |"
            f" {r['avg_set_size']} | {r['single_rate']} | {r['both_rate']} | {r['empty_rate']} |"
            f" {r['positive_coverage']} | {r['negative_coverage']} |")
    lines += [
        "",
        "At the primary alpha = 0.10 the held-out coverage gap against the 1 - alpha",
        "target is the quantity of interest for generalization.",
        "",
    ]
    (REPORTS / "stage07_conformal_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(df[df["mode"] == "validation_locked"].to_string())
    print("saved conformal tables")


if __name__ == "__main__":
    main()
