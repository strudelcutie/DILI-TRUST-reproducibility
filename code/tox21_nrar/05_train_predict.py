
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import MATRICES, PREDICTIONS, REPORTS, SEEDS, ensure_dirs

from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

MODELS = {
    "HistGB": lambda s: HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=200, max_leaf_nodes=31, random_state=s),
    "ET": lambda s: ExtraTreesClassifier(
        n_estimators=200, max_depth=15, min_samples_leaf=3,
        class_weight="balanced", random_state=s, n_jobs=-1),
}

SPLITS = ["scaffold", "cluster_out"]


def load_matrix(split, partition):
    df = pd.read_csv(MATRICES / split / f"model_ready_{split}_{partition}.csv")
    cols = [c for c in df.columns if c not in ("sample_id", "label")]
    return df["sample_id"].values, df[cols].values.astype(np.float32), df["label"].values.astype(int)


def main():
    ensure_dirs()
    lines = ["# Stage 05 - training report", "",
             "Model hyper-parameters are copied verbatim from DILI-TRUST; nothing is tuned.",
             "", "| split | model | seed | train n | train P | valid AUROC | valid AUPRC | fit s |",
             "|---|---|---|---|---|---|---|---|"]

    for split in SPLITS:
        ids_tr, X_tr, y_tr = load_matrix(split, "train")
        parts = {p: load_matrix(split, p) for p in ["valid", "test"]}
        print(f"{split}: train={X_tr.shape} P={int(y_tr.sum())}/{len(y_tr) - int(y_tr.sum())}")

        for model_name, factory in MODELS.items():
            seed_scores = {"valid": [], "test": []}
            for seed in SEEDS:
                t0 = time.time()
                model = factory(seed)
                model.fit(X_tr, y_tr)
                fit_s = time.time() - t0

                for partition, (ids, X, y) in parts.items():
                    score = model.predict_proba(X)[:, 1]
                    seed_scores[partition].append(score)
                    pd.DataFrame({"sample_id": ids, "y_true": y,
                                  "y_score": np.round(score, 6)}).to_csv(
                        PREDICTIONS / f"{split}_{model_name}_seed{seed}_{partition}_predictions.csv",
                        index=False)

                vi, Xv, yv = parts["valid"]
                vs = seed_scores["valid"][-1]
                lines.append(
                    f"| {split} | {model_name} | {seed} | {len(y_tr)} | {int(y_tr.sum())} |"
                    f" {roc_auc_score(yv, vs):.4f} | {average_precision_score(yv, vs):.4f} | {fit_s:.1f} |")
                print(f"  {split}/{model_name}/seed{seed}: fit {fit_s:.1f}s "
                      f"valid AUROC={roc_auc_score(yv, vs):.4f} AUPRC={average_precision_score(yv, vs):.4f}")

            for partition, (ids, X, y) in parts.items():
                arr = np.vstack(seed_scores[partition])
                ens = arr.mean(axis=0)
                pd.DataFrame({
                    "sample_id": ids, "y_true": y,
                    "y_score": np.round(ens, 6),
                    "y_score_std": np.round(arr.std(axis=0), 6),
                    "seed_pos_vote_frac": np.round((arr >= 0.5).mean(axis=0), 4),
                }).to_csv(PREDICTIONS / f"{split}_{model_name}_ensemble_{partition}_predictions.csv",
                          index=False)

                lines.append(
                    f"| {split} | {model_name} | ensemble | - | - |"
                    f" {roc_auc_score(y, ens):.4f} | {average_precision_score(y, ens):.4f} | - |")

    (REPORTS / "stage05_training_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"saved predictions to {PREDICTIONS}")


if __name__ == "__main__":
    main()
