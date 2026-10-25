
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (DATA, FEATURES, PREDICTIONS, REPORTS, TABLES,
                    ensure_dirs)
from common import SPLITS as SPLITS_DIR

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.metrics import brier_score_loss, roc_auc_score

RDLogger.DisableLog("rdApp.*")

SPLITS = ["scaffold", "cluster_out"]
MODELS = ["HistGB", "ET"]
BINS = [0.0, 0.3, 0.5, 0.7, 1.0001]
BIN_LABELS = ["<0.3", "0.3-0.5", "0.5-0.7", ">=0.7"]
AD_WARNING = 0.3
OOD_EXTREME = 0.15


def binary_matrix(ids, fps_index, fps_arr):
    return fps_arr[[fps_index[c] for c in ids]].astype(np.float64)


def max_tanimoto_to_train(query, train):
    inter = query @ train.T
    q_sum = query.sum(axis=1, keepdims=True)
    t_sum = train.sum(axis=1, keepdims=True).T
    union = q_sum + t_sum - inter
    union[union == 0] = 1.0
    return (inter / union).max(axis=1)


def main():
    ensure_dirs()
    curated = pd.read_csv(DATA / "nrar_curated.csv")
    smiles = dict(zip(curated["compound_id"], curated["smiles"]))

    morgan = pd.read_csv(FEATURES / "morgan.csv").set_index("compound_id")
    fps_index = {cid: i for i, cid in enumerate(morgan.index)}
    fps_arr = morgan.values.astype(np.uint8)

    scaffold_of = {}
    for cid, smi in smiles.items():
        mol = Chem.MolFromSmiles(smi)
        core = MurckoScaffold.GetScaffoldForMol(mol)
        scaffold_of[cid] = Chem.MolToSmiles(core, canonical=True) if core is not None else ""

    summary_rows, strat_rows = [], []

    for split in SPLITS:
        sp = pd.read_csv(SPLITS_DIR / f"{split}_split.csv")
        train_ids = sp[sp["split"] == "train"]["compound_id"].tolist()
        train = binary_matrix(train_ids, fps_index, fps_arr)

        all_ids = sp["compound_id"].tolist()
        row_of = {c: i for i, c in enumerate(all_ids)}
        all_mat = binary_matrix(all_ids, fps_index, fps_arr)
        sims = max_tanimoto_to_train(all_mat, train)

        ad = pd.DataFrame({
            "sample_id": all_ids,
            "max_tanimoto_to_train": np.round(sims, 4),
            "ad_warning_flag": sims < AD_WARNING,
            "ood_extreme_flag": sims < OOD_EXTREME,
        }).merge(sp[["compound_id", "split"]].rename(columns={"compound_id": "sample_id"}),
                 on="sample_id", how="left")

        pred = pd.read_csv(PREDICTIONS / f"{split}_HistGB_locked_calibrated_test.csv")
        ad.to_csv(PREDICTIONS / f"ad_features_{split}.csv", index=False)

        test_ids = sp[sp["split"] == "test"]["compound_id"].tolist()
        test_sims = sims[[row_of[c] for c in test_ids]]


        train_scaffolds = {scaffold_of[c] for c in train_ids if scaffold_of[c]}
        test_scaffolds = {scaffold_of[c] for c in test_ids if scaffold_of[c]}
        overlap = train_scaffolds & test_scaffolds
        n_acyclic_test = sum(1 for c in test_ids if not scaffold_of[c])

        summary_rows.append({
            "split": split,
            "n_train": len(train_ids), "n_test": len(test_ids),
            "max_tanimoto_median": round(float(np.median(test_sims)), 4),
            "max_tanimoto_q25": round(float(np.percentile(test_sims, 25)), 4),
            "max_tanimoto_q75": round(float(np.percentile(test_sims, 75)), 4),
            "max_tanimoto_min": round(float(test_sims.min()), 4),
            "max_tanimoto_max": round(float(test_sims.max()), 4),
            "n_ad_warning_lt0.3": int((test_sims < AD_WARNING).sum()),
            "n_ood_extreme_lt0.15": int((test_sims < OOD_EXTREME).sum()),
            "train_scaffolds_unique": len(train_scaffolds),
            "test_scaffolds_unique": len(test_scaffolds),
            "n_acyclic_test": n_acyclic_test,
            "overlapping_scaffolds": len(overlap),
            "scaffold_overlap_ratio": round(len(overlap) / len(test_scaffolds), 4) if test_scaffolds else 0.0,
            "test_samples_with_shared_scaffold": sum(1 for c in test_ids if scaffold_of[c] in train_scaffolds),
        })

        merged = pd.DataFrame({"sample_id": test_ids, "max_tanimoto": test_sims}).merge(
            pred[["sample_id", "y_true", "calibrated_score"]], on="sample_id", how="inner")

        for i in range(len(BINS) - 1):
            lo, hi = BINS[i], BINS[i + 1]
            m = (merged["max_tanimoto"] >= lo) & (merged["max_tanimoto"] < hi)
            sub = merged[m]
            n = len(sub)
            pos = int(sub["y_true"].sum()) if n else 0
            auroc = (round(float(roc_auc_score(sub["y_true"], sub["calibrated_score"])), 4)
                     if n >= 5 and 0 < pos < n else np.nan)
            brier = round(float(brier_score_loss(sub["y_true"], sub["calibrated_score"])), 4) if n else np.nan
            strat_rows.append({
                "split": split, "tanimoto_bin": BIN_LABELS[i], "n": n,
                "n_positive": pos, "n_negative": n - pos,
                "prevalence": round(pos / n, 4) if n else np.nan,
                "mean_max_tanimoto": round(float(sub["max_tanimoto"].mean()), 4) if n else np.nan,
                "auroc": auroc, "brier": brier,
            })

        strat_rows.append({
            "split": split, "tanimoto_bin": "overall", "n": len(merged),
            "n_positive": int(merged["y_true"].sum()),
            "n_negative": len(merged) - int(merged["y_true"].sum()),
            "prevalence": round(float(merged["y_true"].mean()), 4),
            "mean_max_tanimoto": round(float(merged["max_tanimoto"].mean()), 4),
            "auroc": round(float(roc_auc_score(merged["y_true"], merged["calibrated_score"])), 4),
            "brier": round(float(brier_score_loss(merged["y_true"], merged["calibrated_score"])), 4),
        })

    pd.DataFrame(summary_rows).to_csv(TABLES / "ad_tanimoto_summary.csv", index=False)
    pd.DataFrame(strat_rows).to_csv(TABLES / "ad_stratified_performance.csv", index=False)

    lines = [
        "# Stage 08 - applicability domain",
        "",
        "| split | n_train | n_test | median max Tanimoto | Q1 | Q3 | min | n < 0.3 | n < 0.15 | test scaffolds | overlap | acyclic test |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in summary_rows:
        lines.append(
            f"| {r['split']} | {r['n_train']} | {r['n_test']} | {r['max_tanimoto_median']} |"
            f" {r['max_tanimoto_q25']} | {r['max_tanimoto_q75']} | {r['max_tanimoto_min']} |"
            f" {r['n_ad_warning_lt0.3']} | {r['n_ood_extreme_lt0.15']} |"
            f" {r['test_scaffolds_unique']} | {r['overlapping_scaffolds']} | {r['n_acyclic_test']} |")
    lines += [
        "",
        "## Stratified held-out performance (HistGB, locked calibration)",
        "",
        "| split | bin | n | positives | prevalence | AUROC | Brier |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in strat_rows:
        lines.append(f"| {r['split']} | {r['tanimoto_bin']} | {r['n']} | {r['n_positive']} |"
                     f" {r['prevalence']} | {r['auroc']} | {r['brier']} |")
    lines += [
        "",
        "Scaffold overlap counts exclude acyclic compounds (empty Murcko scaffold);",
        "those compounds carry no shared-scaffold information and are listed",
        "separately in the `acyclic test` column.",
        "",
        "The cluster-out held-out cohort is the chemically disjoint (external-like)",
        "cohort: its distance to the training set is the analogue of the DILIst",
        "external cohort distance in the DILI analysis.",
        "",
    ]
    (REPORTS / "stage08_applicability_domain_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(pd.DataFrame(summary_rows).to_string())
    print("saved AD tables")


if __name__ == "__main__":
    main()
