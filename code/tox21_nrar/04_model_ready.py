
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, FEATURES, MATRICES, REPORTS, SPLITS, ensure_dirs

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

SPLIT_FILES = {"scaffold": "scaffold_split.csv", "cluster_out": "cluster_out_split.csv"}


def main():
    ensure_dirs()
    curated = pd.read_csv(DATA / "nrar_curated.csv")
    label_map = dict(zip(curated["compound_id"], curated["label"]))

    rdkit = pd.read_csv(FEATURES / "rdkit_2d.csv").set_index("compound_id")
    morgan = pd.read_csv(FEATURES / "morgan.csv").set_index("compound_id")
    maccs = pd.read_csv(FEATURES / "maccs.csv").set_index("compound_id")

    rdkit_cols_all = list(rdkit.columns)
    morgan_cols = list(morgan.columns)
    maccs_cols = list(maccs.columns)
    assert list(rdkit.index) == list(morgan.index) == list(maccs.index)

    X_rdkit = rdkit.values.astype(np.float64)
    X_rdkit[~np.isfinite(X_rdkit)] = np.nan

    manifest = []
    for split_name, fname in SPLIT_FILES.items():
        sp = pd.read_csv(SPLITS / fname)

        train_ids = sp[sp["split"] == "train"]["compound_id"].tolist()
        X_tr = rdkit.loc[train_ids].values.astype(np.float64)
        X_tr[~np.isfinite(X_tr)] = np.nan

        keep = [j for j in range(X_tr.shape[1])
                if len(np.unique(X_tr[~np.isnan(X_tr[:, j]), j])) > 1]
        cont_cols = [rdkit_cols_all[j] for j in keep]

        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()
        scaler.fit(imputer.fit_transform(X_tr[:, keep]))

        out_dir = MATRICES / split_name
        out_dir.mkdir(parents=True, exist_ok=True)

        for partition in ["train", "valid", "test"]:
            ids = sp[sp["split"] == partition]["compound_id"].tolist()
            Xp = X_rdkit[[rdkit.index.get_loc(c) for c in ids]][:, keep]
            Xp = scaler.transform(imputer.transform(Xp))
            mor = morgan.loc[ids, morgan_cols].values.astype(np.float64)
            mac = maccs.loc[ids, maccs_cols].values.astype(np.float64)

            block = np.nan_to_num(np.hstack([Xp, mor, mac]),
                                  nan=0.0, posinf=0.0, neginf=0.0)
            out = pd.DataFrame(block, columns=cont_cols + morgan_cols + maccs_cols)
            out.insert(0, "label", [int(label_map[c]) for c in ids])
            out.insert(0, "sample_id", ids)
            out.to_csv(out_dir / f"model_ready_{split_name}_{partition}.csv",
                       index=False, float_format="%.6g")

            manifest.append({
                "split": split_name, "partition": partition,
                "n_samples": len(ids), "n_positives": int(out["label"].sum()),
                "n_features": len(cont_cols) + len(morgan_cols) + len(maccs_cols),
                "n_continuous": len(cont_cols),
                "n_morgan": len(morgan_cols), "n_maccs": len(maccs_cols),
            })
            print(f"  {split_name}/{partition}: {out.shape} (P={int(out['label'].sum())})")

        with open(MATRICES / split_name / "feature_columns.json", "w") as f:
            json.dump({"continuous": cont_cols, "morgan": morgan_cols, "maccs": maccs_cols}, f)

    pd.DataFrame(manifest).to_csv(MATRICES / "matrix_manifest.csv", index=False)

    lines = [
        "# Stage 04 - model-ready matrix QC",
        "",
        "| split | partition | n | positives | features | continuous | morgan | maccs |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in manifest:
        lines.append(f"| {r['split']} | {r['partition']} | {r['n_samples']} | {r['n_positives']} |"
                     f" {r['n_features']} | {r['n_continuous']} | {r['n_morgan']} | {r['n_maccs']} |")
    lines += [
        "",
        "- Constant-descriptor filter and imputer/scaler are fitted on the train",
        "  partition of each split only; valid and test are transformed with the",
        "  locked development artefacts.",
        "- +/-inf descriptor values were coerced to missing before imputation.",
        "",
    ]
    (REPORTS / "stage04_matrix_qc.md").write_text("\n".join(lines), encoding="utf-8")
    print("saved matrices + matrix_manifest.csv")


if __name__ == "__main__":
    main()
