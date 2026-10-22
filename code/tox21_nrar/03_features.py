
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, FEATURES, MORGAN_NBITS, MORGAN_RADIUS, REPORTS, ensure_dirs

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Descriptors, MACCSkeys

RDLogger.DisableLog("rdApp.*")


def bitvect_to_array(fp, n_bits):
    buf = np.zeros(n_bits, dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, buf)
    return buf.astype(np.uint8)


def main():
    ensure_dirs()
    df = pd.read_csv(DATA / "nrar_curated.csv")
    n = len(df)

    rdkit_descs = [d[0] for d in Descriptors._descList]
    n_desc = len(rdkit_descs)

    rdkit_arr = np.full((n, n_desc), np.nan, dtype=np.float64)
    morgan_arr = np.zeros((n, MORGAN_NBITS), dtype=np.uint8)
    maccs_arr = np.zeros((n, 167), dtype=np.uint8)

    n_parse_fail = 0
    for k, smi in enumerate(df["smiles"].values):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            n_parse_fail += 1
            continue
        for j, name in enumerate(rdkit_descs):
            try:
                rdkit_arr[k, j] = getattr(Descriptors, name)(mol)
            except Exception:
                rdkit_arr[k, j] = np.nan

        fp = AllChem.GetMorganFingerprintAsBitVect(mol, MORGAN_RADIUS, nBits=MORGAN_NBITS)
        morgan_arr[k] = bitvect_to_array(fp, MORGAN_NBITS)

        maccs = MACCSkeys.GenMACCSKeys(mol)
        maccs_arr[k] = bitvect_to_array(maccs, 167)

        if (k + 1) % 1000 == 0:
            print(f"  {k + 1}/{n}")

    ids = df["compound_id"].values

    df_rdkit = pd.DataFrame(rdkit_arr, columns=rdkit_descs)
    df_rdkit.insert(0, "compound_id", ids)
    df_morgan = pd.DataFrame(morgan_arr, columns=[f"morgan_{i}" for i in range(MORGAN_NBITS)])
    df_morgan.insert(0, "compound_id", ids)
    df_maccs = pd.DataFrame(maccs_arr, columns=[f"maccs_{i + 1}" for i in range(167)])
    df_maccs.insert(0, "compound_id", ids)

    df_rdkit.to_csv(FEATURES / "rdkit_2d.csv", index=False, float_format="%.6g")
    df_morgan.to_csv(FEATURES / "morgan.csv", index=False)
    df_maccs.to_csv(FEATURES / "maccs.csv", index=False)

    missing_rate = df_rdkit[rdkit_descs].isna().mean()
    high_missing = sorted(missing_rate[missing_rate > 0.5].index.tolist())
    non_finite = int(np.sum(~np.isfinite(rdkit_arr)))

    lines = [
        "# Stage 03 - structure feature QC",
        "",
        f"- Compounds processed: {n - n_parse_fail} / {n}",
        f"- RDKit parse failures: {n_parse_fail}",
        "",
        "| modality | dimension | samples |",
        "|---|---|---|",
        f"| RDKit 2D descriptors | {n_desc} | {n - n_parse_fail} |",
        f"| Morgan (r={MORGAN_RADIUS}, {MORGAN_NBITS} bits) | {MORGAN_NBITS} | {n - n_parse_fail} |",
        f"| MACCS keys | 167 | {n - n_parse_fail} |",
        "",
        f"- Non-finite RDKit descriptor entries: {non_finite}",
        f"- RDKit descriptors with >50% missing: {len(high_missing)}",
        f"  {', '.join(high_missing) if high_missing else '(none)'}",
        "",
        "Constant-descriptor filtering and scaling are applied per split on the",
        "development (train) partition only, in stage 04.",
        "",
    ]
    (REPORTS / "stage03_feature_qc.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"rdkit={df_rdkit.shape} morgan={df_morgan.shape} maccs={df_maccs.shape}")
    print(f"high-missing rdkit descriptors: {len(high_missing)}  non-finite: {non_finite}")


if __name__ == "__main__":
    main()
