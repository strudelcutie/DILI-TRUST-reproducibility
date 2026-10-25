
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (CURATED, ENDPOINT, ENDPOINT_DESCRIPTIONS,
                    REPORTS, SLUG, ensure_dirs)

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

def main(src):
    ensure_dirs()
    src = Path(src)
    if not src.exists():
        raise SystemExit(f"input not found: {src}")

    raw = pd.read_csv(src)
    n_raw = len(raw)
    if ENDPOINT not in raw.columns:
        raise SystemExit(f"column {ENDPOINT} missing from {src}")

    for col in ("mol_id", "smiles"):
        if col not in raw.columns:
            raise SystemExit(f"column {col} missing from {src}")

    df = raw[["mol_id", "smiles", ENDPOINT]].rename(
        columns={"mol_id": "compound_id", ENDPOINT: f"{SLUG}_raw"}
    )

    n_missing = int(df[f"{SLUG}_raw"].isna().sum())
    df = df[df[f"{SLUG}_raw"].notna()].copy()
    n_after_missing = len(df)

    df[f"{SLUG}_raw"] = df[f"{SLUG}_raw"].astype(float)
    unexpected = sorted(set(df[f"{SLUG}_raw"].unique()) - {0.0, 1.0})
    df["label"] = df[f"{SLUG}_raw"].map({1.0: 1, 0.0: 0}).astype(int)

    df["smiles"] = df["smiles"].astype(str).str.strip()
    bad_smiles = int(df["smiles"].isin(["", "nan", "None"]).sum())

    keep_mask = []
    for smi in df["smiles"].values:
        keep_mask.append(Chem.MolFromSmiles(smi) is not None)
    keep_mask = np.array(keep_mask, dtype=bool)
    n_unparseable = int((~keep_mask).sum())

    df = df[keep_mask].copy()
    df = df.drop_duplicates(subset=["compound_id"]).reset_index(drop=True)

    out = df[["compound_id", "smiles", "label"]].copy()
    for c in out.columns:
        assert c in ("compound_id", "smiles", "label")
    out.to_csv(CURATED, index=False)

    n = len(out)
    n_pos = int(out["label"].sum())
    n_neg = n - n_pos

    lines = [
        f"# Stage 01 - {ENDPOINT} curation QC",
        "",
        f"- Source file: `{src}`",
        f"- Endpoint: `{ENDPOINT}` (Tox21 {ENDPOINT_DESCRIPTIONS.get(ENDPOINT, 'assay')})",
        "",
        "## Funnel",
        "",
        "| Step | n |",
        "|---|---|",
        f"| Raw rows in source file | {n_raw} |",
        f"| Dropped: {ENDPOINT} missing | -{n_missing} |",
        f"| After missing-label removal | {n_after_missing} |",
        f"| Dropped: non-parseable / empty SMILES | -{n_unparseable} |",
        f"| Final curated compounds | {n} |",
        "",
        "## Label distribution",
        "",
        "| Label | n | share |",
        "|---|---|---|",
        f"| active (1) | {n_pos} | {n_pos / n:.4f} |",
        f"| inactive (0) | {n_neg} | {n_neg / n:.4f} |",
        "",
        f"- Prevalence: {n_pos / n:.4f}",
        f"- Unexpected raw label values: {unexpected if unexpected else 'none'}",
        f"- Empty-SMILES strings encountered before parsing: {bad_smiles}",
        "",
        "## Output contract",
        "",
        f"`data/{CURATED.name}` with columns `compound_id`, `smiles`, `label`,",
        "identical to the DILI-TRUST modeling input contract.",
        "",
    ]
    (REPORTS / "stage01_curation_qc.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"curated: n={n} pos={n_pos} neg={n_neg} prevalence={n_pos / n:.4f}")
    print(f"saved: {CURATED}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).resolve().parents[2] / "data" / "tox21" / "tox21.csv"))
