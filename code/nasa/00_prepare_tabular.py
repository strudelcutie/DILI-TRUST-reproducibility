
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import MATRICES, PREDICTIONS, RAW, REPORTS, TASKS, ensure_dirs

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SEED = 20260521
FRACTIONS = (0.70, 0.15, 0.15)


CANONICAL_FEATURES = [
    "loc", "v(g)", "ev(g)", "iv(g)", "n", "v", "l", "d", "i", "e", "b", "t",
    "locode", "locomment", "loblank", "loccodeandcomment", "uniq_op",
    "uniq_opnd", "total_op", "total_opnd", "branchcount",
]

FEATURE_ALIASES = {"locodeandcomment": "loccodeandcomment"}

DEVELOPMENT_PROJECT = "kc1"
EXTERNAL_PROJECTS = ["pc1", "jm1", "kc2"]

AD_WARNING = 0.3
OOD_EXTREME = 0.15



def read_arff(path):
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    attrs = re.findall(r"^\s*@attribute\s+(\S+)", text, flags=re.MULTILINE)
    if not attrs:
        raise SystemExit(f"no @attribute declarations in {path}")
    body = text.split("@data", 1)[1]
    rows = [r.split(",") for r in body.splitlines()
            if r.strip() and not r.strip().startswith("%")]
    if len(rows[0]) != len(attrs):
        raise SystemExit(f"{path}: {len(attrs)} attributes but {len(rows[0])} fields per row")
    return pd.DataFrame(rows, columns=attrs), attrs[-1]


def load_project(project):
    df, target = read_arff(RAW / f"{project}.arff")
    raw_names = [c for c in df.columns if c != target]
    renamed = {}
    for c in raw_names:
        key = c.strip().lower()
        renamed[c] = FEATURE_ALIASES.get(key, key)
    X = df[raw_names].rename(columns=renamed).apply(pd.to_numeric, errors="coerce")
    y = df[target].astype(str).str.strip().str.lower().isin(["true", "1", "yes"]).astype(int)

    missing = [f for f in CANONICAL_FEATURES if f not in X.columns]
    if missing:
        raise SystemExit(f"{project}: missing canonical features {missing}")
    extra = [c for c in X.columns if c not in CANONICAL_FEATURES]
    if extra:
        raise SystemExit(f"{project}: unexpected extra features {extra}")
    return X[CANONICAL_FEATURES], y


def three_way_split(y, fractions, seed):
    idx = np.arange(len(y))
    tr_va, te = train_test_split(idx, test_size=fractions[2], random_state=seed, stratify=y)
    valid_share = fractions[1] / (fractions[0] + fractions[1])
    tr, va = train_test_split(tr_va, test_size=valid_share, random_state=seed,
                              stratify=y[tr_va])
    return tr, va, te



def max_cosine_to_train(query, train, chunk=512):
    t = np.asarray(train, dtype=np.float64)
    tn = np.linalg.norm(t, axis=1, keepdims=True)
    tn[tn == 0] = 1.0
    t = t / tn

    q = np.asarray(query, dtype=np.float64)
    out = np.empty(q.shape[0], dtype=np.float64)
    for start in range(0, q.shape[0], chunk):
        block = q[start:start + chunk]
        qn = np.linalg.norm(block, axis=1, keepdims=True)
        qn[qn == 0] = 1.0
        out[start:start + chunk] = ((block / qn) @ t.T).max(axis=1)
    return (out + 1.0) / 2.0



def build_task(name, parts):
    X_tr, y_tr, id_tr = parts["train"]
    X_va, y_va, id_va = parts["valid"]
    X_te, y_te, id_te = parts["test"]
    X_tr, X_va, X_te = (f.reset_index(drop=True) for f in (X_tr, X_va, X_te))

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    Z_tr = scaler.fit_transform(imputer.fit_transform(X_tr.astype(float)))
    Z_va = scaler.transform(imputer.transform(X_va.astype(float)))
    Z_te = scaler.transform(imputer.transform(X_te.astype(float)))

    cols = [f"f{i:03d}" for i in range(Z_tr.shape[1])]
    out_dir = MATRICES / name
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    for partition, Z, y, ids in [("train", Z_tr, y_tr, id_tr),
                                 ("valid", Z_va, y_va, id_va),
                                 ("test", Z_te, y_te, id_te)]:
        frame = pd.DataFrame(Z, columns=cols)
        frame.insert(0, "label", np.asarray(y, dtype=int))
        frame.insert(0, "sample_id", list(ids))
        frame.to_csv(out_dir / f"model_ready_{name}_{partition}.csv", index=False)
        manifest.append({"split": name, "partition": partition,
                         "n_samples": len(y), "n_positives": int(np.asarray(y).sum()),
                         "n_features": Z.shape[1]})


    sims = max_cosine_to_train(np.vstack([Z_tr, Z_va, Z_te]), Z_tr)
    ad = pd.DataFrame({
        "sample_id": list(id_tr) + list(id_va) + list(id_te),
        "max_tanimoto_to_train": np.round(sims, 6),
        "ad_warning_flag": sims < AD_WARNING,
        "ood_extreme_flag": sims < OOD_EXTREME,
    })
    ad["partition"] = ["train"] * len(y_tr) + ["valid"] * len(y_va) + ["test"] * len(y_te)
    ad["label"] = np.concatenate([y_tr, y_va, y_te]).astype(int)
    ad.to_csv(PREDICTIONS / f"ad_features_{name}.csv", index=False)

    return manifest, {
        "split": name,
        "n_train": len(y_tr), "pos_train": int(np.sum(y_tr)),
        "n_valid": len(y_va), "pos_valid": int(np.sum(y_va)),
        "n_test": len(y_te), "pos_test": int(np.sum(y_te)),
        "n_features": Z_tr.shape[1],
        "sim_median": round(float(np.median(sims)), 4),
        "n_ad_warning": int((sims < AD_WARNING).sum()),
        "n_ood_extreme": int((sims < OOD_EXTREME).sum()),
    }


def main():
    ensure_dirs()


    Xk, yk = load_project(DEVELOPMENT_PROJECT)
    tr, va, te = three_way_split(yk.values, FRACTIONS, SEED)
    dev = {
        "train": (Xk.iloc[tr], yk.values[tr], [f"kc1_train_{i:05d}" for i in range(len(tr))]),
        "valid": (Xk.iloc[va], yk.values[va], [f"kc1_valid_{i:05d}" for i in range(len(va))]),
        "test": (Xk.iloc[te], yk.values[te], [f"kc1_test_{i:05d}" for i in range(len(te))]),
    }
    assert len(dev["train"][1]) + len(dev["valid"][1]) + len(dev["test"][1]) == len(yk)

    print(f"development {DEVELOPMENT_PROJECT.upper()}: train {len(tr)}/{int(yk.values[tr].sum())} pos, "
          f"valid {len(va)}/{int(yk.values[va].sum())} pos, "
          f"test {len(te)}/{int(yk.values[te].sum())} pos")

    manifest, qc = [], []


    m, q = build_task("kc1_internal", dev)
    q["detail"] = f"held-out stratified {FRACTIONS[2]:.0%} of KC1"
    manifest += m
    qc.append(q)
    print(f"kc1_internal: test {q['n_test']}/{q['pos_test']} pos")


    for proj in EXTERNAL_PROJECTS:
        Xe, ye = load_project(proj)
        parts = dict(dev)
        parts["test"] = (Xe, ye.values, [f"{proj}_test_{i:05d}" for i in range(len(ye))])
        name = f"kc1_{proj}"
        m, q = build_task(name, parts)
        q["detail"] = (f"external project {proj.upper()}; "
                       f"prevalence {ye.mean():.4f} vs development {yk.mean():.4f}")
        manifest += m
        qc.append(q)
        print(f"{name}: test {q['n_test']}/{q['pos_test']} pos (prevalence {ye.mean():.4f})")

    pd.DataFrame(manifest).to_csv(MATRICES / "matrix_manifest.csv", index=False)

    lines = ["# Stage 00 - tabular preparation QC", "",
             "Cross-project software defect screening. One development project (KC1)",
             "split into train/valid/test; three external projects as held-out cohorts.",
             "All four tasks share the same development partition. Transforms are",
             "fitted on the training partition only.", "",
             "| split | shift | n train | pos | n valid | pos | n test | pos | features | median cos-sim | AD warn | OOD |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for q in qc:
        lines.append(
            f"| {q['split']} | {TASKS[q['split']]['shift']} | {q['n_train']} | {q['pos_train']} |"
            f" {q['n_valid']} | {q['pos_valid']} | {q['n_test']} | {q['pos_test']} |"
            f" {q['n_features']} | {q['sim_median']} | {q['n_ad_warning']} | {q['n_ood_extreme']} |")
    lines += ["", "## Notes", ""]
    for q in qc:
        lines.append(f"- **{q['split']}**: {q['detail']}")
    lines += [
        "",
        "The applicability-domain similarity is the maximum cosine similarity to any",
        "training row in the standardised feature space, rescaled from [-1, 1] to",
        "[0, 1]. Unlike Morgan-Tanimoto similarity it has no natural scale: the",
        f"original flag thresholds ({AD_WARNING} / {OOD_EXTREME}) are retained only",
        "for continuity with the molecular pipeline and are not calibrated here.",
        "The observed distribution is reported above so the transfer can be judged.",
        "",
    ]
    (REPORTS / "stage00_preparation_qc.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"saved: {MATRICES} and {PREDICTIONS}")


if __name__ == "__main__":
    main()
