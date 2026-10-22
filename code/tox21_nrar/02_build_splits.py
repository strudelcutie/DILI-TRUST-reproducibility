
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, MORGAN_NBITS, MORGAN_RADIUS, REPORTS, SPLITS, ensure_dirs

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

SEED = 20260521
FRACTIONS = (0.70, 0.15, 0.15)








TANIMOTO_DISTANCE_CUTOFF = 0.70


def scaffold_keys(smiles):
    keys, types = [], []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            keys.append(None)
            types.append("scaffold_failed_review")
            continue
        try:
            core = MurckoScaffold.GetScaffoldForMol(mol)
            core_smi = Chem.MolToSmiles(core, canonical=True) if core is not None else ""
        except Exception:
            keys.append(None)
            types.append("scaffold_failed_review")
            continue
        if core_smi:
            keys.append(core_smi)
            types.append("murcko_scaffold")
        else:
            keys.append(f"ACYCLIC::{smi}")
            types.append("acyclic_no_murcko_scaffold")
    return keys, types


def greedy_group_split(groups, sizes_in_order):
    total = sum(sizes_in_order)
    target_train = int(total * FRACTIONS[0])
    target_valid = int(total * FRACTIONS[1])

    buckets = {"train": [], "valid": [], "test": []}
    n = {"train": 0, "valid": 0, "test": 0}

    for key, size in groups:
        t_room = max(0, target_train - n["train"])
        v_room = max(0, target_valid - n["valid"])
        ts_room = max(0, total - target_train - target_valid - n["test"])
        if t_room >= max(v_room, ts_room):
            buckets["train"].append(key)
            n["train"] += size
        elif v_room >= ts_room:
            buckets["valid"].append(key)
            n["valid"] += size
        else:
            buckets["test"].append(key)
            n["test"] += size

    mapping = {}
    for split_name, keys in buckets.items():
        for k in keys:
            mapping[k] = split_name
    return mapping


def morgan_fps(smiles):
    fps, idx = [], []
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fps.append(AllChem.GetMorganFingerprintAsBitVect(
            mol, MORGAN_RADIUS, nBits=MORGAN_NBITS))
        idx.append(i)
    return fps, idx


def pairwise_distances(fps):
    n = len(fps)
    total = n * (n - 1) // 2
    dists = np.empty(total, dtype=np.float32)
    pos = 0
    for i in range(1, n):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists[pos:pos + i] = 1.0 - np.asarray(sims, dtype=np.float32)
        pos += i
    return dists


def butina_clusters(fps, cutoff):
    n = len(fps)
    dists = pairwise_distances(fps)
    offs = np.array([i * (i - 1) // 2 for i in range(n + 1)], dtype=np.int64)

    flat = np.flatnonzero(dists <= cutoff)
    rows = np.searchsorted(offs, flat, side="right") - 1
    cols = flat - offs[rows]

    neighbours = [[] for _ in range(n)]
    for a, b in zip(rows.tolist(), cols.tolist()):
        neighbours[a].append(b)
        neighbours[b].append(a)

    counts = np.array([len(x) for x in neighbours])
    order = np.argsort(-counts, kind="stable")
    assigned = np.zeros(n, dtype=bool)
    clusters = []
    for idx in order:
        if assigned[idx]:
            continue
        cluster = [int(idx)]
        assigned[idx] = True
        for nb in neighbours[idx]:
            if not assigned[nb]:
                assigned[nb] = True
                cluster.append(int(nb))
        clusters.append(cluster)

    diag = {
        "n_pairs_within_cutoff": int(flat.size),
        "pairs_within_cutoff_fraction": round(float(flat.size / max(len(dists), 1)), 5),
        "distance_10th_percentile": round(float(np.percentile(dists, 10)), 4),
    }
    del dists
    return clusters, diag


def describe(df, split_col="split"):
    out = []
    for s in ["train", "valid", "test"]:
        sub = df[df[split_col] == s]
        p = int(sub["label"].sum())
        out.append((s, len(sub), p, len(sub) - p,
                    p / len(sub) if len(sub) else 0.0))
    return out


def main():
    ensure_dirs()
    df = pd.read_csv(DATA / "nrar_curated.csv")
    n_total = len(df)


    keys, types = scaffold_keys(df["smiles"].tolist())
    df["scaffold_key"] = keys
    df["scaffold_type"] = types

    n_failed = int((df["scaffold_type"] == "scaffold_failed_review").sum())
    n_acyclic = int((df["scaffold_type"] == "acyclic_no_murcko_scaffold").sum())

    sc = df[df["scaffold_type"] != "scaffold_failed_review"].copy()
    group_sizes = sc.groupby("scaffold_key").size().sort_values(ascending=False)
    groups = list(group_sizes.items())
    mapping = greedy_group_split(groups, group_sizes.tolist())
    sc["split"] = sc["scaffold_key"].map(mapping)
    sc["seed"] = SEED
    sc["split_type"] = "scaffold"

    tr = set(sc[sc["split"] == "train"]["scaffold_key"])
    vl = set(sc[sc["split"] == "valid"]["scaffold_key"])
    te = set(sc[sc["split"] == "test"]["scaffold_key"])

    out_cols = ["compound_id", "label", "scaffold_key", "scaffold_type", "split", "seed", "split_type"]
    sc[out_cols].to_csv(SPLITS / "scaffold_split.csv", index=False)

    scaff_stats = describe(sc)


    fps, idx = morgan_fps(df["smiles"].tolist())
    clusters, cl_diag = butina_clusters(fps, TANIMOTO_DISTANCE_CUTOFF)
    cluster_assign = {}
    for ci, cluster in enumerate(clusters):
        for local in cluster:
            cluster_assign[df.index[idx[local]]] = ci
    df["cluster_id"] = df.index.map(cluster_assign)
    df["cluster_id"] = df["cluster_id"].fillna(-1).astype(int)

    cl = df[df["cluster_id"] >= 0].copy()
    cluster_sizes = cl.groupby("cluster_id").size().sort_values(ascending=False)
    cmap = greedy_group_split(list(cluster_sizes.items()), cluster_sizes.tolist())
    cl["split"] = cl["cluster_id"].map(cmap)
    cl["seed"] = SEED
    cl["split_type"] = "cluster_out_butina"

    cl_out_cols = ["compound_id", "label", "cluster_id", "split", "seed", "split_type"]
    cl[cl_out_cols].to_csv(SPLITS / "cluster_out_split.csv", index=False)

    cluster_stats = describe(cl)
    n_clusters = len(clusters)
    max_cluster = int(max(cluster_sizes.tolist())) if len(cluster_sizes) else 0
    n_singleton = int((cluster_sizes == 1).sum())

    lines = [
        "# Stage 02 - split QC",
        "",
        f"- Curated compounds: {n_total}",
        f"- Seed: {SEED}",
        f"- Target fractions train/valid/test: {FRACTIONS[0]:.2f}/{FRACTIONS[1]:.2f}/{FRACTIONS[2]:.2f}",
        "",
        "## Scaffold split (Bemis-Murcko)",
        "",
        f"- Unique Murcko scaffold groups: {len(group_sizes)}",
        f"- Acyclic compounds (own scaffold group): {n_acyclic}",
        f"- RDKit scaffold failures excluded: {n_failed}",
        f"- Compounds in split: {len(sc)}",
        f"- Scaffold overlap train/valid: {len(tr & vl)}",
        f"- Scaffold overlap train/test: {len(tr & te)}",
        f"- Scaffold overlap valid/test: {len(vl & te)}",
        "",
        "| split | n | positives | negatives | prevalence |",
        "|---|---|---|---|---|",
    ]
    for s, n, p, ng, prev in scaff_stats:
        lines.append(f"| {s} | {n} | {p} | {ng} | {prev:.4f} |")

    lines += [
        "",
        "## Cluster-out split (Butina, Morgan r=2 2048 bit)",
        "",
        f"- Butina distance cutoff: {TANIMOTO_DISTANCE_CUTOFF:.2f} (Tanimoto similarity >= {1 - TANIMOTO_DISTANCE_CUTOFF:.2f})",
        f"- Number of clusters: {n_clusters}",
        f"- Singleton clusters: {n_singleton} ({n_singleton / n_clusters:.2f})",
        f"- Largest cluster: {max_cluster} compounds ({max_cluster / len(cl):.3f} of the library)",
        f"- Pairs within cutoff: {cl_diag['n_pairs_within_cutoff']} ({cl_diag['pairs_within_cutoff_fraction']:.5f} of all pairs)",
        f"- Compounds in split: {len(cl)}",
        "",
        "| split | n | positives | negatives | prevalence |",
        "|---|---|---|---|---|",
    ]
    for s, n, p, ng, prev in cluster_stats:
        lines.append(f"| {s} | {n} | {p} | {ng} | {prev:.4f} |")

    lines += [
        "",
        "## Notes",
        "",
        "- Both splits are compound-disjoint by construction (scaffold groups and",
        "  Butina clusters never cross split boundaries).",
        "- Tox21 carries no temporal annotation; no temporal split was fabricated.",
        "- The DILI-TRUST Butina cutoff rule (10th percentile of pairwise Tanimoto",
        f"  distance = {cl_diag['distance_10th_percentile']:.4f} here) is not transferable to the chemically much more",
        f"  diverse Tox21 library; a fixed cutoff of {TANIMOTO_DISTANCE_CUTOFF:.2f} was used instead. All other",
        "  cluster-out split mechanics (Butina clustering, greedy 70/15/15 over",
        "  clusters) are unchanged.",
        "- The cluster-out test partition doubles as the external-like (chemically",
        "  disjoint) held-out cohort consumed by the DILI-TRUST-style external",
        "  evaluation interface.",
        "",
    ]
    (REPORTS / "stage02_split_qc.md").write_text("\n".join(lines), encoding="utf-8")

    print("scaffold:", scaff_stats)
    print("cluster_out:", cluster_stats)
    print(f"butina cutoff={TANIMOTO_DISTANCE_CUTOFF} n_clusters={n_clusters} "
          f"max_cluster={max_cluster} singletons={n_singleton}")
    print(f"saved: {SPLITS}")


if __name__ == "__main__":
    main()
