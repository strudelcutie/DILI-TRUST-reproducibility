#!/usr/bin/env python3

import csv, hashlib, json, os, sys, warnings
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, SaltRemover, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.ML.Cluster import Butina

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
CUR = ROOT / "data_curated"
RAW = ROOT / "raw_data"
SPLITS = ROOT / "splits"
REPORTS = ROOT / "reports" / "stage5"
NOW = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
SEED = 20260521
np.random.seed(SEED)

COMP_MASTER = CUR / "compounds/compound_master.csv"
EVID_TABLE = CUR / "evidence/evidence_table.csv"
EVID_GRAPH = CUR / "evidence/evidence_graph.csv"
DILIRANK_LBL = CUR / "labels/dilirank_labels_curated.csv"
DILIST_LBL = CUR / "labels/dilist_labels_curated.csv"
TDC_LBL = CUR / "labels/tdc_dili_labels_curated.csv"
LABEL_CONFLICT = CUR / "labels/label_conflict.csv"
DILIRANK_RAW = RAW / "dilirank2/Drug Induced Liver Injury Rank (DILIrank 2.0) Dataset  FDA.xlsx"

LEAKAGE_ISSUES = []

def leakage(severity, split_name, detail):
    LEAKAGE_ISSUES.append({"severity": severity, "split": split_name, "detail": detail})
    print(f"  [LEAKAGE-{severity}] {split_name}: {detail[:100]}")




def build_modeling_base():
    print("=" * 72)
    print("STAGE 5.0: Modeling Base Table (Parent-Level)")
    print("=" * 72)

    master = pd.read_csv(COMP_MASTER)
    evid = pd.read_csv(EVID_TABLE)
    dr_lbl = pd.read_csv(DILIRANK_LBL)
    dl_lbl = pd.read_csv(DILIST_LBL)
    lc = pd.read_csv(LABEL_CONFLICT) if LABEL_CONFLICT.exists() else pd.DataFrame()


    evid_lookup = {}
    for _, r in evid.iterrows():
        evid_lookup[r["compound_id"]] = dict(r)


    dr_map = {}
    for _, r in dr_lbl.iterrows():
        nm = str(r.get("compound_name","")).strip().lower()
        lbl = str(r.get("dili_concern_curated",""))
        if nm: dr_map[nm] = lbl


    dl_map = {}
    for _, r in dl_lbl.iterrows():
        nm = str(r.get("compound_name","")).strip().lower()
        lbl = str(r.get("dilist_curated",""))
        if nm: dl_map[nm] = lbl


    conflict_names = set()
    if not lc.empty:
        conflict_names = set(str(c).lower() for c in lc.get("compound_name", []))


    groups = defaultdict(list)
    for _, r in master.iterrows():
        ik = str(r.get("parent_inchikey",""))
        if not ik or ik == "nan":
            ik = f"NO_IK_{r['compound_id']}"
        groups[ik].append(dict(r))

    parent_rows = []
    pid = 0

    for ik, comps in groups.items():
        pid += 1
        parent_id = f"PARENT_{pid:06d}"
        cids = [c["compound_id"] for c in comps]
        pnames = list(set(c["primary_name"] for c in comps))
        all_names_list = []
        for c in comps:
            for nm in str(c.get("source_drug_names","")).split(";"):
                nm = nm.strip()
                if nm: all_names_list.append(nm)
        all_names_list = list(set(all_names_list))

        srcs = set()
        for c in comps:
            for s in str(c.get("sources","")).split(";"):
                s = s.strip()
                if s: srcs.add(s)


        first = comps[0]
        parent_smi = first.get("parent_smiles","")
        canon_smi = first.get("canonical_smiles","")
        scaff_key = first.get("scaffold_split_key","")
        scaff_type = first.get("scaffold_type","")
        scaff_smi = first.get("scaffold_smiles","")
        ik_block = first.get("parent_inchikey_first_block","")
        struct_ok = any(c.get("structure_model_eligible", False) in (True, "True") for c in comps)
        evid_ok = any(c.get("evidence_model_eligible", False) in (True, "True") for c in comps)
        large_mol = any(str(c.get("large_molecule_flag","")) == "True" for c in comps)
        needs_rev = any(c.get("needs_manual_review", False) in (True, "True") for c in comps)


        dr_labels = set()
        dl_labels = set()
        for c in comps:
            pn = str(c.get("primary_name","")).lower()
            if pn in dr_map: dr_labels.add(dr_map[pn])
            if pn in dl_map: dl_labels.add(dl_map[pn])

            for nm in str(c.get("source_drug_names","")).split(";"):
                nm = nm.strip().lower()
                if nm in dr_map: dr_labels.add(dr_map[nm])
                if nm in dl_map: dl_labels.add(dl_map[nm])


        ev = evid_lookup.get(cids[0], {})
        has_conflict = any(str(c.get("primary_name","")).lower() in conflict_names for c in comps) or ev.get("has_label_conflict", False)
        consensus = ev.get("evidence_direction_consensus", "no_evidence")
        has_fda = ev.get("has_fda_label_signal", False)
        has_tdc = ev.get("has_tdc_label", False)
        has_lt = ev.get("has_livertox_evidence", False)


        dr_positive = any("most" in l for l in dr_labels) or any("less" in l for l in dr_labels)
        dr_negative = any("no" in l for l in dr_labels)
        dr_ambiguous = any("ambiguous" in l for l in dr_labels)

        taskA_label = ""
        taskA_excl = ""
        include_A = False

        if dr_positive and dr_negative:
            taskA_label = "conflict"
            taskA_excl = "parent_level_label_conflict"
        elif dr_positive:
            taskA_label = "positive"
            include_A = True
        elif dr_negative:
            taskA_label = "negative"
            include_A = True
        elif dr_ambiguous:
            taskA_label = "ambiguous"
            taskA_excl = "ambiguous_label"
        else:
            taskA_label = "no_dilirank_label"
            taskA_excl = "no_dilirank_label"

        if has_conflict:
            taskA_excl += ";label_conflict" if taskA_excl else "label_conflict"
            include_A = False
        if not struct_ok:
            taskA_excl += ";structure_ineligible" if taskA_excl else "structure_ineligible"
            include_A = False


        dr_bin = 1 if dr_positive else (0 if dr_negative else None)
        dl_bin = 1 if any("positive" in l for l in dl_labels) else (0 if any("negative" in l for l in dl_labels) else None)

        parent_rows.append({
            "parent_sample_id": parent_id,
            "compound_ids": ";".join(cids),
            "primary_name": ";".join(pnames),
            "all_names": ";".join(all_names_list),
            "sources": ";".join(sorted(srcs)),
            "source_count": len(srcs),
            "parent_inchikey": ik,
            "parent_inchikey_first_block": ik_block,
            "parent_smiles": parent_smi,
            "canonical_smiles": canon_smi,
            "scaffold_split_key": scaff_key,
            "scaffold_type": scaff_type,
            "murcko_scaffold_smiles": scaff_smi,
            "structure_model_eligible": struct_ok,
            "evidence_model_eligible": evid_ok,
            "large_molecule_flag": large_mol,
            "needs_manual_review": needs_rev,
            "has_label_conflict": has_conflict,
            "evidence_direction_consensus": consensus,
            "dilirank_label": ";".join(sorted(dr_labels)) if dr_labels else "",
            "dilirank_binary_label": dr_bin,
            "dilist_label": ";".join(sorted(dl_labels)) if dl_labels else "",
            "dilist_binary_label": dl_bin,
            "include_in_taskA_main": include_A,
            "taskA_label": taskA_label,
            "taskA_exclusion_reason": taskA_excl,
        })

    df_base = pd.DataFrame(parent_rows)

    df_base = pd.concat([
        df_base[df_base["include_in_taskA_main"]].sort_values("parent_sample_id"),
        df_base[~df_base["include_in_taskA_main"]].sort_values("parent_sample_id"),
    ]).reset_index(drop=True)

    out = SPLITS / "modeling_base/modeling_base_table.csv"
    df_base.to_csv(out, index=False, encoding="utf-8")
    print(f"  [SAVED] {out.name} ({len(df_base)} parent-level rows)")

    n_inc = df_base["include_in_taskA_main"].sum()
    n_pos = int((df_base["taskA_label"] == "positive").sum())
    n_neg = int((df_base["taskA_label"] == "negative").sum())
    n_amb = int((df_base["taskA_exclusion_reason"].str.contains("ambiguous", na=False)).sum())
    n_conf = int((df_base["taskA_exclusion_reason"].str.contains("conflict", na=False)).sum())
    n_struct = int((df_base["taskA_exclusion_reason"].str.contains("structure", na=False)).sum())


    qc = []
    qc.append("# Modeling Base QC Report")
    qc.append(f"\n- Total compound rows in master: {len(master)}")
    qc.append(f"- Parent-level samples: {len(df_base)}")
    qc.append(f"- Duplicate parent groups collapsed: {len(master) - len(df_base)}")
    qc.append(f"- Task A included: {n_inc}")
    qc.append(f"  - Positive: {n_pos}")
    qc.append(f"  - Negative: {n_neg}")
    qc.append(f"- Excluded ambiguous: {n_amb}")
    qc.append(f"- Excluded conflict: {n_conf}")
    qc.append(f"- Excluded structure ineligible: {n_struct}")
    with open(SPLITS / "modeling_base/modeling_base_qc_report.md", "w") as f:
        f.write("\n".join(qc))
    print(f"  [SAVED] modeling_base_qc_report.md")

    print(f"  Task A: {n_inc} included ({n_pos}P / {n_neg}N)")
    return df_base





def build_random_split(base):
    print("\n" + "=" * 72)
    print("STAGE 5.1: Random Split")
    print("=" * 72)

    taskA = base[base["include_in_taskA_main"]].copy()
    y = taskA["taskA_label"].map({"positive": 1, "negative": 0})

    train_idx, tmp_idx = train_test_split(
        np.arange(len(taskA)), test_size=0.30, random_state=SEED, stratify=y)
    valid_idx, test_idx = train_test_split(
        tmp_idx, test_size=0.50, random_state=SEED,
        stratify=y.iloc[tmp_idx])

    split_map = {}
    for i in train_idx: split_map[i] = "train"
    for i in valid_idx: split_map[i] = "valid"
    for i in test_idx: split_map[i] = "test"

    taskA = taskA.reset_index(drop=True)
    taskA["split"] = taskA.index.map(split_map)
    taskA["seed"] = SEED
    taskA["split_type"] = "random_stratified"

    out_cols = ["parent_sample_id","compound_ids","primary_name","parent_inchikey",
                "parent_inchikey_first_block","scaffold_split_key","taskA_label",
                "split","seed","split_type"]
    taskA[out_cols].to_csv(SPLITS / "random/taskA/random_split_seed20260521.csv", index=False)

    for s in ["train","valid","test"]:
        n = (taskA["split"]==s).sum()
        p = (taskA[taskA["split"]==s]["taskA_label"]=="positive").sum()
        neg = n - p
        print(f"  {s}: {n} ({p}P / {neg}N)")


    ik_overlap = check_overlap(taskA, "parent_inchikey")
    rpt = f"# Random Split QC\n\n- Train/valid/test parent_inchikey overlap: {ik_overlap}\n- Seed: {SEED}\n- Split type: random_stratified\n"
    with open(SPLITS / "random/taskA/random_split_qc_report.md", "w") as f: f.write(rpt)

    return taskA[out_cols]


def check_overlap(df, col):
    sets = {}
    for s in ["train","valid","test"]:
        if s in df["split"].values:
            sets[s] = set(df[df["split"]==s][col].dropna())
    overlaps = {}
    for a, b in [("train","valid"),("train","test"),("valid","test")]:
        if a in sets and b in sets:
            overlaps[f"{a}_vs_{b}"] = len(sets[a] & sets[b])
    return overlaps





def build_scaffold_split(base):
    print("\n" + "=" * 72)
    print("STAGE 5.2: Scaffold Split")
    print("=" * 72)

    taskA = base[base["include_in_taskA_main"]].copy()

    review = taskA[taskA["scaffold_type"] == "scaffold_failed_review"]
    taskA = taskA[taskA["scaffold_type"] != "scaffold_failed_review"]

    scaffolds = taskA.groupby("scaffold_split_key")
    scaff_list = [(k, len(g), g["taskA_label"].value_counts().to_dict())
                  for k, g in scaffolds]


    scaff_list.sort(key=lambda x: -x[1])

    train, valid, test = [], [], []
    target_train = int(len(taskA) * 0.70)
    target_valid = int(len(taskA) * 0.15)


    for scaff_key, n, label_dist in scaff_list:
        pos = label_dist.get("positive", 0)
        neg = label_dist.get("negative", 0)


        t_n = len(train) if isinstance(train, list) else 0
        v_n = len(valid) if isinstance(valid, list) else 0
        ts_n = len(test) if isinstance(test, list) else 0


        t_room = max(0, target_train - t_n)
        v_room = max(0, target_valid - v_n)
        ts_room = max(0, len(taskA) - target_train - target_valid - ts_n)

        if t_room >= max(v_room, ts_room):
            train.append(scaff_key)
        elif v_room >= ts_room:
            valid.append(scaff_key)
        else:
            test.append(scaff_key)


    split_map = {}
    for k in train: split_map[k] = "train"
    for k in valid: split_map[k] = "valid"
    for k in test: split_map[k] = "test"

    taskA["split"] = taskA["scaffold_split_key"].map(split_map)
    taskA["seed"] = SEED
    taskA["split_type"] = "scaffold"


    scaff_train = set(train)
    scaff_valid = set(valid)
    scaff_test = set(test)
    scaff_overlap_tv = scaff_train & scaff_valid
    scaff_overlap_tt = scaff_train & scaff_test
    scaff_overlap_vt = scaff_valid & scaff_test

    print(f"  Scaffold overlap: train/valid={len(scaff_overlap_tv)}, train/test={len(scaff_overlap_tt)}, valid/test={len(scaff_overlap_vt)}")
    if scaff_overlap_tt:
        leakage("HIGH", "scaffold_split", f"{len(scaff_overlap_tt)} scaffolds overlap train/test")
    if scaff_overlap_tv:
        leakage("MEDIUM", "scaffold_split", f"{len(scaff_overlap_tv)} scaffolds overlap train/valid")

    out_cols = ["parent_sample_id","compound_ids","primary_name","parent_inchikey",
                "parent_inchikey_first_block","scaffold_split_key","scaffold_type",
                "taskA_label","split","seed","split_type"]
    taskA[out_cols].to_csv(SPLITS / "scaffold/taskA/scaffold_split_seed20260521.csv", index=False)

    if len(review) > 0:
        review[out_cols].to_csv(SPLITS / "scaffold/taskA/scaffold_split_excluded_review.csv", index=False)

    for s in ["train","valid","test"]:
        n = (taskA["split"]==s).sum()
        p = (taskA[taskA["split"]==s]["taskA_label"]=="positive").sum()
        acyclic = (taskA[(taskA["split"]==s) & (taskA["scaffold_type"]=="acyclic_no_murcko_scaffold")]).shape[0]
        print(f"  {s}: {n} ({p}P/{n-p}N, acyclic={acyclic})")

    ik_ov = check_overlap(taskA, "parent_inchikey")
    fb_ov = check_overlap(taskA, "parent_inchikey_first_block")

    rpt = f"# Scaffold Split QC\n\n- Scaffolds: {len(scaff_list)}\n- Scaffold overlap train/test: {len(scaff_overlap_tt)}\n- IK overlap: {ik_ov}\n- First block overlap: {fb_ov}\n- Acyclic samples: {int((taskA['scaffold_type']=='acyclic_no_murcko_scaffold').sum())}\n"
    with open(SPLITS / "scaffold/taskA/scaffold_split_qc_report.md", "w") as f: f.write(rpt)

    return taskA[out_cols]





def build_dilist_external(base):
    print("\n" + "=" * 72)
    print("STAGE 5.3: DILIst Non-Overlap External Split")
    print("=" * 72)

    dl_lbl = pd.read_csv(DILIST_LBL)
    master = pd.read_csv(COMP_MASTER)


    train_base = base[base["include_in_taskA_main"]]
    train_iks = set(train_base["parent_inchikey"].dropna())
    train_ik_blocks = set(train_base["parent_inchikey_first_block"].dropna())


    dl_rows = []
    for _, r in dl_lbl.iterrows():
        nm = str(r.get("compound_name","")).strip()
        lbl = str(r.get("dilist_curated",""))
        if not nm: continue


        matched = master[master["primary_name"].str.lower() == nm.lower()]
        if matched.empty:

            matched = master[master["source_drug_names"].str.lower().str.contains(nm.lower(), na=False)]
        if matched.empty:
            continue

        ik = str(matched.iloc[0].get("parent_inchikey",""))
        ik_block = str(matched.iloc[0].get("parent_inchikey_first_block",""))
        parent_id_match = base[base["parent_inchikey"] == ik]
        psid = parent_id_match.iloc[0]["parent_sample_id"] if not parent_id_match.empty else ""

        overlap = ik in train_iks
        overlap_block = ik_block in train_ik_blocks
        overlap_reason = ""
        if overlap: overlap_reason = "parent_inchikey_in_train"
        elif overlap_block: overlap_reason = "parent_inchikey_first_block_in_train"

        in_conflict = nm.lower() in set(str(c).lower() for c in pd.read_csv(LABEL_CONFLICT).get("compound_name", []))

        dl_rows.append({
            "parent_sample_id": psid,
            "compound_name": nm,
            "parent_inchikey": ik,
            "parent_inchikey_first_block": ik_block,
            "dilist_label": lbl,
            "external_label": 1 if "positive" in lbl else 0,
            "overlap_removed": overlap or overlap_block,
            "overlap_reason": overlap_reason,
            "conflict_flag": in_conflict,
        })

    df_dl = pd.DataFrame(dl_rows)
    df_dl["split"] = "external_test"
    df_dl["seed"] = SEED
    df_dl["split_type"] = "dilist_nonoverlap_external"


    clean_test = df_dl[~df_dl["overlap_removed"] & ~df_dl["conflict_flag"]]
    removed = df_dl[df_dl["overlap_removed"]]
    conflict_review = df_dl[df_dl["conflict_flag"] & ~df_dl["overlap_removed"]]

    out_cols = ["parent_sample_id","compound_name","parent_inchikey","parent_inchikey_first_block",
                "dilist_label","external_label","split","overlap_removed","overlap_reason",
                "conflict_flag","seed","split_type"]

    clean_test[out_cols].to_csv(SPLITS / "external_dilist/taskA/dilist_nonoverlap_external_split.csv", index=False)
    removed[out_cols].to_csv(SPLITS / "external_dilist/taskA/dilist_nonoverlap_removed_overlap.csv", index=False)
    conflict_review[out_cols].to_csv(SPLITS / "external_dilist/taskA/dilist_conflict_external_review.csv", index=False)

    print(f"  DILIst total matched: {len(df_dl)}")
    print(f"  Clean external test: {len(clean_test)}")
    print(f"  Removed (overlap): {len(removed)}")
    print(f"  Conflict review: {len(conflict_review)}")

    rpt = f"# DILIst Non-Overlap QC\n\n- Total matched: {len(df_dl)}\n- Clean test: {len(clean_test)}\n- Removed overlap: {len(removed)}\n- Conflict review: {len(conflict_review)}\n"
    with open(SPLITS / "external_dilist/taskA/dilist_nonoverlap_qc_report.md", "w") as f: f.write(rpt)

    ik_leak = len(set(clean_test["parent_inchikey"]) & train_iks)
    if ik_leak > 0:
        leakage("HIGH", "dilist_external", f"{ik_leak} parent_inchikey overlap between DILIst test and DILIrank train")

    return clean_test





def build_temporal_split(base):
    print("\n" + "=" * 72)
    print("STAGE 5.4: Temporal Old-to-New Split")
    print("=" * 72)

    dr_raw = pd.read_excel(DILIRANK_RAW, header=1)
    print(f"  DILIrank 2.0 raw: {len(dr_raw)} rows")
    print(f"  Comment distribution: {dict(dr_raw['Comment'].value_counts())}")


    def map_temporal(comment):
        c = str(comment).strip()
        if c == "Unchanged":
            return "Unchanged", True, False, False, False
        elif c == "Revised":
            return "Revised", True, False, True, True
        elif c in ("New", "Added", "Newly added", "New Drug", "Added in 2.0"):
            return "New", False, True, False, False
        else:
            return "unknown", False, False, False, False

    membership = []
    for _, r in dr_raw.iterrows():
        lt = str(r.get("LTKBID",""))
        nm = str(r.get("CompoundName","")).strip()
        comment = str(r.get("Comment","")).strip()
        role, in_1, added_2, revised, drift = map_temporal(comment)
        membership.append({
            "ltkbid": lt,
            "compound_name": nm,
            "dilirank2_label": str(r.get("vDILI-Concern","")),
            "dilirank2_comment": comment,
            "in_dilirank1_original": in_1,
            "dilirank2_added": added_2,
            "dilirank2_revised_from_1": revised,
            "label_drift_possible": drift,
            "temporal_role": f"old_train_candidate{'_label_drift' if drift else ''}" if in_1 else
                             ("temporal_test_candidate" if added_2 else "membership_unknown_review"),
        })

    df_membership = pd.DataFrame(membership)
    df_membership.to_csv(SPLITS / "temporal_1to2/dilirank_version_membership.csv", index=False)
    print(f"  [SAVED] dilirank_version_membership.csv")

    n_unchanged = int((df_membership["dilirank2_comment"]=="Unchanged").sum())
    n_revised = int((df_membership["dilirank2_comment"]=="Revised").sum())
    n_new = int((df_membership["dilirank2_comment"]=="New").sum())
    print(f"  Unchanged: {n_unchanged}, Revised: {n_revised}, New: {n_new}")


    taskA = base[(base["include_in_taskA_main"]) & (base["structure_model_eligible"])].copy()


    matched_rows = []
    for _, mb in taskA.iterrows():
        pnames = str(mb.get("primary_name","")).lower()
        all_ns = str(mb.get("all_names","")).lower()

        best_match = None
        for _, mem in df_membership.iterrows():
            mem_nm = mem["compound_name"].lower()
            if mem_nm == pnames or mem_nm in all_ns or pnames in mem_nm:
                best_match = mem
                break

        if best_match is not None:
            row = dict(mb)
            row.update({
                "dilirank_ltkbid": best_match["ltkbid"],
                "dilirank2_label": best_match["dilirank2_label"],
                "dilirank2_comment": best_match["dilirank2_comment"],
                "in_dilirank1_original": best_match["in_dilirank1_original"],
                "dilirank2_added": best_match["dilirank2_added"],
                "dilirank2_revised_from_1": best_match["dilirank2_revised_from_1"],
                "label_drift_possible": best_match["label_drift_possible"],
                "temporal_role": best_match["temporal_role"],
            })
            matched_rows.append(row)

    df_temporal = pd.DataFrame(matched_rows)
    print(f"  Matched to modeling base: {len(df_temporal)}")


    old = df_temporal[df_temporal["temporal_role"].str.contains("old_train")]
    new = df_temporal[df_temporal["temporal_role"] == "temporal_test_candidate"]
    unknown = df_temporal[~df_temporal.index.isin(old.index) & ~df_temporal.index.isin(new.index)]

    print(f"  Old train candidates: {len(old)} (Revised={int((old['dilirank2_comment']=='Revised').sum())})")
    print(f"  Temporal test candidates: {len(new)}")


    old_y = old["taskA_label"].map({"positive":1,"negative":0})
    train_idx, valid_idx = train_test_split(
        np.arange(len(old)), test_size=0.20, random_state=SEED, stratify=old_y)

    old_train = old.iloc[train_idx].copy()
    old_valid = old.iloc[valid_idx].copy()
    old_train["split"] = "train"
    old_valid["split"] = "valid"
    new["split"] = "temporal_test"

    df_all = pd.concat([old_train, old_valid, new])
    df_all["seed"] = SEED
    df_all["split_type"] = "dilirank2_harmonized_old_to_new_temporal_holdout"

    out_cols = ["parent_sample_id","compound_ids","primary_name","parent_inchikey",
                "parent_inchikey_first_block","scaffold_split_key",
                "dilirank_ltkbid","dilirank2_label","taskA_label","dilirank2_comment",
                "in_dilirank1_original","dilirank2_added","dilirank2_revised_from_1",
                "label_drift_possible","temporal_role","split","seed","split_type"]

    df_all[out_cols].to_csv(SPLITS / "temporal_1to2/temporal_old_to_new_split.csv", index=False)

    if len(unknown) > 0:
        unknown[out_cols].to_csv(SPLITS / "temporal_1to2/temporal_excluded_review.csv", index=False)


    train_val_iks = set(old["parent_inchikey"])
    test_iks = set(new["parent_inchikey"])
    ik_leak = train_val_iks & test_iks
    train_val_blocks = set(old["parent_inchikey_first_block"])
    test_blocks = set(new["parent_inchikey_first_block"])
    block_leak = train_val_blocks & test_blocks

    print(f"  IK overlap train/val vs temporal_test: {len(ik_leak)}")
    print(f"  First block overlap: {len(block_leak)}")

    if ik_leak:
        leakage("HIGH", "temporal", f"{len(ik_leak)} parent_inchikey overlap temporal_test/train")
    if block_leak:
        leakage("MEDIUM", "temporal", f"{len(block_leak)} first-block overlap (analog risk)")


    train_scaffs = set(old["scaffold_split_key"])
    test_scaffs = set(new["scaffold_split_key"])
    scaff_overlap = train_scaffs & test_scaffs
    if scaff_overlap:
        leakage("MEDIUM", "temporal", f"{len(scaff_overlap)} scaffold keys overlap (analog_leakage_risk)")


    rpt = f"""# Temporal Split Readiness Report

**Split name:** DILIrank2_harmonized_old_to_new_temporal_holdout

**Important:** This temporal holdout uses harmonized DILIrank 2.0 labels.
It is an old-to-new added-drug temporal holdout, NOT a strict original DILIrank 1.0-label training design.

- Old train+valid candidates: {len(old)} (Unchanged={n_unchanged}, Revised={n_revised})
- Temporal test candidates: {len(new)} (New={n_new})
- IK overlap train/val vs test: {len(ik_leak)}
- First block overlap: {len(block_leak)}
- Scaffold overlap: {len(scaff_overlap)}
"""
    with open(SPLITS / "temporal_1to2/temporal_split_readiness_report.md", "w") as f: f.write(rpt)

    drift_plan = f"""# Temporal Label Drift Sensitivity Plan

**Revised drugs (n={n_revised}):** These drugs existed in DILIrank 1.0 but had their labels revised in 2.0.
They are kept as old_train_candidate with DILIrank 2.0 labels but flagged `label_drift_possible=True`.

**Sensitivity analyses (deferred):**
1. Train without Revised drugs; test on temporal_test only
2. Compare model predictions on Revised vs Unchanged old drugs
3. Assess whether Revised labels introduce systematic bias

**Note:** Without DILIrank 1.0 original label table, direct label drift quantification is not possible.
"""
    with open(SPLITS / "temporal_1to2/temporal_label_drift_sensitivity_plan.md", "w") as f: f.write(drift_plan)

    print(f"  [SAVED] temporal split outputs")
    return df_all





def build_post2021_lockbox(base):
    print("\n" + "=" * 72)
    print("STAGE 5.5: Post-2021 Lockbox")
    print("=" * 72)


    submissions = pd.read_csv(RAW / "fda_approvals/drugsatfda/Submissions.txt", sep="\t", encoding="latin-1")
    products = pd.read_csv(RAW / "fda_approvals/drugsatfda/Products.txt", sep="\t", encoding="latin-1")
    applications = pd.read_csv(RAW / "fda_approvals/drugsatfda/Applications.txt", sep="\t", encoding="latin-1")


    submissions["approval_date"] = pd.to_datetime(submissions["SubmissionStatusDate"], errors="coerce")
    submissions["approval_year"] = submissions["approval_date"].dt.year



    prod_sub = products.merge(submissions[["ApplNo","SubmissionType","approval_year"]], on="ApplNo", how="left")
    orig_only = prod_sub[prod_sub["SubmissionType"] == "ORIG"]
    first_approval = orig_only.groupby("ActiveIngredient")["approval_year"].min().reset_index()
    first_approval.columns = ["ActiveIngredient", "first_approval_year"]


    truly_new_post2021 = first_approval[first_approval["first_approval_year"] >= 2022]
    post2021_ingredients = truly_new_post2021["ActiveIngredient"].dropna().unique()
    post2021_years = dict(zip(truly_new_post2021["ActiveIngredient"], truly_new_post2021["first_approval_year"]))

    if len(post2021_ingredients) == 0:
        print("  No truly new post-2021 ActiveIngredients found (all existed before)")

        pd.DataFrame(columns=["lockbox_id","drug_name","active_ingredient","approval_year",
            "mapped_compound_id","parent_sample_id","parent_inchikey",
            "structure_model_eligible","evidence_maturity","hard_label_allowed",
            "lockbox_role","manual_review_required","notes"]).to_csv(
            SPLITS / "post2021_lockbox/post2021_lockbox.csv", index=False)
        return None

    print(f"  Truly new post-2021 ActiveIngredients (first approval >= 2022): {len(post2021_ingredients)}")
    print(f"  (Excluded existing drugs with new post-2021 submissions)")


    truly_new_post2021.to_csv(SPLITS / "post2021_lockbox/post2021_candidate_raw.csv", index=False)


    master = pd.read_csv(COMP_MASTER)
    lockbox = []
    for ing in post2021_ingredients:
        ing_lower = str(ing).lower().strip()
        if not ing_lower: continue

        matched = master[
            master["primary_name"].str.lower() == ing_lower
        ]
        if matched.empty:
            matched = master[
                master["source_drug_names"].str.lower().str.contains(ing_lower, na=False)
            ]

        cid = matched.iloc[0]["compound_id"] if not matched.empty else ""
        psid = ""; ik = ""; struct_ok = False
        if not matched.empty:
            r = matched.iloc[0]
            cid = r["compound_id"]
            ik = str(r.get("parent_inchikey",""))
            struct_ok = r.get("structure_model_eligible",False) in (True, "True")
            psm = base[base["parent_inchikey"] == ik]
            if not psm.empty:
                psid = psm.iloc[0]["parent_sample_id"]

        lockbox.append({
            "lockbox_id": f"LB_{len(lockbox)+1:04d}",
            "drug_name": str(ing),
            "active_ingredient": str(ing),
            "application_number": "",
            "approval_date": "",
            "approval_year": int(post2021_years.get(ing, 2022)),
            "mapped_compound_id": cid,
            "parent_sample_id": psid,
            "parent_inchikey": ik,
            "structure_model_eligible": struct_ok,
            "evidence_maturity": "emerging",
            "hard_label_allowed": False,
            "lockbox_role": "post2021_lockbox_test_only",
            "manual_review_required": True,
            "notes": "Post-2021 first-approved ActiveIngredient; not eligible for training/validation/tuning. Evidence maturity insufficient.",
        })

    df_lb = pd.DataFrame(lockbox)
    df_lb.to_csv(SPLITS / "post2021_lockbox/post2021_lockbox.csv", index=False)
    df_lb.to_csv(SPLITS / "post2021_lockbox/post2021_lockbox_mapping_status.csv", index=False)


    maturity_rows = []
    for _, lb in df_lb.iterrows():
        cid = lb["mapped_compound_id"]
        if not cid: continue
        ev = pd.read_csv(EVID_TABLE)
        ev_match = ev[ev["compound_id"] == cid]
        maturity_rows.append({
            "compound_id": cid,
            "parent_sample_id": lb["parent_sample_id"],
            "drug_name": lb["drug_name"],
            "active_ingredient": lb["active_ingredient"],
            "approval_year": lb["approval_year"],
            "years_since_approval": 2026 - int(lb["approval_year"]) if lb["approval_year"] else 0,
            "has_livertox_record": ev_match.iloc[0]["has_livertox_evidence"] if not ev_match.empty else False,
            "has_fda_hepatotoxicity_label_signal": ev_match.iloc[0]["has_fda_label_signal"] if not ev_match.empty else False,
            "evidence_maturity": "emerging",
            "hard_label_allowed": False,
            "maturity_reason": "Post-2021 approval; insufficient time for mature DILI signal accumulation",
            "manual_review_required": True,
        })

    pd.DataFrame(maturity_rows).to_csv(CUR / "evidence/post2021_evidence_maturity.csv", index=False)

    n_matched = int((df_lb["mapped_compound_id"] != "").sum())
    print(f"  Lockbox entries: {len(df_lb)}")
    print(f"  Matched to compound_master: {n_matched}")
    print(f"  [SAVED] post2021 lockbox outputs")


    rpt = f"""# Post-2021 Lockbox Readiness Report

- Total lockbox entries: {len(df_lb)}
- Matched to compound_master: {n_matched}
- Evidence maturity: all "emerging" (post-2021, insufficient time)
- hard_label_allowed: False for all (no definitive DILI signal yet)
- lockbox_role: post2021_lockbox_test_only
- These compounds must NOT appear in any training/validation/tuning set.

**Warning:** Post-2021 compounds represent exploratory/lockbox evaluation only.
Do not claim generalization to future approvals without prospective validation.
"""
    with open(SPLITS / "post2021_lockbox/post2021_lockbox_readiness_report.md", "w") as f: f.write(rpt)

    return df_lb





def build_cluster_out_split(base):
    print("\n" + "=" * 72)
    print("STAGE 5.6: Cluster-Out Split")
    print("=" * 72)

    taskA = base[base["include_in_taskA_main"]].copy()


    fps = []
    valid_idx = []
    for i, (_, r) in enumerate(taskA.iterrows()):
        smi = str(r.get("parent_smiles",""))
        if not smi or smi == "nan":
            continue
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                fps.append(fp)
                valid_idx.append(i)
        except:
            continue

    print(f"  Fingerprints generated: {len(fps)}/{len(taskA)}")


    n_fps = len(fps)
    dists = []
    for i in range(n_fps):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        for s in sims:
            dists.append(1.0 - s)

    if len(dists) > 0 and len(dists) == n_fps * (n_fps - 1) // 2:
        cutoff = np.percentile(dists, 10) if len(dists) > 0 else 0.35
        try:
            clusters = Butina.ClusterData(dists, n_fps, cutoff, isDistData=True)
            cluster_assign = {}
            for ci, cluster in enumerate(clusters):
                for idx in cluster:
                    cluster_assign[valid_idx[idx]] = ci
            n_clusters = len(clusters)
        except Exception as e:
            print(f"  Butina failed: {e}, using fallback")
            cluster_assign = {vi: vi for vi in valid_idx}
            n_clusters = n_fps
    else:
        cluster_assign = {vi: vi for vi in valid_idx}
        n_clusters = n_fps

    print(f"  Clusters: {n_clusters} (Butina cutoff~{cutoff:.3f})")

    taskA["cluster_id"] = taskA.index.map(cluster_assign)
    taskA["cluster_id"] = taskA["cluster_id"].fillna(-1).astype(int)


    clust_groups = taskA.groupby("cluster_id")
    clust_list = [(k, len(g)) for k, g in clust_groups]
    clust_list.sort(key=lambda x: -x[1])

    train_clust, valid_clust, test_clust = [], [], []
    t_n, v_n, ts_n = 0, 0, 0
    target = len(taskA)
    target_tr = int(target * 0.70)
    target_vl = int(target * 0.15)

    for cid, n in clust_list:
        if t_n + n <= target_tr:
            train_clust.append(cid); t_n += n
        elif v_n + n <= target_vl:
            valid_clust.append(cid); v_n += n
        else:
            test_clust.append(cid); ts_n += n

    cmap = {}
    for c in train_clust: cmap[c] = "train"
    for c in valid_clust: cmap[c] = "valid"
    for c in test_clust: cmap[c] = "test"
    taskA["split"] = taskA["cluster_id"].map(cmap)
    taskA["seed"] = SEED
    taskA["split_type"] = "cluster_out_butina"

    out_cols = ["parent_sample_id","parent_inchikey","parent_smiles","cluster_id",
                "taskA_label","split","seed","split_type"]
    taskA[out_cols].to_csv(SPLITS / "cluster_out/taskA/cluster_out_split_seed20260521.csv", index=False)

    for s in ["train","valid","test"]:
        n = (taskA["split"]==s).sum()
        print(f"  {s}: {n} in {taskA[taskA['split']==s]['cluster_id'].nunique()} clusters")

    ik_ov = check_overlap(taskA, "parent_inchikey")
    rpt = f"# Cluster-Out Split QC\n\n- Clusters: {n_clusters}\n- IK overlap: {ik_ov}\n"
    with open(SPLITS / "cluster_out/taskA/cluster_out_qc_report.md", "w") as f: f.write(rpt)

    return taskA[out_cols]





def build_leakage_audit(base, scaffold_split, temporal_split, cluster_split, dilist_clean):
    print("\n" + "=" * 72)
    print("STAGE 5.7+5.8: Tanimoto NN + Leakage Audit")
    print("=" * 72)


    taskA = base[base["include_in_taskA_main"]]
    fp_dict = {}
    smi_dict = {}
    for _, r in taskA.iterrows():
        smi = str(r.get("parent_smiles",""))
        if not smi or smi == "nan": continue
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                fp_dict[r["parent_sample_id"]] = fp
                smi_dict[r["parent_sample_id"]] = smi
        except: pass

    print(f"  Fingerprints: {len(fp_dict)}")


    split_configs = [
        ("scaffold", scaffold_split),
        ("temporal", temporal_split),
        ("cluster_out", cluster_split),
    ]

    all_nn_rows = []

    for split_name, split_df in split_configs:
        if split_df is None or len(split_df) == 0: continue
        if "split" not in split_df.columns: continue

        train_ids = set(split_df[split_df["split"]=="train"]["parent_sample_id"])
        test_ids = set(split_df[split_df["split"].isin(["test","temporal_test"])]["parent_sample_id"])

        print(f"  {split_name}: train={len(train_ids)}, test={len(test_ids)}")

        for test_id in test_ids:
            if test_id not in fp_dict: continue
            test_fp = fp_dict[test_id]
            test_ik = taskA[taskA["parent_sample_id"]==test_id].iloc[0].get("parent_inchikey","") if not taskA[taskA["parent_sample_id"]==test_id].empty else ""
            test_name = taskA[taskA["parent_sample_id"]==test_id].iloc[0].get("primary_name","") if not taskA[taskA["parent_sample_id"]==test_id].empty else ""

            best_sim = 0
            best_train_id = ""
            for train_id in train_ids:
                if train_id not in fp_dict: continue
                sim = DataStructs.TanimotoSimilarity(test_fp, fp_dict[train_id])
                if sim > best_sim:
                    best_sim = sim
                    best_train_id = train_id

            if best_sim > 0:
                train_ik = taskA[taskA["parent_sample_id"]==best_train_id].iloc[0].get("parent_inchikey","") if not taskA[taskA["parent_sample_id"]==best_train_id].empty else ""
                train_name = taskA[taskA["parent_sample_id"]==best_train_id].iloc[0].get("primary_name","") if not taskA[taskA["parent_sample_id"]==best_train_id].empty else ""

                test_ik_block = test_ik.split("-")[0] if test_ik and "-" in str(test_ik) else ""
                train_ik_block = train_ik.split("-")[0] if train_ik and "-" in str(train_ik) else ""

                analog_flag = ""
                if best_sim >= 0.90: analog_flag = "very_high_similarity_neighbor"
                elif best_sim >= 0.85: analog_flag = "high_similarity_neighbor"

                all_nn_rows.append({
                    "split_name": split_name,
                    "test_parent_sample_id": test_id,
                    "test_parent_inchikey": test_ik,
                    "test_primary_name": test_name,
                    "nearest_train_parent_sample_id": best_train_id,
                    "nearest_train_parent_inchikey": train_ik,
                    "nearest_train_name": train_name,
                    "max_tanimoto": round(best_sim, 4),
                    "same_scaffold": str(taskA[taskA["parent_sample_id"]==test_id].iloc[0].get("scaffold_split_key","")) == str(taskA[taskA["parent_sample_id"]==best_train_id].iloc[0].get("scaffold_split_key","")) if not taskA[taskA["parent_sample_id"]==test_id].empty and not taskA[taskA["parent_sample_id"]==best_train_id].empty else False,
                    "same_inchikey_first_block": test_ik_block == train_ik_block,
                    "analog_leakage_flag": analog_flag,
                })

    df_nn = pd.DataFrame(all_nn_rows)
    df_nn.to_csv(SPLITS / "leakage_audit/tanimoto_nearest_neighbor_report.csv", index=False)
    print(f"  [SAVED] tanimoto_nearest_neighbor_report.csv ({len(df_nn)} rows)")


    n_vhigh = int((df_nn["max_tanimoto"] >= 0.90).sum()) if not df_nn.empty else 0
    n_high = int((df_nn["max_tanimoto"] >= 0.85).sum()) if not df_nn.empty else 0
    print(f"  Very high similarity (>=0.90): {n_vhigh}")
    print(f"  High similarity (>=0.85): {n_high}")

    if n_vhigh > 0:
        leakage("MEDIUM", "tanimoto_nn", f"{n_vhigh} test samples have train neighbor with Tanimoto >=0.90")
    if n_high > 20:
        leakage("LOW", "tanimoto_nn", f"{n_high} test samples have train neighbor with Tanimoto >=0.85")


    lb_path = SPLITS / "post2021_lockbox/post2021_lockbox.csv"
    if lb_path.exists():
        lb = pd.read_csv(lb_path)
        lb_iks = set(lb["parent_inchikey"].dropna())
        lb_cids = set(lb["mapped_compound_id"].dropna())

        for split_name, split_df in [("scaffold", scaffold_split), ("temporal", temporal_split),
                                      ("cluster_out", cluster_split)]:
            if split_df is None or len(split_df) == 0: continue
            split_iks = set(split_df["parent_inchikey"].dropna())
            contamin = lb_iks & split_iks & {ik for ik in split_iks if ik and ik != "nan"}
            if contamin:
                leakage("HIGH", f"post2021_contamination_{split_name}",
                        f"{len(contamin)} post-2021 lockbox parent_inchikeys found in {split_name} split — REMOVING from train/valid, moving to lockbox")

                split_df.loc[split_df["parent_inchikey"].isin(contamin) & split_df["split"].isin(["train","valid"]), "split"] = "post2021_removed"


    df_leak = pd.DataFrame(LEAKAGE_ISSUES)
    df_leak.to_csv(SPLITS / "leakage_audit/leakage_blocking_issues.csv", index=False)

    high_n = int((df_leak["severity"]=="HIGH").sum()) if not df_leak.empty else 0
    medium_n = int((df_leak["severity"]=="MEDIUM").sum()) if not df_leak.empty else 0

    if high_n > 0: leak_verdict = "FAIL_leakage_detected"
    elif medium_n > 2: leak_verdict = "CONDITIONAL_PASS_minor_warnings"
    else: leak_verdict = "PASS_stage5_splits_ready"


    with open(SPLITS / "leakage_audit/leakage_audit_report.json", "w") as f:
        json.dump({"verdict": leak_verdict, "issues": LEAKAGE_ISSUES,
                    "tanimoto_summary": {"very_high_ge_0.90": n_vhigh, "high_ge_0.85": n_high}}, f, indent=2)


    rpt = ["# Leakage Audit Report", f"\n**Verdict:** `{leak_verdict}`", f"\n- HIGH: {high_n}, MEDIUM: {medium_n}"]
    for l in LEAKAGE_ISSUES:
        rpt.append(f"- [{l['severity']}] {l['split']}: {l['detail'][:120]}")
    with open(SPLITS / "leakage_audit/leakage_audit_report.md", "w") as f:
        f.write("\n".join(rpt))

    print(f"\n  Leakage verdict: {leak_verdict}")
    return leak_verdict, df_leak





def main():
    print(f"Stage 5: Splits & Leakage Audit")
    print(f"Start: {NOW}\n")


    base = build_modeling_base()


    random_df = build_random_split(base)


    scaffold_df = build_scaffold_split(base)


    dilist_clean = build_dilist_external(base)


    temporal_df = build_temporal_split(base)


    lockbox_df = build_post2021_lockbox(base)


    cluster_df = build_cluster_out_split(base)


    leak_verdict, leak_df = build_leakage_audit(base, scaffold_df, temporal_df, cluster_df, dilist_clean)


    high_leaks = int((leak_df["severity"]=="HIGH").sum()) if not leak_df.empty else 0
    if high_leaks > 0: stage5_verdict = "FAIL_leakage_detected"
    else: stage5_verdict = "PASS_stage5_splits_ready"

    n_inc = base["include_in_taskA_main"].sum()
    n_pos = int((base["taskA_label"]=="positive").sum())
    n_neg = int((base["taskA_label"]=="negative").sum())


    summary = f"""# Stage 5 Split Summary Report

**Generated:** {NOW}
**Final Verdict:** `{stage5_verdict}`

Modeling Base
- Parent-level samples: {len(base)}
- Task A included: {n_inc} ({n_pos} positive, {n_neg} negative)

Random Split
- Train/valid/test: {int((random_df['split']=='train').sum())}/{int((random_df['split']=='valid').sum())}/{int((random_df['split']=='test').sum())}

Scaffold Split
- Train/valid/test: {int((scaffold_df['split']=='train').sum())}/{int((scaffold_df['split']=='valid').sum())}/{int((scaffold_df['split']=='test').sum())}

DILIst External
- Clean test: {len(dilist_clean)}

Temporal Old-to-New
- Old train candidates: {int((temporal_df['temporal_role'].str.contains('old_train')).sum())}
- Temporal test: {int((temporal_df['split']=='temporal_test').sum())}
- **Important:** This is DILIrank 2.0 harmonized old-to-new temporal holdout, NOT strict DILIrank 1.0 original-label training.

Post-2021 Lockbox
- Entries: {len(lockbox_df) if lockbox_df is not None else 0}

Cluster-Out Split
- Train/valid/test: {int((cluster_df['split']=='train').sum())}/{int((cluster_df['split']=='valid').sum())}/{int((cluster_df['split']=='test').sum())}

Leakage Audit
- Verdict: {leak_verdict}
- HIGH: {int((leak_df['severity']=='HIGH').sum()) if not leak_df.empty else 0}
- MEDIUM: {int((leak_df['severity']=='MEDIUM').sum()) if not leak_df.empty else 0}

Stage 6 Readiness
- Can proceed to Stage 6: {'YES' if stage5_verdict.startswith('PASS') else 'CONDITIONAL'}
"""
    with open(REPORTS / "stage5_split_summary_report.md", "w") as f: f.write(summary)
    print(f"\n[SAVED] stage5_split_summary_report.md")


    print(f"\n{'='*72}")
    print("STAGE 5 FINAL OUTPUT")
    print(f"{'='*72}")
    print(f"  1.  modeling_base_table rows:             {len(base)}")
    print(f"  2.  taskA included:                       {n_inc} ({n_pos}P / {n_neg}N)")
    print(f"  3.  random split T/V/T:                   {int((random_df['split']=='train').sum())}/{int((random_df['split']=='valid').sum())}/{int((random_df['split']=='test').sum())}")
    print(f"  4.  scaffold split T/V/T:                 {int((scaffold_df['split']=='train').sum())}/{int((scaffold_df['split']=='valid').sum())}/{int((scaffold_df['split']=='test').sum())}")
    print(f"  5.  DILIst external clean test:           {len(dilist_clean)}")
    print(f"  6.  temporal old train candidate:         {int((temporal_df['temporal_role']=='old_train_candidate').sum())}")
    print(f"  7.  temporal revised old candidate:       {int((temporal_df['dilirank2_comment']=='Revised').sum())}")
    print(f"  8.  temporal test candidate:              {int((temporal_df['split']=='temporal_test').sum())}")
    print(f"  9.  post2021 lockbox count:               {len(lockbox_df) if lockbox_df is not None else 0}")
    print(f"  10. cluster-out T/V/T:                    {int((cluster_df['split']=='train').sum())}/{int((cluster_df['split']=='valid').sum())}/{int((cluster_df['split']=='test').sum())}")
    print(f"  11. parent_inchikey leakage (HIGH):       {high_leaks}")
    print(f"  12. scaffold leakage in scaffold split:   (see leakage report)")
    print(f"  13. post2021 contamination:               (see leakage report)")
    print(f"  14. DILIst overlap leakage:               (see leakage report)")
    print(f"  15. leakage audit verdict:                {leak_verdict}")
    print(f"  16. final Stage 5 verdict:                {stage5_verdict}")
    print(f"{'='*72}")


if __name__ == "__main__":
    main()
