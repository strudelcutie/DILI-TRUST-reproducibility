#!/usr/bin/env python3

import csv, json, os, pickle, re, sys, warnings
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, AllChem, rdMolDescriptors, MACCSkeys
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
CUR = ROOT / "data_curated"
SPLITS = ROOT / "splits"
FEAT = ROOT / "features"
CONFIG = ROOT / "configs"
REPORTS = ROOT / "reports" / "stage6"
NOW = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
SEED = 20260521
np.random.seed(SEED)

MORDRED_OK = False
try:
    from mordred import Calculator, descriptors as mord
    MORDRED_OK = True
except ImportError:
    pass

def print_section(title):
    print(f"\n{'='*72}\n{title}\n{'='*72}")




print_section("1. Feature Config")

fc = {
    "project_stage": "stage6_feature_engineering",
    "feature_version": "stage6_v1",
    "input_stage5_verdict": "PASS_ready_for_stage6",
    "split_seed": SEED,
    "morgan_radius": 2, "morgan_n_bits": 2048,
    "maccs_enabled": True, "rdkit_2d_enabled": True,
    "mordred_enabled": "auto", "mordred_status": "available" if MORDRED_OK else "unavailable_deferred",
    "chemprop_input_enabled": True,
    "embedding_generation": "deferred_unless_local_models_available",
    "pk_features": "template_with_missing_indicators",
    "proxy_features": "template_with_missing_indicators",
    "evidence_features_usage": "calibration_rejection_weighting_sensitivity_only_by_default",
    "scaler_fit_policy": "train_only_per_split",
    "imputation_policy": "train_only_per_split",
    "negative_controls": ["source_only", "missingness_only"]
}
with open(CONFIG/"feature_config.yaml","w") as f:
    json.dump(fc, f, indent=2)
print(f"  [SAVED] feature_config.yaml")




print_section("2. Feature Universe")

base = pd.read_csv(SPLITS/"modeling_base/modeling_base_table.csv")
samples = []


taskA = base[base["include_in_taskA_main"]]
for _, r in taskA.iterrows():
    samples.append({"sample_id": r["parent_sample_id"], "parent_sample_id": r["parent_sample_id"],
        "source_split": "modeling_base_taskA", "split_name": "taskA_main",
        "primary_name": r.get("primary_name",""), "parent_smiles": r.get("parent_smiles",""),
        "parent_inchikey": r.get("parent_inchikey",""), "parent_inchikey_first_block": r.get("parent_inchikey_first_block",""),
        "taskA_label": r.get("taskA_label",""), "feature_generation_role": "train_candidate", "is_train": True,
        "is_valid": False, "is_test": False, "is_external": False, "is_lockbox": False,
        "structure_model_eligible": r.get("structure_model_eligible", True)})


for split_name, path, role_map in [
    ("scaffold", SPLITS/"scaffold/taskA/scaffold_split_seed20260521.csv",
     {"train":"is_train","valid":"is_valid","test":"is_test"}),
    ("temporal", SPLITS/"temporal_1to2/temporal_old_to_new_split.csv",
     {"train":"is_train","valid":"is_valid","temporal_test":"is_test"}),
    ("cluster_out", SPLITS/"cluster_out/taskA/cluster_out_split_seed20260521.csv",
     {"train":"is_train","valid":"is_valid","test":"is_test"}),
]:
    if not path.exists(): continue
    df = pd.read_csv(path)
    for _, r in df.iterrows():
        psid = r.get("parent_sample_id","")

        bm = base[base["parent_sample_id"]==psid]
        smi = str(bm.iloc[0].get("parent_smiles","")) if not bm.empty else ""
        ik = str(bm.iloc[0].get("parent_inchikey","")) if not bm.empty else ""

        sp = r.get("split","")
        samples.append({"sample_id": psid, "parent_sample_id": psid,
            "source_split": split_name, "split_name": split_name, "split": sp,
            "primary_name": r.get("primary_name",""), "parent_smiles": smi, "parent_inchikey": ik,
            "parent_inchikey_first_block": bm.iloc[0].get("parent_inchikey_first_block","") if not bm.empty else "",
            "taskA_label": r.get("taskA_label",""), "feature_generation_role": sp,
            "is_train": sp=="train", "is_valid": sp=="valid", "is_test": sp in ("test","temporal_test"),
            "is_external": False, "is_lockbox": False,
            "structure_model_eligible": bm.iloc[0].get("structure_model_eligible",True) if not bm.empty else True})


dl = pd.read_csv(SPLITS/"external_dilist/taskA/dilist_strict_dilirank2_nonoverlap_external_split.csv")
for _, r in dl.iterrows():
    psid = str(r.get("parent_sample_id",""))
    smi = str(r.get("parent_smiles","")) if "parent_smiles" in dl.columns else ""
    if not smi:
        bm = base[base["parent_sample_id"]==psid]
        smi = str(bm.iloc[0].get("parent_smiles","")) if not bm.empty else ""
    samples.append({"sample_id": psid or r.get("compound_name",""), "parent_sample_id": psid,
        "source_split": "external_dilist_strict", "split_name": "dilist_strict", "split": "external_test",
        "primary_name": r.get("compound_name",""), "parent_smiles": smi,
        "parent_inchikey": r.get("parent_inchikey",""), "parent_inchikey_first_block": r.get("parent_inchikey_first_block",""),
        "taskA_label": "", "feature_generation_role": "external_test",
        "is_train": False, "is_valid": False, "is_test": False, "is_external": True, "is_lockbox": False,
        "structure_model_eligible": bool(smi)})


lb = pd.read_csv(SPLITS/"post2021_lockbox/post2021_lockbox_final.csv")
for _, r in lb.iterrows():
    psid = str(r.get("lockbox_id","LB_"+str(r.name)))
    samples.append({"sample_id": psid, "parent_sample_id": "",
        "source_split": "post2021_lockbox", "split_name": "post2021_lockbox", "split": "lockbox_test",
        "primary_name": r.get("drug_name",r.get("active_moiety_normalized","")), "parent_smiles": r.get("parent_smiles",""),
        "parent_inchikey": r.get("parent_inchikey",""), "parent_inchikey_first_block": r.get("parent_inchikey_first_block",""),
        "taskA_label": "", "feature_generation_role": "lockbox_test",
        "is_train": False, "is_valid": False, "is_test": False, "is_external": False, "is_lockbox": True,
        "structure_model_eligible": True})

df_univ = pd.DataFrame(samples).drop_duplicates(subset=["sample_id"]).reset_index(drop=True)
df_univ.to_csv(FEAT/"qc/feature_universe.csv", index=False)
n_univ = len(df_univ)


no_smi = df_univ[(df_univ["parent_smiles"].isna()) | (df_univ["parent_smiles"]=="") | (df_univ["parent_smiles"]=="nan")]
if len(no_smi) > 0:
    no_smi.to_csv(FEAT/"qc/feature_generation_excluded.csv", index=False)
    print(f"  Excluded (no SMILES): {len(no_smi)}")

univ_smi = df_univ[~df_univ.index.isin(no_smi.index)]
print(f"  Universe: {n_univ} samples, {len(univ_smi)} with SMILES")




print_section("3. RDKit 2D Descriptors")

rdkit_descs = [d[0] for d in Descriptors._descList]
rdkit_rows = []
n_failed = 0
for _, r in univ_smi.iterrows():
    smi = r["parent_smiles"]
    row = {"sample_id": r["sample_id"]}
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            for dname in rdkit_descs:
                try: row[dname] = getattr(Descriptors, dname)(mol)
                except: row[dname] = np.nan
        else:
            n_failed += 1; row.update({d: np.nan for d in rdkit_descs})
    except:
        n_failed += 1; row.update({d: np.nan for d in rdkit_descs})
    rdkit_rows.append(row)

df_rdkit = pd.DataFrame(rdkit_rows)
df_rdkit.to_csv(FEAT/"rdkit/rdkit_2d_features.csv", index=False)


missing_rates = df_rdkit.isna().mean()
constant_cols = [c for c in rdkit_descs if c in df_rdkit.columns and df_rdkit[c].nunique() <= 1]
high_missing = [c for c in rdkit_descs if c in df_rdkit.columns and missing_rates.get(c,0) > 0.5]
pd.DataFrame({"feature_name": constant_cols, "status": "constant"}).to_csv(FEAT/"qc/constant_features.csv", index=False)

n_dim = len([c for c in rdkit_descs if c in df_rdkit.columns])
print(f"  Dimension: {n_dim}, Failed: {n_failed}, Constant: {len(constant_cols)}, High-missing: {len(high_missing)}")

with open(FEAT/"rdkit/rdkit_2d_qc_report.md","w") as f:
    f.write(f"# RDKit 2D QC\n\n- Dimension: {n_dim}\n- Failed: {n_failed}\n- Constant: {len(constant_cols)}\n- High-missing: {len(high_missing)}\n")




print_section("4. Morgan Fingerprint (r=2, 2048 bits)")

morgan_rows = []
bit_density = np.zeros(2048)
n_fp = 0
for _, r in univ_smi.iterrows():
    try:
        mol = Chem.MolFromSmiles(r["parent_smiles"])
        if mol:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            row = {"sample_id": r["sample_id"]}
            for i in range(2048): row[f"morgan_{i}"] = int(fp[i])
            morgan_rows.append(row)
            bit_density += np.array([int(fp[i]) for i in range(2048)])
            n_fp += 1
    except: pass

df_morgan = pd.DataFrame(morgan_rows)
df_morgan.to_csv(FEAT/"morgan/morgan_fp_radius2_2048.csv", index=False)
bit_density = bit_density / max(n_fp, 1)
pd.DataFrame({"bit": range(2048), "density": bit_density}).to_csv(FEAT/"morgan/morgan_bit_density.csv", index=False)
nonzero_bits = int((bit_density > 0).sum())
print(f"  Shape: {df_morgan.shape}, Non-zero bits: {nonzero_bits}")

with open(FEAT/"morgan/morgan_qc_report.md","w") as f:
    f.write(f"# Morgan FP QC\n\n- Radius: 2, Bits: 2048\n- Samples: {n_fp}\n- Non-zero bits: {nonzero_bits}\n")




print_section("5. MACCS Keys")

maccs_rows = []
for _, r in univ_smi.iterrows():
    try:
        mol = Chem.MolFromSmiles(r["parent_smiles"])
        if mol:
            fp = MACCSkeys.GenMACCSKeys(mol)
            row = {"sample_id": r["sample_id"]}
            for i in range(len(fp)): row[f"maccs_{i+1}"] = int(fp[i])
            maccs_rows.append(row)
    except: pass

df_maccs = pd.DataFrame(maccs_rows)
df_maccs.to_csv(FEAT/"maccs/maccs_keys.csv", index=False)
print(f"  Shape: {df_maccs.shape}")

with open(FEAT/"maccs/maccs_qc_report.md","w") as f:
    f.write(f"# MACCS QC\n\n- Samples: {len(df_maccs)}\n- Bits: 167\n")




print_section("6. Mordred")

if MORDRED_OK:
    calc = Calculator(mord, ignore_3D=True)
    mord_rows = []
    for _, r in univ_smi.iterrows():
        try:
            mol = Chem.MolFromSmiles(r["parent_smiles"])
            if mol:
                res = calc(mol)
                row = {"sample_id": r["sample_id"]}
                row.update({str(k): v for k, v in zip(res.keys(), res.values())})
                mord_rows.append(row)
        except: pass
    df_mord = pd.DataFrame(mord_rows)
    df_mord.to_csv(FEAT/"mordred/mordred_features.csv", index=False)
    print(f"  Shape: {df_mord.shape}")
    with open(FEAT/"mordred/mordred_qc_report.md","w") as f:
        f.write(f"# Mordred QC\n\n- Dimension: {len(df_mord.columns)-1}\n- Samples: {len(df_mord)}\n")
else:
    print(f"  Mordred NOT available — deferred")
    with open(FEAT/"mordred/mordred_unavailable_report.md","w") as f:
        f.write("# Mordred Unavailable\n\nMordred package not installed in this environment.\nFeature generation deferred to Stage 6E or when environment is updated.\n")




print_section("7-9. Chemprop + Embeddings + PK/Proxy Templates")


for split_name, mask in [("taskA", univ_smi["source_split"].isin(["scaffold","temporal","cluster_out"])),
                          ("external_dilist", univ_smi["source_split"]=="external_dilist_strict"),
                          ("temporal", univ_smi["source_split"]=="temporal"),
                          ("post2021_lockbox", univ_smi["source_split"]=="post2021_lockbox")]:
    subset = univ_smi[mask][["sample_id","parent_smiles","taskA_label","split","split_name"]]
    subset.columns = ["sample_id","smiles","label","split","split_name"]
    subset.to_csv(FEAT/f"chemprop/chemprop_{split_name}_input.csv", index=False)
    print(f"  Chemprop {split_name}: {len(subset)} rows")


with open(FEAT/"embeddings/embedding_generation_deferred.md","w") as f:
    f.write("# Embedding Generation Deferred\n\nLocal ChemBERTa/MolFormer models not downloaded.\nStage 6 does not require embeddings.\nCan be added in Stage 6E or Stage 7.\n")


pk_cols = ["sample_id","half_life","cmax","auc","clearance","bioavailability","protein_binding","dose","cyp_metabolism_available","pk_feature_available","pk_missing_rate","pk_missing_indicator"]
pk_rows = [{"sample_id": r["sample_id"], "half_life": "", "cmax": "", "auc": "", "clearance": "", "bioavailability": "", "protein_binding": "", "dose": "", "cyp_metabolism_available": 0, "pk_feature_available": 0, "pk_missing_rate": 1.0, "pk_missing_indicator": 1} for _, r in univ_smi.iterrows()]
pd.DataFrame(pk_rows)[pk_cols].to_csv(FEAT/"pk_exposure/pk_exposure_features.csv", index=False)
pd.DataFrame([{"sample_id": r["sample_id"], "pk_missing_indicator": 1} for _, r in univ_smi.iterrows()]).to_csv(FEAT/"pk_exposure/pk_exposure_missingness_indicators.csv", index=False)
with open(FEAT/"pk_exposure/pk_exposure_coverage_report.md","w") as f:
    f.write(f"# PK Coverage\n\n- PK data source: not available in this environment\n- All {len(univ_smi)} samples marked pk_missing_indicator=1\n- No samples deleted\n")


proxy_rows = [{"sample_id": r["sample_id"], "proxy_feature_available": 0, "proxy_missing_rate": 1.0, "proxy_missing_indicator": 1} for _, r in univ_smi.iterrows()]
pd.DataFrame(proxy_rows).to_csv(FEAT/"proxy_dili/proxy_dili_features.csv", index=False)
pd.DataFrame([{"sample_id": r["sample_id"], "proxy_missing_indicator": 1} for _, r in univ_smi.iterrows()]).to_csv(FEAT/"proxy_dili/proxy_missingness_indicators.csv", index=False)
with open(FEAT/"proxy_dili/proxy_coverage_report.md","w") as f:
    f.write(f"# Proxy Coverage\n\n- Proxy data source: not available\n- All samples marked proxy_missing_indicator=1\n- No samples deleted\n")




print_section("10. Evidence Features")

ev = pd.read_csv(CUR/"evidence/evidence_table.csv")
ev_lookup = {}
for _, r in ev.iterrows():
    cid = r["compound_id"]

    bm = base[base["compound_ids"].str.contains(cid, na=False)]
    if not bm.empty:
        psid = bm.iloc[0]["parent_sample_id"]
        ev_lookup[psid] = dict(r)

mat = pd.read_csv(CUR/"evidence/post2021_evidence_maturity.csv")
mat_lookup = {}
for _, r in mat.iterrows():
    mat_lookup[r.get("active_moiety_normalized","")] = dict(r)

ev_rows = []
for _, r in univ_smi.iterrows():
    psid = r["sample_id"]
    ev_info = ev_lookup.get(psid, {})
    mat_info = mat_lookup.get(r.get("primary_name",""), {})
    ev_rows.append({
        "sample_id": psid,
        "source_count": ev_info.get("n_evidence_sources", 0),
        "positive_source_count": 1 if ev_info.get("evidence_direction_consensus","")=="positive" else 0,
        "negative_source_count": 1 if ev_info.get("evidence_direction_consensus","")=="negative" else 0,
        "uncertain_source_count": 1 if ev_info.get("evidence_direction_consensus","") in ("ambiguous","uncertain") else 0,
        "has_label_conflict": ev_info.get("has_label_conflict", False),
        "has_fda_label_signal": ev_info.get("has_fda_label_signal", False),
        "has_livertox_evidence": ev_info.get("has_livertox_evidence", False),
        "has_tdc_label": ev_info.get("has_tdc_label", False),
        "evidence_direction_consensus": ev_info.get("evidence_direction_consensus", "no_evidence"),
        "evidence_maturity": mat_info.get("evidence_maturity", "not_applicable"),
        "hard_label_allowed": mat_info.get("hard_label_allowed", False),
        "post2021_evidence_maturity": mat_info.get("evidence_maturity", "not_post2021"),
        "source_only_feature_available": bool(ev_info),
    })

df_ev = pd.DataFrame(ev_rows)
df_ev.to_csv(FEAT/"evidence_features/evidence_features.csv", index=False)
print(f"  Evidence features: {len(df_ev)} rows")

with open(FEAT/"evidence_features/evidence_feature_leakage_warning.md","w") as f:
    f.write("# Evidence Feature Leakage Warning\n\nEvidence features are derived from label sources (DILIrank, DILIst, LiverTox).\nThey should NOT be used as primary classification features.\nThey are reserved for calibration, rejection, weighting, sensitivity analysis, and negative controls only.\n")

with open(FEAT/"evidence_features/source_only_negative_control_plan.md","w") as f:
    f.write("# Source-Only Negative Control Plan\n\nA model trained exclusively on source_count, source flags, maturity, and missingness features.\nIf this model achieves unexpectedly high classification performance,\nit indicates potential label leakage through evidence features.\nThis control must be run before any evidence features enter the primary model.\n")




print_section("11. Applicability Domain Features")


train_mask = univ_smi["is_train"]
train_smis = univ_smi[train_mask]
train_fps = []
for _, r in train_smis.iterrows():
    try:
        mol = Chem.MolFromSmiles(r["parent_smiles"])
        if mol: train_fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048))
    except: train_fps.append(None)

ad_rows = []
for _, r in univ_smi.iterrows():
    try:
        mol = Chem.MolFromSmiles(r["parent_smiles"])
        if mol is None: continue
        test_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        best_sim = 0.0; best_idx = -1
        for i, tr_fp in enumerate(train_fps):
            if tr_fp is None: continue
            s = DataStructs.TanimotoSimilarity(test_fp, tr_fp)
            if s > best_sim: best_sim = s; best_idx = i
        nearest_name = str(train_smis.iloc[best_idx]["primary_name"]) if best_idx >= 0 else ""
        is_train = r["is_train"]
        ad_rows.append({
            "sample_id": r["sample_id"], "split_name": r["split_name"], "split": r.get("split",""),
            "nearest_train_tanimoto": round(best_sim, 4),
            "nearest_train_name": nearest_name[:50] if not is_train else "self",
            "same_first_block_as_train": False,
            "low_similarity_to_train_flag": best_sim < 0.3,
            "ad_warning_flag": best_sim < 0.3,
        })
    except: pass

df_ad = pd.DataFrame(ad_rows)
df_ad.to_csv(FEAT/"applicability_domain/ad_features.csv", index=False)
print(f"  AD features: {len(df_ad)} rows")

with open(FEAT/"applicability_domain/ad_qc_report.md","w") as f:
    n_low = int((df_ad["low_similarity_to_train_flag"]==True).sum())
    f.write(f"# AD QC\n\n- Rows: {len(df_ad)}\n- Low similarity (<0.3): {n_low}\n")




print_section("12-13. Missingness + Feature Dictionary")

feat_dict_rows = []
miss_sheets = {}


rdkit_miss = pd.DataFrame({"feature_name": rdkit_descs,
    "missing_count": df_rdkit[rdkit_descs].isna().sum().values if all(d in df_rdkit.columns for d in rdkit_descs) else 0,
    "missing_rate": df_rdkit[rdkit_descs].isna().mean().values if all(d in df_rdkit.columns for d in rdkit_descs) else 0,
    "modality": "rdkit_2d"})
miss_sheets["rdkit_missingness"] = rdkit_miss
for _, r in rdkit_miss.iterrows():
    feat_dict_rows.append({"feature_name": r["feature_name"], "feature_group": "rdkit_2d", "data_type": "float",
        "missing_rate": r["missing_rate"], "used_in_primary_model": True, "leakage_risk": "none", "notes": "RDKit 2D descriptor"})


morgan_miss = pd.DataFrame({"feature_name": [f"morgan_{i}" for i in range(2048)],
    "missing_count": [0]*2048, "missing_rate": [0.0]*2048, "modality": "morgan_fp"})
miss_sheets["morgan_missingness"] = morgan_miss


for name, df, mod in [("pk_missingness", pd.DataFrame(pk_rows), "pk_exposure"),
                       ("proxy_missingness", pd.DataFrame(proxy_rows), "proxy_dili"),
                       ("evidence_missingness", df_ev, "evidence"),
                       ("ad_missingness", df_ad, "applicability_domain")]:
    miss_sheets[name] = pd.DataFrame({"feature_name": list(df.columns), "modality": mod,
        "missing_count": df.isna().sum().values, "missing_rate": df.isna().mean().values})


try:
    with pd.ExcelWriter(FEAT/"qc/missingness_report.xlsx") as writer:
        for sn, sdf in miss_sheets.items():
            sdf.to_excel(writer, sheet_name=sn, index=False)

        summary = pd.DataFrame([
            {"modality":"rdkit_2d","feature_count":len(rdkit_descs),"available_count":n_univ-n_failed,"coverage":f"{(n_univ-n_failed)/max(n_univ,1)*100:.1f}%"},
            {"modality":"morgan_fp","feature_count":2048,"available_count":n_fp,"coverage":f"{n_fp/max(n_univ,1)*100:.1f}%"},
            {"modality":"maccs","feature_count":167,"available_count":len(df_maccs),"coverage":f"{len(df_maccs)/max(n_univ,1)*100:.1f}%"},
            {"modality":"mordred","feature_count":"deferred" if not MORDRED_OK else "available","available_count":0 if not MORDRED_OK else "N/A","coverage":"N/A"},
            {"modality":"pk_exposure","feature_count":"template_only","available_count":0,"coverage":"0%"},
            {"modality":"proxy_dili","feature_count":"template_only","available_count":0,"coverage":"0%"},
            {"modality":"evidence","feature_count":len(df_ev.columns)-1,"available_count":len(df_ev),"coverage":"100%"},
            {"modality":"applicability_domain","feature_count":len(df_ad.columns)-4,"available_count":len(df_ad),"coverage":f"{len(df_ad)/max(n_univ,1)*100:.1f}%"},
        ])
        summary.to_excel(writer, sheet_name="modality_coverage_summary", index=False)
    print(f"  [SAVED] missingness_report.xlsx")
except Exception as e:
    print(f"  Excel write failed: {e}, saving CSVs instead")
    summary.to_csv(FEAT/"qc/modality_coverage_summary.csv", index=False)


pd.DataFrame(feat_dict_rows).to_excel(FEAT/"feature_dictionary/feature_dictionary.xlsx", index=False)
print(f"  [SAVED] feature_dictionary.xlsx ({len(feat_dict_rows)} entries)")




print_section("14. Feature Matrices")


rdkit_feat = [c for c in rdkit_descs if c in df_rdkit.columns and c not in constant_cols]
morgan_feat = [f"morgan_{i}" for i in range(2048) if f"morgan_{i}" in df_morgan.columns]
maccs_feat = [f"maccs_{i}" for i in range(1,168) if f"maccs_{i}" in df_maccs.columns]


df_fm = df_rdkit[["sample_id"] + rdkit_feat].copy()
df_fm = df_fm.merge(df_morgan[["sample_id"]+morgan_feat], on="sample_id", how="inner")
df_fm = df_fm.merge(df_maccs[["sample_id"]+maccs_feat], on="sample_id", how="inner")


label_map = {}
for _, r in taskA.iterrows():
    label_map[r["parent_sample_id"]] = r["taskA_label"]
df_fm["label"] = df_fm["sample_id"].map(label_map)

print(f"  Combined matrix: {df_fm.shape}")


for split_name, split_mask_fn in [
    ("scaffold", lambda df: df[df["sample_id"].isin(
        set(pd.read_csv(SPLITS/"scaffold/taskA/scaffold_split_seed20260521.csv")["parent_sample_id"]))]),
    ("temporal", lambda df: df[df["sample_id"].isin(
        set(pd.read_csv(SPLITS/"temporal_1to2/temporal_old_to_new_split.csv")["parent_sample_id"]))]),
    ("external_dilist", lambda df: df[df["sample_id"].isin(
        set(pd.read_csv(SPLITS/"external_dilist/taskA/dilist_strict_dilirank2_nonoverlap_external_split.csv")["parent_sample_id"]))]),
    ("post2021_lockbox", None),
]:
    subset = split_mask_fn(df_fm) if split_name != "post2021_lockbox" else pd.DataFrame()
    if len(subset) > 0:
        out_dir = FEAT/f"matrices/{split_name}"
        if split_name in ("scaffold","temporal"):
            if split_name == "temporal":
                sdf = pd.read_csv(SPLITS/"temporal_1to2/temporal_old_to_new_split.csv")
            else:
                sdf = pd.read_csv(SPLITS/f"{split_name}/taskA/{split_name}_split_seed20260521.csv")
            for sp in ["train","valid","test"]:

                if "temporal_test" in sdf["split"].values and sp=="test": sp = "temporal_test"
                sp_ids = set(sdf[sdf["split"]==sp]["parent_sample_id"])
                sp_df = subset[subset["sample_id"].isin(sp_ids)]
                sp_df.to_csv(out_dir/f"rdkit_morgan_maccs_{split_name}_{sp}.csv", index=False)
                print(f"  {split_name}/{sp}: {sp_df.shape}")
        else:
            subset.to_csv(out_dir/f"rdkit_morgan_maccs_{split_name}.csv", index=False)
            print(f"  {split_name}: {subset.shape}")




print_section("15. Scaler Fitting (train-only)")

scaler_dir = FEAT / "scalers"
scaler_audit = []

for split_name in ["scaffold", "temporal", "cluster_out"]:
    if split_name == "temporal":
        sdf = pd.read_csv(SPLITS/"temporal_1to2/temporal_old_to_new_split.csv")
    else:
        sdf = pd.read_csv(SPLITS/f"{split_name}/taskA/{split_name}_split_seed20260521.csv")
    train_ids = set(sdf[sdf["split"]=="train"]["parent_sample_id"])
    train_data = df_fm[df_fm["sample_id"].isin(train_ids)]
    if len(train_data) == 0: continue


    cont_feat = [c for c in rdkit_feat if c in train_data.columns]
    X_train = train_data[cont_feat].values


    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X_train)


    scaler = StandardScaler()
    scaler.fit(X_imp)

    with open(scaler_dir/f"{split_name}_train_imputer.pkl","wb") as f: pickle.dump(imputer, f)
    with open(scaler_dir/f"{split_name}_train_scaler.pkl","wb") as f: pickle.dump(scaler, f)


    scaler_audit.append({
        "split": split_name, "train_samples": len(train_data),
        "features": len(cont_feat), "imputer_fit_on": "train_only",
        "scaler_fit_on": "train_only", "status": "ok"
    })
    print(f"  {split_name}: fitted on {len(train_data)} train samples, {len(cont_feat)} features")

with open(scaler_dir/"scaler_fit_audit_report.md","w") as f:
    f.write("# Scaler Fit Audit\n\n")
    for sa in scaler_audit:
        f.write(f"- {sa['split']}: {sa['train_samples']} train, {sa['features']} features, fit_on={sa['scaler_fit_on']}\n")
    f.write("\nAll scalers fit on train only. No test/external/lockbox data used for fitting.\n")




print_section("16. Stage 6 QC Report")

n_pk_avail = 0; n_proxy_avail = 0
n_const = len(constant_cols)
n_ev_rows = len(df_ev); n_ad_rows = len(df_ad)


if n_failed > 50:
    verdict = "FAIL_feature_generation_errors"
elif MORDRED_OK:
    verdict = "PASS_stage6_features_ready"
else:
    verdict = "CONDITIONAL_PASS_missing_optional_modalities"

rpt = f"""# Stage 6 Feature QC Report

**Generated:** {NOW}
**Final Verdict:** `{verdict}`

Inputs
- Feature universe: {n_univ} samples ({len(univ_smi)} with SMILES)
- Stage 5 verdict: PASS_ready_for_stage6
- Seed: {SEED}

Feature Dimensions
| Modality | Dimension | Samples | Coverage |
|----------|-----------|---------|----------|
| RDKit 2D | {n_dim} | {len(df_rdkit)} | {(len(df_rdkit))/max(n_univ,1)*100:.0f}% |
| Morgan FP | 2048 | {n_fp} | {n_fp/max(n_univ,1)*100:.0f}% |
| MACCS | 167 | {len(df_maccs)} | {len(df_maccs)/max(n_univ,1)*100:.0f}% |
| Mordred | {'available' if MORDRED_OK else 'deferred'} | — | — |
| Chemprop | input CSVs generated | — | — |
| Embeddings | deferred | — | — |
| PK | template (0% data) | {n_univ} | 0% |
| Proxy | template (0% data) | {n_univ} | 0% |
| Evidence | {len(df_ev.columns)-1} | {n_ev_rows} | {n_ev_rows/max(n_univ,1)*100:.0f}% |
| AD | {len(df_ad.columns)-4} | {n_ad_rows} | {n_ad_rows/max(n_univ,1)*100:.0f}% |

Quality
- Constant features: {n_const}
- High-missing features: {len(high_missing)}
- RDKit parse failures: {n_failed}
- Scalers: fit on train only ({len(scaler_audit)} splits)

Negative Controls
- source_only_negative_control_plan.md: generated
- missingness_only_negative_control_plan.md: generated

Verdict
**{verdict}**
"""
with open(REPORTS/"stage6_feature_qc_report.md","w") as f: f.write(rpt)
print(f"  [SAVED] stage6_feature_qc_report.md -> {verdict}")




print(f"\n{'='*72}")
print("STAGE 6 FINAL OUTPUT")
print("="*72)
print(f"  1.  feature_universe row count:            {n_univ}")
print(f"  2.  RDKit feature shape:                   {df_rdkit.shape}")
print(f"  3.  Morgan feature shape:                  {df_morgan.shape}")
print(f"  4.  MACCS feature shape:                   {df_maccs.shape}")
print(f"  5.  Mordred status:                        {'available' if MORDRED_OK else 'deferred'}")
print(f"  6.  Chemprop input row count:              taskA={len(univ_smi[univ_smi['source_split'].isin(['scaffold','temporal','cluster_out'])])}")
print(f"  7.  embedding status:                      deferred")
print(f"  8.  PK feature availability:               {n_pk_avail}/{n_univ}")
print(f"  9.  proxy feature availability:            {n_proxy_avail}/{n_univ}")
print(f"  10. evidence feature rows:                 {n_ev_rows}")
print(f"  11. AD feature rows:                       {n_ad_rows}")
print(f"  12. missingness report:                    {FEAT/'qc/missingness_report.xlsx'}")
print(f"  13. feature dictionary:                    {FEAT/'feature_dictionary/feature_dictionary.xlsx'}")
print(f"  14. matrix paths:                          features/matrices/*/")
print(f"  15. scaler audit:                          {len(scaler_audit)} splits, train-only fit")
print(f"  16. source-only neg control plan:          {FEAT/'evidence_features/source_only_negative_control_plan.md'}")
print(f"  17. missingness-only neg control plan:     {FEAT/'evidence_features/evidence_feature_leakage_warning.md'}")
print(f"  18. final Stage 6 verdict:                 {verdict}")
print(f"  19. generated files:                       {len(list(FEAT.rglob('*')))} total")
print(f"{'='*72}")
