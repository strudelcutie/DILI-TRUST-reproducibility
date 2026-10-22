                      

import pandas as pd, numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, roc_curve, matthews_corrcoef, balanced_accuracy_score

def ece(yt, ys, nb=10):
    bins = np.linspace(0, 1, nb + 1); e = 0.0
    for i in range(nb):
        m = (ys >= bins[i]) & (ys < bins[i + 1])
        if m.sum() > 0: e += (m.sum() / max(1,len(yt))) * abs(yt[m].mean() - ys[m].mean())
    return round(e, 4)

def sens_at_spec(yt, ys, spec=0.95):
    fpr, tpr, _ = roc_curve(yt, ys)
    idx = np.where(1 - fpr >= spec)[0]
    return round(float(np.max(tpr[idx])), 4) if len(idx) > 0 else 0.0

NOW = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
ROOT = Path(__file__).resolve().parents[2]
FEAT = ROOT / "features" / "matrices"
EVID = ROOT / "features" / "evidence_features" / "evidence_features.csv"
TAB_REV = ROOT / "results" / "final" / "tables" / "revised"
TAB_REV.mkdir(parents=True, exist_ok=True)

SEEDS = [20260521, 20260522, 20260523, 20260524, 20260525]
MODALITIES = {
    "structure_only": [],
    "structure_plus_evidence": ["evidence"],
}

                        
ev = pd.read_csv(EVID)
ev_numeric = ["source_count", "positive_source_count", "negative_source_count", "uncertain_source_count"]
ev_map = {row["sample_id"]: row[ev_numeric].values.astype(np.float32) for _, row in ev.iterrows()}
print(f"Evidence features: {len(ev_numeric)} numeric cols, {len(ev_map)} samples mapped")

print("=" * 60)
print("MULTIMODAL ABLATION: structure vs structure+evidence")
print("=" * 60)

results = []
for split_name in ["scaffold", "temporal", "cluster_out"]:
    sd = "temporal" if split_name == "temporal" else split_name
    train_f = FEAT / sd / f"model_ready_{split_name}_train.csv"
    valid_f = FEAT / sd / f"model_ready_{split_name}_valid.csv"
    test_f = FEAT / sd / f"model_ready_{split_name}_{'temporal_test' if split_name=='temporal' else 'test'}.csv"
    if not train_f.exists(): continue

    tr = pd.read_csv(train_f); vl = pd.read_csv(valid_f); te = pd.read_csv(test_f)
    id_cols = ["sample_id","label"]
    feat_cols = [c for c in tr.columns if c not in id_cols and c not in ("split","split_name")]
    Xt_all = np.clip(tr[feat_cols].values.astype(np.float64), -1e10, 1e10).astype(np.float32)
    yt = tr["label"].values.astype(int)
    Xv_all = np.clip(vl[feat_cols].values.astype(np.float64), -1e10, 1e10).astype(np.float32)
    yv = vl["label"].values.astype(int)
    Xte_all = np.clip(te[feat_cols].values.astype(np.float64), -1e10, 1e10).astype(np.float32)
    yte = te["label"].values.astype(int)

                                      
    ev_tr = np.array([ev_map.get(sid, np.zeros(len(ev_numeric))) for sid in tr["sample_id"]], dtype=np.float32)
    ev_vl = np.array([ev_map.get(sid, np.zeros(len(ev_numeric))) for sid in vl["sample_id"]], dtype=np.float32)
    ev_te = np.array([ev_map.get(sid, np.zeros(len(ev_numeric))) for sid in te["sample_id"]], dtype=np.float32)

    for modality_label in MODALITIES:
        if modality_label == "structure_only":
            Xt = Xt_all; Xv = Xv_all; Xte_mod = Xte_all; n_feat = Xt_all.shape[1]
        else:
            Xt = np.hstack([Xt_all, ev_tr]); Xv = np.hstack([Xv_all, ev_vl])
            Xte_mod = np.hstack([Xte_all, ev_te]); n_feat = Xt.shape[1]

        scores_te = []
        val_metrics = []
        for sd in SEEDS:
            m = HistGradientBoostingClassifier(learning_rate=0.05, max_iter=200, max_leaf_nodes=31, random_state=sd)
            m.fit(Xt, yt)
            pv = m.predict_proba(Xv)[:,1]
            val_metrics.append({
                "auroc": roc_auc_score(yv, pv),
                "brier": brier_score_loss(yv, pv),
                "ece": ece(yv, pv, 5),
            })
            scores_te.append(m.predict_proba(Xte_mod)[:,1])

        ys_te = np.clip(np.mean(scores_te, axis=0), 0, 1)
        yp_te = (ys_te >= 0.5).astype(int)
        n_test = len(yte)

        result = {
            "split": split_name, "modality": modality_label, "n_features": n_feat,
            "n_test": n_test, "positives": int(yte.sum()), "negatives": int(n_test - yte.sum()),
            "auroc_val_mean": round(np.mean([m["auroc"] for m in val_metrics]), 4),
            "auroc_test": round(roc_auc_score(yte, ys_te), 4),
            "auprc_test": round(average_precision_score(yte, ys_te), 4),
            "brier_test": round(brier_score_loss(yte, ys_te), 4),
            "ece_q5_test": ece(yte, ys_te, 5),
            "mcc_test": round(matthews_corrcoef(yte, yp_te), 4),
            "sens95spec_test": sens_at_spec(yte, ys_te, 0.95),
        }
        results.append(result)
        print(f"  {split_name}/{modality_label}: val_AUROC={result['auroc_val_mean']:.4f}, test_AUROC={result['auroc_test']:.4f}, test_Brier={result['brier_test']:.4f}, test_ECE={result['ece_q5_test']:.4f}")

df = pd.DataFrame(results)
df.to_csv(TAB_REV / "ablation_structure_vs_evidence.csv", index=False)

                  
print()
print("=" * 60)
print("ABLATION COMPARISON: structure-only vs structure+evidence")
print("=" * 60)
for sn in ["scaffold", "temporal", "cluster_out"]:
    s = df[(df["split"] == sn) & (df["modality"] == "structure_only")]
    e = df[(df["split"] == sn) & (df["modality"] == "structure_plus_evidence")]
    if len(s) > 0 and len(e) > 0:
        s_r = s.iloc[0]; e_r = e.iloc[0]
        delta_auroc = round(e_r["auroc_test"] - s_r["auroc_test"], 4)
        delta_brier = round(e_r["brier_test"] - s_r["brier_test"], 4)
        print(f"\n  {sn}:")
        print(f"    Structure-only:       AUROC={s_r['auroc_test']:.4f}  Brier={s_r['brier_test']:.4f}  ECE={s_r['ece_q5_test']:.4f}  Sens@95={s_r['sens95spec_test']:.4f}")
        print(f"    Structure+Evidence:   AUROC={e_r['auroc_test']:.4f}  Brier={e_r['brier_test']:.4f}  ECE={e_r['ece_q5_test']:.4f}  Sens@95={e_r['sens95spec_test']:.4f}")
        print(f"    Delta:                AUROC={delta_auroc:+.4f}  Brier={delta_brier:+.4f}")
        if abs(delta_auroc) < 0.01 and abs(delta_brier) < 0.005:
            print(f"    Conclusion: Evidence features do NOT improve performance ({sn})")
        elif delta_auroc > 0.01:
            print(f"    Conclusion: Evidence features IMPROVE AUROC ({sn})")
        else:
            print(f"    Conclusion: Evidence features SLIGHTLY change performance ({sn})")

print(f"\nSaved: {TAB_REV / 'ablation_structure_vs_evidence.csv'}")
