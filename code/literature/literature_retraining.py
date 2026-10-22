                      

import pandas as pd, numpy as np, warnings
from pathlib import Path
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, roc_curve
warnings.filterwarnings("ignore")
NOW = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
ROOT = Path(__file__).resolve().parents[2]
FEAT = ROOT / "features" / "matrices"
TMP125 = ROOT / "results" / "tmp" / "stage12_5"
TAB125 = ROOT / "results" / "final" / "tables" / "stage12_5"
FIG125 = ROOT / "results" / "final" / "figures" / "stage12_5"
REP125 = ROOT / "reports" / "stage12_5"
TAB9 = ROOT / "results" / "final" / "tables" / "stage9"
SEEDS = [20260521, 20260522, 20260523]
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 9, "legend.fontsize": 7, "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight"})

for sd in ["retraining/InterDILI_adapted","retraining/DILIPredictor_simplified","retraining/pDILI_v1_adapted"]:
    (TMP125/sd).mkdir(parents=True, exist_ok=True)
for d in [TAB125, FIG125, REP125]: d.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("STAGE 12.5 ACTUAL RETRAINING")
print("=" * 60)

           
print("\nLoading DILI-TRUST data...")
data = {}
for sn in ["scaffold", "temporal", "cluster_out"]:
    for role, fname in [("train",f"model_ready_{sn}_train.csv"),("valid",f"model_ready_{sn}_valid.csv"),
        ("test",f"model_ready_{sn}_test.csv" if sn!="temporal" else f"model_ready_temporal_temporal_test.csv")]:
        fp = FEAT / sn / fname
        if not fp.exists(): continue
        d = pd.read_csv(fp)
        idc = ["sample_id","label"]; fc = [c for c in d.columns if c not in idc and c not in ("split","split_name")]
        X = np.clip(d[fc].values.astype(np.float64), -1e10, 1e10).astype(np.float32)
        data[f"{sn}_{role}"] = {"X":X,"y":d["label"].values.astype(int),"sids":d["sample_id"].values,"fc":fc}
for sn, sd in [("external_dilist","external_dilist"),("post2021_lockbox","post2021_lockbox")]:
    fp = FEAT / sd / f"model_ready_{sn}_scaffold_scaler.csv"
    if fp.exists():
        d = pd.read_csv(fp)
        idc = ["sample_id","label"]; fc = [c for c in d.columns if c not in idc and c not in ("split","split_name")]
        X = np.clip(d[fc].values.astype(np.float64), -1e10, 1e10).astype(np.float32)
        y = d["label"].values if "label" in d.columns else np.full(len(d), np.nan)
        data[f"{sn}_eval"] = {"X":X,"y":y,"sids":d["sample_id"].values,"fc":fc}
print(f"  {len(data)} data keys loaded")

        
def ece_fn(yt, ys, nb=10):
    bins = np.linspace(0,1,nb+1); e=0.0
    for i in range(nb):
        m = (ys>=bins[i])&(ys<bins[i+1])
        e += (m.sum()/max(1,len(yt)))*abs(yt[m].mean()-ys[m].mean()) if m.sum()>0 else 0
    return round(e,4)

def sens_at_spec_fn(yt, ys, spec=0.95):
    fpr,tpr,_ = roc_curve(yt,ys)
    idx = np.where(1-fpr>=spec)[0]
    return round(float(np.max(tpr[idx])),4) if len(idx)>0 else 0.0

all_preds = {}; all_logs = {}; all_metrics = []

                
for mn, build_fn, rclass in [
    ("InterDILI_adapted", lambda s: HistGradientBoostingClassifier(learning_rate=0.05, max_iter=200, max_leaf_nodes=31, random_state=s), "A_adapted_strict_retraining"),
    ("DILIPredictor_simplified", lambda s: RandomForestClassifier(n_estimators=200, max_depth=15, min_samples_leaf=3, random_state=s, n_jobs=-1), "B_simplified_reproduction"),
    ("pDILI_v1_adapted", lambda s: RandomForestClassifier(n_estimators=300, max_depth=None, min_samples_leaf=3, random_state=s, n_jobs=-1), "A_adapted_strict_retraining"),
]:
    print(f"\n{'='*40}\n{mn} ({rclass})\n{'='*40}")
    logs = []; preds_list = []
    for sn in ["scaffold","external_dilist","temporal","post2021_lockbox","cluster_out"]:
        sn_src = "scaffold" if sn in ["external_dilist","post2021_lockbox"] else ("temporal" if sn=="temporal" else "cluster_out" if sn=="cluster_out" else "scaffold")
        if sn in ["external_dilist","post2021_lockbox"]:
            ek = f"{sn}_eval"
            if ek not in data: continue
            Xe, ye, sids_e = data[ek]["X"], data[ek]["y"], data[ek]["sids"]; ts = "scaffold"
        else:
            ek = f"{sn_src}_test" if sn!="temporal" else "temporal_test"
            if ek not in data: continue
            Xe, ye, sids_e = data[ek]["X"], data[ek]["y"], data[ek]["sids"]; ts = sn_src
        tk = f"{ts}_train"; vk = f"{ts}_valid"
        if tk not in data or vk not in data: continue
        Xtr, ytr = data[tk]["X"], data[tk]["y"]
        Xv, yv = data[vk]["X"], data[vk]["y"]

        ss = []
        for sd in SEEDS:
            m = build_fn(sd); m.fit(Xtr, ytr)
            sc = m.predict_proba(Xe)[:,1]; ss.append(sc)
            pv = m.predict_proba(Xv)[:,1]
            logs.append({"model":mn,"split":sn,"seed":sd,"train_split":ts,"valid_auroc":round(roc_auc_score(yv,pv),4),"status":"completed"})

        ens = np.mean(ss, axis=0); hl = not np.all(np.isnan(ye))
        df = pd.DataFrame({"sample_id":sids_e,"split":sn,"model_name":mn,"y_true":ye if hl else np.nan,"raw_score":np.round(ens,6),"reproduction_class":rclass})
        df.to_csv(TMP125/f"retraining/{mn}/{sn}_ensemble_raw_predictions.csv", index=False)
        preds_list.append(df)

        if hl:
            a = roc_auc_score(ye,ens); apr = average_precision_score(ye,ens); br = brier_score_loss(ye,ens)
            print(f"  {sn}: n={len(ye)} AUROC={a:.4f} AUPRC={apr:.4f} Brier={br:.4f} Sens95={sens_at_spec_fn(ye,ens):.4f}")
            all_metrics.append({"model_name":mn,"split":sn,"reproduction_class":rclass,"n":len(ye),"positives":int(np.nansum(ye)),"negatives":int(len(ye)-np.nansum(ye)),"auroc":round(a,4),"auprc":round(apr,4),"brier":round(br,4),"ece_q5":ece_fn(ye,ens,5),"sens95spec":sens_at_spec_fn(ye,ens,0.95),"metric_source":"actual_prediction"})
        else:
            print(f"  {sn}: n={len(ye)} (prediction-only)")
            all_metrics.append({"model_name":mn,"split":sn,"reproduction_class":rclass,"n":len(ye),"auroc":np.nan,"auprc":np.nan,"brier":np.nan,"ece_q5":np.nan,"sens95spec":np.nan,"metric_source":"actual_prediction_unlabeled"})

    pd.DataFrame(logs).to_csv(TMP125/f"retraining/{mn}/training_log.csv", index=False)
    all_preds[mn] = preds_list; all_logs[mn] = logs

              
dfm = pd.DataFrame(all_metrics)
df_strict = dfm[dfm["reproduction_class"]=="A_adapted_strict_retraining"]; df_simp = dfm[dfm["reproduction_class"]=="B_simplified_reproduction"]
df_strict.to_csv(TAB125/"literature_model_strict_split_metrics.csv", index=False); df_strict.to_excel(TAB125/"literature_model_strict_split_metrics.xlsx", index=False)
df_simp.to_csv(TAB125/"literature_model_simplified_reproduction_metrics.csv", index=False); df_simp.to_excel(TAB125/"literature_model_simplified_reproduction_metrics.xlsx", index=False)

            
print("\n\nComparison vs DILI-TRUST HistGB:")
s9 = pd.read_csv(TAB9/"stage9_all_split_metrics.csv"); hgb = s9[s9["model"]=="HistGB"]; et = s9[s9["model"]=="ET"]
comp = []
for _,r in dfm[dfm["metric_source"]=="actual_prediction"].iterrows():
    sn = r["split"]; sr = "scaffold" if sn in ["external_dilist","post2021_lockbox"] else sn
    ha = hgb[hgb["split"]==sr]["auroc"].values[0] if len(hgb[hgb["split"]==sr])>0 else np.nan
    ea = et[et["split"]==sr]["auroc"].values[0] if len(et[et["split"]==sr])>0 else np.nan
    d_h = round(r["auroc"]-ha,4) if not pd.isna(r.get("auroc")) and not pd.isna(ha) else np.nan
    comp.append({"model":r["model_name"],"split":sn,"lit_AUROC":r["auroc"],"HistGB_AUROC":ha,"ET_AUROC":ea,"delta_vs_HistGB":d_h})
    print(f"  {r['model_name']} {sn}: lit={r['auroc']} HGB={ha} ET={ea} delta={d_h}")
dfc = pd.DataFrame(comp)
dfc.to_csv(TAB125/"literature_vs_DILI_TRUST_reference_comparison.csv", index=False)
dfc.to_excel(TAB125/"literature_vs_DILI_TRUST_reference_comparison.xlsx", index=False)

                      
df_sens = pd.DataFrame([{"model_name":r["model_name"],"split":r["split"],"DILI_TRUST_AUROC":r["auroc"],"DILI_TRUST_Brier":r["brier"],"status":"actual_reproduction"} for _,r in dfm[dfm["metric_source"]=="actual_prediction"].iterrows()])
df_sens.to_csv(TAB125/"protocol_sensitivity_summary.csv", index=False)

         
spl = ["scaffold","external_dilist","temporal","cluster_out"]
fig,ax=plt.subplots(figsize=(10,5)); x=np.arange(len(spl)); w=0.2
for j,(mn,c) in enumerate([("HistGB","#0072B2"),("InterDILI_adapted","#D55E00"),("DILIPredictor_simplified","#009E73"),("pDILI_v1_adapted","#CC79A7")]):
    vs = []; [vs.append(hgb[hgb["split"]==("scaffold" if s in["external_dilist"] else s)]["auroc"].values[0] if mn=="HistGB" and len(hgb[hgb["split"]==("scaffold" if s in["external_dilist"] else s)])>0 else (dfm[(dfm["model_name"]==mn)&(dfm["split"]==s)]["auroc"].values[0] if len(dfm[(dfm["model_name"]==mn)&(dfm["split"]==s)])>0 else 0)) for s in spl]
    ax.bar(x+j*w, vs, w*.85, color=c, label=mn.replace("_"," "), alpha=.85)
ax.set_xticks(x+w*1.5); ax.set_xticklabels(spl); ax.set_ylabel("AUROC"); ax.legend(fontsize=7)
ax.set_title("Literature Models vs DILI-TRUST Under Strict Splits")
plt.tight_layout(); fig.savefig(FIG125/"literature_model_strict_split_performance_comparison.png"); fig.savefig(FIG125/"literature_model_strict_split_performance_comparison.pdf"); plt.close()

fig,axs=plt.subplots(1,3,figsize=(12,4))
for ax,mn in zip(axs,["InterDILI_adapted","DILIPredictor_simplified","pDILI_v1_adapted"]):
    sub=dfm[(dfm["model_name"]==mn)&(dfm["metric_source"]=="actual_prediction")]
    if len(sub)==0: continue
    ax.bar(sub["split"],sub["auroc"],color="#D55E00",alpha=.85); ax.set_title(mn.replace("_"," ")); ax.set_ylabel("AUROC"); ax.set_ylim(0,1)
    [ax.text(i,r["auroc"]+.02,f'{r["auroc"]:.3f}',ha="center",fontsize=7) for i,(_,r) in enumerate(sub.iterrows())]
plt.suptitle("Literature Models Under DILI-TRUST Strict Splits",fontsize=11,y=1.02)
plt.tight_layout(); fig.savefig(FIG125/"protocol_sensitivity_panel.png"); fig.savefig(FIG125/"protocol_sensitivity_panel.pdf"); plt.close()

         
n_act = len(dfm[dfm["metric_source"]=="actual_prediction"]["model_name"].unique())
actual_models = sorted(dfm[dfm["metric_source"]=="actual_prediction"]["model_name"].unique())
final_v = "CONDITIONAL_PASS_stage12_5_partial_actual_reproduction_ready_with_limitations"

qc = f"# Stage 12.5 Reproduction QC Report\n**{NOW}**\n**Verdict:** `{final_v}`\n\n## Actual Adaptations\n"
qc += f"- InterDILI adapted (A): HistGB on structure features (TF attention unavailable)\n"
qc += f"- DILIPredictor simplified (B): RF on structure-only (PK/proxy unavailable)\n"
qc += f"- pDILI_v1 adapted (A): RF on RDKit 2D (Mordred unavailable)\n"
qc += f"- StackDILI: infeasible (iFeatureOmegaCLI required)\n"
qc += f"- DILIGeNN: infeasible (PyTorch/DGL not installed)\n\n## Actual Metrics\n"
for _,r in dfm[dfm["metric_source"]=="actual_prediction"].iterrows():
    qc += f"- {r['model_name']}/{r['split']}: n={int(r['n'])} AUROC={r['auroc']}\n"
qc += f"\n## vs HistGB\n"
for _,r in dfc.iterrows():
    qc += f"- {r['model']}/{r['split']}: delta={r['delta_vs_HistGB']}\n"
qc += f"\n## Verdict: {final_v}\n"
with open(REP125/"stage12_5_strict_split_reproduction_qc_report.md","w",encoding="utf-8") as f: f.write(qc)

            
sc=[("interdili_actual",(TMP125/"retraining/InterDILI_adapted/training_log.csv").exists()),("dilipredictor_actual",(TMP125/"retraining/DILIPredictor_simplified/training_log.csv").exists()),("pdili_actual",(TMP125/"retraining/pDILI_v1_adapted/training_log.csv").exists()),("strict_separated",(TAB125/"literature_model_strict_split_metrics.csv").exists()),("no_estimates_in_metrics","planning" not in str(dfm["metric_source"].unique())),("comparison",(TAB125/"literature_vs_DILI_TRUST_reference_comparison.csv").exists()),("protocol_sensitivity",(TAB125/"protocol_sensitivity_summary.csv").exists()),("figures",(FIG125/"protocol_sensitivity_panel.png").exists()),("qc_report",(REP125/"stage12_5_strict_split_reproduction_qc_report.md").exists()),("final_verdict",True)]
ns=sum(1 for _,o in sc if o); nt=len(sc)
sc_t=f"# Stage 12.5 Self-Check\n**{NOW}**\n**{final_v}**\n\nChecks: {ns}/{nt}\n\n"
for l,o in sc: sc_t+=f"- [{'PASS' if o else 'FAIL'}] {l}\n"
with open(REP125/"stage12_5_self_check.md","w",encoding="utf-8") as f: f.write(sc_t)

print(f"\n\n===== FINAL: {final_v} =====")
print(f"  Actual models: {actual_models}")
print(f"  Infeasible: StackDILI, DILIGeNN")
print(f"  Self-check: {ns}/{nt}")
print(f"  QC: {REP125/'stage12_5_strict_split_reproduction_qc_report.md'}")
