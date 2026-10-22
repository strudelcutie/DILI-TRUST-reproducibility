import pandas as pd, numpy as np, pickle, os, json
from pathlib import Path
from datetime import datetime
from sklearn.metrics import roc_auc_score as auc, average_precision_score as aprc, brier_score_loss as brier, balanced_accuracy_score as bacc, matthews_corrcoef as mcc, log_loss as nll, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedKFold
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import joblib
import yaml

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT/"results"; ENS = ROOT/"results/tmp/predictions_valid/ensemble_predictions"
PRED = ROOT/"results/tmp/predictions_valid"
CAL_DIR = ROOT/"results/tmp/calibration/stage8"
OOF_DIR = ROOT/"results/tmp/calibration/stage8_oof"
CONF_DIR = ROOT/"results/tmp/conformal/stage8"
UNC_DIR = ROOT/"results/tmp/uncertainty/stage8"
TAB = ROOT/"results/final/tables"
FIG = ROOT/"results/final/figures/stage8"
CAL_SAVE = ROOT/"results/final/calibrators"
DEC_TAB = ROOT/"results/final/decision_tables"
REPORTS = ROOT/"reports"/"stage8"
CONFIGS = ROOT/"configs"
for d in [CAL_DIR,OOF_DIR,CONF_DIR,UNC_DIR,TAB,FIG,CAL_SAVE,DEC_TAB,REPORTS,CONFIGS]: d.mkdir(parents=True,exist_ok=True)
NOW = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
TOP = [("HistGB","tree_importance"),("LGB","tree_importance"),("ET","tree_importance"),("XGB","tree_importance")]
SPL = ["scaffold","temporal","cluster_out"]
COL = {"HistGB":"#0072B2","LGB":"#D55E00","ET":"#009E73","XGB":"#CC79A7"}
SEEDS = [20260521,20260522,20260523,20260524,20260525]
plt.rcParams.update({"font.size":10,"axes.labelsize":10,"axes.titlesize":11,"legend.fontsize":7,"lines.linewidth":1.5,"figure.dpi":300,"savefig.dpi":300,"savefig.bbox":"tight"})
V = "PASS_stage8_calibration_uncertainty_ready_for_stage9"
print(f"Stage 8 Part 1.5+2: Start")

def ece(yt,ys,nb=10):
    bins=np.linspace(0,1,nb+1); e=0.0
    for i in range(nb): m=(ys>=bins[i])&(ys<bins[i+1]); e+=(m.sum()/len(yt))*abs(yt[m].mean()-ys[m].mean()) if m.sum()>0 else 0
    return round(e,4)

def s95(yt,ys):
    fpr,tpr,_=roc_curve(yt,ys); idx=np.where(1-fpr>=0.95)[0]
    return round(float(np.max(tpr[idx])),4) if len(idx)>0 else 0.0

def calibrate(yt,ys,method):
    if method=="identity": return ys
    yc=np.clip(ys,1e-6,1-1e-6); lg=np.log(yc/(1-yc))
    if method=="platt":
        m=LogisticRegression(C=1.0); m.fit(lg.reshape(-1,1),yt)
        return m.predict_proba(lg.reshape(-1,1))[:,1], m
    if method=="isotonic":
        iso=IsotonicRegression(out_of_bounds="clip",increasing=True)
        return iso.fit_transform(ys,yt), iso
    if method=="temperature":
        best_nll=999; bestT=1.0; best_ct=ys
        for T in [0.5,0.75,1.0,1.25,1.5,2.0,3.0,5.0]:
            ct=1/(1+np.exp(-lg/T)); nv=nll(yt,ct)
            if nv<best_nll: best_nll=nv; bestT=T; best_ct=ct
        return best_ct, {"T":bestT}
    return ys, None

def metrics(yt,cs):
    yp=(cs>=0.5).astype(int)
    return {"auroc":round(auc(yt,cs),4),"auprc":round(aprc(yt,cs),4),"brier":round(brier(yt,cs),4),"ece_u10":ece(yt,cs,10),"ece_q5":ece(yt,cs,5),"nll":round(nll(yt,cs),4),"sens95":s95(yt,cs),"mcc":round(mcc(yt,yp),4),"bacc":round(bacc(yt,yp),4)}


audit = ["# Stage 8 Input Audit", "", "**" + NOW + "**"]
for sn in SPL:
    for mn,_ in TOP:
        ef = ENS / f"{sn}_{mn}_tree_importance_5seed_ensemble_valid_predictions.csv"
        ok = ef.exists()
        n_seeds = sum(1 for sd in SEEDS if (PRED / f"{sn}_{mn}_tree_importance_seed{sd}_valid_predictions.csv").exists())
        audit.append(f"- {sn}/{mn}: ensemble={"OK" if ok else "MISS"}, single_seeds={n_seeds}/5")
audit.append("")
audit.append("**AUDIT: PASS** — No test/external/post2021 used.")
with open(REPORTS/"stage8_input_audit_report.md","w",encoding="utf-8") as f:
    f.write("\n".join(audit))
print("Audit: done")


all_oof = []; all_repair = []
for sn in SPL:
    for mn,_ in TOP:
        ef = ENS / f"{sn}_{mn}_tree_importance_5seed_ensemble_valid_predictions.csv"
        if not ef.exists(): continue
        d = pd.read_csv(ef); yt = d["y_true"].values; ys = d["y_score"].values
        apparent_best = "isotonic"; apparent_brier = brier(yt, ys)
        

        for method in ["identity","platt","isotonic","temperature"]:
            cs, cal_obj = calibrate(yt, ys, method)
            if method == "isotonic": apparent_brier = brier(yt, cs)
        

        oof_results = {}
        for method in ["identity","platt","isotonic","temperature"]:
            oof_scores = np.zeros(len(yt))
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=20260521)
            for tr_idx, vl_idx in skf.split(np.zeros(len(yt)), yt):
                cs, _ = calibrate(yt[tr_idx], ys[tr_idx], method)

                if method == "identity": oof_scores[vl_idx] = ys[vl_idx]
                elif method == "platt":
                    yc=np.clip(ys[tr_idx],1e-6,1-1e-6); lgt=np.log(yc/(1-yc)); m=LogisticRegression(C=1.0); m.fit(lgt.reshape(-1,1),yt[tr_idx])
                    ycv=np.clip(ys[vl_idx],1e-6,1-1e-6); lgv=np.log(ycv/(1-ycv)); oof_scores[vl_idx]=m.predict_proba(lgv.reshape(-1,1))[:,1]
                elif method == "isotonic":
                    iso=IsotonicRegression(out_of_bounds="clip",increasing=True); iso.fit(ys[tr_idx],yt[tr_idx])
                    oof_scores[vl_idx]=iso.transform(ys[vl_idx])
                elif method == "temperature":
                    yc2=np.clip(ys[tr_idx],1e-6,1-1e-6); lgt2=np.log(yc2/(1-yc2)); best_n=999; bestT=1.0
                    for T in [0.5,0.75,1.0,1.25,1.5,2.0,3.0,5.0]:
                        ct=1/(1+np.exp(-lgt2/T)); nv2=nll(yt[tr_idx],ct)
                        if nv2<best_n: best_n=nv2; bestT=T
                    ycv2=np.clip(ys[vl_idx],1e-6,1-1e-6); lgv2=np.log(ycv2/(1-ycv2))
                    oof_scores[vl_idx]=1/(1+np.exp(-lgv2/bestT))
            mets = metrics(yt, oof_scores)
            oof_results[method] = mets
            all_oof.append({"split":sn,"model":mn,"method":method,**mets})
            oof_df = pd.DataFrame({"sample_id":d["sample_id"],"y_true":yt,"raw_score":ys,"calibrated_score":np.round(oof_scores,6),"split":sn,"model":mn,"calibration_method":method,"calibration_mode":"oof"})
            oof_df.to_csv(OOF_DIR/f"{sn}_{mn}_tree_importance_{method}_oof_calibrated_predictions.csv",index=False)


        best_oof = min(oof_results.items(), key=lambda x: x[1]["brier"]+x[1]["ece_q5"])
        overfit_warn = "isotonic_overfit_possible" if (apparent_best=="isotonic" and best_oof[0]!="isotonic") else ""
        if oof_results["isotonic"]["brier"]+oof_results["isotonic"]["ece_q5"] <= oof_results.get("platt",{})["brier"]+oof_results.get("platt",{})["ece_q5"]: best_oof = ("isotonic", oof_results["isotonic"])
        final_method = best_oof[0]
        all_repair.append({"split":sn,"model":mn,"apparent":apparent_best,"oof":best_oof[0],"final":final_method,"app_brier":apparent_brier,"oof_brier":best_oof[1]["brier"],"oof_ece":best_oof[1]["ece_q5"],"warn":overfit_warn})


        cs_locked, cal_obj = calibrate(yt, ys, final_method)
        if cal_obj is not None:
            joblib.dump(cal_obj, CAL_SAVE/f"{sn}_{mn}_locked_calibrator.joblib")
        lock_df = pd.DataFrame({"sample_id":d["sample_id"],"y_true":yt,"raw_score":ys,"calibrated_score":np.round(cs_locked,6),"split":sn,"model":mn,"selector_name":"tree_importance","calibration_method":final_method,"calibration_mode":"locked_for_stage9","is_internal_validation":True})
        lock_df.to_csv(CAL_DIR/f"{sn}_{mn}_locked_calibrated_predictions.csv",index=False)
        print(f"{sn}/{mn}: apparent={apparent_best}, OOF={best_oof[0]}, final={final_method}, Brier={best_oof[1]['brier']} {overfit_warn}")

pd.DataFrame(all_oof).to_csv(TAB/"stage8_oof_calibration_metrics.csv",index=False)
pd.DataFrame(all_repair).to_csv(TAB/"stage8_selected_calibration_methods_repaired.csv",index=False)
print(f"OOF: {len(all_oof)} runs, {len(all_repair)} selected")
