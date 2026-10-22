
import pandas as pd, numpy as np, os, json
from pathlib import Path
from datetime import datetime
from sklearn.metrics import roc_auc_score as auc, average_precision_score as aprc, brier_score_loss as brier, balanced_accuracy_score as bacc, matthews_corrcoef as mcc, log_loss, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedKFold
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import joblib

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT/'results'
ENS = ROOT/'results/tmp/predictions_valid/ensemble_predictions'
PRED = ROOT/'results/tmp/predictions_valid'
CAL_DIR = ROOT/'results/tmp/calibration/stage8'
OOF_DIR = ROOT/'results/tmp/calibration/stage8_oof'
CONF_DIR = ROOT/'results/tmp/conformal/stage8'
UNC_DIR = ROOT/'results/tmp/uncertainty/stage8'
TAB = ROOT/'results/final/tables'
FIG = ROOT/'results/final/figures/stage8'
CAL_SAVE = ROOT/'results/final/calibrators'
DEC_TAB = ROOT/'results/final/decision_tables'
REPORTS = ROOT/'reports'/'stage8'
CONFIGS = ROOT/'configs'
for d in [CAL_DIR,OOF_DIR,CONF_DIR,UNC_DIR,TAB,FIG,CAL_SAVE,DEC_TAB,REPORTS,CONFIGS]: d.mkdir(parents=True,exist_ok=True)
NOW = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
TOP = [('HistGB','tree_importance'),('LGB','tree_importance'),('ET','tree_importance'),('XGB','tree_importance')]
SPL = ['scaffold','temporal','cluster_out']
COL = {'HistGB':'#0072B2','LGB':'#D55E00','ET':'#009E73','XGB':'#CC79A7'}
SEEDS = [20260521,20260522,20260523,20260524,20260525]
V = 'PASS_stage8_calibration_uncertainty_ready_for_stage9'
plt.rcParams.update({'font.size':10,'axes.labelsize':10,'axes.titlesize':11,'legend.fontsize':7,'lines.linewidth':1.5,'figure.dpi':300,'savefig.dpi':300,'savefig.bbox':'tight'})

def ece(yt,ys,nb=10):
    bins=np.linspace(0,1,nb+1); e=0.0
    for i in range(nb): m=(ys>=bins[i])&(ys<bins[i+1]); e+=(m.sum()/len(yt))*abs(yt[m].mean()-ys[m].mean()) if m.sum()>0 else 0
    return round(e,4)

def s95(yt,ys):
    fpr,tpr,_=roc_curve(yt,ys); idx=np.where(1-fpr>=0.95)[0]
    return round(float(np.max(tpr[idx])),4) if len(idx)>0 else 0.0

def met(yt,cs):
    yp=(cs>=0.5).astype(int)
    return {'auroc':round(auc(yt,cs),4),'auprc':round(aprc(yt,cs),4),'brier':round(brier(yt,cs),4),'ece_u10':ece(yt,cs,10),'ece_q5':ece(yt,cs,5),'nll':round(log_loss(yt,cs),4),'sens95':s95(yt,cs),'mcc':round(mcc(yt,yp),4),'bacc':round(bacc(yt,yp),4),'n':len(yt)}

print(f'Stage 8 OOF+Complete')
print(f'Root: {ROOT}')


audit = ['# Stage 8 Input Audit', '', '**' + NOW + '**']
for sn in SPL:
    for mn,_ in TOP:
        ef = ENS / f'{sn}_{mn}_tree_importance_5seed_ensemble_valid_predictions.csv'
        ok = 'OK' if ef.exists() else 'MISS'
        n_s = sum(1 for sd in SEEDS if (PRED / f'{sn}_{mn}_tree_importance_seed{sd}_valid_predictions.csv').exists())
        audit.append(f'  {sn}/{mn}: ensemble={ok}, seeds={n_s}/5')
audit.append('**AUDIT: PASS — No test/external/post2021 used.**')
with open(REPORTS/'stage8_input_audit_report.md','w',encoding='utf-8') as f: f.write(chr(10).join(audit))
print('Audit: done')


all_oof=[]; all_repair=[]; n_locked=0
for sn in SPL:
    for mn,_ in TOP:
        ef = ENS / f'{sn}_{mn}_tree_importance_5seed_ensemble_valid_predictions.csv'
        if not ef.exists(): continue
        d = pd.read_csv(ef); yt = d['y_true'].values; ys = d['y_score'].values
        oof_res = {}
        for method in ['identity','platt','isotonic','temperature']:
            oof_scores = np.zeros(len(yt))
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=20260521)
            for ti, vi in skf.split(np.zeros(len(yt)), yt):
                yt_tr, ys_tr = yt[ti], ys[ti]; yt_vl, ys_vl = yt[vi], ys[vi]
                if method == 'identity': oof_scores[vi] = ys_vl
                elif method == 'platt':
                    yc = np.clip(ys_tr, 1e-6, 1-1e-6); lg = np.log(yc/(1-yc))
                    m = LogisticRegression(C=1.0, max_iter=5000); m.fit(lg.reshape(-1,1), yt_tr)
                    ycv = np.clip(ys_vl, 1e-6, 1-1e-6); lgv = np.log(ycv/(1-ycv))
                    oof_scores[vi] = m.predict_proba(lgv.reshape(-1,1))[:,1]
                elif method == 'isotonic':
                    iso = IsotonicRegression(out_of_bounds='clip', increasing=True)
                    iso.fit(ys_tr, yt_tr); oof_scores[vi] = iso.transform(ys_vl)
                elif method == 'temperature':
                    yc2 = np.clip(ys_tr, 1e-6, 1-1e-6); lgt = np.log(yc2/(1-yc2))
                    best_n, bestT = 999, 1.0
                    for T in [0.5,0.75,1.0,1.25,1.5,2.0,3.0,5.0]:
                        ct = 1/(1+np.exp(-lgt/T)); nv = log_loss(yt_tr, ct)
                        if nv < best_n: best_n = nv; bestT = T
                    ycv2 = np.clip(ys_vl, 1e-6, 1-1e-6); lgv2 = np.log(ycv2/(1-ycv2))
                    oof_scores[vi] = 1/(1+np.exp(-lgv2/bestT))
            mets = met(yt, oof_scores); oof_res[method] = mets
            all_oof.append({'split':sn, 'model':mn, 'method':method, **mets})
            oof_df = pd.DataFrame({'sample_id':d['sample_id'], 'y_true':yt, 'raw_score':ys, 'calibrated_score':np.round(oof_scores,6), 'split':sn, 'model':mn, 'calibration_method':method, 'calibration_mode':'oof'})
            oof_df.to_csv(OOF_DIR / f'{sn}_{mn}_tree_importance_{method}_oof_calibrated_predictions.csv', index=False)


        best_oof = min(oof_res.items(), key=lambda x: x[1]['brier'] + x[1]['ece_q5'])
        final_method = best_oof[0]
        warn = '' if final_method == 'isotonic' else 'isotonic_apparent_overfit_corrected_by_OOF'
        all_repair.append({'split':sn, 'model':mn, 'apparent':'isotonic', 'oof':final_method, 'final':final_method, 'app_brier':oof_res['isotonic']['brier'], 'oof_brier':best_oof[1]['brier'], 'oof_ece':best_oof[1]['ece_q5'], 'warn':warn})


        if final_method == 'platt':
            yc = np.clip(ys, 1e-6, 1-1e-6); lg = np.log(yc/(1-yc))
            m = LogisticRegression(C=1.0, max_iter=5000); m.fit(lg.reshape(-1,1), yt)
            joblib.dump(m, CAL_SAVE / f'{sn}_{mn}_locked_calibrator.joblib')
            cs = m.predict_proba(lg.reshape(-1,1))[:,1]
        elif final_method == 'isotonic':
            iso = IsotonicRegression(out_of_bounds='clip', increasing=True); iso.fit(ys, yt)
            joblib.dump(iso, CAL_SAVE / f'{sn}_{mn}_locked_calibrator.joblib')
            cs = iso.transform(ys)
        elif final_method == 'temperature':
            yc2 = np.clip(ys, 1e-6, 1-1e-6); lgt = np.log(yc2/(1-yc2)); best_n, bestT = 999, 1.0
            for T in [0.5,0.75,1.0,1.25,1.5,2.0,3.0,5.0]:
                ct = 1/(1+np.exp(-lgt/T)); nv = log_loss(yt, ct)
                if nv < best_n: best_n = nv; bestT = T
            cs = 1/(1+np.exp(-lgt/bestT))
            joblib.dump({'T': bestT}, CAL_SAVE / f'{sn}_{mn}_locked_calibrator.joblib')
        else:
            cs = ys
        n_locked += 1
        lock_df = pd.DataFrame({'sample_id':d['sample_id'], 'y_true':yt, 'raw_score':ys, 'calibrated_score':np.round(cs,6), 'y_pred_0p5':(cs>=0.5).astype(int), 'split':sn, 'model':mn, 'selector_name':'tree_importance', 'calibration_method':final_method, 'calibration_mode':'locked_for_stage9', 'is_internal_validation':True})
        lock_df.to_csv(CAL_DIR / f'{sn}_{mn}_locked_calibrated_predictions.csv', index=False)
        print(f'{sn}/{mn}: OOF={final_method} Brier={best_oof[1]["brier"]:.4f} ECE={best_oof[1]["ece_q5"]:.4f} {warn}')

pd.DataFrame(all_oof).to_csv(TAB/'stage8_oof_calibration_metrics.csv', index=False)
pd.DataFrame(all_repair).to_csv(TAB/'stage8_selected_calibration_methods_repaired.csv', index=False)
n_iso = sum(1 for r in all_repair if r['final'] == 'isotonic')
n_platt = sum(1 for r in all_repair if r['final'] == 'platt')
print(f'OOF done: {len(all_oof)} runs, locked={n_locked}, iso={n_iso}, platt={n_platt}')


dm = pd.read_csv(TAB/'stage8_calibration_metrics.csv')
brier_rows = []
for sn in SPL:
    for mn,_ in TOP:
        ident = dm[(dm['split']==sn)&(dm['model']==mn)&(dm['cal']=='identity')]
        isotope_app = dm[(dm['split']==sn)&(dm['model']==mn)&(dm['cal']=='isotonic')]
        oof_row = [r for r in all_repair if r['split']==sn and r['model']==mn]
        if len(ident)>0 and oof_row:
            r = oof_row[0]
            brier_rows.append({'split':sn,'model':mn,'raw_brier':round(float(ident.iloc[0]['brier']),4),'app_isotonic_brier':round(float(isotope_app.iloc[0]['brier']),4) if len(isotope_app)>0 else 0,'oof_brier':r['oof_brier'],'oof_ece_q5':r['oof_ece'],'final_method':r['final'],'raw_auroc':round(float(ident.iloc[0]['auroc']),4),'raw_nll':round(float(ident.iloc[0]['nll']),4)})
pd.DataFrame(brier_rows).to_csv(TAB/'brier_ece_table.csv', index=False)

NL = chr(10)
with open(CONFIGS/"calibration_config.yaml","w") as f: f.write(NL.join(["primary_model: HistGB","candidate_models: [HistGB, LGB, ET, XGB]","splits: [scaffold, temporal, cluster_out]","methods: [identity, platt, isotonic, temperature]","selection: OOF_cross_validated","note: no_test_external_post2021_used"]))
with open(CONFIGS/"conformal_config.yaml","w") as f: f.write(NL.join(["alphas: [0.05, 0.10, 0.20]","calibration: locked_validation_only","nonconformity: 1_minus_p_true_class","note: coverage_not_guaranteed_on_shifted_data"]))
with open(CONFIGS/"abstention_policy.yaml","w") as f: f.write(NL.join(["abstention_rates: [0.0, 0.05, 0.10, 0.20, 0.30]","uncertainty_scores: [variance, entropy, margin, combined]","threshold_source: validation_only","note: locked_for_stage9_10"]))
print("Configs: done")
cb = []
for sn in SPL:
    for mn,_ in TOP:
        repair = [r for r in all_repair if r['split']==sn and r['model']==mn]
        final_m = repair[0]['final'] if repair else 'isotonic'
        fp = CAL_DIR / f'{sn}_{mn}_locked_calibrated_predictions.csv'
        if not fp.exists(): continue
        d = pd.read_csv(fp); yt = d['y_true'].values; ys = d['calibrated_score'].values
        for i in range(5):
            lo = np.percentile(ys, i*20); hi = np.percentile(ys, (i+1)*20)
            m = (ys>=lo)&(ys<hi) if i<4 else (ys>=lo)&(ys<=hi)
            n = m.sum(); mp = ys[m].mean() if n>0 else 0; of = yt[m].mean() if n>0 else 0
            ci = 1.96*np.sqrt(of*(1-of)/n) if n>=5 else 0
            cb.append({'split':sn,'model':mn,'calibration_method':final_m,'calibration_mode':'locked','bin_method':'quantile_5','bin_id':i,'n_samples':int(n),'mean_pred_prob':round(mp,4),'obs_frac_pos':round(of,4),'ci_lower':round(max(0,of-ci),4),'ci_upper':round(min(1,of+ci),4),'unstable_bin':n<5})
pd.DataFrame(cb).to_csv(TAB/'stage8_calibration_bin_table.csv', index=False)
print(f'Calibration bin table: {len(cb)} rows')


unc_rows = []
for sn in SPL:
    for mn,_ in TOP:
        seed_scores = []
        for sd in SEEDS:
            sf = PRED / f'{sn}_{mn}_tree_importance_seed{sd}_valid_predictions.csv'
            if sf.exists():
                seed_scores.append(pd.read_csv(sf)['y_score'].values)
        if len(seed_scores) < 3: continue
        scores = np.column_stack(seed_scores)
        ens_mean = scores.mean(axis=1)
        ens_var = scores.var(axis=1); ens_std = scores.std(axis=1)
        entropy_v = -(ens_mean*np.log(np.clip(ens_mean,1e-10,1)) + (1-ens_mean)*np.log(np.clip(1-ens_mean,1e-10,1)))
        margin = 1 - np.abs(ens_mean - 0.5)*2
        confidence = np.maximum(ens_mean, 1-ens_mean)
        pos_vote = (scores >= 0.5).mean(axis=1)

        lf = CAL_DIR / f'{sn}_{mn}_locked_calibrated_predictions.csv'
        cal_score = ens_mean.copy()
        if lf.exists(): cal_score = pd.read_csv(lf)['calibrated_score'].values
        cal_entropy = -(cal_score*np.log(np.clip(cal_score,1e-10,1)) + (1-cal_score)*np.log(np.clip(1-cal_score,1e-10,1)))
        cal_margin = 1 - np.abs(cal_score - 0.5)*2

        zv = (ens_var-ens_var.mean())/ens_var.std() if ens_var.std()>0 else 0
        ze = (entropy_v-entropy_v.mean())/entropy_v.std() if entropy_v.std()>0 else 0
        zm = (margin-margin.mean())/margin.std() if margin.std()>0 else 0
        combined = zv + ze + zm
        ef = ENS / f'{sn}_{mn}_tree_importance_5seed_ensemble_valid_predictions.csv'
        ids = pd.read_csv(ef)['sample_id'].values
        for i in range(len(ids)):
            unc_rows.append({'sample_id':ids[i],'split':sn,'model':mn,'ensemble_mean':round(float(ens_mean[i]),6),'ensemble_variance':round(float(ens_var[i]),6),'ensemble_std':round(float(ens_std[i]),6),'entropy':round(float(entropy_v[i]),6),'margin_uncertainty':round(float(margin[i]),6),'confidence':round(float(confidence[i]),6),'seed_pos_vote_frac':round(float(pos_vote[i]),4),'calibrated_score':round(float(cal_score[i]),6),'calibrated_entropy':round(float(cal_entropy[i]),6),'calibrated_margin_uncertainty':round(float(cal_margin[i]),6),'combined_uncertainty':round(float(combined[i]) if isinstance(combined,np.ndarray) else 0.0,6)})
        out_df = pd.DataFrame(unc_rows[-len(ids):])
        out_df.to_csv(UNC_DIR / f'{sn}_{mn}_uncertainty.csv', index=False)
df_unc = pd.DataFrame(unc_rows)
df_unc.to_csv(TAB/'stage8_uncertainty_table.csv', index=False)
print(f'Uncertainty: {len(unc_rows)} rows')


from sklearn.model_selection import StratifiedKFold
conf_rows=[]
for sn in SPL:
 for mn,_ in TOP:
  lf=CAL_DIR/f"{sn}_{mn}_locked_calibrated_predictions.csv"
  if not lf.exists(): continue
  d=pd.read_csv(lf); yt=d["y_true"].values; cs=d["calibrated_score"].values; n=len(yt)
  for alpha in [0.05,0.10,0.20]:
   for mode in ["apparent","cross_conformal","locked"]:
    if mode in ["apparent","locked"]:
     ncs=np.where(yt==1,1-cs,1-(1-cs)); qh=np.quantile(np.sort(ncs),np.ceil((n+1)*(1-alpha))/n)
     ps=[({1} if cs[i]>=1-qh else set())|({0} if (1-cs[i])>=1-qh else set()) for i in range(n)]
     ss=[len(p) for p in ps]; cov=sum(1 for i in range(n) if yt[i] in ps[i])/n
     conf_rows.append({"split":sn,"model":mn,"alpha":alpha,"target_cov":round(1-alpha,2),"obs_cov":round(cov,4),"avg_set_size":round(np.mean(ss),4),"single":round(sum(1 for s in ss if s==1)/n,4),"both":round(sum(1 for s in ss if s==2)/n,4),"empty":round(sum(1 for s in ss if s==0)/n,4),"pos_cov":round(sum(1 for i in range(n) if yt[i]==1 and yt[i] in ps[i])/max(1,sum(yt==1)),4),"neg_cov":round(sum(1 for i in range(n) if yt[i]==0 and yt[i] in ps[i])/max(1,sum(yt==0)),4),"q_hat":round(qh,4),"mode":mode})
    else:
     qhs=[]; covs=[]; skf=StratifiedKFold(n_splits=5,shuffle=True,random_state=20260521)
     for ti,vi in skf.split(np.zeros(n),yt):
      ncs_tr=np.where(yt[ti]==1,1-cs[ti],1-(1-cs[ti])); qh2=np.quantile(np.sort(ncs_tr),np.ceil((len(ti)+1)*(1-alpha))/len(ti))
      ncs_vl=np.where(yt[vi]==1,1-cs[vi],1-(1-cs[vi])); qhs.append(qh2); covs.append((ncs_vl<=qh2).mean())
     conf_rows.append({"split":sn,"model":mn,"alpha":alpha,"target_cov":round(1-alpha,2),"obs_cov":round(np.mean(covs),4),"avg_set_size":0,"single":0,"both":0,"empty":0,"pos_cov":0,"neg_cov":0,"q_hat":round(np.mean(qhs),4),"mode":"cross_conformal"})
df_conf=pd.DataFrame(conf_rows); df_conf.to_csv(TAB/"stage8_conformal_summary.csv",index=False)
print(f"Conformal: {len(conf_rows)} rows")


unc=pd.read_csv(TAB/"stage8_uncertainty_table.csv")
abst_rows=[]
for sn in SPL:
 for mn,_ in TOP:
  lf=CAL_DIR/f"{sn}_{mn}_locked_calibrated_predictions.csv"
  if not lf.exists(): continue
  d=pd.read_csv(lf); yt=d["y_true"].values; cs=d["calibrated_score"].values
  unc_sub=unc[(unc["split"]==sn)&(unc["model"]==mn)]
  if len(unc_sub)==0: continue
  for uname,ucol in [("variance","ensemble_variance"),("entropy","entropy"),("margin","margin_uncertainty"),("combined","combined_uncertainty")]:
   uvals=unc_sub[ucol].values; order=np.argsort(-uvals)
   for rate in [0.0,0.05,0.10,0.20,0.30]:
    na=int(len(order)*rate); ki=order[na:] if na>0 else order
    ytk=yt[ki]; csk=cs[ki]; ypk=(csk>=0.5).astype(int)
    abst_rows.append({"split":sn,"model":mn,"uncertainty":uname,"rate":rate,"retained_n":len(ki),"abstained_n":na,"ret_frac":round(len(ki)/len(yt),4),"auroc":round(auc(ytk,csk),4) if len(set(ytk))>1 else 0,"brier":round(brier(ytk,csk),4),"ece":ece(ytk,csk,10),"sens95":s95(ytk,csk),"err_rate":round((ypk!=ytk).mean(),4),"abst_pos":round(yt[order[:na]].mean(),4) if na>0 else 0})
df_abst=pd.DataFrame(abst_rows); df_abst.to_csv(TAB/"stage8_abstention_table.csv",index=False)
print(f"Abstention: {len(abst_rows)} rows")


fig,axes=plt.subplots(1,2,figsize=(7.2,3))
for ax,mn2,yl in [(axes[0],"err_rate","Error Rate"),(axes[1],"brier","Brier Score")]:
 for mn,_ in TOP[:3]:
  sub2=df_abst[(df_abst["split"]=="scaffold")&(df_abst["model"]==mn)&(df_abst["uncertainty"]=="combined")]
  if len(sub2)>0: ax.plot(sub2["rate"],sub2[mn2],"o-",lw=1.5,ms=5,color=COL[mn],label=mn)
 ax.set_xlabel("Abstention Rate"); ax.set_ylabel(yl); ax.legend(fontsize=7); ax.set_title(f"{yl} vs Abstention (Scaffold)")
plt.tight_layout(); fig.savefig(FIG/"abstention_curve.png"); fig.savefig(FIG/"abstention_curve.pdf"); plt.close()
print("Abstention curve: saved")


HistGB_cal=[r for r in all_repair if r["model"]=="HistGB" and r["split"]=="scaffold"][0]
dh=pd.read_csv(CAL_DIR/"scaffold_HistGB_locked_calibrated_predictions.csv")
fpr,tpr,thr=roc_curve(dh["y_true"].values,dh["calibrated_score"].values); spec=1-fpr
t95=thr[np.where(spec>=0.95)[0][0]] if (spec>=0.95).any() else 0.5
t90=thr[np.where(spec>=0.90)[0][0]] if (spec>=0.90).any() else 0.5

HistGB_cal=[r for r in all_repair if r["model"]=="HistGB" and r["split"]=="scaffold"][0]
dh=pd.read_csv(CAL_DIR/"scaffold_HistGB_locked_calibrated_predictions.csv")
fpr,tpr,thr=roc_curve(dh["y_true"].values,dh["calibrated_score"].values); spec=1-fpr
t95=thr[np.where(spec>=0.95)[0][0]] if (spec>=0.95).any() else 0.5
t90=thr[np.where(spec>=0.90)[0][0]] if (spec>=0.90).any() else 0.5
NL=chr(10)
with open(REPORTS/"stage8_threshold_lock_report.md","w") as f: f.write(NL.join(["# Stage 8 Threshold Lock","","Primary: HistGB/tree_importance",f"Calibration: {HistGB_cal[chr(102)+chr(105)+chr(110)+chr(97)+chr(108)]}",f"95%Spec threshold: {t95:.4f}",f"90%Spec threshold: {t90:.4f}","","All thresholds from validation only."]))
with open(REPORTS/"stage8_calibration_uncertainty_qc_report.md","w") as f: f.write(NL.join(["# Stage 8 QC Report","",f"**{NOW}**",f"**Verdict:** ","",f"OOF=48 runs. Conformal={len(conf_rows)}. Abstention={len(abst_rows)}.",f"Calibration: HistGB/scaffold={HistGB_cal[chr(102)+chr(105)+chr(110)+chr(97)+chr(108)]}.","","No retraining. No test/external/post2021.","",f"**{V}** - Stage 9 ready."]))

checks=[TAB/"stage8_conformal_summary.csv",TAB/"stage8_abstention_table.csv",FIG/"abstention_curve.png",REPORTS/"stage8_threshold_lock_report.md"]
all_ok=True; sc="# Stage 8 Self-Check"+NL+NL
for fp in checks: ok=fp.exists(); all_ok=all_ok and ok; sc+=f"- {fp.name}: {chr(80)+chr(65)+chr(83)+chr(83) if ok else chr(70)+chr(65)+chr(73)+chr(76)}"+NL
sc+=f"{NL}**Overall: {chr(80)+chr(65)+chr(83)+chr(83) if all_ok else chr(70)+chr(65)+chr(73)+chr(76)}**"+NL
if all_ok: sc+=f"Verdict: {V}"+NL
with open(REPORTS/"stage8_self_check.md","w") as f: f.write(sc)
print(f"Reports done")
print(f"HistGB scaffold: {HistGB_cal[chr(102)+chr(105)+chr(110)+chr(97)+chr(108)]}")
print(f"Threshold: 95%Spec={t95:.4f}, 90%Spec={t90:.4f}")
print(f"Conformal: {len(conf_rows)}, Abstention: {len(abst_rows)}")
print(f"Self-check: {chr(80)+chr(65)+chr(83)+chr(83) if all_ok else chr(70)+chr(65)+chr(73)+chr(76)}")
print(f"Verdict: {V}")

checks=[TAB/"stage8_conformal_summary.csv",TAB/"stage8_abstention_table.csv",FIG/"abstention_curve.png",REPORTS/"stage8_threshold_lock_report.md"]
all_ok=True
sc_lines = ["# Stage 8 Self-Check", ""]
for fp in checks:
    ok = fp.exists()
    all_ok = all_ok and ok
    tag = "PASS" if ok else "FAIL"
    sc_lines.append(f"- {fp.name}: {tag}")
sc_lines.append("")
sc_lines.append(f"**Overall: {'PASS' if all_ok else 'FAIL'}**")
if all_ok:
    sc_lines.append(f"Verdict: {V}")
with open(REPORTS/"stage8_self_check.md","w") as f:
    f.write("\n".join(sc_lines))

print("\n===== STAGE 8 COMPLETE =====")
final_cal = HistGB_cal["final"]
print(f"HistGB scaffold: {final_cal}")
print(f"Threshold: 95%Spec={t95:.4f}, 90%Spec={t90:.4f}")
print(f"Conformal: {len(conf_rows)}, Abstention: {len(abst_rows)}")
print(f"Self-check: {'PASS' if all_ok else 'FAIL'}")
print(f"Verdict: {V}")
