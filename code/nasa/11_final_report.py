
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (CALIBRATORS, CURATED, FIGURES, MATRICES, PREDICTIONS, REPORTS,
                    ROOT, TABLES)
from common import (ENDPOINT, ENDPOINT_DESCRIPTIONS,
                    SLUG)
from common import SPLITS as SPLITS_DIR

SPLITS = ["scaffold", "cluster_out"]
MODELS = ["HistGB", "ET"]


def md_table(df, cols, headers=None, float_fmt=None):
    headers = headers or cols
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    for _, r in df.iterrows():
        vals = []
        for c in cols:
            v = r[c]
            if float_fmt and isinstance(v, float):
                v = float_fmt.format(v)
            vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return lines


def ci(lo, hi):
    return f"{lo} to {hi}"


def main():
    curated = pd.read_csv(CURATED)
    manifest = pd.read_csv(MATRICES / "matrix_manifest.csv")
    _tp = manifest[manifest["partition"] == "test"]["n_positives"]
    test_pos_lo, test_pos_hi = int(_tp.min()), int(_tp.max())
    t1 = pd.read_csv(TABLES / "task1_ranking.csv")
    t2 = pd.read_csv(TABLES / "task2_probability_reliability.csv")
    t3 = pd.read_csv(TABLES / "task3_decision_utility.csv")
    conf = pd.read_csv(TABLES / "conformal_coverage.csv")
    tri = pd.read_csv(TABLES / "triage_summary.csv")
    unc = pd.read_csv(TABLES / "uncertainty_error_discrimination.csv")
    ad = pd.read_csv(TABLES / "ad_tanimoto_summary.csv")
    thr = pd.read_csv(TABLES / "threshold_lock_table.csv")

    n = len(curated)
    n_pos = int(curated["label"].sum())
    n_neg = n - n_pos

    L = []
    A = L.append
    A("# Generalization to an independent non-hepatic toxicity endpoint")
    A("")
    A(f"## Tox21 {ENDPOINT} independent validation of the DILI-TRUST reliability framework")
    A("")
    A("This report evaluates whether the DILI-TRUST three-task reliability")
    A("assessment (ranking / probability reliability / decision utility) transfers")
    A("unchanged from DILI prediction to a chemically unrelated Tox21 toxicity")
    A("endpoint. No DILI analysis script was modified, no model")
    A("architecture was changed, and no new algorithm was introduced: the feature")
    A("block, the two models, the calibration protocol, the conformal protocol, the")
    A("uncertainty score and the decision policy are reused verbatim.")
    A("")
    A("---")
    A("")
    A("## 1. Data")
    A("")
    A("| item | value |")
    A("|---|---|")
    A(f"| Source | Tox21 10k library (`tox21.csv`), {ENDPOINT} assay column |")
    A(f"| Endpoint | {ENDPOINT} ({ENDPOINT_DESCRIPTIONS.get(ENDPOINT, 'assay')}) |")
    A(f"| Curated compounds | {n} |")
    A(f"| Active (label = 1) | {n_pos} |")
    A(f"| Inactive (label = 0) | {n_neg} |")
    A(f"| Prevalence | {n_pos / n:.4f} |")
    A("| Retained fields | `compound_id` (mol_id), `smiles`, `label` |")
    A("")
    A(f"Compounds with a missing {ENDPOINT} label were dropped, and labels were mapped")
    A("active = 1 / inactive = 0, producing the same `compound_id`, `smiles`,")
    A("`label` contract the DILI pipeline consumes.")
    A("")
    A("## 2. Splits")
    A("")
    A("| split | partition | n | positives | negatives | prevalence |")
    A("|---|---|---|---|---|---|")
    for split in SPLITS:
        for _, r in manifest[manifest["split"] == split].iterrows():
            A(f"| {split} | {r['partition']} | {r['n_samples']} | {r['n_positives']} |"
              f" {r['n_samples'] - r['n_positives']} | {r['n_positives'] / r['n_samples']:.4f} |")
    A("")
    A("- **Scaffold split** (Bemis-Murcko groups, greedy 70/15/15): the internal")
    A("  held-out design, directly analogous to the DILI scaffold split.")
    A("- **Cluster-out split** (Butina on Morgan r=2 2048 bit, greedy 70/15/15):")
    A("  the chemically disjoint held-out design. Its test partition is used as the")
    A("  external-like evaluation cohort that the DILI-TRUST external evaluation")
    A("  interface consumes.")
    A(f"- **No temporal split.** Tox21 {ENDPOINT} carries no time annotation, so a")
    A("  temporal split was not fabricated.")
    A("")
    A("### Applicability domain of the held-out cohorts")
    A("")
    A("| split | n train | n held-out | median max Tanimoto | Q1 | Q3 | n < 0.3 | n < 0.15 | scaffold overlap |")
    A("|---|---|---|---|---|---|---|---|---|")
    for _, r in ad.iterrows():
        A(f"| {r['split']} | {r['n_train']} | {r['n_test']} | {r['max_tanimoto_median']} |"
          f" {r['max_tanimoto_q25']} | {r['max_tanimoto_q75']} | {r['n_ad_warning_lt0.3']} |"
          f" {r['n_ood_extreme_lt0.15']} | {r['overlapping_scaffolds']}/{r['test_scaffolds_unique']} |")
    A("")
    A("## 3. Features and models (unchanged from DILI-TRUST)")
    A("")
    A("| item | value |")
    A("|---|---|")
    A("| RDKit 2D descriptors | non-constant subset of `Descriptors._descList` |")
    A("| Morgan fingerprint | radius 2, 2048 bits |")
    A("| MACCS keys | 167 bits |")
    A("| Feature block | 215 continuous + 2048 Morgan + 167 MACCS = 2430 |")
    A("| Preprocessing | median imputation + standardisation fitted on train only |")
    A("| Primary model | `HistGradientBoostingClassifier(learning_rate=0.05, max_iter=200, max_leaf_nodes=31)` |")
    A("| Robustness model | `ExtraTreesClassifier(n_estimators=200, max_depth=15, min_samples_leaf=3, class_weight='balanced')` |")
    A("| Seeds | 20260521-20260525, scores averaged over the five seeds |")
    A("")
    A("## 4. Parameter locking")
    A("")
    A("| parameter | locked on | value |")
    A("|---|---|---|")
    for _, r in thr.iterrows():
        A(f"| calibration method | validation ({r['split']}/{r['model']}) | {r['calibration_method']} |")
    for _, r in thr.iterrows():
        A(f"| th@95% specificity, th@95% sensitivity, U80 ({r['split']}/{r['model']}) |"
          f" validation | {r['th_spec95']}, {r['th_sens95']}, {r['u80_threshold']} |")
    A("")
    A("Conformal q-hat is computed on the validation nonconformity scores at")
    A("quantile ceil((n+1)(1-alpha))/n and applied unchanged to the held-out")
    A("cohort. The held-out partitions never influenced a calibration method, a")
    A("threshold, a q-hat, or an uncertainty cut-off.")
    A("")
    A("---")
    A("")
    A("## 5. Task I - Ranking")
    A("")
    A("Held-out cohorts (bootstrap 95% CI, 2000 resamples).")
    A("")
    A("| split | model | score | n | positives | AUROC (95% CI) | AUPRC (95% CI) |")
    A("|---|---|---|---|---|---|---|")
    for _, r in t1[t1["partition"] == "test"].iterrows():
        A(f"| {r['split']} | {r['model']} | {r['score']} | {r['n']} | {r['positives']} |"
          f" {r['auroc']} ({ci(r['auroc_ci_lower'], r['auroc_ci_upper'])}) |"
          f" {r['auprc']} ({ci(r['auprc_ci_lower'], r['auprc_ci_upper'])}) |")
    A("")
    A("## 6. Task II - Probability reliability")
    A("")
    A("| split | model | Brier (95% CI) | Brier skill score | bias | intercept | slope | ECE(q5) | NLL |")
    A("|---|---|---|---|---|---|---|---|---|")
    for _, r in t2[t2["partition"] == "test"].iterrows():
        A(f"| {r['split']} | {r['model']} | {r['brier']} ({ci(r['brier_ci_lower'], r['brier_ci_upper'])}) |"
          f" {r['brier_skill_score']} | {r['calibration_bias']} | {r['calibration_intercept']} |"
          f" {r['calibration_slope']} | {r['ece_q5']} | {r['nll']} |")
    A("")
    A("Calibration intercept 0 and slope 1 indicate perfect calibration; a slope")
    A("below 1 indicates over-confident probabilities, a negative intercept a")
    A("systematic over-prediction of the positive class.")
    A("")
    A("## 6b. Calibration-collapse verification")
    A("")
    A("Pooled held-out calibration is intact, but a collapse is present once the")
    A("cohort is decomposed. Uncalibrated ExtraTrees scores already fail on the")
    A("held-out cohorts, and after calibration the remote chemical bin of the")
    A("cluster-out cohort still fails.")
    A("")
    try:
        df_a = pd.read_csv(TABLES / "calibration_raw_vs_calibrated.csv")
        df_b = pd.read_csv(TABLES / "calibration_by_chemical_distance.csv")
        df_c = pd.read_csv(TABLES / "calibration_prior_shift.csv")
        A("Raw versus locked calibrated:")
        A("")
        A("| split | model | score | BSS | bias | intercept | slope |")
        A("|---|---|---|---|---|---|---|")
        for _, r in df_a.iterrows():
            A(f"| {r['split']} | {r['model']} | {r['score']} | {r['brier_skill_score']} |"
              f" {r['calibration_bias']} | {r['calibration_intercept']} | {r['calibration_slope']} |")
        A("")
        A("By chemical distance (calibrated scores):")
        A("")
        A("| split | model | Tanimoto bin | n | positives | BSS | bias | intercept | slope |")
        A("|---|---|---|---|---|---|---|---|---|")
        for _, r in df_b.iterrows():
            A(f"| {r['split']} | {r['model']} | {r['tanimoto_bin']} | {r['n']} | {r['positives']} |"
              f" {r['brier_skill_score']} | {r['calibration_bias']} |"
              f" {r['calibration_intercept']} | {r['calibration_slope']} |")
        A("")
        A("Prior-shift stress test (locked map, reweighted evaluation cohort):")
        A("")
        A("| split | model | target prevalence | achieved | BSS | bias | slope |")
        A("|---|---|---|---|---|---|---|")
        for _, r in df_c.iterrows():
            A(f"| {r['split']} | {r['model']} | {r['target_prevalence']} | {r['prevalence']} |"
              f" {r['brier_skill_score']} | {r['calibration_bias']} | {r['calibration_slope']} |")
    except FileNotFoundError:
        A("(run `12_calibration_collapse_check.py` to populate this section)")
    A("")
    A("## 7. Task III - Decision utility")
    A("")
    A("### 7.1 Conformal coverage (validation-locked q-hat)")
    A("")
    A("| split | model | alpha | partition | q-hat | target | observed | set size | single | both | empty | pos cov | neg cov |")
    A("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    cf = conf[conf["mode"] == "validation_locked"]
    for _, r in cf.iterrows():
        A(f"| {r['split']} | {r['model']} | {r['alpha']} | {r['evaluation_partition']} |"
          f" {r['q_hat']} | {r['target_coverage']} | {r['observed_coverage']} | {r['avg_set_size']} |"
          f" {r['single_rate']} | {r['both_rate']} | {r['empty_rate']} | {r['positive_coverage']} |"
          f" {r['negative_coverage']} |")
    A("")
    A("### 7.2 Uncertainty and domain guards")
    A("")
    A("| split | model | partition | no-call (U80) | conformal both | conformal empty | OOD < 0.15 | AD warning < 0.3 |")
    A("|---|---|---|---|---|---|---|---|")
    for _, r in t3.iterrows():
        A(f"| {r['split']} | {r['model']} | {r['partition']} | {r['no_call_rate_u80']} |"
          f" {r['conformal_both_rate']} | {r['conformal_empty_rate']} | {r['ood_extreme_rate']} |"
          f" {r['ad_warning_rate_lt0.3']} |")
    A("")
    A("### 7.3 Fixed-threshold triage (held-out cohorts)")
    A("")
    A("| split | model | high-risk n (%) | precision | recall (positives) | low-score n (%) | intermediate n (%) | no-call n (%) |")
    A("|---|---|---|---|---|---|---|---|")
    for _, r in tri.iterrows():
        A(f"| {r['split']} | {r['model']} | {r['high_risk_n']} ({r['high_risk_rate']}) |"
          f" {r['high_risk_precision']} | {r['high_risk_recall']} |"
          f" {r['low_score_n']} ({r['low_score_rate']}) |"
          f" {r['intermediate_n']} ({r['intermediate_rate']}) |"
          f" {r['no_call_n']} ({r['no_call_rate']}) |")
    A("")
    A("### 7.4 Does the uncertainty score identify uncertain predictions?")
    A("")
    A("| split | model | partition | error rate | AUROC(U -> error) | error rate retained | error rate abstained |")
    A("|---|---|---|---|---|---|---|")
    for _, r in unc.iterrows():
        A(f"| {r['split']} | {r['model']} | {r['partition']} | {r['error_rate']} |"
          f" {r['auroc_u_for_error']} | {r['error_rate_retained_u80']} | {r['error_rate_abstained_u80']} |")
    A("")
    A("---")
    A("")
    A("## 8. Correspondence with the DILI-TRUST analysis")
    A("")
    A(f"| DILI-TRUST element | Tox21 {ENDPOINT} analogue |")
    A("|---|---|")
    A(f"| DILIrank-based modeling base | Tox21 {ENDPOINT} curated actives/inactives |")
    A("| Scaffold split (train/valid/test) | `scaffold_split.csv`, identical greedy procedure |")
    A("| Cluster-out split | `cluster_out_split.csv`, Butina clustering + greedy 70/15/15 |")
    A("| External DILIst cohort | cluster-out held-out cohort (chemically disjoint) |")
    A("| Post-2021 lockbox | not applicable (no time axis) |")
    A("| RDKit + Morgan + MACCS matrix | `matrices/<split>/model_ready_<split>_<partition>.csv` |")
    A("| HistGB primary / ET robustness | `05_train_predict.py`, identical hyper-parameters |")
    A("| Stage 8 calibration lock | `06_calibrate_and_lock.py` -> `threshold_lock_table.csv` |")
    A("| Stage 8 U(p) = H(p) + 0.5 M(p), U80 rule | `task3_decision_utility.csv` |")
    A("| Stage 9 conformal (validation-locked q-hat) | `conformal_coverage.csv` |")
    A("| Stage 9 bootstrap CI (2000 resamples) | Task I and Task II tables |")
    A("| Stage 10 conservative decision policy | `triage_table.csv`, `triage_summary.csv` |")
    A("")
    A("## 9. Output inventory")
    A("")
    A("| artefact | path |")
    A("|---|---|")
    for label, p in [
        ("Curated data", CURATED),
        ("Splits", SPLITS_DIR),
        ("Features", ROOT / "features"),
        ("Model-ready matrices", MATRICES),
        ("Predictions and conformal sets", PREDICTIONS),
        ("Locked calibrators", CALIBRATORS),
        ("Tables", TABLES),
        ("Figures", FIGURES),
        ("Reports", REPORTS),
    ]:
        A(f"| {label} | `{p}` |")
    A("")
    A("### Tables")
    A("")
    table_desc = {
        "Table_S1": "data and split statistics (DILI Table S2 analogue)",
        "Table_S2": "ranking metrics with bootstrap CI",
        "Table_S3": "probability reliability metrics",
        "Table_S4": "conformal coverage, marginal and class conditional (DILI Table S7 analogue)",
        "Table_S5": "fixed-threshold triage summary (DILI Table S19 analogue)",
        "Table_S6": "bootstrap CI in long form (DILI Table S8 analogue)",
        "Table_S7": "Tanimoto-stratified held-out performance (DILI Table S33 analogue)",
        "Table_S8": "Tanimoto distance summary (DILI Table S34 analogue)",
        "Table_S9": "locked thresholds and uncertainty cut-off (DILI Table S10 analogue)",
    }
    for f in sorted(TABLES.glob("*.csv")):
        desc = table_desc.get(f.stem)
        suffix = f" - {desc}" if desc else ""
        A(f"- `tables/{f.name}`{suffix}")
    A("")
    A("### Figures")
    A("")
    fig_desc = {
        "Figure_SA_ranking": "ROC and precision-recall on the held-out cohorts",
        "Figure_SB_reliability": "reliability curves and Cox calibration on the held-out cohorts",
        "Figure_SC_conformal": "conformal coverage vs target, and class-conditional coverage",
        "Figure_SD_uncertainty_abstention": "ranking and error rate as a function of abstention",
        "Figure_SE_applicability_domain": "distance to the development set and AUROC by similarity bin (DILI Figure S4 analogue)",
        "Figure_SF_triage": "triage composition and high-risk group performance",
        "Figure_SG_calibration_collapse": "calibration-collapse verification: pooled, by chemical distance, and under prior shift",
    }
    for f in sorted(FIGURES.glob("*.pdf")):
        desc = fig_desc.get(f.stem)
        suffix = f" - {desc}" if desc else ""
        A(f"- `figures/{f.name}`{suffix}")
    A("")
    A("## 10. Limitations")
    A("")
    A(f"- Tox21 {ENDPOINT} is imbalanced ({(n_pos / n) * 100:.1f}% actives); the held-out")
    A(f"  partitions contain {test_pos_lo}-{test_pos_hi} positives, so held-out")
    A("  confidence intervals are wide and stratified-by-domain bins are small.")
    A(f"- No second database exists for {ENDPOINT}, so 'external' here means chemically")
    A("  disjoint (cluster-out), not a different data source.")
    A("- The Butina cutoff of the DILI cluster-out procedure (10th percentile of")
    A("  pairwise distances) is not transferable to the much more diverse Tox21")
    A("  library; a fixed Tanimoto similarity threshold of 0.30 was used instead,")
    A("  documented in `reports/stage02_split_qc.md`.")
    A("- The DILI release stores a conservative `near_zero_locked_threshold` whose")
    A("  computation is not present in the released scripts; the validation-locked")
    A("  95%-sensitivity threshold is used as the low-score cut instead.")
    A("")

    out = REPORTS / f"FINAL_REPORT_tox21_{SLUG}_validation.md"
    out.write_text("\n".join(L), encoding="utf-8")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
