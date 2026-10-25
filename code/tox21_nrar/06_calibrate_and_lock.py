
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (CALIBRATORS, PREDICTIONS, REPORTS, TABLES, ensure_dirs,
                    ece, nll, uncertainty_score)

from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_curve
from sklearn.model_selection import StratifiedKFold

SPLITS = ["scaffold", "cluster_out"]
MODELS = ["HistGB", "ET"]
METHODS = ["identity", "platt", "isotonic", "temperature"]
TEMPERATURES = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0]
U80_PERCENTILE = 80.0


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def apply_temperature(scores, T):
    return 1.0 / (1.0 + np.exp(-logit(scores) / T))


def fit_temperature(scores, y):
    best_T, best_brier = 1.0, np.inf
    for T in TEMPERATURES:
        b = brier_score_loss(y, apply_temperature(scores, T))
        if b < best_brier:
            best_brier, best_T = b, T
    return best_T


def fit_calibrator(method, scores, y):
    if method == "identity":
        return None
    if method == "platt":
        m = LogisticRegression(C=1.0, max_iter=5000)
        m.fit(logit(scores).reshape(-1, 1), y)
        return m
    if method == "isotonic":
        iso = IsotonicRegression(out_of_bounds="clip", increasing=True)
        iso.fit(scores, y)
        return iso
    if method == "temperature":
        return {"T": fit_temperature(scores, y), "method": "temperature"}
    raise ValueError(method)


def apply_calibrator(method, calibrator, scores):
    if method == "identity":
        return scores
    if method == "platt":
        return calibrator.predict_proba(logit(scores).reshape(-1, 1))[:, 1]
    if method == "isotonic":
        return calibrator.transform(scores)
    if method == "temperature":
        return apply_temperature(scores, calibrator["T"])
    raise ValueError(method)


def oof_scores(method, scores, y, seed=20260521):
    oof = np.zeros(len(y), dtype=float)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for tr, vl in skf.split(np.zeros(len(y)), y):
        cal = fit_calibrator(method, scores[tr], y[tr])
        oof[vl] = apply_calibrator(method, cal, scores[vl])
    return oof


def thresholds(yt, cs):
    fpr, tpr, ths = roc_curve(yt, cs)
    out = {}
    idx = np.where(1 - fpr >= 0.95)[0]
    out["th_spec95"] = round(float(ths[idx[-1]]), 4) if len(idx) else 0.5
    out["sens_at_spec95"] = round(float(tpr[idx[-1]]), 4) if len(idx) else 0.0
    idx = np.where(tpr >= 0.95)[0]
    out["th_sens95"] = round(float(ths[idx[0]]), 4) if len(idx) else 0.5
    out["spec_at_sens95"] = round(float(1 - fpr[idx[0]]), 4) if len(idx) else 0.0
    j = tpr - fpr
    best = int(np.argmax(j))
    out["th_youden"] = round(float(ths[best]), 4)
    out["sens_at_youden"] = round(float(tpr[best]), 4)
    out["spec_at_youden"] = round(float(1 - fpr[best]), 4)
    return out


def main():
    ensure_dirs()
    oof_rows, th_rows = [], []
    summary = {}

    for split in SPLITS:
        for model_name in MODELS:
            valid = pd.read_csv(PREDICTIONS / f"{split}_{model_name}_ensemble_valid_predictions.csv")
            test = pd.read_csv(PREDICTIONS / f"{split}_{model_name}_ensemble_test_predictions.csv")
            yv = valid["y_true"].values.astype(int)
            sv = valid["y_score"].values.astype(float)
            yt = test["y_true"].values.astype(int)
            st = test["y_score"].values.astype(float)

            best = None
            for method in METHODS:
                oof = oof_scores(method, sv, yv)
                row = {
                    "split": split, "model": model_name, "method": method,
                    "brier": round(float(brier_score_loss(yv, oof)), 4),
                    "ece_q5": ece(yv, oof, 5),
                    "nll": nll(yv, oof),
                }
                row["rank_score"] = round(row["brier"] + 0.5 * row["ece_q5"] + 0.1 * row["nll"], 4)
                oof_rows.append(row)
                if best is None or row["rank_score"] < best["rank_score"]:
                    best = row
            selected = best["method"]
            print(f"{split}/{model_name}: selected calibrator = {selected} "
                  f"(rank={best['rank_score']})")

            calibrator = fit_calibrator(selected, sv, yv)
            joblib.dump({"method": selected, "calibrator": calibrator},
                        CALIBRATORS / f"{split}_{model_name}_locked_calibrator.joblib")

            cv = np.clip(apply_calibrator(selected, calibrator, sv), 0.0, 1.0)
            ct = np.clip(apply_calibrator(selected, calibrator, st), 0.0, 1.0)

            for part, df, scores, y, cal in [
                ("valid", valid, sv, yv, cv), ("test", test, st, yt, ct)]:
                pd.DataFrame({
                    "sample_id": df["sample_id"].values,
                    "y_true": y, "raw_score": np.round(scores, 6),
                    "calibrated_score": np.round(cal, 6),
                    "y_pred_0p5": (cal >= 0.5).astype(int),
                    "y_score_std": df["y_score_std"].values,
                    "seed_pos_vote_frac": df["seed_pos_vote_frac"].values,
                    "split": split, "model": model_name,
                    "calibration_method": selected,
                    "partition": part,
                }).to_csv(PREDICTIONS / f"{split}_{model_name}_locked_calibrated_{part}.csv",
                          index=False)

            th = thresholds(yv, cv)
            u = uncertainty_score(cv)
            th["u80_threshold"] = round(float(np.percentile(u, U80_PERCENTILE)), 6)
            n_pos = int(yv.sum())
            th_rows.append({
                "split": split, "model": model_name, "calibration_method": selected,
                "n_valid": len(yv), "n_pos": n_pos, "n_neg": len(yv) - n_pos,
                "prevalence": round(n_pos / len(yv), 4),
                "threshold_source": "validation partition only",
                **th,
            })
            summary[f"{split}_{model_name}"] = {
                "calibration_method": selected,
                "thresholds": th,
                "oof_ranking": {k: v for k, v in best.items() if k not in ("split", "model")},
            }

    pd.DataFrame(oof_rows).to_csv(TABLES / "calibration_oof_method_selection.csv", index=False)
    pd.DataFrame(th_rows).to_csv(TABLES / "threshold_lock_table.csv", index=False)
    with open(REPORTS / "stage06_calibration_lock.json", "w") as f:
        json.dump(summary, f, indent=2)

    lines = [
        "# Stage 06 - calibration and threshold lock",
        "",
        "## Selected calibrators (OOF ranking on validation)",
        "",
        "| split | model | method | OOF brier | OOF ECE(q5) | OOF NLL | rank |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in oof_rows:
        if r["method"] == summary[f"{r['split']}_{r['model']}"]["calibration_method"]:
            lines.append(f"| {r['split']} | {r['model']} | {r['method']} | {r['brier']} |"
                         f" {r['ece_q5']} | {r['nll']} | {r['rank_score']} |")
    lines += [
        "",
        "## Locked thresholds (validation partition only)",
        "",
        "| split | model | th@95%spec | sens@95%spec | th@95%sens | spec@95%sens | th_youden | U80 threshold |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in th_rows:
        lines.append(f"| {r['split']} | {r['model']} | {r['th_spec95']} | {r['sens_at_spec95']} |"
                     f" {r['th_sens95']} | {r['spec_at_sens95']} | {r['th_youden']} |"
                     f" {r['u80_threshold']} |")
    lines += [
        "",
        "All calibrators, thresholds and the uncertainty cut-off were fixed on the",
        "validation partition. The test partitions were never used to select a",
        "calibration method, a threshold, or an uncertainty cut-off.",
        "",
    ]
    (REPORTS / "stage06_calibration_lock_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("saved calibration + threshold lock tables")


if __name__ == "__main__":
    main()
