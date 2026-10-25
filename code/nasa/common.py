
import os

import numpy as np
from pathlib import Path

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    roc_curve,
    confusion_matrix,
    balanced_accuracy_score,
    log_loss,
)

ROOT = Path(__file__).resolve().parents[1]

RAW = ROOT / "raw"





TASKS = {
    "kc1_internal": {
        "label": "NASA KC1 (internal held-out)",
        "shift": f"same project, held-out modules (prevalence {0.1546:.1%})",
    },
    "kc1_pc1": {
        "label": "NASA PC1 (external project)",
        "shift": "cross-project; external prevalence 6.9% (down from 15.5%)",
    },
    "kc1_jm1": {
        "label": "NASA JM1 (external project)",
        "shift": "cross-project; external prevalence 19.4% (up from 15.5%)",
    },
    "kc1_kc2": {
        "label": "NASA KC2 (external project)",
        "shift": "cross-project; external prevalence 20.5% (up from 15.5%)",
    },
}
SPLIT_NAMES = list(TASKS)

DATA = ROOT / "data"
SPLITS = ROOT / "splits"
FEATURES = ROOT / "features"
MATRICES = ROOT / "matrices"
PREDICTIONS = ROOT / "predictions"
CALIBRATORS = ROOT / "calibrators"
TABLES = ROOT / "tables"
FIGURES = ROOT / "figures"
REPORTS = ROOT / "reports"

SEEDS = [20260521, 20260522, 20260523, 20260524, 20260525]
ALPHAS = [0.05, 0.10, 0.20]
PRIMARY_ALPHA = 0.10
N_BOOT = 2000

COLORS = {"HistGB": "#0072B2", "ET": "#009E73"}
SPLIT_LABELS = {k: v["label"] for k, v in TASKS.items()}
SPLIT_COLORS = {"kc1_internal": "#666666", "kc1_pc1": "#D55E00",
                "kc1_jm1": "#0072B2", "kc1_kc2": "#009E73"}


def ensure_dirs():
    for d in [DATA, SPLITS, FEATURES, MATRICES, PREDICTIONS,
              CALIBRATORS, TABLES, FIGURES, REPORTS]:
        d.mkdir(parents=True, exist_ok=True)


def ece(yt, ys, nb=10):
    bins = np.linspace(0, 1, nb + 1)
    e = 0.0
    for i in range(nb):
        m = (ys >= bins[i]) & (ys < bins[i + 1])
        if m.sum() > 0:
            e += (m.sum() / max(1, len(yt))) * abs(yt[m].mean() - ys[m].mean())
    return round(float(e), 4)


def sens_at_spec(yt, ys, spec=0.95):
    fpr, tpr, _ = roc_curve(yt, ys)
    idx = np.where(1 - fpr >= spec)[0]
    return round(float(np.max(tpr[idx])), 4) if len(idx) > 0 else 0.0


def spec_at_sens(yt, ys, sens=0.95):
    fpr, tpr, _ = roc_curve(yt, ys)
    idx = np.where(tpr >= sens)[0]
    return round(float(1 - fpr[idx[0]]), 4) if len(idx) > 0 else 0.0


def nll(yt, ys):
    yc = np.clip(ys, 1e-15, 1 - 1e-15)
    return round(float(-np.mean(yt * np.log(yc) + (1 - yt) * np.log(1 - yc))), 4)


def brier_skill_score(yt, ys):
    bs = brier_score_loss(yt, ys)
    prev = float(np.mean(yt))
    bs_ref = brier_score_loss(yt, np.full(len(yt), prev))
    return round(float(1.0 - bs / bs_ref), 4) if bs_ref > 0 else np.nan


def reliability_curve(yt, ys, nb=10):
    bins = np.linspace(0, 1, nb + 1)
    rows = []
    for i in range(nb):
        m = (ys >= bins[i]) & (ys < bins[i + 1])
        rows.append({
            "bin": i + 1,
            "bin_lower": round(float(bins[i]), 3),
            "bin_upper": round(float(bins[i + 1]), 3),
            "n": int(m.sum()),
            "mean_predicted": round(float(ys[m].mean()), 4) if m.sum() > 0 else np.nan,
            "observed_frequency": round(float(yt[m].mean()), 4) if m.sum() > 0 else np.nan,
        })
    return rows


def calibration_curve_logistic(yt, ys):
    from sklearn.linear_model import LogisticRegression
    yc = np.clip(ys, 1e-6, 1 - 1e-6)
    lg = np.log(yc / (1 - yc)).reshape(-1, 1)
    m = LogisticRegression(C=1e6, max_iter=5000)
    m.fit(lg, yt)
    return round(float(m.intercept_[0]), 4), round(float(m.coef_[0][0]), 4)


def calibration_bias(yt, ys):
    return round(float(np.mean(ys) - np.mean(yt)), 4)


def uncertainty_score(cs):
    p = np.clip(cs, 1e-15, 1 - 1e-15)
    H = -(p * np.log(p) + (1 - p) * np.log(1 - p))
    M = 1 - np.abs(2 * p - 1)
    return H + 0.5 * M


def bootstrap_ci(yt, ys, metric, nboot=N_BOOT, seed=20260521):
    rng = np.random.default_rng(seed)
    n = len(yt)
    vals = []
    for _ in range(nboot):
        idx = rng.choice(n, n, replace=True)
        ytb, ysb = yt[idx], ys[idx]
        if len(set(ytb)) < 2:
            continue
        try:
            if metric == "auroc":
                vals.append(roc_auc_score(ytb, ysb))
            elif metric == "auprc":
                vals.append(average_precision_score(ytb, ysb))
            elif metric == "brier":
                vals.append(brier_score_loss(ytb, ysb))
            elif metric == "ece":
                vals.append(ece(ytb, ysb, 5))
            elif metric == "mcc":
                ypb = (ysb >= 0.5).astype(int)
                tnb, fpb, fnb, tpb = confusion_matrix(ytb, ypb).ravel()
                vals.append(((tpb * tnb) - (fpb * fnb)) /
                            np.sqrt((tpb + fpb) * (tpb + fnb) * (tnb + fpb) * (tnb + fnb) + 1e-10))
            elif metric == "sens95spec":
                vals.append(sens_at_spec(ytb, ysb, 0.95))
            elif metric == "bss":
                vals.append(brier_skill_score(ytb, ysb))
        except Exception:
            continue
    if len(vals) == 0:
        return np.nan, np.nan
    return round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4)


def metric_set(yt, ys, yp=None, nboot=N_BOOT, seed=20260521):
    if yp is None:
        yp = (ys >= 0.5).astype(int)
    yt = np.asarray(yt).astype(int)
    ys = np.asarray(ys, dtype=float)
    n = len(yt)
    p = int(yt.sum())
    ng = n - p
    has_both = (p > 0) and (ng > 0)

    m = {"n": n, "positives": p, "negatives": ng,
         "prevalence": round(p / n, 4) if n > 0 else 0.0}
    m["auroc"] = round(float(roc_auc_score(yt, ys)), 4) if has_both else np.nan
    m["auprc"] = round(float(average_precision_score(yt, ys)), 4) if has_both else np.nan
    m["brier"] = round(float(brier_score_loss(yt, ys)), 4)
    m["brier_skill_score"] = brier_skill_score(yt, ys)
    m["ece_q5"] = ece(yt, ys, 5)
    m["ece_u10"] = ece(yt, ys, 10)
    m["nll"] = nll(yt, ys)
    m["calibration_bias"] = calibration_bias(yt, ys)
    ci, cs = calibration_curve_logistic(yt, ys)
    m["calibration_intercept"] = ci
    m["calibration_slope"] = cs
    tn, fp, fn, tp = confusion_matrix(yt, yp).ravel()
    m["mcc"] = round(float(((tp * tn) - (fp * fn)) /
                           np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) + 1e-10)), 4)
    m["bacc"] = round(float(balanced_accuracy_score(yt, yp)), 4)
    m["sensitivity"] = round(float(tp / (tp + fn + 1e-10)), 4)
    m["specificity"] = round(float(tn / (tn + fp + 1e-10)), 4)
    m["sens95spec"] = sens_at_spec(yt, ys, 0.95) if has_both else np.nan
    m["spec95sens"] = spec_at_sens(yt, ys, 0.95) if has_both else np.nan

    for metric, key in [("auroc", "auroc"), ("auprc", "auprc"), ("brier", "brier"),
                        ("bss", "brier_skill_score"), ("ece", "ece_q5"),
                        ("mcc", "mcc"), ("sens95spec", "sens95spec")]:
        val = m.get(key)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            lo = hi = np.nan
        else:
            lo, hi = bootstrap_ci(yt, ys, metric, nboot, seed)
        m[f"{key}_ci_lower"] = lo
        m[f"{key}_ci_upper"] = hi
    return m
