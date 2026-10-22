#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PREDICTIONS = ROOT / "results" / "predictions" / "prediction_scores_388.csv"
THRESHOLDS = ROOT / "tables_archive" / "low_risk_threshold.csv"
SUMMARY = ROOT / "results" / "recomputed_triage.csv"

SPLITS = ["scaffold", "external_dilist", "temporal", "cluster_out"]

HIGH_LABEL = "High-score high-risk"
LOW_LABEL = "Low-score candidate group (not automatic low-risk exclusion)"
NO_CALL_LABEL = "Uncertain / no-call"
OOD_LABEL = "OOD"


def load_low_thresholds():
    table = pd.read_csv(THRESHOLDS, skiprows=1)
    table = table[table["Threshold rule"] == "th_sens95"]
    return {(row.Split, row.Model): float(row.Cutoff) for row in table.itertuples()}


def classify(df, low_thresholds):
    categories, codes, reasons = [], [], []
    for row in df.itertuples(index=False):
        score = float(row.calibrated_score)
        high_cut = float(row.high_risk_threshold)
        source = str(row.threshold_source_split)
        low_cut = low_thresholds.get((source, str(row.model)))
        if low_cut is None:
            low_cut = low_thresholds.get((str(row.split_name), str(row.model)))

        ood = bool(row.ood_flag)
        both = str(row.conformal_set_alpha_0_10) == "both"
        high_u = bool(row.high_uncertainty_flag)
        intermediate = low_cut <= score < high_cut

        if ood:
            categories.append(OOD_LABEL)
            codes.append("OOD")
            reasons.append("OOD")
        elif both or high_u or intermediate:
            categories.append(NO_CALL_LABEL)
            codes.append("UNCERTAIN_NO_CALL")
            note = []
            if both:
                note.append("conformal_both")
            if high_u:
                note.append("high_uncertainty")
            if intermediate:
                note.append("intermediate_score")
            reasons.append(";".join(note))
        elif score >= high_cut:
            categories.append(HIGH_LABEL)
            codes.append("HS_HR")
            reasons.append("none")
        elif score < low_cut and str(row.conformal_set_alpha_0_10) == "negative":
            categories.append(LOW_LABEL)
            codes.append("LS_CANDIDATE")
            reasons.append("none")
        else:
            categories.append(NO_CALL_LABEL)
            codes.append("UNCERTAIN_NO_CALL")
            reasons.append("indeterminate_score")

    out = df.copy()
    out["triage_category"] = categories
    out["triage_category_code"] = codes
    out["no_call_reasons"] = reasons
    return out


def summarize(df):
    rows = []
    for split in SPLITS:
        for model in ["HistGB", "ET"]:
            sub = df[(df.split_name == split) & (df.model == model)]
            if sub.empty:
                continue
            high = int((sub.triage_category == HIGH_LABEL).sum())
            low = int((sub.triage_category == LOW_LABEL).sum())
            ood = int((sub.triage_category == OOD_LABEL).sum())
            no_call = int(sub.triage_category.isin([NO_CALL_LABEL, OOD_LABEL]).sum())
            rows.append(dict(split=split, model=model, n=len(sub), high=high,
                             low=low, no_call=no_call, ood=ood))
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--update-predictions", action="store_true")
    args = parser.parse_args()

    df = pd.read_csv(PREDICTIONS)
    df = classify(df, load_low_thresholds())
    summary = summarize(df)
    print(summary.to_string(index=False))
    summary.to_csv(SUMMARY, index=False)

    histgb = summary[summary.model == "HistGB"]
    low = df[(df.model == "HistGB") & (df.triage_category == LOW_LABEL)
              & (df.split_name.isin(SPLITS))]
    print(f"\nHistGB totals: high={int(histgb.high.sum())} "
          f"low={int(histgb.low.sum())} no-call={int(histgb.no_call.sum())}")
    print(f"Low-score tier: {len(low)} compounds, {int(low.y_true.sum())} DILI-positive")
    external = low[low.split_name == "external_dilist"]
    print(f"External low-score tier: {len(external)} compounds, "
          f"{int(external.y_true.sum())} DILI-positive")

    if args.update_predictions:
        df.to_csv(PREDICTIONS, index=False)
        print(f"Updated {PREDICTIONS}")


if __name__ == "__main__":
    main()
