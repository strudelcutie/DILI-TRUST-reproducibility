# DILI-TRUST reproducibility package

Code, curated input data, and final result tables for the manuscript
"One score, three decisions: why ranking well does not mean usable risk or justified exclusion".

The study asks whether a single screening score can support three different
decisions — ranking candidates, estimating risk, and excluding low-score cases — and evaluates this question in drug-induced liver injury (DILI) data, two Tox21
assays, and four NASA software-defect projects.

## Contents

```
code/dili/          DILI pipeline (splits, features, models, calibration, evaluation, decisions) and configuration files
code/literature/    literature-model adaptation and reproduction scripts (S1)
code/tox21_nrar/    Tox21 NR-AR validation pipeline (01-09)
code/tox21_nrer/    Tox21 NR-ER validation pipeline (01-09)
code/nasa/          NASA MDP cross-domain pipeline (00-13)
data/dili/          curated labels, compound master, and split assignments
data/tox21/         Tox21 raw assay table (NR-AR, NR-ER, and other endpoints)
data/nasa/          KC1, PC1, JM1, and KC2 ARFF files
results/            final supplementary tables, figure source data, and cross-domain summaries
tables_archive/     detailed source records moved out of the supplement
```

## Data sources

| Dataset | Source | Notes |
|---|---|---|
| DILIrank 2.0 | US FDA | Task A subset, 818 drugs |
| DILIst | FDA / NCBI | curated to 388 small-molecule external compounds |
| Tox21 | MoleculeNet (Wu et al., 2018) | NR-AR 7,432 and NR-ER 6,309 curated compounds |
| NASA MDP | PROMISE repository (Sayyad Shirabad and Menzies, 2005) | KC1 2,109; PC1 1,109; JM1 10,885; KC2 522 modules |

Curated extracts used in the analysis are included. Raw FDA label downloads and
third-party repository mirrors are not redistributed; they can be regenerated
from the sources above.
The external split file in data/dili/dilist_strict_dilirank2_nonoverlap_external_split.csv lists 399 candidates before the small-molecule eligibility filter; the final 388-compound cohort applies that filter (Supplementary Table S15).
## Environment

Python 3.12, with RDKit. Install dependencies with:

pip install -r requirements.txt

The literature-model adaptation code additionally used DGL and PyTorch; those
environments are not pinned here because they were specific to the original
model repositories.

## What can be reproduced

1. Recompute the main-text DILI tables. Run
   python code/reproduce_main_tables.py
   This reads `results/predictions/prediction_scores_388.csv` 
   (calibrated scores for the 388-compound external cohort and the three internal
   settings) and recomputes Table 1 (AUROC, AUPRC, Brier, BSS), Table 2 (bias,
   null Brier, ECE, calibration intercept and slope), Table 3 (conformal coverage
   and set composition), and Table 4 (high-score precision with Wilson intervals,
   recall, enrichment). It also recomputes the low-score tier (conformal-negative
   and below the locked 95%-sensitivity threshold): 56 low-score records (55 unique compounds) across the four
   DILI settings, 25 DILI-positive; external DILIst 38, of which 21 positive.
   Values match the manuscript to rounding; small differences can arise across
   library versions.
2. Inspect final results. All manuscript tables are in
  `results/Data S1.xlsx`; the cross-domain summaries are
  provided as CSV files in `results/`, and detailed source records (cohort
  construction, feature importance, label-definition mapping, threshold
  sensitivity, split attribution, and training log) are in `tables_archive/`.
3. Rebuild features and models (DILI). Scripts in `code/dili/` construct
   splits, generate the 2,427-feature block with RDKit, train HistGB/ExtraTrees,
   fit calibration, and evaluate under the locked protocol.
4. Re-run decision and triage analyses. `triage.py` recomputes the three-group triage used in the manuscript
   (122 high-score, 56 low-score, 604 no-call) from the frozen predictions; `screening_simulation.py` reads
   calibrated predictions and produces the screening tables.
5. Re-run cross-domain validation. `code/tox21_nrar/`, `code/tox21_nrer/`,
   and `code/nasa/` reproduce the Tox21 and NASA analyses from the included raw
   data.

## Reproduction levels

1. Recompute reported metrics from the released predictions (`code/reproduce_main_tables.py`, `code/dili/triage.py`).
2. Rebuild features and retrain models from the included curated inputs (`code/dili/`, `code/tox21_*`, `code/nasa/`).
3. Rebuild inputs from public sources; raw downloads are not redistributed, and the retrieval and curation records are in `tables_archive/` and Supplementary Table S15.

## Traceability

| Step | Source |
|---|---|---|
| Compound records | DILIrank 2.0, DILIst|
| Structures and labels | PubChem, LiverTox |
| Splits | Bemis-Murcko scaffold, Butina clusters |
| Features | RDKit 2D, Morgan, MACCS | 
| Calibration | validation partitions | 
| Thresholds | validation partitions |
| Predictions | released CSV | results/predictions |

The ablation script (`code/dili/ablation.py`) and the literature-model scripts (`code/literature/`) are included, but they require the regenerated feature matrices and, for the literature models, the original repositories. The label-definition sensitivity (Most-DILI versus No-DILI, Less-DILI removed) can be rerun with `code/dili/label_sensitivity.py`; the recorded run is in `tables_archive/label_sensitivity_recorded.csv`. The script reproduces the direction of the result (AUROC 0.665/0.703/0.768 versus 0.669/0.702/0.753 recorded).

The scripts were written for the study's working tree (`data/`, `features/`,
`results/` at the repository root) and are provided as analysis records. Paths
may need adjustment when running them in a different layout; intermediate
matrices and predictions are regenerated by the pipeline rather than shipped.
The appendix tables are outputs of this pipeline, not independent scripts: a
full regeneration of every supplementary table requires running the pipeline and
re-fitting the models, and small numerical differences can arise across RDKit,
scikit-learn, and BLAS versions.

## Figures

The frozen tables from which Figures 2–5 were plotted are in
`results/figure_data/`. `Figure_5_source.csv` is the assembled per-setting
matrix plotted in Figure 5 (AUROC, BSS, calibration slope, and low-score
enrichment for all 12 settings); `dili/`, `tox21/`, and `nasa/` hold the
underlying metric, conformal, prediction, and triage tables. Figure files
themselves and the plotting code are not distributed with this repository.

## Key numbers

- DILI external cohort: 388 compounds (286 positives, 102 negatives), AUROC 0.606,
  Brier skill score −0.173.
- Feature block: 2,427 features = 212 RDKit 2D descriptors + 2,048 Morgan bits
  + 167 MACCS keys.
- Tox21: 7,258 unique NR-AR and 6,186 unique NR-ER structures; 6,108 shared.
- NASA: KC1 development project (train 1,475 / valid 317 / test 317); PC1, JM1,
  and KC2 used as external projects without retuning.

## License

MIT. See `LICENSE`.

## Citation

If you use this code or data, please cite the manuscript above. Dataset-specific
citations are listed in the manuscript's reference list (DILIrank 2.0, DILIst,
MoleculeNet/Tox21, and the PROMISE repository).
