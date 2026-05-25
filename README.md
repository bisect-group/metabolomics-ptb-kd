# MUDAR-PTB: Metabolomics-based Preterm Birth Prediction

A two-stage ML pipeline for preterm birth (PTB) prediction using longitudinal maternal serum metabolomics data, stratified across trimesters (T1, T2, T3).

**Stage 1** runs an exhaustive search over feature selection methods, class-imbalance strategies, and classifiers.  
**Stage 2** applies Knowledge Distillation (KD) — using the best Stage 1 model as teacher — to transfer predictive signal to models trained on earlier trimester data.

---

## Repository Structure

```
mudar-ptb/
├── notebooks/
│   ├── 01_data_selection.ipynb          # Raw data filtering & metabolite column selection (MOMI-specific format)
│   └── 02_longitudinal_data_prep.ipynb  # Train/test split, null filtering, class-conditional imputation
├── pipeline/
│   └── metabolomics_pipeline.py         # Main Stage 1 + Stage 2 pipeline (entry point)
├── data/                                # Place preprocessed CSVs here (gitignored)
├── output/                              # All results, figures, and logs written here (gitignored)
├── requirements.txt
└── README.md
```

---

## Data Requirements & Workflow

There are **two entry points** depending on the format of your data:

### Path A — Starting from raw MOMI-format data

Run the notebooks **in order** before the pipeline:

```
notebooks/01_data_selection.ipynb
        ↓  produces: All_Trimesters.csv
notebooks/02_longitudinal_data_prep.ipynb
        ↓  produces: T1_train.csv, T1_test.csv,
                     T2_train.csv, T2_test.csv,
                     T3_train.csv, T3_test.csv
pipeline/metabolomics_pipeline.py
```

**01_data_selection.ipynb** expects:
- `MOMI_derived_data_Oct4_mapping.csv` — the raw MOMI sample mapping file with a `SITE` column (filtered to `AMANHIT`).
- Metabolite intensity columns following the `maternal_plasma_*` naming convention.
- Metadata columns: `ORIG_ID`, `PTB_NEW`, `Trimester`.

It outputs `All_Trimesters.csv` with samples from all three trimesters merged.

**02_longitudinal_data_prep.ipynb** expects:
- `All_Trimesters.csv` (from the previous step) with a `Trimester` column containing values `first`, `second`, `third`.

It performs a **global stratified train/test split on `ORIG_ID`** (preventing patient-level leakage across trimesters), null-rate filtering per class, and class-conditional median imputation. It outputs six CSVs: `T{1,2,3}_train.csv` and `T{1,2,3}_test.csv`.

---

### Path B — Starting from pre-split, pre-imputed CSVs

If you already have train/test CSVs with null values handled, skip the notebooks entirely and run the pipeline directly:

```
pipeline/metabolomics_pipeline.py
```

Expected files in the working directory (or specify via `--datasets`):
```
T1_train.csv   T1_test.csv
T2_train.csv   T2_test.csv
T3_train.csv   T3_test.csv
```

Each CSV must contain numeric metabolite feature columns plus a `PTB_NEW` binary target column (`0` / `1`).

---

## Installation

```bash
git clone https://github.com/<your-username>/mudar-ptb.git
cd mudar-ptb
pip install -r requirements.txt
```

Python 3.10+ is recommended.

---

## Running the Pipeline

```bash
cd mudar-ptb
python pipeline/metabolomics_pipeline.py \
    --datasets T1 T2 T3 \
    --target_col PTB_NEW \
    --output_dir output \
    --log_file output/pipeline_log.txt
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--datasets` | `T1 T2 T3 T1_T2 T2_T3 T1_T3` | Dataset prefixes to run (must have matching `_train.csv` / `_test.csv`) |
| `--target_col` | `PTB_NEW` | Name of the binary target column |
| `--output_dir` | `output` | Directory for all figures, CSVs, and logs |
| `--log_file` | `pipeline_log.txt` | Path for the run log |

---

## Pipeline Details

### Stage 1 — Conventional Methods

For each dataset, an exhaustive grid search is run over:
- **Feature selection:** Variance threshold, ANOVA F-test with FDR correction, RFECV with Random Forest
- **Class imbalance:** None, SMOTE
- **Classifiers:** Logistic Regression, SVM, Gaussian Naive Bayes, Random Forest, XGBoost

Evaluation uses stratified 5-fold CV with **average precision (AP)** as the primary metric. The optimal classification threshold is selected via Youden's J statistic. Bootstrap 95% CIs are reported for ROC-AUC, AP, F1, precision, and recall.

Per-combination outputs include ROC curves, PR curves, confusion matrices, and calibration plots.

### Stage 2 — Knowledge Distillation

The dataset with the highest Stage 1 ROC-AUC is selected as the **teacher**. For each remaining dataset (**student**), KD is run in two branches:

- **Branch 1:** Student trained in the teacher's feature space
- **Branch 2:** Student trained in the union of teacher + student feature spaces

A grid search over the distillation weight α is performed. Branch comparison plots and a summary `stage2_kd_results.csv` are saved to `output/`.

---

## Outputs

```
output/
├── T1/                        # Per-dataset Stage 1 figures and CSVs
├── T2/
├── T3/
├── stage1_results_summary.csv
├── stage2_kd_results.csv
├── stage2_kd_overview.png
└── pipeline_log.txt
```

---

## Citation

If you use this pipeline in your work, please cite:

```
[Add your paper citation here]
```

---

## License

[Add your license here]
