# DataCo Smart Supply Chain: Late-Delivery Risk

**IT3091 Machine Learning · Group 2026-AI-05 · Guided Data Track (Logistics & Supply Chain)**

| Member | IT Number | Focus |
|---|---|---|
| Bolonne B.R.M. | IT24102050 | EDA + Preprocessing |
| Kodagoda D.S. | IT24101613 | EDA + Baseline model training|
| Jayasekara C.D. | IT24103494 | EDA + Hyper Parameter Tuning|
| Liyanage J.L.K.L. | IT24101927 | EDA + Evaluation and recommendation |

---

## Live demo

**[Open the app](https://it3091mlassignment-a5pssekjqfbdbprx7e7chu.streamlit.app/)**

Five tabs: the findings behind our recommendation, scoring a real order from the
held-out test set, building your own order, model performance, and the recommendation
itself. Built with Streamlit from the same model the notebooks produce.

---

## The problem

For every order, using only information known when the order is placed, predict whether it is
at risk of being delivered late, early enough for operations to intervene.

Supervised binary classification. Target: `Late_delivery_risk` (1 = late).

## Headline result

| | Validation ROC-AUC |
|---|---|
| Majority class | 0.5000 |
| `Shipping Mode` rule (4 columns) | 0.7387 |
| Logistic Regression (baseline) | 0.7471 |
| **Random Forest (tuned)** | **0.7704** |

**Final test ROC-AUC: 0.7807**

The more useful finding is not the model. **First Class orders are late 100% of the time** -
every one takes exactly 2 days against a 1-day promise. Second Class and Standard Class have
identical actual transit times (2-6 days, median 4) yet are late 80% and 40%. The lateness is
caused by the delivery promise, not the delivery network.

## Repository contents

| Path | What it is |
|---|---|
| `01_Preprocessing_FINAL.ipynb` | EDA and preprocessing. 63 code cells, 31 figures. Runs in ~4 min |
| `02_Model_Training.ipynb` | Baseline + 4 alternatives, tuning, evaluation. ~17 min |
| `artifacts/` | Splits, fitted pipeline, decision log, model comparison, manifest |
| `figures/` | 41 charts produced by the notebooks |
| `IT3091 Final Report - 2026-AI-05.docx` | The final report, including the workflow diagram and AI-use declaration |

## Not in this repository

The raw dataset (`DataCoSupplyChainDataset.csv`, 91.5 MB) is excluded. It is too close to
GitHub's 100 MB file limit, and it is public data anyway. Download it from the link in the
assignment descriptor. To confirm you have the same file we used:

```
SHA-256  fa6d022ed437155e1a2f0378710602848703c8a7f203f7ff5d77805bf8480aa6
Bytes    95,910,149
```

`artifacts/final_model.joblib`, `artifacts/processed_dataset.csv` and the encoded CSVs are
also excluded. All are rebuilt by running the notebooks.

## How to reproduce

```bash
pip install pandas numpy scikit-learn scipy matplotlib joblib pyarrow xgboost
```

1. Put `DataCoSupplyChainDataset.csv` and `DescriptionDataCoSupplyChain.csv` in the repo root.
2. Run `01_Preprocessing_FINAL.ipynb` top to bottom. It writes `artifacts/` and `figures/`.
3. Run `02_Model_Training.ipynb` top to bottom.

Random seed is 42 throughout. Both notebooks are committed with their outputs saved, so the
results can be checked without re-running anything.

## Method notes

- **Leakage was proven, not assumed.** The rule `real days > scheduled days` reproduces the
  label on 97.55% of rows, so four post-shipment columns were dropped.
- **The split is chronological, not random.** One order spans up to 5 rows sharing a single
  label, and the real task is forward in time. A random split leaks on both counts.
- **Cross-validation is order-aware.** Plain `TimeSeriesSplit` cut through 2 orders, so a
  custom splitter drops validation rows whose order already appeared in training.
- **PCA was tested and rejected.** At 25 inputs, feature selection beat it by 0.020 ROC-AUC
  and stayed interpretable.
- **The top three models are statistically tied.** The gap between them (0.0022) is smaller
  than the fold-to-fold variation within a single model (0.0030).

## AI use

An AI assistant (Claude) was used for code, checks and report drafting. The full declaration,
including eight specific errors it made that we caught, is in the report.
