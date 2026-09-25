# Model Doctor audit: High audit risk

**Health score: 66/100.** Model Doctor returns 4 findings: 1 high and 3 medium.

**Audit coverage:** 13 of 19 checks run, 6 not applicable, 0 errors.

- **Model:** LogisticRegression
- **Predicting:** `default` (classification)
- **Goal:** Predict which loan applicants will default, at application time.
- **Data:** 3,000 training rows, 1,000 test rows, 7 input columns
- **Training code:** included in the audit

## Findings (most serious first)

### 1. Possible target leakage in 'collections_flag'
**High** · Data leakage · check `LEAK-001` · evidence confidence: **80%**

**What we find.** Using only 'collections_flag', a simple train-only rule reaches 96.3% normalized balanced-accuracy skill (0% = chance-level balanced accuracy, 100% = perfect) when predicting 'default'.

**Why it matters.** A feature this strong can contain information that becomes available only after the outcome is known. It can also be a valid measurement, so the score alone does not prove leakage.

**What to do.** Check what 'collections_flag' means and when it becomes available. Remove it and retrain only if it is unavailable at prediction time or is derived from the outcome.

### 2. Preprocessing (StandardScaler) runs before the evaluation split
**Medium** · Data leakage · check `LEAK-003` · evidence confidence: **90%**

**What we find.** In the training code, StandardScaler is fitted on line(s) 22, before the first train/test split or cross-validation run on line 24.

**Why it matters.** A learned step that runs before the evaluation boundary can use information from rows that later act as validation or test data. Scaling or imputation usually has a smaller effect, but the evaluation is still no longer fully independent.

**What to do.** Create the split first. Fit learned preprocessing on training data only, and keep fold-specific preprocessing inside the cross-validation Pipeline.

### 3. The data has a full-dataset scaling fingerprint
**Medium** · Data leakage · check `LEAK-002` · evidence confidence: **80%**

**What we find.** 6 numeric column(s) are almost perfectly standard-scaled only when training and test rows are combined.

**Why it matters.** This pattern strongly suggests that scaling is fitted before the split. It is a fingerprint rather than direct proof, so the original data preparation should be checked.

**What to do.** Return to the raw, unscaled data. Split it first, fit the scaler on training rows only, transform the test rows with that fitted scaler, and retrain the model.

### 4. Accuracy alone can hide poor rare-class performance
**Medium** · Misleading metrics · check `METRIC-001` · evidence confidence: **70%**

**What we find.** The model is judged by accuracy from the stated reported metric: 98.9%. The minority class '1' makes up 13.0% of training rows, while an always-majority prediction already scores 85.8% on this test set. The model recalls 97.2% of the '1' test cases.

**Why it matters.** With imbalanced classes, accuracy can stay high even when the model performs poorly on the rare class. Using accuracy alone can therefore hide behaviour that matters for the prediction goal.

**What to do.** Report recall, precision and F1 for '1', and include PR AUC when probability scores are available. Compare accuracy with the always-majority baseline instead of using it alone.

## Before and after: reference rebuild

Model Doctor builds a comparison pipeline with the same estimator type, train-only preprocessing, and supported automatic repairs. It does not edit the original training code.

The balanced accuracy is similar: 98.2% for the original setup and 98.2% for the reference rebuild on the same evaluation rows.

_Model Doctor can re-audit the rebuilt model for up to three repair rounds. Because later repair decisions can use results from the current evaluation data, the before/after values are diagnostic results, not an untouched final performance estimate._

**Repair history:**

*Repair round 1*

- Builds train-only preprocessing for the reference model: fills missing values, scales numeric columns, and one-hot encodes categorical columns with unknown-category handling. (standard step of the rebuild)
- Reports recall, precision, F1 and AUC for the rare class instead of accuracy alone (a reporting change only; it does not change training). (reason: METRIC-001)

Final re-audit: still open: Possible target leakage in 'collections_flag' (LEAK-001).

| Measure | Before | After | Change |
|---|---:|---:|---:|
| Accuracy | 98.9% | 98.9% | +0.0 pts |
| Balanced accuracy | 98.2% | 98.2% | +0.0 pts |
| Rare cases caught | 97.2% | 97.2% | +0.0 pts |
| Alerts that are right | 95.2% | 95.2% | +0.0 pts |
| F1 for the rare class | 96.2% | 96.2% | +0.0 pts |
| Ranking quality (ROC AUC) | 0.995 | 0.995 | -0.000 |
| Precision-recall AUC | 0.979 | 0.979 | -0.000 |

Original audit: **4** findings. Reference rebuild: **1** finding still open. Checks that read the original training code do not run on the reference rebuild.

## What Model Doctor checks

| Check | Area | Result |
|---|---|---|
| Model runs on the data | Data quality | No issue detected |
| Missing values handled differently in train and test | Data quality | No issue detected |
| Missing values disguised as numbers | Data quality | No issue detected |
| New categories that only appear in test data | Data quality | Not applicable (No text/category columns.) |
| Category codes that mean different things in train and test | Data quality | Not applicable (The data has no number-coded category columns.) |
| Test data looks different from training data | Data quality | No issue detected |
| Columns that give away the answer | Data leakage | Finding raised |
| Preprocessing that learns from the test data | Data leakage | Finding raised |
| Leaky steps in the training code | Data leakage | Finding raised |
| Training on the future (time leakage) | Data leakage | Not applicable (The data has no date/time column.) |
| ID columns used as features | Data leakage | No issue detected |
| Same rows in training and test data | Train/test contamination | No issue detected |
| Same people/entities in training and test data | Train/test contamination | Not applicable (The data has no repeated group or ID column to compare.) |
| Cross-validation that ignores groups or time | Train/test contamination | Not applicable (The audit has no confirmed group or time structure that requires special validation.) |
| Accuracy hiding poor results on rare cases | Misleading metrics | Finding raised |
| Much better on training data than on test data | Overfitting | No issue detected |
| Suspiciously perfect test score | Overfitting | No issue detected |
| Model ignores the rare class | Class imbalance | No issue detected |
| Imbalanced data with no counter-measure | Class imbalance | Not applicable (Smallest class is 13.0% of the training data; this check treats that as outside severe imbalance.) |

_This audit covers common ML failure patterns in the data, model and training code. It does not check fairness, security, privacy, latency or business fit, so it is not a deployment approval. Evidence confidence is rule-based, not a statistical probability. When available, the health score is a heuristic summary for prioritising, not a certification. It is hidden when no checks complete or when any check ends with an error._
