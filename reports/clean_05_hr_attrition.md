# Model Doctor audit: No issues detected in completed checks

**Health score: 100/100.** The 14 completed checks detect no issues within their scope.

**Audit coverage:** 14 of 19 checks run, 5 not applicable, 0 errors.

- **Model:** Pipeline: ColumnTransformer -> RandomForestClassifier
- **Predicting:** `left_company` (classification)
- **Goal:** Predict which employees are likely to leave in the next year.
- **Data:** 2,250 training rows, 750 test rows, 6 input columns
- **Training code:** included in the audit

## Findings (most serious first)

No issues are detected in the completed checks.

## How the model scores today

| Measure | Value |
|---|---:|
| Accuracy | 66.8% |
| Balanced accuracy | 64.9% |
| Rare cases caught | 61.5% |
| Alerts that are right | 37.0% |
| F1 for the rare class | 46.2% |
| Ranking quality (ROC AUC) | 0.705 |
| Precision-recall AUC | 0.409 |

## What Model Doctor checks

| Check | Area | Result |
|---|---|---|
| Model runs on the data | Data quality | No issue detected |
| Missing values handled differently in train and test | Data quality | No issue detected |
| Missing values disguised as numbers | Data quality | No issue detected |
| New categories that only appear in test data | Data quality | No issue detected |
| Category codes that mean different things in train and test | Data quality | No issue detected |
| Test data looks different from training data | Data quality | No issue detected |
| Columns that give away the answer | Data leakage | No issue detected |
| Preprocessing that learns from the test data | Data leakage | No issue detected |
| Leaky steps in the training code | Data leakage | No issue detected |
| Training on the future (time leakage) | Data leakage | Not applicable (The data has no date/time column.) |
| ID columns used as features | Data leakage | No issue detected |
| Same rows in training and test data | Train/test contamination | No issue detected |
| Same people/entities in training and test data | Train/test contamination | Not applicable (The data has no repeated group or ID column to compare.) |
| Cross-validation that ignores groups or time | Train/test contamination | Not applicable (The audit has no confirmed group or time structure that requires special validation.) |
| Accuracy hiding poor results on rare cases | Misleading metrics | Not applicable (Classes are reasonably balanced (smallest training class = 23.2%).) |
| Much better on training data than on test data | Overfitting | No issue detected |
| Suspiciously perfect test score | Overfitting | No issue detected |
| Model ignores the rare class | Class imbalance | No issue detected |
| Imbalanced data with no counter-measure | Class imbalance | Not applicable (Smallest class is 23.2% of the training data; this check treats that as outside severe imbalance.) |

_This audit covers common ML failure patterns in the data, model and training code. It does not check fairness, security, privacy, latency or business fit, so it is not a deployment approval. Evidence confidence is rule-based, not a statistical probability. When available, the health score is a heuristic summary for prioritising, not a certification. It is hidden when no checks complete or when any check ends with an error._
