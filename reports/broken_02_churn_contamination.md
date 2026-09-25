# Model Doctor audit: High audit risk

**Health score: 36/100.** Model Doctor returns 4 findings: 4 high.

**Audit coverage:** 15 of 19 checks run, 4 not applicable, 0 errors.

- **Model:** RandomForestClassifier
- **Predicting:** `churned` (classification)
- **Goal:** Predict which customers will churn next month.
- **Data:** 3,691 training rows, 1,231 test rows, 9 input columns
- **Training code:** included in the audit

## Findings (most serious first)

### 1. Test feature rows also appear in the training data
**High** · Train/test contamination · check `CONT-001` · evidence confidence: **95%**

**What we find.** 37.0% of test rows (456 of 1231) have the same feature values as a training row. For these matched feature rows, the test label also appears with the training copy 100% of the time.

**Why it matters.** Repeated examples across training and evaluation reduce the independence of the test set. Flexible models can memorise these examples, which can make the reported test score look better than performance on genuinely unseen cases.

**What to do.** Check why the same records appear on both sides of the split. When they are duplicate records, remove duplicates before splitting and create the train/test split again.

### 2. The same 'customer_id' values appear in training and test data
**High** · Train/test contamination · check `CONT-002` · evidence confidence: **90%**

**What we find.** 99.3% of test rows belong to a 'customer_id' value that also appears in the training data (about 5.3 training rows per 'customer_id').

**Why it matters.** 'customer_id' is configured as the grouping column, so rows from the same group should stay on one side of a group-independent evaluation. The current overlap allows information about the same entity to appear in both training and test data.

**What to do.** Split by 'customer_id' with GroupShuffleSplit or GroupKFold so one group does not appear on both sides of the evaluation.

### 3. Cross-validation does not respect groups
**High** · Train/test contamination · check `CONT-003` · evidence confidence: **85%**

**What we find.** The code uses KFold on line 24, while the data requires validation that respects groups.

**Why it matters.** Ordinary or time-only cross-validation can place rows from the same entity in both training and validation folds. That can make validation performance optimistic when the goal is group-independent generalization.

**What to do.** Use GroupKFold or StratifiedGroupKFold with groups=df['customer_id'].

### 4. A large train-test gap suggests overfitting
**High** · Overfitting · check `OVERFIT-001` · evidence confidence: **85%**

**What we find.** Balanced accuracy is 100.0% on training data and 76.7% on test data, a gap of 23.3%.

**Why it matters.** A large gap is consistent with the model learning patterns that do not generalize to the test data. Data-quality or distribution problems can also increase this gap, so those findings should be checked before treating overfitting as the only cause.

**What to do.** First check whether train and test data are comparable. If they are, reduce model complexity: limit tree depth (max_depth); require more samples per leaf (min_samples_leaf). Tune changes with appropriate cross-validation.

## Before and after: reference rebuild

Model Doctor builds a comparison pipeline with the same estimator type, train-only preprocessing, and supported automatic repairs. It does not edit the original training code.

The original setup scores 76.7% balanced accuracy; the reference rebuild scores 69.3%. Because the repair changes which rows form the evaluation set, these values are diagnostic and should not be interpreted as a direct performance gain or loss.

_Model Doctor can re-audit the rebuilt model for up to three repair rounds. Because later repair decisions can use results from the current evaluation data, the before/after values are diagnostic results, not an untouched final performance estimate. The original and rebuilt scores use different evaluation rows because the original split requires repair. The values therefore show diagnostic change rather than a direct like-for-like performance comparison._

**Repair history:**

*Repair round 1*

- Re-splits by 'customer_id' so each customer id is only in training or only in test. (reason: CONT-002)
- Makes the estimator more regularized: max_depth=6, min_samples_leaf=10. (reason: OVERFIT-001)
- Builds train-only preprocessing for the reference model: fills missing values, scales numeric columns, and one-hot encodes categorical columns with unknown-category handling. (standard step of the rebuild)

Re-audit: The model almost always predicts '0' (IMB-001). A supported new repair is added for the next round.

*Repair round 2*

- Gives the rare class extra weight (class_weight='balanced'). (reason: IMB-001)

Final re-audit: no findings remain.

| Measure | Before | After | Change |
|---|---:|---:|---:|
| Accuracy | 84.8% | 70.9% | -13.9 pts |
| Balanced accuracy | 76.7% | 69.3% | -7.4 pts |
| Rare cases caught | 59.4% | 65.7% | +6.4 pts |
| Alerts that are right | 77.8% | 45.9% | -31.9 pts |
| F1 for the rare class | 67.4% | 54.1% | -13.3 pts |
| Ranking quality (ROC AUC) | 0.873 | 0.770 | -0.103 |
| Precision-recall AUC | 0.776 | 0.573 | -0.203 |

Original audit: **4** findings. Reference rebuild: **0** findings still open. Checks that read the original training code do not run on the reference rebuild.

## What Model Doctor checks

| Check | Area | Result |
|---|---|---|
| Model runs on the data | Data quality | No issue detected |
| Missing values handled differently in train and test | Data quality | No issue detected |
| Missing values disguised as numbers | Data quality | No issue detected |
| New categories that only appear in test data | Data quality | Not applicable (No text/category columns.) |
| Category codes that mean different things in train and test | Data quality | No issue detected |
| Test data looks different from training data | Data quality | No issue detected |
| Columns that give away the answer | Data leakage | No issue detected |
| Preprocessing that learns from the test data | Data leakage | No issue detected |
| Leaky steps in the training code | Data leakage | No issue detected |
| Training on the future (time leakage) | Data leakage | Not applicable (The data has no date/time column.) |
| ID columns used as features | Data leakage | No issue detected |
| Same rows in training and test data | Train/test contamination | Finding raised |
| Same people/entities in training and test data | Train/test contamination | Finding raised |
| Cross-validation that ignores groups or time | Train/test contamination | Finding raised |
| Accuracy hiding poor results on rare cases | Misleading metrics | Not applicable (Classes are reasonably balanced (smallest training class = 25.5%).) |
| Much better on training data than on test data | Overfitting | Finding raised |
| Suspiciously perfect test score | Overfitting | No issue detected |
| Model ignores the rare class | Class imbalance | No issue detected |
| Imbalanced data with no counter-measure | Class imbalance | Not applicable (Smallest class is 25.5% of the training data; this check treats that as outside severe imbalance.) |

_This audit covers common ML failure patterns in the data, model and training code. It does not check fairness, security, privacy, latency or business fit, so it is not a deployment approval. Evidence confidence is rule-based, not a statistical probability. When available, the health score is a heuristic summary for prioritising, not a certification. It is hidden when no checks complete or when any check ends with an error._
