# Model Doctor audit: Critical audit issues

**Health score: 36/100.** Model Doctor returns 5 findings: 1 critical, 1 high and 3 medium.

**Audit coverage:** 15 of 19 checks run, 4 not applicable, 0 errors.

- **Model:** GradientBoostingClassifier
- **Predicting:** `is_fraud` (classification)
- **Goal:** Flag fraudulent card transactions in real time.
- **Data:** 15,000 training rows, 5,000 test rows, 7 input columns
- **Training code:** included in the audit

## Findings (most serious first)

### 1. The model almost always predicts '0'
**Critical** · Class imbalance · check `IMB-001` · evidence confidence: **90%**

**What we find.** The model predicts '0' for 99.6% of test cases, while 98.6% actually belong to that class. It recalls only 2.9% of the '1' cases.

**Why it matters.** The model is strongly biased toward the common class and misses most examples of the rare class. High overall accuracy can hide this behaviour.

**What to do.** Compare class weighting, resampling, or a different decision threshold, and judge the result with rare-class recall, precision and F1.

### 2. Accuracy alone can hide poor rare-class performance
**High** · Misleading metrics · check `METRIC-001` · evidence confidence: **90%**

**What we find.** The model is judged by accuracy from the stated reported metric: 98.3%. The minority class '1' makes up 1.4% of training rows, while an always-majority prediction already scores 98.6% on this test set. The model recalls 2.9% of the '1' test cases.

**Why it matters.** With imbalanced classes, accuracy can stay high even when the model performs poorly on the rare class. Using accuracy alone can therefore hide behaviour that matters for the prediction goal.

**What to do.** Report recall, precision and F1 for '1', and include PR AUC when probability scores are available. Compare accuracy with the always-majority baseline instead of using it alone.

### 3. Placeholder numbers may represent missing values
**Medium** · Data quality · check `DQ-002` · evidence confidence: **75%**

**What we find.** These columns contain a repeated code far outside the usual training range: 'distance_from_home_km' uses -999 (train 5.0%, test 4.8%; usual range 0.88 to 84.172).

**Why it matters.** A placeholder such as -999 can act like a real extreme measurement. This can distort scaling, averages, and model decisions when the code really means 'unknown'.

**What to do.** Confirm that each code means a missing value. If it does, replace it with NaN, add a missing-value flag when useful, and impute inside the Pipeline.

### 4. Severe class imbalance has no explicit handling
**Medium** · Class imbalance · check `IMB-002` · evidence confidence: **70%**

**What we find.** '1' makes up only 1.4% of the training data, and no class weighting, resampling, or sample-weight handling is detected. The model recalls 2.9% of the '1' test cases (F1 4.4%).

**Why it matters.** Without an explicit imbalance strategy, many models can favour common classes. Whether weighting or resampling helps should be checked against rare-class metrics rather than assumed.

**What to do.** Compare class_weight='balanced' or an appropriate resampling method, then evaluate recall, precision and F1 for the rare class.

### 5. A large train-test gap suggests overfitting
**Medium** · Overfitting · check `OVERFIT-001` · evidence confidence: **65%**

**What we find.** Balanced accuracy is 65.4% on training data and 51.2% on test data, a gap of 14.1%.

**Why it matters.** A large gap is consistent with the model learning patterns that do not generalize to the test data. Data-quality or distribution problems can also increase this gap, so those findings should be checked before treating overfitting as the only cause.

**What to do.** First check whether train and test data are comparable. If they are, reduce model complexity: require more samples per leaf (min_samples_leaf). Tune changes with appropriate cross-validation.

## Before and after: reference rebuild

Model Doctor builds a comparison pipeline with the same estimator type, train-only preprocessing, and supported automatic repairs. It does not edit the original training code.

The reference rebuild scores 72.6% balanced accuracy, compared with 51.2% for the original setup. On the same evaluation rows, the repaired setup performs better, but this diagnostic comparison is not an untouched final estimate.

_Model Doctor can re-audit the rebuilt model for up to three repair rounds. Because later repair decisions can use results from the current evaluation data, the before/after values are diagnostic results, not an untouched final performance estimate._

**Repair history:**

*Repair round 1*

- Gives the rare class extra weight with balanced sample weights. (reason: IMB-001, IMB-002)
- Makes the estimator more regularized: min_samples_leaf=10. (reason: OVERFIT-001)
- Builds train-only preprocessing for the reference model: fills missing values, scales numeric columns, and one-hot encodes categorical columns with unknown-category handling. (standard step of the rebuild)
- Reports recall, precision, F1 and AUC for the rare class instead of accuracy alone (a reporting change only; it does not change training). (reason: METRIC-001)

Final re-audit: still open: Placeholder numbers may represent missing values (DQ-002); A large train-test gap suggests overfitting (OVERFIT-001).

| Measure | Before | After | Change |
|---|---:|---:|---:|
| Accuracy | 98.3% | 82.1% | -16.2 pts |
| Balanced accuracy | 51.2% | 72.6% | +21.4 pts |
| Rare cases caught | 2.9% | 62.9% | +60.0 pts |
| Alerts that are right | 10.0% | 4.8% | -5.2 pts |
| F1 for the rare class | 4.4% | 8.9% | +4.5 pts |
| Ranking quality (ROC AUC) | 0.817 | 0.803 | -0.014 |
| Precision-recall AUC | 0.084 | 0.127 | +0.044 |

Original audit: **5** findings. Reference rebuild: **2** findings still open. Checks that read the original training code do not run on the reference rebuild.

## What Model Doctor checks

| Check | Area | Result |
|---|---|---|
| Model runs on the data | Data quality | No issue detected |
| Missing values handled differently in train and test | Data quality | No issue detected |
| Missing values disguised as numbers | Data quality | Finding raised |
| New categories that only appear in test data | Data quality | Not applicable (No text/category columns.) |
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
| Accuracy hiding poor results on rare cases | Misleading metrics | Finding raised |
| Much better on training data than on test data | Overfitting | Finding raised |
| Suspiciously perfect test score | Overfitting | No issue detected |
| Model ignores the rare class | Class imbalance | Finding raised |
| Imbalanced data with no counter-measure | Class imbalance | Finding raised |

_This audit covers common ML failure patterns in the data, model and training code. It does not check fairness, security, privacy, latency or business fit, so it is not a deployment approval. Evidence confidence is rule-based, not a statistical probability. When available, the health score is a heuristic summary for prioritising, not a certification. It is hidden when no checks complete or when any check ends with an error._
