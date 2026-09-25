# Model Doctor audit: High audit risk

**Health score: 30/100.** Model Doctor returns 6 findings: 4 high, 1 medium and 1 low.

**Audit coverage:** 15 of 19 checks run, 4 not applicable, 0 errors.

- **Model:** HistGradientBoostingClassifier
- **Predicting:** `high_demand` (classification)
- **Goal:** Forecast whether tomorrow will be a high-demand day for each store.
- **Data:** 6,950 training rows, 1,738 test rows, 6 input columns
- **Training code:** included in the audit

## Findings (most serious first)

### 1. Category codes in 'store_type' do not match between training and test data
**High** · Data quality · check `DQ-004` · evidence confidence: **90%**

**What we find.** In 'store_type', the numeric codes behave differently between training and test data. The two sets fit much better when 3 codes are matched differently (test code -> training code: 0->2, 1->0, 2->1). The training code also converts this column to numbers separately on the test data (line 27: pd.factorize). Reading the codes with the suggested mapping changes the model's test balanced accuracy by +30.8 points.

**Why it matters.** This pattern can appear when categories turn into numbers separately for training and test data, such as fitting LabelEncoder twice or using pd.factorize separately. The same number can then represent different categories, which changes predictions.

**What to do.** Fit one encoder on the training data and reuse it everywhere, or use OneHotEncoder(handle_unknown='ignore') inside the Pipeline. Re-create this column from the original category values.

### 2. Training data includes dates after the test period begins
**High** · Data leakage · check `LEAK-004` · evidence confidence: **90%**

**What we find.** 99.9% of training rows are dated after the first test date (train: 2023-01-08 to 2024-12-31, test: 2023-01-08 to 2024-12-31). The training and test periods overlap.

**Why it matters.** When the goal depends on time order, evaluation should reproduce the information available at prediction time. Training on later dates can make the test score too optimistic.

**What to do.** Sort by 'date', train on the earlier period, and test on the most recent period. Use TimeSeriesSplit for cross-validation when appropriate.

### 3. Cross-validation does not respect time order
**High** · Train/test contamination · check `CONT-003` · evidence confidence: **85%**

**What we find.** The code uses KFold on line 30, while the data requires validation that respects time order.

**Why it matters.** Ordinary or group-only cross-validation can train on rows that occur after the validation period. That makes the validation score optimistic for a future-looking prediction problem.

**What to do.** Use TimeSeriesSplit or another forward-chaining validation design that trains on earlier periods and validates on later periods.

### 4. The train/test split shuffles time-ordered data
**High** · Train/test contamination · check `CONT-003` · evidence confidence: **85%**

**What we find.** train_test_split on line 22 shuffles rows, while 'date' is used as the time structure for this prediction problem.

**Why it matters.** Shuffling can place later observations in training while earlier observations remain in test data. For future-looking evaluation, that mixes past and future.

**What to do.** Sort by 'date' and create an earlier-training/later-test split. Use TimeSeriesSplit for cross-validation when appropriate.

### 5. A large train-test gap suggests overfitting
**Medium** · Overfitting · check `OVERFIT-001` · evidence confidence: **55%**

**What we find.** Balanced accuracy is 95.9% on training data and 60.9% on test data, a gap of 35.1%.

**Why it matters.** A large gap is consistent with the model learning patterns that do not generalize to the test data. Data-quality or distribution problems can also increase this gap, so those findings should be checked before treating overfitting as the only cause. Another finding in this audit ("Category codes in 'store_type' do not match between training and test data") can also increase the train-test gap by reducing test performance. This makes overfitting a less certain explanation.

**What to do.** First check whether train and test data are comparable. If they are, reduce model complexity: limit tree depth (max_depth). Tune changes with appropriate cross-validation.

### 6. Missing values are filled using the whole dataset
**Low** · Data leakage · check `LEAK-003` · evidence confidence: **80%**

**What we find.** The code fills missing values with a mean, median, or mode computed before the evaluation split (line 19).

**Why it matters.** The fill value uses information from rows that later act as validation or test data, so the evaluation is not fully independent.

**What to do.** Use an imputer inside the Pipeline so its values come from training rows only.

## Before and after: reference rebuild

Model Doctor builds a comparison pipeline with the same estimator type, train-only preprocessing, and supported automatic repairs. It does not edit the original training code.

The original setup scores 60.9% balanced accuracy; the reference rebuild scores 91.0%. Because the repair changes which rows form the evaluation set, these values are diagnostic and should not be interpreted as a direct performance gain or loss.

_Model Doctor can re-audit the rebuilt model for up to three repair rounds. Because later repair decisions can use results from the current evaluation data, the before/after values are diagnostic results, not an untouched final performance estimate. The original and rebuilt scores use different evaluation rows because the original split requires repair. The values therefore show diagnostic change rather than a direct like-for-like performance comparison._

**Repair history:**

*Repair round 1*

- Excludes 'store_type' from the reference rebuild because the category codes cannot be trusted. Re-encode the original category values before adding the column back. (reason: DQ-004)
- Re-splits by date on 'date': trains on earlier rows and tests on the most recent rows. (reason: CONT-003, LEAK-004)
- Makes the estimator more regularized: max_depth=6. (reason: OVERFIT-001)
- Builds train-only preprocessing for the reference model: fills missing values, scales numeric columns, and one-hot encodes categorical columns with unknown-category handling. (standard step of the rebuild)

Final re-audit: still open: Some numeric columns look different in the test data (DQ-005).

### Needs a human

- Re-encode 'store_type' from the original category values with one encoder that learns from training data only, then add it back.

| Measure | Before | After | Change |
|---|---:|---:|---:|
| Accuracy | 75.8% | 91.8% | +15.9 pts |
| Balanced accuracy | 60.9% | 91.0% | +30.2 pts |
| Rare cases caught | 26.1% | 89.3% | +63.2 pts |
| Alerts that are right | 70.5% | 83.9% | +13.4 pts |
| F1 for the rare class | 38.1% | 86.5% | +48.4 pts |
| Ranking quality (ROC AUC) | 0.815 | 0.974 | +0.159 |
| Precision-recall AUC | 0.629 | 0.942 | +0.312 |

Original audit: **6** findings. Reference rebuild: **1** finding still open. Checks that read the original training code do not run on the reference rebuild.

## What Model Doctor checks

| Check | Area | Result |
|---|---|---|
| Model runs on the data | Data quality | No issue detected |
| Missing values handled differently in train and test | Data quality | No issue detected |
| Missing values disguised as numbers | Data quality | No issue detected |
| New categories that only appear in test data | Data quality | Not applicable (No text/category columns.) |
| Category codes that mean different things in train and test | Data quality | Finding raised |
| Test data looks different from training data | Data quality | No issue detected |
| Columns that give away the answer | Data leakage | No issue detected |
| Preprocessing that learns from the test data | Data leakage | No issue detected |
| Leaky steps in the training code | Data leakage | Finding raised |
| Training on the future (time leakage) | Data leakage | Finding raised |
| ID columns used as features | Data leakage | No issue detected |
| Same rows in training and test data | Train/test contamination | No issue detected |
| Same people/entities in training and test data | Train/test contamination | Not applicable (The data has no repeated group or ID column to compare.) |
| Cross-validation that ignores groups or time | Train/test contamination | Finding raised |
| Accuracy hiding poor results on rare cases | Misleading metrics | Not applicable (Classes are reasonably balanced (smallest training class = 30.5%).) |
| Much better on training data than on test data | Overfitting | Finding raised |
| Suspiciously perfect test score | Overfitting | No issue detected |
| Model ignores the rare class | Class imbalance | No issue detected |
| Imbalanced data with no counter-measure | Class imbalance | Not applicable (Smallest class is 30.5% of the training data; this check treats that as outside severe imbalance.) |

_This audit covers common ML failure patterns in the data, model and training code. It does not check fairness, security, privacy, latency or business fit, so it is not a deployment approval. Evidence confidence is rule-based, not a statistical probability. When available, the health score is a heuristic summary for prioritising, not a certification. It is hidden when no checks complete or when any check ends with an error._
