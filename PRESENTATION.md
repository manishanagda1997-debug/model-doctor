# Model Doctor Demo Guide

## 1. The Problem

A machine learning model can look good during testing and still give unreliable results.

Common reasons include:

- data leakage
- train/test contamination
- misleading metrics
- overfitting
- data-quality problems
- class imbalance

Finding these problems manually takes time, and some of them are easy to miss.

Model Doctor brings these checks together in one audit.

---

## 2. What Model Doctor Does

Model Doctor checks:

- the dataset
- the trained model, when available
- the training code, when available

It runs **19 checks** across six main problem areas.

For every finding, the report explains:

- what Model Doctor finds
- why it matters
- what to do next

When a supported repair is safe to apply, Model Doctor can also build a reference version of the pipeline and run the audit again for comparison.

The original model and training code stay unchanged.

---

## 3. Start with the Demo Summary

Open:

```text
reports/index.html
```

The demo includes:

- 4 deliberately broken pipelines
- 1 clean control pipeline

The four broken pipelines contain **14 root problems**.

Some problems are detected by more than one check, so the demo expects **16 detector checks** in total.

Model Doctor catches:

**16 / 16 expected detector checks**

The clean control finishes with:

**0 findings**

---

## 4. Show the Loan Default Report

Open:

```text
reports/broken_01_loan_leakage.html
```

The report finds four problems.

One important finding is:

```text
Possible target leakage in 'collections_flag'
```

This feature is very strong, but Model Doctor does not automatically remove it.

A strong feature can be genuine. The report asks the user to check what the column means and when it becomes available.

The report also finds preprocessing before the evaluation split and a misleading metric problem.

After the reference rebuild:

```text
Original audit: 4 findings
Still open in the reference rebuild: 1
Balanced accuracy: 98.2% → 98.2%
```

The rebuild fits the scaler on training rows only, so the scaling leak is removed. The balanced accuracy does not change, which shows that the scaling leak has almost no effect here.

The large effect comes from `collections_flag`. In a manual check, the same model without this column gives 56.0% balanced accuracy instead of 98.2%.

The possible target-leakage finding stays open for human review.

This shows that Model Doctor does not make an automatic change when the evidence is not strong enough.

---

## 5. Show the Fraud Report

The fraud example shows why accuracy alone can be misleading when one class is rare.

Its balanced accuracy changes from:

```text
51.2% → 72.6%
```

The report also shows rare-class metrics such as recall, precision, and F1.

---

## 6. Run One Audit from the Command Line

Example:

```bash
python -m model_doctor audit \
    --config fixtures/output/broken_04_store_demand_time/config.json \
    --fix
```

This runs Model Doctor using a ready-made config file.

---

## 7. Why the Results Are Useful

Model Doctor does not depend on only one type of evidence.

It can check:

- model behaviour
- train and test data
- training code

The project also includes tests for:

- audit checks
- code analysis
- CLI behaviour
- broken and clean pipelines
- reference rebuild behaviour
- classification
- regression
- different model types

Each finding includes:

- severity
- evidence confidence

Evidence confidence is not a probability.

It shows how strongly the available evidence supports the finding.

Model Doctor also separates:

- checks that pass
- checks that are not applicable
- checks that cannot run because of an error

---

## 8. Key Features to Show

Model Doctor includes:

- 19 audit checks
- all 6 main problem areas
- severity levels
- evidence confidence
- HTML reports
- Markdown reports
- JSON reports
- command-line support
- `.pkl` and `.joblib` model support
- CSV input
- Python script analysis
- notebook analysis
- supported automatic repairs
- before/after reference comparison
- classification support
- regression support

---

# Likely Judge Questions

## How do you know a strong feature is leakage?

We do not assume that every strong feature is leakage.

Model Doctor checks how strongly the feature predicts the target and looks at other available evidence.

If the evidence does not prove that the feature is unavailable at prediction time, Model Doctor reports it as possible target leakage and leaves the final decision to the user.

It does not automatically delete the feature only because it is strong.

---

## Does Model Doctor change the original model or code?

No.

The original model and source code stay unchanged.

When `--fix` is used, Model Doctor creates a separate reference rebuild for comparison.

---

## Does Model Doctor run the training code?

No.

It reads Python scripts and notebooks using static code analysis.

It does not execute the supplied training code.

---

## Does it work only with one model type?

No.

Model Doctor works with the scikit-learn estimator interface.

The project tests different classification and regression models instead of depending on one model family.

Other estimators that closely follow the same interface may also work, but compatibility outside the tested set is not guaranteed.

---

## Why do some findings remain after the reference rebuild?

Some problems are not safe to fix automatically.

For example, Model Doctor cannot know the real-world meaning or timing of every feature from the data alone.

When the evidence is not strong enough for a safe automatic action, the finding stays open for human review.

---

## Why is there no health score after the reference rebuild?

Checks that read the original training code do not run on the reference rebuild, because the rebuild does not use that code.

A health score for the rebuild would therefore not measure the same things as the original score. The report shows the original health score and the findings that are still open instead.

---

## Is the health score a model-performance score?

No.

The health score helps prioritise audit findings.

It is not:

- model accuracy
- a probability
- a production certification
- a deployment approval

---

## Are the before/after results final model-performance estimates?

No.

They are diagnostic comparison results from the reference rebuild.

They help show the effect of supported repairs, but they are not an untouched final estimate of real-world performance.
