# Model Doctor

Model Doctor is a Python toolkit for checking common problems in machine learning pipelines.

It looks at the dataset, trained model, and training code when available, and helps find problems that can make model results unreliable.

It explains:

- what the problem is
- why it matters
- what can be done about it

Model Doctor can also build a reference version of the pipeline with supported fixes and run the audit again for comparison.

---

## What Model Doctor Checks

Model Doctor includes **19 checks** across six main areas:

- Data leakage
- Train/test contamination
- Misleading metrics
- Overfitting
- Data quality
- Class imbalance

Some examples include:

- features that may reveal the target
- preprocessing before the train/test split
- duplicate rows across train and test data
- the same people or groups appearing in both splits
- incorrect cross-validation for time or grouped data
- accuracy hiding poor rare-class performance
- large train/test performance gaps
- missing-value problems
- unseen categories
- encoding mismatch
- distribution shift
- models that ignore the rare class

---

## Main Features

- Works with scikit-learn-compatible models
- Supports classification and regression
- Accepts `.pkl` and `.joblib` models
- Works with separate train/test CSV files or one CSV with a split column
- Can inspect Python training scripts and notebooks
- Does not execute the supplied training code during static analysis
- Gives severity and evidence confidence for findings
- Creates HTML, Markdown, and JSON reports
- Supports command-line use
- Supports JSON configuration files
- Can build a reference pipeline with supported fixes
- Runs the audit again after repair for before/after comparison

Model Doctor does not automatically change something when the evidence is not strong enough. 

---

## Demo

The project includes:

- 4 deliberately broken ML pipelines
- 1 clean control pipeline

The broken pipelines cover problems such as leakage, contamination, class imbalance, poor evaluation choices, time leakage, and encoding problems.

The current demo catches:

**16 / 16 expected detector checks**

The clean control finishes with:

**0 findings**

Run the full demo with:

```bash
python scripts/run_demo.py
```

Then open:

```text
reports/index.html
```

The report shows the results for all five pipelines.

---

## Install

Model Doctor requires **Python 3.10 or newer**.

Install the required packages:

```bash
pip install -r requirements.txt
```

You can also install the project as a package:

```bash
pip install .
```

---

## Run an Audit

Example:

```bash
python -m model_doctor audit \
    --model model.joblib \
    --train train.csv \
    --test test.csv \
    --target churned \
    --out reports/churn
```

Include the training code:

```bash
python -m model_doctor audit \
    --model model.joblib \
    --train train.csv \
    --test test.csv \
    --target churned \
    --code train.py \
    --out reports/churn
```

Build a reference version with supported fixes:

```bash
python -m model_doctor audit \
    --model model.joblib \
    --train train.csv \
    --test test.csv \
    --target churned \
    --fix \
    --out reports/churn
```

---

## Run the Tests

```bash
python -m unittest discover -s tests -v
```

The test suite covers the main checks, code analysis, CLI behaviour, fixture pipelines, repair behaviour, classification, regression, and different estimator types.

---

## Reports

Model Doctor can create three report formats:

### HTML

A browser report with findings, explanations, evidence, recommendations, health score, coverage, and check results.

### Markdown

A text-friendly version for GitHub, review, or sharing.

### JSON

A machine-readable version for other tools or CI workflows.

---

## Project Structure

```text
model_doctor/
    core.py
    auditor.py
    fixer.py
    report.py
    cli.py
    code_analysis.py
    checks/

fixtures/
    4 broken pipelines
    1 clean control

tests/
    project test suite

scripts/
    run_demo.py

reports/
    generated audit reports

README.md
START_HERE.md
PRESENTATION.md
REQUIREMENTS_CHECKLIST.md
pyproject.toml
requirements.txt
```

---

## Important Notes

The health score is a simple score for prioritising findings. It is not a probability or production approval.

The health score is based on individual findings. Related evidence from different checks can affect the score separately.

When no fitted model is provided and the target is ambiguous, set the task with `--task classification` or `--task regression`.

The reference rebuild is used for comparison. Its result is not an untouched final estimate of real-world model performance.

Static code analysis covers common pandas and scikit-learn patterns, but very complex custom code may need manual review.

Model Doctor checks common ML pipeline problems. It does not check fairness, security, privacy, latency, or business suitability.

Only load `.pkl` or `.joblib` model files from sources you trust.

---

## Author

**Manisha Nagda**
