# Start Here

**Model Doctor** checks a machine learning model, its data, and the training code when available.

It runs **19 checks** for common problems that can make model results unreliable.

For every finding, the report explains:

- what the problem is
- why it matters
- what can be done about it

---

## 1. See the Results

Open:

```text
reports/index.html
```

This page shows the results for:

- 4 deliberately broken pipelines
- 1 clean control pipeline

Each pipeline also has its own full audit report.

The demo catches:

**16 / 16 expected detector checks**

The clean control finishes with:

**0 findings**

---

## 2. Install

Model Doctor requires **Python 3.10 or newer**.

Install the required packages:

```bash
pip install -r requirements.txt
```

You can also install the project itself:

```bash
pip install .
```

---

## 3. Run the Full Demo

```bash
python scripts/run_demo.py
```

The demo:

- builds the five example pipelines
- runs Model Doctor on each one
- applies supported repairs when possible
- runs the audit again
- writes the results to `reports/`

A successful run ends with:

```text
Expected detector checks caught: 16/16. Summary: reports/index.html
```

---

## 4. Audit One Model

Example:

```bash
python -m model_doctor audit \
    --model model.joblib \
    --train train.csv \
    --test test.csv \
    --target churned \
    --code train.py \
    --reported-metric accuracy \
    --goal "Predict whether a customer will leave" \
    --fix \
    --out reports/my_audit
```

You can also use one of the ready-made fixture configs. Run the full demo (step 3) first: it rebuilds the fixture models with your installed scikit-learn version.

```bash
python -m model_doctor audit \
    --config fixtures/output/broken_01_loan_leakage/config.json \
    --fix
```

---

## 5. Run the Tests

```bash
python -m unittest discover -s tests -v
```

The test suite checks the main audit rules, code analysis, CLI behaviour, fixture pipelines, repair behaviour, classification, regression, and different model types.

---

## Key Demo Results

| Pipeline | Expected detector checks | Original health score | Findings in original audit | Still open after reference rebuild |
|---|---:|---:|---:|---:|
| Loan default | 4 / 4 | 66 | 4 | 1 |
| Customer churn | 4 / 4 | 36 | 4 | 0 |
| Card fraud | 4 / 4 | 36 | 5 | 2 |
| Store demand | 4 / 4 | 30 | 6 | 1 |
| HR attrition (clean control) | none expected | 100 | 0 | 0 |

The health score is shown for the original audit only. Checks that read the original training code do not run on the reference rebuild, so a score for the rebuild does not measure the same things.

The four broken pipelines contain **14 root problems**.

Some root problems are detected by more than one check, so the demo expects **16 detector checks** in total.

The fraud and store-demand pipelines also produce an additional `OVERFIT-001` finding beyond their expected checks.

Some findings remain open after the reference rebuild because Model Doctor does not automatically make a change when the evidence is not strong enough.

---

## Where Things Are

| Path | What it contains |
|---|---|
| `model_doctor/` | main audit code, checks, code analysis, fixer, reports, and CLI |
| `fixtures/` | four broken example pipelines and one clean control |
| `tests/` | project test suite |
| `scripts/run_demo.py` | runs the complete demo |
| `reports/` | generated audit reports |
| `reports/index.html` | demo summary |
| `README.md` | main project documentation |
| `REQUIREMENTS_CHECKLIST.md` | hackathon requirements and where each one is covered |
| `PRESENTATION.md` | presentation and demo guide |

---

## Important Note

The reference rebuild is used for comparison.

Its before/after results are diagnostic results and are not an untouched final estimate of real-world model performance.
