# Requirements Checklist

This file shows how Model Doctor covers the requirements in the hackathon problem statement and where to find the proof.

---

## Main Requirements

| Requirement | How Model Doctor Covers It | Where to Check |
|---|---|---|
| Accept a dataset, training pipeline or trained model, and prediction task or goal | The CLI accepts train/test data or one dataset, a trained model, optional training code, task type, target, and goal. The Python API also supports these inputs. | `model_doctor/cli.py`, `model_doctor/auditor.py` |
| Use automatic programmatic checks | Model Doctor registers 19 checks and also performs static analysis of Python scripts and notebooks. | `model_doctor/checks/`, `model_doctor/code_analysis.py` |
| Produce a structured audit report | Every finding explains what Model Doctor finds, why it matters, and what to do. Reports also show severity, evidence confidence, coverage, and technical details. | `model_doctor/report.py`, `reports/` |
| Prove the tool on 3–4 deliberately broken pipelines | The project includes 4 broken pipelines and 1 clean control pipeline. | `fixtures/`, `reports/index.html` |
| Show before/after results | When supported repairs are available, Model Doctor builds a separate reference version and runs the audit again. | `model_doctor/fixer.py`, generated reports |

---

## Issue Types

The problem requires at least five issue types.

Model Doctor covers **all six**.

| Issue Type | Model Doctor Checks |
|---|---|
| Data leakage | `LEAK-001`, `LEAK-002`, `LEAK-003`, `LEAK-004`, `LEAK-005` |
| Train/test contamination | `CONT-001`, `CONT-002`, `CONT-003` |
| Misleading metrics | `METRIC-001` |
| Overfitting | `OVERFIT-001`, `OVERFIT-002` |
| Data quality | `DQ-000`, `DQ-001`, `DQ-002`, `DQ-003`, `DQ-004`, `DQ-005` |
| Class imbalance | `IMB-001`, `IMB-002` |

There are **19 checks** in total.

---

## Data Leakage

| Check | What It Looks For |
|---|---|
| `LEAK-001` | A feature that may reveal the target |
| `LEAK-002` | Preprocessing that appears to use train and test data together |
| `LEAK-003` | Leaky preprocessing or other learned steps in the training code |
| `LEAK-004` | Training on future data when time order matters |
| `LEAK-005` | ID-like columns used as model features |

---

## Train/Test Contamination

| Check | What It Looks For |
|---|---|
| `CONT-001` | The same feature rows appearing in train and test data |
| `CONT-002` | The same people, customers, devices, or other groups appearing in both splits |
| `CONT-003` | Cross-validation that does not respect group or time structure |

---

## Misleading Metrics

| Check | What It Looks For |
|---|---|
| `METRIC-001` | Accuracy being used when class imbalance can hide poor rare-class performance |

---

## Overfitting

| Check | What It Looks For |
|---|---|
| `OVERFIT-001` | A large performance gap between training and test data |
| `OVERFIT-002` | A suspiciously high test score that needs leakage or contamination review |

---

## Data Quality

| Check | What It Looks For |
|---|---|
| `DQ-000` | The model cannot make predictions on the supplied data |
| `DQ-001` | Missing-value behaviour differs between train and test |
| `DQ-002` | Values such as `-999` or `9999` that may represent missing data |
| `DQ-003` | Categories in test data that do not appear in training data |
| `DQ-004` | Possible category-encoding mismatch between training and test data |
| `DQ-005` | Numeric distribution shift between train and test |

---

## Class Imbalance

| Check | What It Looks For |
|---|---|
| `IMB-001` | The model ignores a rare class or predicts mainly the majority class |
| `IMB-002` | Severe class imbalance without a clear counter-measure |

---

## Additional Requirements

| Requirement | How Model Doctor Covers It | Where to Check |
|---|---|---|
| Python-based solution | The project uses Python, pandas, NumPy, scikit-learn, SciPy, and joblib. | `requirements.txt`, `pyproject.toml` |
| Work with at least 3 model types | The four broken fixtures already use LogisticRegression, RandomForestClassifier, GradientBoostingClassifier, and HistGradientBoostingClassifier. The tests cover additional model families. | `fixtures/`, `tests/test_integration.py` |
| Work beyond one dataset | Generalisation tests use different datasets, including breast cancer, wine, and diabetes datasets. | `tests/test_integration.py` |
| Work with classification and regression | The test suite includes both classification and regression models. | `tests/test_integration.py` |
| Include 3–4 broken pipelines | The project includes exactly 4 deliberately broken pipelines. | `fixtures/broken_01_loan_leakage.py` through `fixtures/broken_04_store_demand_time.py` |
| Include proof that the auditor detects the problems | Fixture tests compare expected detector IDs with the findings returned by Model Doctor. | `tests/test_integration.py`, `expected_checks.json` files |
| Produce a report for non-technical readers | Reports use simple sections such as What we find, Why it matters, and What to do. | `model_doctor/report.py`, `reports/*.html` |
| Human-readable output | Model Doctor creates HTML and Markdown reports. HTML can also be printed to PDF from a browser. | `reports/` |
| Test the auditing tool itself | The test suite covers audit checks, code analysis, fixtures, repair behaviour, CLI behaviour, reports, clean cases, and edge cases. | `tests/` |

---

## Demo Proof

The demo includes:

- 4 deliberately broken pipelines
- 1 clean control pipeline
- 14 root problems across the broken pipelines
- 16 expected detector checks

The current demo catches:

**16 / 16 expected detector checks**

The clean control returns:

**0 findings**

Open:

```text
reports/index.html
```

or run:

```bash
python scripts/run_demo.py
```

---

## Bonus Features

### Supported Repairs and Before/After Comparison

Using `--fix`, Model Doctor can apply supported repairs when there is enough evidence for an automatic change.

It then runs the audit again and shows before/after diagnostic results.

Some findings stay open when they need human judgement.

The before/after numbers are comparison results, not an untouched final estimate of real-world performance.

**Proof:**

- `model_doctor/fixer.py`
- generated reports
- `tests/test_integration.py`

---

### Severity

Each finding receives one of four severity levels:

```text
critical
high
medium
low
```

The report also gives a health score for prioritising findings.

The health score is not a probability or production approval. It is based on individual findings, so related evidence from different checks can affect the score separately.

**Proof:**

- `model_doctor/core.py`
- `model_doctor/auditor.py`
- generated reports

---

### Evidence Confidence

Each finding includes an evidence-confidence score.

The score shows how strongly the available evidence supports the finding.

It is not a statistical probability.

Model Doctor can also use evidence from related checks when deciding how strongly to present a finding.

**Proof:**

- `model_doctor/auditor.py`
- individual check files
- generated reports

---

### Command-Line Interface

Model Doctor supports command-line audits using trusted `.pkl` or `.joblib` models with CSV data.

Example:

```bash
python -m model_doctor audit \
    --model model.joblib \
    --train train.csv \
    --test test.csv \
    --target churned \
    --out reports/churn
```

The CLI also supports:

- one CSV with a split column
- Python training scripts
- notebooks
- JSON config files
- data-only audits
- HTML, Markdown, and JSON output
- `--fix`
- `--fail-on` for CI use

**Proof:**

- `model_doctor/cli.py`
- `tests/test_integration.py`

---

## Test Coverage

The project has three main test files:

```text
tests/test_checks.py
tests/test_code_analysis.py
tests/test_integration.py
```

They cover:

- registered audit checks
- positive and clean cases
- edge cases
- static code analysis
- fixture detection
- clean-control behaviour
- automatic repair
- report wording
- audit coverage
- CLI behaviour
- config handling
- single-CSV mode
- different estimator families
- classification
- multiclass classification
- regression
- data-only audits

Run the complete suite with:

```bash
python -m unittest discover -s tests -v
```

---

## Known Limits

Model Doctor focuses on common technical ML pipeline problems.

Important limits include:

- a very strong real feature can look similar to target leakage
- data alone cannot always separate an encoding mistake from real data drift
- static code analysis cannot understand every custom or dynamic Python program
- some findings need human review instead of an automatic fix
- reference-rebuild results are diagnostic comparisons, not untouched final performance estimates
- Model Doctor does not check fairness, security, privacy, latency, or business suitability

See `README.md` and the generated reports for more details.