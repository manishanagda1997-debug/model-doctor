"""End-to-end tests for fixtures, auto-fixes, generalisation, CLI, and reports.

Run with: python -m unittest tests.test_integration -v
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model_doctor import REGISTRY, audit  # noqa: E402
from model_doctor.auditor import AuditResult, CheckRun  # noqa: E402
from model_doctor.cli import (
    _merge_config,
    build_parser,
    run_audit,
    split_by_column,
)  # noqa: E402
from model_doctor.report import render_html, render_markdown  # noqa: E402

warnings.simplefilter("ignore")
FIXTURES = sorted((ROOT / "fixtures").glob("*_0*.py"))
_BUILT = {}
_RUNS = {}


def build(script: Path) -> Path:
    """Run one fixture generator once and cache its temporary output folder."""
    if script not in _BUILT:
        out = Path(tempfile.mkdtemp()) / script.stem
        subprocess.run(
            [sys.executable, str(script), str(out)],
            check=True,
            capture_output=True,
            text=True,
        )
        _BUILT[script] = out
    return _BUILT[script]


def fixture_metadata(out: Path) -> dict:
    """Read a generated fixture config."""
    return json.loads((out / "config.json").read_text(encoding="utf-8"))


def fixture_expected_checks(out: Path) -> set:
    """Read the test-only expected detector IDs for a generated fixture."""
    return set(json.loads((out / "expected_checks.json").read_text(encoding="utf-8")))


def run_fixture(out: Path, fix=False):
    """Run one fixture through the same config merge used by the CLI.

       Results are cached because these tests only read them."""
    key = (out, bool(fix))
    if key in _RUNS:
        return _RUNS[key]

    config = fixture_metadata(out)
    audit_config = out / "audit_config.json"
    audit_config.write_text(json.dumps(config, indent=2), encoding="utf-8")

    args = ["audit", "--config", str(audit_config)]
    if fix:
        args.append("--fix")

    result = run_audit(_merge_config(build_parser().parse_args(args)))
    _RUNS[key] = result
    return result


class TestFixtures(unittest.TestCase):
    """Prove planted failures are detected and the clean control stays clean."""

    def test_each_fixture_catches_its_planted_bugs(self):
        for script in FIXTURES:
            with self.subTest(fixture=script.stem):
                out = build(script)
                expected = fixture_expected_checks(out)
                result = run_fixture(out)
                found = {finding.check_id for finding in result.findings}
                self.assertTrue(expected <= found, f"missing {expected - found}")
                self.assertFalse([run for run in result.runs if run.status == "error"])

    def test_clean_pipeline_has_no_serious_false_alarms(self):
        out = build(ROOT / "fixtures" / "clean_05_hr_attrition.py")
        result = run_fixture(out)
        serious = [
            finding
            for finding in result.findings
            if finding.severity in ("critical", "high", "medium")
        ]
        self.assertEqual(serious, [], [finding.title for finding in serious])
        self.assertGreaterEqual(result.health_score, 90)

    def test_autofix_reduces_open_findings_on_broken_fixtures(self):
        for script in FIXTURES:
            if script.stem.startswith("clean"):
                continue
            with self.subTest(fixture=script.stem):
                fix_result = run_fixture(build(script), fix=True).fix_result
                self.assertIsNotNone(fix_result)
                self.assertIsNone(fix_result.error)
                self.assertTrue(fix_result.model_rebuilt)
                self.assertGreater(len(fix_result.applied), 0)

                # Some findings intentionally need human review, so reduction is enough.
                self.assertLess(
                    len(fix_result.after_findings), len(fix_result.before_findings)
                )

    def test_fix_history_records_why_a_later_repair_is_added(self):
        fix_result = run_fixture(
            build(ROOT / "fixtures" / "broken_02_churn_contamination.py"), fix=True
        ).fix_result

        first_changes = " ".join(
            action["change"] for action in fix_result.rounds[0]["changes"]
        )
        self.assertNotIn("class_weight", first_changes)

        first_reaudit_ids = {
            finding["check_id"] for finding in fix_result.rounds[0]["reaudit"]
        }
        self.assertIn("IMB-001", first_reaudit_ids)

        second_changes = fix_result.rounds[1]["changes"]
        self.assertTrue(
            any(
                "class_weight" in action["change"] and action["because"] == ["IMB-001"]
                for action in second_changes
            )
        )
        self.assertEqual(fix_result.rounds[-1]["reaudit"], [])

    def test_metric_finding_alone_changes_reporting_not_training(self):
        from model_doctor.core import Finding
        from model_doctor.fixer import auto_fix

        frame = datasets.load_breast_cancer(as_frame=True).frame
        train, test = train_test_split(frame, random_state=0)
        model = LogisticRegression(max_iter=3000).fit(
            train.drop(columns="target"),
            train.target,
        )

        result = audit(
            train,
            test,
            "target",
            model=model,
        )

        finding = Finding(
            "METRIC-001",
            "metrics",
            "Accuracy hides rare-class performance",
            "medium",
            0.9,
            "w",
            "y",
            "f",
            fix_action={"type": "report_balanced_metrics"},
        )

        fix_result = auto_fix(
            result.ctx,
            [finding],
        )

        self.assertIsNone(fix_result.error)
        self.assertIs(fix_result.new_model, model)
        self.assertFalse(fix_result.model_rebuilt)
        self.assertTrue(fix_result.reporting_changed)
        self.assertIsNone(model.get_params()["class_weight"])

    def test_autofix_improves_fraud_recall(self):
        fix_result = run_fixture(
            build(ROOT / "fixtures" / "broken_03_fraud_imbalance.py"),
            fix=True,
        ).fix_result

        self.assertGreater(
            fix_result.after_metrics["minority_recall"],
            fix_result.before_metrics["minority_recall"] + 0.3,
        )

    def test_reference_rebuild_exposes_inflated_leaky_score(self):
        fix_result = run_fixture(
            build(ROOT / "fixtures" / "broken_01_loan_leakage.py"),
            fix=True,
        ).fix_result

        self.assertFalse(fix_result.eval_rows_changed)
        self.assertLess(
            fix_result.after_metrics["roc_auc"],
            fix_result.before_metrics["roc_auc"],
        )


class TestGeneralisation(unittest.TestCase):
    """Run on unseen datasets and several estimator families without check errors."""

    def _cases(self):
        breast_cancer = datasets.load_breast_cancer(as_frame=True).frame
        wine = datasets.load_wine(as_frame=True).frame
        diabetes = datasets.load_diabetes(as_frame=True).frame

        yield "bc-lr", breast_cancer, "target", make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=3000),
        )

        yield "bc-svc-no-proba", breast_cancer, "target", make_pipeline(
            StandardScaler(),
            SVC(),
        )

        yield "bc-knn", breast_cancer, "target", KNeighborsClassifier()

        yield "bc-nb", breast_cancer, "target", GaussianNB()

        yield (
            "wine-et-multiclass",
            wine,
            "target",
            ExtraTreesClassifier(max_depth=4, random_state=0),
        )

        yield (
            "wine-hgb",
            wine,
            "target",
            HistGradientBoostingClassifier(max_depth=3),
        )

        yield "diabetes-ridge", diabetes, "target", Ridge()

        yield (
            "diabetes-gbr",
            diabetes,
            "target",
            GradientBoostingRegressor(random_state=0),
        )

        try:
            from xgboost import XGBClassifier

            yield (
                "bc-xgb",
                breast_cancer,
                "target",
                XGBClassifier(n_estimators=50, max_depth=3),
            )

        except ImportError:
            pass

    def test_runs_everywhere_without_check_errors(self):
        for name, frame, target, model in self._cases():
            with self.subTest(case=name):
                train, test = train_test_split(
                    frame,
                    test_size=0.25,
                    random_state=0,
                )

                features = [column for column in frame.columns if column != target]

                model.fit(
                    train[features],
                    train[target],
                )

                result = audit(
                    train,
                    test,
                    target,
                    model=model,
                )

                self.assertFalse(
                    [run.note for run in result.runs if run.status == "error"],
                    name,
                )

                self.assertFalse(
                    [
                        finding
                        for finding in result.findings
                        if finding.severity == "critical"
                    ],
                    name,
                )

    def test_regression_task_detected(self):
        frame = datasets.load_diabetes(as_frame=True).frame
        train, test = train_test_split(
            frame,
            random_state=0,
        )

        model = Ridge().fit(
            train.drop(columns="target"),
            train.target,
        )

        result = audit(
            train,
            test,
            "target",
            model=model,
        )

        self.assertEqual(
            result.ctx.task,
            "regression",
        )

    def test_planted_leak_on_real_data(self):
        frame = datasets.load_breast_cancer(as_frame=True).frame

        frame["lab_result_flag"] = frame.target + np.random.default_rng(0).normal(
            0,
            0.1,
            len(frame),
        )

        train, test = train_test_split(
            frame,
            random_state=0,
        )

        model = LogisticRegression(max_iter=3000).fit(
            train.drop(columns="target"),
            train.target,
        )

        result = audit(
            train,
            test,
            "target",
            model=model,
        )

        leak = [
            finding for finding in result.findings if finding.check_id == "LEAK-001"
        ]

        self.assertTrue(leak)

        self.assertEqual(
            leak[0].evidence["column"],
            "lab_result_flag",
        )

    def test_data_only_audit_without_model(self):
        frame = datasets.load_breast_cancer(as_frame=True).frame

        train, test = train_test_split(
            frame,
            random_state=0,
        )

        result = audit(
            train,
            test,
            "target",
        )

        self.assertIn(
            "skipped",
            {run.status for run in result.runs},
        )

        self.assertFalse([run for run in result.runs if run.status == "error"])


class TestCLIAndReports(unittest.TestCase):
    def test_cli_writes_all_report_formats_and_fail_on(self):
        out = build(ROOT / "fixtures" / "broken_03_fraud_imbalance.py")

        prefix = Path(tempfile.mkdtemp()) / "audit"

        command = [
            sys.executable,
            "-m",
            "model_doctor",
            "audit",
            "--model",
            str(out / "model.joblib"),
            "--train",
            str(out / "train.csv"),
            "--test",
            str(out / "test.csv"),
            "--target",
            "is_fraud",
            "--reported-metric",
            "accuracy",
            "--fix",
            "--out",
            str(prefix),
            "--fail-on",
            "high",
        ]

        process = subprocess.run(
            command,
            cwd=ROOT,
            capture_output=True,
            text=True,
        )

        self.assertEqual(
            process.returncode,
            1,
            process.stderr,
        )

        html = prefix.with_suffix(".html").read_text(encoding="utf-8")

        self.assertIn(
            "What we find",
            html,
        )

        self.assertIn(
            "Before and after: reference rebuild",
            html,
        )

        self.assertTrue(prefix.with_suffix(".md").exists())

        data = json.loads(prefix.with_suffix(".json").read_text(encoding="utf-8"))

        self.assertIn(
            "IMB-001",
            {finding["check_id"] for finding in data["findings"]},
        )

        self.assertIsNotNone(data["health_score"])

    def test_cli_single_csv_mode(self):
        out = build(ROOT / "fixtures" / "clean_05_hr_attrition.py")

        full = pd.concat(
            [
                pd.read_csv(out / "train.csv").assign(split="train"),
                pd.read_csv(out / "test.csv").assign(split="test"),
            ]
        )

        temp = Path(tempfile.mkdtemp())

        full.to_csv(
            temp / "all.csv",
            index=False,
        )

        process = subprocess.run(
            [
                sys.executable,
                "-m",
                "model_doctor",
                "audit",
                "--model",
                str(out / "model.joblib"),
                "--data",
                str(temp / "all.csv"),
                "--split-col",
                "split",
                "--target",
                "left_company",
                "--out",
                str(temp / "r"),
                "--formats",
                "json",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )

        self.assertEqual(
            process.returncode,
            0,
            process.stderr,
        )

        self.assertTrue((temp / "r.json").exists())

    def test_config_values_survive_and_explicit_cli_values_win(self):
        temp = Path(tempfile.mkdtemp())

        config_path = temp / "config.json"

        config_path.write_text(
            json.dumps(
                {
                    "target": "y",
                    "test_size": 0.4,
                    "fix": True,
                    "formats": ["json"],
                    "out": "reports/from_config",
                }
            ),
            encoding="utf-8",
        )

        config_only = _merge_config(
            build_parser().parse_args(
                [
                    "audit",
                    "--config",
                    str(config_path),
                ]
            )
        )

        self.assertEqual(
            config_only["test_size"],
            0.4,
        )

        self.assertTrue(config_only["fix"])

        self.assertEqual(
            config_only["formats"],
            ["json"],
        )

        self.assertEqual(
            Path(config_only["out"]),
            temp / "reports" / "from_config",
        )

        overridden = _merge_config(
            build_parser().parse_args(
                [
                    "audit",
                    "--config",
                    str(config_path),
                    "--test-size",
                    "0.2",
                    "--no-fix",
                    "--formats",
                    "html,md",
                ]
            )
        )

        self.assertEqual(
            overridden["test_size"],
            0.2,
        )

        self.assertFalse(overridden["fix"])

        self.assertEqual(
            overridden["formats"],
            ["html", "md"],
        )

    def test_unknown_config_key_is_rejected(self):
        temp = Path(tempfile.mkdtemp())

        config_path = temp / "config.json"

        config_path.write_text(
            json.dumps(
                {
                    "target": "y",
                    "expected_checks": [],
                }
            ),
            encoding="utf-8",
        )

        parsed = build_parser().parse_args(
            [
                "audit",
                "--config",
                str(config_path),
            ]
        )

        with self.assertRaises(SystemExit):
            _merge_config(parsed)


class TestReportWording(unittest.TestCase):
    """Reports must describe diagnostic evidence without deployment overclaims."""

    BANNED = [
        "honest",
        "fair test",
        "really achieve",
        "Not safe to deploy",
        "How sure we are",
        "Fix before deploying",
        "unbiased",
    ]

    def _reports(self, name):
        result = run_fixture(
            build(ROOT / "fixtures" / f"{name}.py"),
            fix=True,
        )

        return (
            result,
            render_html(result),
            render_markdown(result),
        )

    def test_no_overclaiming_words_and_limits_shown(self):
        for name in (
            "broken_01_loan_leakage",
            "broken_02_churn_contamination",
        ):
            with self.subTest(fixture=name):
                result, html, markdown = self._reports(name)

                for text in (
                    html,
                    markdown,
                ):
                    lower = text.lower()

                    for word in self.BANNED:
                        self.assertNotIn(
                            word.lower(),
                            lower,
                        )

                    self.assertIn(
                        "not an untouched final performance estimate",
                        text,
                    )

                    self.assertIn(
                        "not a statistical probability",
                        text.replace(
                            "\n",
                            " ",
                        ),
                    )

                self.assertEqual(
                    "different evaluation rows" in html,
                    result.fix_result.eval_rows_changed,
                )

        self.assertTrue(
            self._reports("broken_02_churn_contamination")[
                0
            ].fix_result.eval_rows_changed
        )

        self.assertFalse(
            self._reports("broken_01_loan_leakage")[0].fix_result.eval_rows_changed
        )

    def test_reporting_only_fix_is_not_called_a_reference_rebuild(self):
        from model_doctor.core import Finding
        from model_doctor.fixer import auto_fix

        frame = datasets.load_breast_cancer(as_frame=True).frame

        train, test = train_test_split(
            frame,
            random_state=0,
        )

        model = LogisticRegression(max_iter=3000).fit(
            train.drop(columns="target"),
            train.target,
        )

        result = audit(
            train,
            test,
            "target",
            model=model,
        )

        finding = Finding(
            "METRIC-001",
            "metrics",
            "Accuracy alone is weak here",
            "medium",
            0.9,
            "w",
            "y",
            "f",
            fix_action={"type": "report_balanced_metrics"},
        )

        result.fix_result = auto_fix(
            result.ctx,
            [finding],
        )

        html = render_html(result)

        markdown = render_markdown(result)

        for text in (
            html,
            markdown,
        ):
            self.assertIn(
                "Reporting update: model unchanged",
                text,
            )

            self.assertNotIn(
                "Before and after: reference rebuild",
                text,
            )


class TestCoverage(unittest.TestCase):
    def _full_result(self):
        frame = datasets.load_breast_cancer(as_frame=True).frame

        train, test = train_test_split(
            frame,
            random_state=0,
        )

        model = LogisticRegression(max_iter=3000).fit(
            train.drop(columns="target"),
            train.target,
        )

        return (
            audit(
                train,
                test,
                "target",
                model=model,
            ),
            audit(
                train,
                test,
                "target",
            ),
        )

    def test_coverage_adds_up_and_is_shown(self):
        full, data_only = self._full_result()

        for result in (
            full,
            data_only,
        ):
            coverage = result.coverage()

            self.assertEqual(
                coverage["total"],
                len(REGISTRY),
            )

            self.assertEqual(
                coverage["run"] + coverage["not_applicable"] + coverage["errors"],
                coverage["total"],
            )

            text = f"{coverage['run']} of " f"{coverage['total']} checks run"

            self.assertIn(
                text,
                render_html(result),
            )

            self.assertIn(
                text,
                render_markdown(result),
            )

        self.assertLess(
            data_only.coverage()["run"],
            full.coverage()["run"],
        )

    def test_incomplete_and_no_applicable_audits_hide_health_score(self):
        full, _ = self._full_result()

        incomplete = AuditResult(
            full.ctx,
            [],
            [
                CheckRun(
                    "TEST-ERR",
                    "Synthetic error",
                    "data_quality",
                    "Synthetic integration check",
                    "error",
                    "RuntimeError: boom",
                )
            ],
        )

        self.assertEqual(
            incomplete.verdict,
            "Audit incomplete",
        )

        self.assertIsNone(incomplete.to_dict()["health_score"])

        self.assertIn(
            "health score unavailable",
            render_html(incomplete).lower(),
        )

        self.assertIn(
            "health score: unavailable",
            render_markdown(incomplete).lower(),
        )

        no_checks = AuditResult(
            full.ctx,
            [],
            [],
        )

        self.assertEqual(
            no_checks.verdict,
            "No applicable checks",
        )

        self.assertIsNone(no_checks.to_dict()["health_score"])

        self.assertIn(
            "health score unavailable",
            render_html(no_checks).lower(),
        )

        self.assertIn(
            "health score: unavailable",
            render_markdown(no_checks).lower(),
        )


class TestSplitColumn(unittest.TestCase):
    def test_split_column_validation(self):
        frame = pd.DataFrame(
            {
                "x": range(6),
                "s": [
                    "train",
                    "Train ",
                    "test",
                    "TEST",
                    "train",
                    "test",
                ],
            }
        )

        train, test = split_by_column(
            frame,
            "s",
        )

        self.assertEqual(
            (
                len(train),
                len(test),
            ),
            (
                3,
                3,
            ),
        )

        bad_cases = (
            [
                "train",
                "validation",
                "test",
            ],
            [
                "train",
                None,
                "test",
            ],
            [
                "train",
                "train",
                "train",
            ],
        )

        for values in bad_cases:
            with self.subTest(values=values):
                with self.assertRaises(SystemExit):
                    split_by_column(
                        pd.DataFrame(
                            {
                                "x": range(3),
                                "s": values,
                            }
                        ),
                        "s",
                    )

        with self.assertRaises(SystemExit):
            split_by_column(
                frame,
                "missing_col",
            )


if __name__ == "__main__":
    unittest.main()
