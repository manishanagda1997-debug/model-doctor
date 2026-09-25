"""Synthetic positive, clean, and edge cases for every registered audit check.

Run with: python -m unittest tests.test_checks -v
"""

import os
import sys
import unittest
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_doctor import REGISTRY, audit  # noqa: E402

warnings.simplefilter("ignore")


EXPECTED_CHECK_IDS = [
    "DQ-000",
    "DQ-001",
    "DQ-002",
    "DQ-003",
    "DQ-004",
    "DQ-005",
    "LEAK-001",
    "LEAK-002",
    "LEAK-003",
    "LEAK-004",
    "LEAK-005",
    "CONT-001",
    "CONT-002",
    "CONT-003",
    "METRIC-001",
    "OVERFIT-001",
    "OVERFIT-002",
    "IMB-001",
    "IMB-002",
]


def ids(result):
    return {finding.check_id for finding in result.findings}


def findings_for(result, check_id):
    return [
        finding
        for finding in result.findings
        if finding.check_id == check_id
    ]


def check_run(result, check_id):
    return next(
        run
        for run in result.runs
        if run.check_id == check_id
    )


def assert_flagged(case, result, check_id):
    case.assertEqual(
        check_run(result, check_id).status,
        "flagged",
    )
    case.assertIn(
        check_id,
        ids(result),
    )


def assert_passed(case, result, check_id):
    case.assertEqual(
        check_run(result, check_id).status,
        "passed",
    )
    case.assertNotIn(
        check_id,
        ids(result),
    )


def base_data(
    n=2000,
    seed=0,
    rate=0.5,
):
    """Return a binary dataset with two informative features and two noise features."""
    rng = np.random.default_rng(
        seed
    )

    features = pd.DataFrame(
        {
            "f1": rng.normal(
                size=n
            ),
            "f2": rng.normal(
                size=n
            ),
            "f3": rng.normal(
                size=n
            ),
            "f4": rng.normal(
                size=n
            ),
        }
    )

    score = (
        1.2
        * features.f1
        - 0.8
        * features.f2
        + np.log(
            rate
            / (
                1
                - rate
            )
        )
    )

    features[
        "y"
    ] = (
        rng.random(
            n
        )
        < 1
        / (
            1
            + np.exp(
                -score
            )
        )
    ).astype(
        int
    )

    return features


def fit_split(
    df,
    model,
    features=None,
    target="y",
    seed=0,
    stratify=True,
):
    """Split, fit, and return train data, test data, and the fitted estimator."""
    train, test = train_test_split(
        df,
        test_size=0.25,
        random_state=seed,
        stratify=(
            df[
                target
            ]
            if stratify
            else None
        ),
    )

    if features is None:
        features = [
            column
            for column in df.columns
            if column != target
        ]

    model.fit(
        train[
            features
        ],
        train[
            target
        ],
    )

    return (
        train,
        test,
        model,
    )


class TestRegistry(
    unittest.TestCase
):
    def test_all_expected_checks_are_registered_in_report_order(
        self,
    ):
        self.assertEqual(
            [
                spec.check_id
                for spec in REGISTRY
            ],
            EXPECTED_CHECK_IDS,
        )

        self.assertEqual(
            len(
                REGISTRY
            ),
            19,
        )


class TestLeakage(
    unittest.TestCase
):
    def test_leak001_target_proxy_flagged(
        self,
    ):
        df = base_data()

        df[
            "final_status"
        ] = (
            df.y
            * 2
            + np.random.default_rng(
                1
            ).normal(
                0,
                0.05,
                len(
                    df
                ),
            )
        )

        train, test, model = fit_split(
            df,
            LogisticRegression(),
        )

        result = audit(
            train,
            test,
            "y",
            model=model,
        )

        assert_flagged(
            self,
            result,
            "LEAK-001",
        )

        finding = findings_for(
            result,
            "LEAK-001",
        )[0]

        self.assertEqual(
            finding.evidence[
                "column"
            ],
            "final_status",
        )

        self.assertEqual(
            finding.severity,
            "critical",
        )

        # A strong proxy still needs semantic/timing confirmation.
        self.assertIsNone(
            finding.fix_action
        )

    def test_leak001_clean(
        self,
    ):
        train, test, model = fit_split(
            base_data(),
            LogisticRegression(),
        )

        result = audit(
            train,
            test,
            "y",
            model=model,
        )

        assert_passed(
            self,
            result,
            "LEAK-001",
        )

    def test_leak002_prescaled_data_flagged(
        self,
    ):
        df = base_data()

        features = [
            "f1",
            "f2",
            "f3",
            "f4",
        ]

        df[
            features
        ] = StandardScaler().fit_transform(
            df[
                features
            ]
        )

        train, test, model = fit_split(
            df,
            LogisticRegression(),
        )

        result = audit(
            train,
            test,
            "y",
            model=model,
        )

        assert_flagged(
            self,
            result,
            "LEAK-002",
        )

    def test_leak002_pipeline_fitted_on_full_data_flagged(
        self,
    ):
        df = base_data()

        features = [
            "f1",
            "f2",
            "f3",
            "f4",
        ]

        train, test = train_test_split(
            df,
            test_size=0.25,
            random_state=0,
        )

        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(),
        )

        model.fit(
            df[
                features
            ],
            df[
                "y"
            ],
        )

        result = audit(
            train,
            test,
            "y",
            model=model,
        )

        assert_flagged(
            self,
            result,
            "LEAK-002",
        )

    def test_leak002_nested_pipeline_fitted_on_full_data_flagged(
        self,
    ):
        df = base_data()

        features = [
            "f1",
            "f2",
            "f3",
            "f4",
        ]

        train, test = train_test_split(
            df,
            test_size=0.25,
            random_state=0,
        )

        prep = ColumnTransformer(
            [
                (
                    "numeric",
                    Pipeline(
                        [
                            (
                                "scale",
                                StandardScaler(),
                            )
                        ]
                    ),
                    features,
                )
            ],
            remainder="drop",
        )

        model = Pipeline(
            [
                (
                    "prep",
                    prep,
                ),
                (
                    "model",
                    LogisticRegression(),
                ),
            ]
        )

        model.fit(
            df[
                features
            ],
            df[
                "y"
            ],
        )

        result = audit(
            train,
            test,
            "y",
            model=model,
        )

        assert_flagged(
            self,
            result,
            "LEAK-002",
        )

    def test_leak002_proper_pipeline_clean(
        self,
    ):
        train, test, model = fit_split(
            base_data(),
            make_pipeline(
                StandardScaler(),
                LogisticRegression(),
            ),
        )

        result = audit(
            train,
            test,
            "y",
            model=model,
        )

        assert_passed(
            self,
            result,
            "LEAK-002",
        )

    def test_leak003_code_fit_before_split(
        self,
    ):
        code = (
            "from sklearn.preprocessing import StandardScaler\n"
            "from sklearn.feature_selection import SelectKBest\n"
            "sc = StandardScaler()\n"
            "X = sc.fit_transform(X)\n"
            "X = SelectKBest(k=5).fit_transform(X, y)\n"
            "X_tr, X_te, y_tr, y_te = train_test_split(X, y)\n"
        )

        train, test, model = fit_split(
            base_data(),
            LogisticRegression(),
        )

        result = audit(
            train,
            test,
            "y",
            model=model,
            pipeline_code=code,
        )

        assert_flagged(
            self,
            result,
            "LEAK-003",
        )

        self.assertEqual(
            findings_for(
                result,
                "LEAK-003",
            )[0].severity,
            "high",
        )

    def test_leak003_code_clean(
        self,
    ):
        code = (
            "X_tr, X_te, y_tr, y_te = train_test_split(X, y)\n"
            "sc = StandardScaler()\n"
            "X_tr = sc.fit_transform(X_tr)\n"
            "X_te = sc.transform(X_te)\n"
        )

        train, test, model = fit_split(
            base_data(),
            LogisticRegression(),
        )

        result = audit(
            train,
            test,
            "y",
            model=model,
            pipeline_code=code,
        )

        assert_passed(
            self,
            result,
            "LEAK-003",
        )

    def test_leak004_temporal_flagged_and_clean(
        self,
    ):
        df = base_data()

        df[
            "date"
        ] = pd.date_range(
            "2022-01-01",
            periods=len(
                df
            ),
            freq="D",
        ).strftime(
            "%Y-%m-%d"
        )

        features = [
            "f1",
            "f2",
            "f3",
            "f4",
        ]

        train, test, model = fit_split(
            df,
            LogisticRegression(),
            features=features,
        )

        bad_result = audit(
            train,
            test,
            "y",
            model=model,
            time_col="date",
        )

        assert_flagged(
            self,
            bad_result,
            "LEAK-004",
        )

        train2 = df.iloc[
            :1500
        ]

        test2 = df.iloc[
            1500:
        ]

        model2 = LogisticRegression().fit(
            train2[
                features
            ],
            train2.y,
        )

        clean_result = audit(
            train2,
            test2,
            "y",
            model=model2,
            time_col="date",
        )

        assert_passed(
            self,
            clean_result,
            "LEAK-004",
        )

    def test_leak005_id_feature(
        self,
    ):
        df = base_data()

        df.insert(
            0,
            "customer_id",
            np.arange(
                100000,
                100000
                + len(
                    df
                ),
            ),
        )

        train, test, model = fit_split(
            df,
            RandomForestClassifier(
                n_estimators=20,
                max_depth=4,
                random_state=0,
            ),
        )

        result = audit(
            train,
            test,
            "y",
            model=model,
        )

        assert_flagged(
            self,
            result,
            "LEAK-005",
        )

        finding = findings_for(
            result,
            "LEAK-005",
        )[0]

        self.assertEqual(
            finding.fix_action[
                "type"
            ],
            "drop_columns",
        )

        self.assertEqual(
            finding.fix_action[
                "columns"
            ],
            [
                "customer_id"
            ],
        )

    def test_leak005_named_entity_id_is_not_auto_dropped_without_sequence_evidence(
        self,
    ):
        df = base_data()

        df[
            "patient_id"
        ] = np.repeat(
            np.arange(
                len(
                    df
                )
                // 4
            ),
            4,
        )

        train, test, model = fit_split(
            df,
            RandomForestClassifier(
                n_estimators=20,
                max_depth=4,
                random_state=0,
            ),
        )

        result = audit(
            train,
            test,
            "y",
            model=model,
        )

        assert_flagged(
            self,
            result,
            "LEAK-005",
        )

        self.assertIsNone(
            findings_for(
                result,
                "LEAK-005",
            )[0].fix_action
        )

    def test_leak005_clean(
        self,
    ):
        df = base_data()

        df[
            "visits_last_month"
        ] = np.random.default_rng(
            4
        ).integers(
            0,
            30,
            len(
                df
            ),
        )

        train, test, model = fit_split(
            df,
            RandomForestClassifier(
                n_estimators=20,
                max_depth=4,
                random_state=0,
            ),
        )

        result = audit(
            train,
            test,
            "y",
            model=model,
        )

        assert_passed(
            self,
            result,
            "LEAK-005",
        )


class TestContamination(
    unittest.TestCase
):
    def test_cont001_duplicates_flagged(
        self,
    ):
        df = base_data()

        df = pd.concat(
            [
                df,
                df.sample(
                    frac=0.4,
                    random_state=1,
                ),
            ],
            ignore_index=True,
        )

        train, test, model = fit_split(
            df,
            RandomForestClassifier(
                n_estimators=30,
                random_state=0,
            ),
        )

        result = audit(
            train,
            test,
            "y",
            model=model,
        )

        assert_flagged(
            self,
            result,
            "CONT-001",
        )

        # Duplicate evidence is reported, but rows are not deleted automatically.
        self.assertIsNone(
            findings_for(
                result,
                "CONT-001",
            )[0].fix_action
        )

    def test_cont001_clean(
        self,
    ):
        train, test, model = fit_split(
            base_data(),
            LogisticRegression(),
        )

        result = audit(
            train,
            test,
            "y",
            model=model,
        )

        assert_passed(
            self,
            result,
            "CONT-001",
        )

    def test_cont001_naturally_repetitive_rows_are_skipped(
        self,
    ):
        rng = np.random.default_rng(
            0
        )

        patterns = pd.DataFrame(
            {
                "f1": [
                    0,
                    0,
                    1,
                    1,
                    2,
                    2,
                ],
                "f2": [
                    0,
                    1,
                    0,
                    1,
                    0,
                    1,
                ],
            }
        )

        df = patterns.iloc[
            rng.integers(
                0,
                len(
                    patterns
                ),
                size=2000,
            )
        ].reset_index(
            drop=True
        )

        df[
            "y"
        ] = (
            df[
                "f1"
            ]
            > 0
        ).astype(
            int
        )

        train, test = train_test_split(
            df,
            test_size=0.25,
            random_state=0,
        )

        model = LogisticRegression().fit(
            train[
                [
                    "f1",
                    "f2",
                ]
            ],
            train[
                "y"
            ],
        )

        result = audit(
            train,
            test,
            "y",
            model=model,
        )

        self.assertNotIn(
            "CONT-001",
            ids(
                result
            ),
        )

        run = check_run(
            result,
            "CONT-001",
        )

        self.assertEqual(
            run.status,
            "skipped",
        )

        self.assertIn(
            "naturally repetitive",
            run.note,
        )

    def test_cont002_group_overlap(
        self,
    ):
        df = base_data()

        df[
            "patient_id"
        ] = np.repeat(
            np.arange(
                len(
                    df
                )
                // 4
            ),
            4,
        )

        features = [
            "f1",
            "f2",
            "f3",
            "f4",
        ]

        train, test, model = fit_split(
            df,
            LogisticRegression(),
            features=features,
            stratify=False,
        )

        bad_result = audit(
            train,
            test,
            "y",
            model=model,
            group_col="patient_id",
        )

        assert_flagged(
            self,
            bad_result,
            "CONT-002",
        )

        self.assertEqual(
            findings_for(
                bad_result,
                "CONT-002",
            )[0].fix_action[
                "type"
            ],
            "group_split",
        )

        train2 = df[
            df.patient_id
            < 375
        ]

        test2 = df[
            df.patient_id
            >= 375
        ]

        model2 = LogisticRegression().fit(
            train2[
                features
            ],
            train2.y,
        )

        clean_result = audit(
            train2,
            test2,
            "y",
            model=model2,
            group_col="patient_id",
        )

        assert_passed(
            self,
            clean_result,
            "CONT-002",
        )

    def test_cont003_naive_cv_on_groups(
        self,
    ):
        df = base_data()

        df[
            "user_id"
        ] = np.repeat(
            np.arange(
                len(
                    df
                )
                // 4
            ),
            4,
        )

        features = [
            "f1",
            "f2",
            "f3",
            "f4",
        ]

        train, test, model = fit_split(
            df,
            LogisticRegression(),
            features=features,
            stratify=False,
        )

        bad = (
            "scores = cross_val_score("
            "model, X, y, cv=KFold(5, shuffle=True))\n"
        )

        good = (
            "scores = cross_val_score("
            "model, X, y, cv=GroupKFold(5), groups=g)\n"
        )

        bad_result = audit(
            train,
            test,
            "y",
            model=model,
            group_col="user_id",
            pipeline_code=bad,
        )

        assert_flagged(
            self,
            bad_result,
            "CONT-003",
        )

        clean_result = audit(
            train,
            test,
            "y",
            model=model,
            group_col="user_id",
            pipeline_code=good,
        )

        assert_passed(
            self,
            clean_result,
            "CONT-003",
        )

    def test_cont003_good_cv_elsewhere_does_not_hide_bad_cv(
        self,
    ):
        df = base_data()

        df[
            "user_id"
        ] = np.repeat(
            np.arange(
                len(
                    df
                )
                // 4
            ),
            4,
        )

        features = [
            "f1",
            "f2",
            "f3",
            "f4",
        ]

        train, test, model = fit_split(
            df,
            LogisticRegression(),
            features=features,
            stratify=False,
        )

        code = (
            "bad = cross_val_score("
            "model, X, y, cv=KFold(5, shuffle=True))\n"
            "good = cross_val_score("
            "model, X, y, cv=GroupKFold(5), groups=g)\n"
        )

        result = audit(
            train,
            test,
            "y",
            model=model,
            group_col="user_id",
            pipeline_code=code,
        )

        assert_flagged(
            self,
            result,
            "CONT-003",
        )


class TestPerformanceChecks(
    unittest.TestCase
):
    def test_metric001_accuracy_on_imbalanced(
        self,
    ):
        train, test, model = fit_split(
            base_data(
                rate=0.05
            ),
            LogisticRegression(),
        )

        bad_result = audit(
            train,
            test,
            "y",
            model=model,
            reported_metric="accuracy",
        )

        assert_flagged(
            self,
            bad_result,
            "METRIC-001",
        )

        clean_result = audit(
            train,
            test,
            "y",
            model=model,
            reported_metric="f1",
        )

        assert_passed(
            self,
            clean_result,
            "METRIC-001",
        )

    def test_metric001_unknown_metric_is_not_applicable(
        self,
    ):
        train, test, model = fit_split(
            base_data(
                rate=0.05
            ),
            LogisticRegression(),
        )

        result = audit(
            train,
            test,
            "y",
            model=model,
        )

        run = check_run(
            result,
            "METRIC-001",
        )

        self.assertEqual(
            run.status,
            "skipped",
        )

        self.assertIn(
            "not known",
            run.note,
        )

    def test_metric001_from_code(
        self,
    ):
        train, test, model = fit_split(
            base_data(
                rate=0.05
            ),
            LogisticRegression(),
        )

        acc_only = (
            "print(accuracy_score("
            "y_te, model.predict(X_te)))\n"
        )

        good = (
            "print(f1_score(y_te, p), "
            "roc_auc_score(y_te, q))\n"
        )

        bad_result = audit(
            train,
            test,
            "y",
            model=model,
            pipeline_code=acc_only,
        )

        assert_flagged(
            self,
            bad_result,
            "METRIC-001",
        )

        clean_result = audit(
            train,
            test,
            "y",
            model=model,
            pipeline_code=good,
        )

        assert_passed(
            self,
            clean_result,
            "METRIC-001",
        )

    def test_overfit001_gap(
        self,
    ):
        df = base_data(
            seed=3
        )

        df[
            "y"
        ] = np.random.default_rng(
            9
        ).integers(
            0,
            2,
            len(
                df
            ),
        )

        train, test, model = fit_split(
            df,
            RandomForestClassifier(
                n_estimators=50,
                random_state=0,
            ),
        )

        bad_result = audit(
            train,
            test,
            "y",
            model=model,
        )

        assert_flagged(
            self,
            bad_result,
            "OVERFIT-001",
        )

        train, test, model = fit_split(
            base_data(),
            LogisticRegression(),
        )

        clean_result = audit(
            train,
            test,
            "y",
            model=model,
        )

        assert_passed(
            self,
            clean_result,
            "OVERFIT-001",
        )

    def test_overfit002_perfect_score(
        self,
    ):
        df = base_data()

        df[
            "result_code"
        ] = df.y.astype(
            float
        )

        train, test, model = fit_split(
            df,
            LogisticRegression(),
        )

        result = audit(
            train,
            test,
            "y",
            model=model,
        )

        assert_flagged(
            self,
            result,
            "OVERFIT-002",
        )

        finding = findings_for(
            result,
            "OVERFIT-002",
        )[0]

        # Independent leakage evidence increases confidence.
        self.assertGreaterEqual(
            finding.confidence,
            0.8,
        )

    def test_overfit002_normal_score_clean(
        self,
    ):
        train, test, model = fit_split(
            base_data(),
            LogisticRegression(),
        )

        result = audit(
            train,
            test,
            "y",
            model=model,
        )

        assert_passed(
            self,
            result,
            "OVERFIT-002",
        )

    def test_overfit002_without_cause_is_low_confidence(
        self,
    ):
        rng = np.random.default_rng(
            0
        )

        df = pd.DataFrame(
            {
                "f1": rng.normal(
                    size=2000
                ),
                "f2": rng.normal(
                    size=2000
                ),
            }
        )

        df[
            "y"
        ] = (
            df.f1
            > 0
        ).astype(
            int
        )

        train, test, model = fit_split(
            df,
            LogisticRegression(
                C=100
            ),
        )

        result = audit(
            train,
            test,
            "y",
            model=model,
        )

        finding = findings_for(
            result,
            "OVERFIT-002",
        )[0]

        self.assertEqual(
            finding.severity,
            "low",
        )

        self.assertLess(
            finding.confidence,
            0.5,
        )

    def test_overfit002_one_class_test_is_not_applicable(
        self,
    ):
        rng = np.random.default_rng(
            0
        )

        train = pd.DataFrame(
            {
                "x": rng.normal(
                    size=200
                ),
                "y": np.r_[
                    np.zeros(
                        180,
                        dtype=int,
                    ),
                    np.ones(
                        20,
                        dtype=int,
                    ),
                ],
            }
        )

        test = pd.DataFrame(
            {
                "x": rng.normal(
                    size=50
                ),
                "y": np.zeros(
                    50,
                    dtype=int,
                ),
            }
        )

        model = LogisticRegression().fit(
            train[
                [
                    "x"
                ]
            ],
            train[
                "y"
            ],
        )

        result = audit(
            train,
            test,
            "y",
            model=model,
        )

        self.assertNotIn(
            "OVERFIT-002",
            ids(
                result
            ),
        )

        self.assertEqual(
            check_run(
                result,
                "OVERFIT-002",
            ).status,
            "skipped",
        )

    def test_overfit002_undefined_r2_is_not_applicable(
        self,
    ):
        rng = np.random.default_rng(
            0
        )

        x_train = rng.normal(
            size=100
        )

        train = pd.DataFrame(
            {
                "x": x_train,
                "y": (
                    2
                    * x_train
                    + rng.normal(
                        0,
                        0.1,
                        size=100,
                    )
                ),
            }
        )

        test = pd.DataFrame(
            {
                "x": [
                    0.25
                ],
                "y": [
                    0.5
                ],
            }
        )

        model = LinearRegression().fit(
            train[
                [
                    "x"
                ]
            ],
            train[
                "y"
            ],
        )

        result = audit(
            train,
            test,
            "y",
            model=model,
            task="regression",
        )

        self.assertNotIn(
            "OVERFIT-002",
            ids(
                result
            ),
        )

        self.assertEqual(
            check_run(
                result,
                "OVERFIT-002",
            ).status,
            "skipped",
        )

    def test_overfit001_downgraded_when_data_problem_explains_gap(
        self,
    ):
        from model_doctor.auditor import _cross_check
        from model_doctor.core import Finding

        def make_finding(
            check_id,
            category,
            severity,
            confidence,
        ):
            return Finding(
                check_id,
                category,
                check_id,
                severity,
                confidence,
                "w",
                "y",
                "f",
            )

        gap = make_finding(
            "OVERFIT-001",
            "overfitting",
            "high",
            0.85,
        )

        _cross_check(
            [
                gap,
                make_finding(
                    "DQ-004",
                    "data_quality",
                    "high",
                    0.9,
                ),
            ]
        )

        self.assertEqual(
            (
                gap.severity,
                round(
                    gap.confidence,
                    2,
                ),
            ),
            (
                "medium",
                0.55,
            ),
        )

        alone = make_finding(
            "OVERFIT-001",
            "overfitting",
            "high",
            0.85,
        )

        _cross_check(
            [
                alone,
                make_finding(
                    "DQ-004",
                    "data_quality",
                    "medium",
                    0.5,
                ),
            ]
        )

        self.assertEqual(
            (
                alone.severity,
                alone.confidence,
            ),
            (
                "high",
                0.85,
            ),
        )

    def test_imb002_low_when_rare_class_is_handled_well(
        self,
    ):
        rng = np.random.default_rng(
            0
        )

        df = pd.DataFrame(
            {
                "f1": rng.normal(
                    size=6000
                ),
                "f2": rng.normal(
                    size=6000
                ),
            }
        )

        df[
            "y"
        ] = (
            (
                3
                * df.f1
                + rng.normal(
                    0,
                    0.5,
                    6000,
                )
            )
            > 5
        ).astype(
            int
        )

        train, test, model = fit_split(
            df,
            LogisticRegression(),
        )

        result = audit(
            train,
            test,
            "y",
            model=model,
        )

        findings = findings_for(
            result,
            "IMB-002",
        )

        self.assertEqual(
            [
                finding.severity
                for finding in findings
            ],
            [
                "low"
            ],
        )

        self.assertIsNone(
            findings[
                0
            ].fix_action
        )

    def test_imb001_majority_collapse(
        self,
    ):
        df = base_data(
            n=4000,
            rate=0.03,
            seed=5,
        )

        df[
            "f1"
        ] = np.random.default_rng(
            2
        ).normal(
            size=len(
                df
            )
        )

        df[
            "f2"
        ] = np.random.default_rng(
            3
        ).normal(
            size=len(
                df
            )
        )

        train, test, model = fit_split(
            df,
            LogisticRegression(),
        )

        result = audit(
            train,
            test,
            "y",
            model=model,
        )

        assert_flagged(
            self,
            result,
            "IMB-001",
        )

        assert_flagged(
            self,
            result,
            "IMB-002",
        )

    def test_imb001_no_minority_examples_in_test_does_not_fake_zero_recall(
        self,
    ):
        rng = np.random.default_rng(
            0
        )

        train = pd.DataFrame(
            {
                "x": rng.normal(
                    size=200
                ),
                "y": np.r_[
                    np.zeros(
                        190,
                        dtype=int,
                    ),
                    np.ones(
                        10,
                        dtype=int,
                    ),
                ],
            }
        )

        test = pd.DataFrame(
            {
                "x": rng.normal(
                    size=60
                ),
                "y": np.zeros(
                    60,
                    dtype=int,
                ),
            }
        )

        model = DummyClassifier(
            strategy="most_frequent"
        ).fit(
            train[
                [
                    "x"
                ]
            ],
            train[
                "y"
            ],
        )

        result = audit(
            train,
            test,
            "y",
            model=model,
        )

        assert_passed(
            self,
            result,
            "IMB-001",
        )

    def test_imb_clean_with_class_weight(
        self,
    ):
        train, test, model = fit_split(
            base_data(
                n=4000,
                rate=0.05,
            ),
            LogisticRegression(
                class_weight="balanced"
            ),
        )

        result = audit(
            train,
            test,
            "y",
            model=model,
        )

        assert_passed(
            self,
            result,
            "IMB-001",
        )

        assert_passed(
            self,
            result,
            "IMB-002",
        )


class TestDataQuality(
    unittest.TestCase
):
    def test_dq000_model_crash(
        self,
    ):
        train, test, model = fit_split(
            base_data(),
            LogisticRegression(),
        )

        test = test.copy()

        test.loc[
            test.index[
                :10
            ],
            "f3",
        ] = np.nan

        result = audit(
            train,
            test,
            "y",
            model=model,
        )

        assert_flagged(
            self,
            result,
            "DQ-000",
        )

    def test_dq000_model_runs_clean(
        self,
    ):
        train, test, model = fit_split(
            base_data(),
            LogisticRegression(),
        )

        result = audit(
            train,
            test,
            "y",
            model=model,
        )

        assert_passed(
            self,
            result,
            "DQ-000",
        )

    def test_dq001_missing_only_in_test(
        self,
    ):
        train, test, _ = fit_split(
            base_data(),
            LogisticRegression(),
        )

        features = [
            "f1",
            "f2",
            "f3",
            "f4",
        ]

        model = HistGradientBoostingClassifier().fit(
            train[
                features
            ],
            train.y,
        )

        shifted_test = test.copy()

        shifted_test.loc[
            shifted_test.sample(
                frac=0.3,
                random_state=0,
            ).index,
            "f1",
        ] = np.nan

        bad_result = audit(
            train,
            shifted_test,
            "y",
            model=model,
        )

        assert_flagged(
            self,
            bad_result,
            "DQ-001",
        )

        train2, test2, model2 = fit_split(
            base_data(),
            LogisticRegression(),
        )

        clean_result = audit(
            train2,
            test2,
            "y",
            model=model2,
        )

        assert_passed(
            self,
            clean_result,
            "DQ-001",
        )

    def test_dq002_sentinel(
        self,
    ):
        df = base_data()

        df.loc[
            df.sample(
                frac=0.05,
                random_state=0,
            ).index,
            "f3",
        ] = -999

        train, test, model = fit_split(
            df,
            LogisticRegression(),
        )

        bad_result = audit(
            train,
            test,
            "y",
            model=model,
        )

        assert_flagged(
            self,
            bad_result,
            "DQ-002",
        )

        # Sentinel values are reported, not changed automatically.
        self.assertIsNone(
            findings_for(
                bad_result,
                "DQ-002",
            )[0].fix_action
        )

        train, test, model = fit_split(
            base_data(),
            LogisticRegression(),
        )

        clean_result = audit(
            train,
            test,
            "y",
            model=model,
        )

        assert_passed(
            self,
            clean_result,
            "DQ-002",
        )

    def test_dq003_unseen_category(
        self,
    ):
        df = base_data()

        df[
            "city"
        ] = np.random.default_rng(
            0
        ).choice(
            [
                "delhi",
                "jaipur",
                "pune",
            ],
            len(
                df
            ),
        )

        train, test = train_test_split(
            df,
            test_size=0.25,
            random_state=0,
        )

        shifted_test = test.copy()

        shifted_test.loc[
            shifted_test.index[
                :40
            ],
            "city",
        ] = "surat"

        bad_result = audit(
            train,
            shifted_test,
            "y",
        )

        assert_flagged(
            self,
            bad_result,
            "DQ-003",
        )

        train2, test2 = train_test_split(
            df,
            test_size=0.25,
            random_state=0,
        )

        clean_result = audit(
            train2,
            test2,
            "y",
        )

        assert_passed(
            self,
            clean_result,
            "DQ-003",
        )

    def test_dq005_distribution_shift(
        self,
    ):
        train, test, model = fit_split(
            base_data(),
            LogisticRegression(),
        )

        shifted_test = test.copy()

        shifted_test[
            "f3"
        ] = (
            shifted_test[
                "f3"
            ]
            * 3
            + 4
        )

        bad_result = audit(
            train,
            shifted_test,
            "y",
            model=model,
        )

        assert_flagged(
            self,
            bad_result,
            "DQ-005",
        )

        self.assertIn(
            "f3",
            findings_for(
                bad_result,
                "DQ-005",
            )[0].evidence[
                "psi_by_column"
            ],
        )

        clean_result = audit(
            train,
            test,
            "y",
            model=model,
        )

        assert_passed(
            self,
            clean_result,
            "DQ-005",
        )

    def _dq004_case(
        self,
        test_map,
        test_rates=None,
        seed=0,
        p=(
            0.1,
            0.2,
            0.3,
            0.4,
        ),
    ):
        rng = np.random.default_rng(
            seed
        )

        categories = [
            "a",
            "b",
            "c",
            "d",
        ]

        rates = {
            "a": 0.1,
            "b": 0.3,
            "c": 0.6,
            "d": 0.85,
        }

        def make(
            n,
            category_rates,
        ):
            category = rng.choice(
                categories,
                n,
                p=p,
            )

            target = (
                rng.random(
                    n
                )
                < pd.Series(
                    category
                ).map(
                    category_rates
                ).to_numpy()
            ).astype(
                int
            )

            return pd.DataFrame(
                {
                    "f1": rng.normal(
                        size=n
                    ),
                    "y": target,
                    "cat": category,
                }
            )

        train = make(
            2800,
            rates,
        )

        test = make(
            1200,
            test_rates
            or rates,
        )

        train[
            "cat"
        ] = train[
            "cat"
        ].map(
            {
                "a": 0,
                "b": 1,
                "c": 2,
                "d": 3,
            }
        )

        test[
            "cat"
        ] = test[
            "cat"
        ].map(
            test_map
        )

        model = RandomForestClassifier(
            n_estimators=30,
            max_depth=4,
            random_state=0,
        ).fit(
            train[
                [
                    "f1",
                    "cat",
                ]
            ],
            train.y,
        )

        return (
            train,
            test,
            model,
        )

    def test_dq004_data_only_is_possible_and_capped_at_medium(
        self,
    ):
        train, test, model = self._dq004_case(
            {
                "a": 3,
                "b": 2,
                "c": 1,
                "d": 0,
            }
        )

        result = audit(
            train,
            test,
            "y",
            model=model,
        )

        findings = findings_for(
            result,
            "DQ-004",
        )

        self.assertTrue(
            findings
        )

        self.assertIn(
            "Possible",
            findings[
                0
            ].title,
        )

        self.assertIn(
            findings[
                0
            ].severity,
            (
                "medium",
                "low",
            ),
        )

        self.assertLessEqual(
            findings[
                0
            ].confidence,
            0.5,
        )

        # Data-only evidence does not justify automatic exclusion.
        self.assertIsNone(
            findings[
                0
            ].fix_action
        )

    def test_dq004_code_evidence_raises_confidence_and_impact_sets_severity(
        self,
    ):
        train, test, model = self._dq004_case(
            {
                "a": 3,
                "b": 2,
                "c": 1,
                "d": 0,
            }
        )

        code = (
            "tr, te = train_test_split(df)\n"
            "te['cat'] = pd.factorize(te['cat'])[0]\n"
        )

        result = audit(
            train,
            test,
            "y",
            model=model,
            pipeline_code=code,
        )

        finding = findings_for(
            result,
            "DQ-004",
        )[0]

        self.assertGreaterEqual(
            finding.confidence,
            0.85,
        )

        self.assertEqual(
            finding.severity,
            "high",
        )

        self.assertGreater(
            finding.evidence[
                "score_change_if_remapped"
            ],
            0.05,
        )

        self.assertEqual(
            finding.fix_action[
                "type"
            ],
            "exclude_untrusted_column",
        )

    def test_dq004_concept_drift_is_never_high(
        self,
    ):
        identity = {
            "a": 0,
            "b": 1,
            "c": 2,
            "d": 3,
        }

        train, test, model = self._dq004_case(
            identity,
            test_rates={
                "a": 0.85,
                "b": 0.6,
                "c": 0.3,
                "d": 0.1,
            },
            p=(
                0.25,
                0.25,
                0.25,
                0.25,
            ),
        )

        findings = findings_for(
            audit(
                train,
                test,
                "y",
                model=model,
            ),
            "DQ-004",
        )

        self.assertTrue(
            findings
        )

        for finding in findings:
            self.assertNotEqual(
                finding.severity,
                "high",
            )

            self.assertLessEqual(
                finding.confidence,
                0.5,
            )

            self.assertIn(
                "Possible",
                finding.title,
            )

    def test_dq004_consistent_encoding_is_clean(
        self,
    ):
        train, test, model = self._dq004_case(
            {
                "a": 0,
                "b": 1,
                "c": 2,
                "d": 3,
            }
        )

        result = audit(
            train,
            test,
            "y",
            model=model,
        )

        assert_passed(
            self,
            result,
            "DQ-004",
        )


if __name__ == "__main__":
    unittest.main()
    