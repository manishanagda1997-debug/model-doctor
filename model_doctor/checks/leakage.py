"""Checks data leakage that can make evaluation results look better than real use."""

from __future__ import annotations

import re
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import balanced_accuracy_score, r2_score
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from ..core import ID_NAME_RE, Finding, SkipCheck, is_categorical, register
from ._utils import detect_time_cols, encode_pair, parse_datetimes, pct


LEAKY_TOKEN_PREFIXES = (
    "outcome",
    "result",
    "label",
    "target",
    "status",
    "final",
    "approv",
    "decision",
    "resolv",
    "closed",
    "paid",
    "default",
    "churn",
    "fraud",
    "after",
    "post",
    "refund",
    "collection",
    "chargeback",
    "diagnos",
    "cancel",
    "settle",
    "writeoff",
)


def _tokens(name: str) -> list[str]:
    """Returns lowercase words from a column or target name."""
    return [
        token
        for token in re.split(r"[_\W]+", str(name).lower())
        if token
    ]


def _name_hint(column: str, target: str) -> bool:
    """Returns True when a feature name looks related to an outcome."""
    column_tokens = _tokens(column)

    target_tokens = {
        token
        for token in _tokens(target)
        if len(token) > 2
    }

    outcome_hint = any(
        token.startswith(prefix)
        for token in column_tokens
        for prefix in LEAKY_TOKEN_PREFIXES
    )

    target_hint = bool(
        set(column_tokens)
        & target_tokens
    )

    return outcome_hint or target_hint


def _categorical_prediction(
    train_feature: pd.Series,
    test_feature: pd.Series,
    y_train: pd.Series,
    classification: bool,
):
    """Builds a simple train-only prediction from one categorical feature."""
    train_key = (
        train_feature
        .astype("string")
        .fillna("<NA>")
    )

    test_key = (
        test_feature
        .astype("string")
        .fillna("<NA>")
    )

    frame = pd.DataFrame(
        {
            "feature": train_key.to_numpy(),
            "target": y_train.to_numpy(),
        }
    )

    if classification:
        default = (
            y_train
            .value_counts()
            .index[0]
        )

        mapping = (
            frame
            .groupby(
                "feature",
                dropna=False,
            )["target"]
            .agg(
                lambda values: values.value_counts().index[0]
            )
        )

        prediction = [
            mapping.get(
                value,
                default,
            )
            for value in test_key
        ]

        try:
            return np.asarray(
                prediction,
                dtype=y_train.to_numpy().dtype,
            )

        except (TypeError, ValueError):
            return np.asarray(
                prediction
            )

    numeric_target = pd.to_numeric(
        frame["target"],
        errors="coerce",
    )

    valid = numeric_target.notna()

    if not valid.any():
        return None

    mapping = (
        pd.DataFrame(
            {
                "feature": frame.loc[
                    valid,
                    "feature",
                ],
                "target": numeric_target[
                    valid
                ].to_numpy(),
            }
        )
        .groupby(
            "feature"
        )["target"]
        .mean()
    )

    default = float(
        numeric_target[
            valid
        ].mean()
    )

    return (
        test_key
        .map(mapping)
        .fillna(default)
        .to_numpy(float)
    )


def _single_feature_skill(
    ctx,
    train: pd.DataFrame,
    test: pd.DataFrame,
    column: str,
) -> float | None:
    """Returns the test skill of one feature using a small train-only model."""
    y_train = train[
        ctx.target
    ]

    y_test = test[
        ctx.target
    ]

    if is_categorical(
        train[column]
    ):
        prediction = _categorical_prediction(
            train[column],
            test[column],
            y_train,
            ctx.is_classification,
        )

        if prediction is None:
            return None

    else:
        x_train, x_test = encode_pair(
            train[column],
            test[column],
        )

        if (
            len(x_train) == 0
            or np.nanstd(x_train) == 0
        ):
            return None

        leaf = max(
            5,
            len(train) // 500,
        )

        if ctx.is_classification:
            model = DecisionTreeClassifier(
                max_depth=4,
                min_samples_leaf=leaf,
                random_state=0,
            )

        else:
            model = DecisionTreeRegressor(
                max_depth=4,
                min_samples_leaf=leaf,
                random_state=0,
            )

        model.fit(
            x_train.reshape(-1, 1),
            y_train,
        )

        prediction = model.predict(
            x_test.reshape(-1, 1)
        )

    with warnings.catch_warnings():
        warnings.simplefilter(
            "ignore"
        )

        if ctx.is_classification:
            class_count = max(
                2,
                y_train.nunique(),
            )

            baseline = (
                1
                / class_count
            )

            score = balanced_accuracy_score(
                y_test,
                prediction,
            )

            return float(
                (score - baseline)
                / (1 - baseline)
            )

        y_test_numeric = pd.to_numeric(
            y_test,
            errors="coerce",
        ).to_numpy(
            dtype=float
        )

        prediction_numeric = np.asarray(
            prediction,
            dtype=float,
        )

        valid = (
            np.isfinite(
                y_test_numeric
            )
            & np.isfinite(
                prediction_numeric
            )
        )

        if valid.sum() < 2:
            return None

        return float(
            r2_score(
                y_test_numeric[
                    valid
                ],
                prediction_numeric[
                    valid
                ],
            )
        )


# LEAK-001: strong single-feature target proxies
@register(
    "LEAK-001",
    "Columns that give away the answer",
    "leakage",
    (
        "Tests every feature on its own. A feature with unusually high test skill "
        "can be a target proxy and needs a timing or meaning check."
    ),
)
def target_proxy_features(ctx):
    train = ctx.sample(
        ctx.train
    )

    test = ctx.sample(
        ctx.test
    )

    scores = []

    for column in ctx.feature_cols:
        try:
            skill = _single_feature_skill(
                ctx,
                train,
                test,
                column,
            )

        except Exception:  # noqa: BLE001
            continue

        if (
            skill is None
            or not np.isfinite(skill)
        ):
            continue

        scores.append(
            (
                column,
                float(skill),
            )
        )

    scores.sort(
        key=lambda item: -item[1]
    )

    top = {
        column: round(
            score,
            3,
        )
        for column, score in scores[:5]
    }

    findings = []

    for column, skill in scores[:5]:
        name_hint = _name_hint(
            column,
            ctx.target,
        )

        exact_target = (
            str(column)
            == str(ctx.target)
        )

        if skill >= 0.97:
            severity = (
                "critical"
                if name_hint
                else "high"
            )

            confidence = (
                0.9
                if name_hint
                else 0.55
            )

        elif skill >= 0.90:
            severity = (
                "high"
                if name_hint
                else "medium"
            )

            confidence = (
                0.8
                if name_hint
                else 0.4
            )

        elif (
            skill >= 0.80
            and name_hint
        ):
            severity = "medium"
            confidence = 0.6

        else:
            continue

        if name_hint:
            title = (
                f"Possible target leakage in '{column}'"
            )

        else:
            title = (
                f"Possible target proxy: '{column}' predicts the target almost on its own"
            )

        if ctx.is_classification:
            skill_text = (
                f"{pct(max(0.0, min(1.0, skill)))} normalized "
                "balanced-accuracy skill "
                "(0% = chance-level balanced accuracy, 100% = perfect)"
            )

        else:
            skill_text = (
                f"R-squared of {pct(max(0.0, min(1.0, skill)))} "
                "(0% = mean baseline, 100% = perfect)"
            )

        findings.append(
            Finding(
                "LEAK-001",
                "leakage",
                title,
                severity,
                confidence,
                what=(
                    f"Using only '{column}', a simple train-only rule reaches "
                    f"{skill_text} when predicting '{ctx.target}'."
                ),
                why=(
                    "A feature this strong can contain information that becomes available "
                    "only after the outcome is known. It can also be a valid measurement, "
                    "so the score alone does not prove leakage."
                ),
                fix=(
                    f"Check what '{column}' means and when it becomes available. "
                    "Remove it and retrain only if it is unavailable at prediction time "
                    "or is derived from the outcome."
                ),
                evidence={
                    "column": column,
                    "single_column_skill": round(
                        skill,
                        3,
                    ),
                    "name_suggests_leak": name_hint,
                    "exact_target_column": exact_target,
                    "top_single_column_scores": top,
                },
                # Only the target column itself is safe to remove automatically.
                fix_action=(
                    {
                        "type": "drop_columns",
                        "columns": [
                            column
                        ],
                    }
                    if exact_target
                    else None
                ),
            )
        )

    return findings


# LEAK-002: preprocessing fitted with test information
_STAT_ATTRS = {
    "StandardScaler": (
        "mean_",
        "mean",
    ),
    "RobustScaler": (
        "center_",
        "median",
    ),
    "MinMaxScaler": (
        "data_min_",
        "min",
    ),
}


def _resolve_columns(
    selector,
    input_cols,
) -> list:
    """Returns raw column names for common ColumnTransformer selectors."""
    columns = list(
        input_cols
    )

    if isinstance(
        selector,
        str,
    ):
        return [
            selector
        ]

    if (
        isinstance(
            selector,
            (int, np.integer),
        )
        and not isinstance(
            selector,
            (bool, np.bool_),
        )
    ):
        index = int(
            selector
        )

        if (
            -len(columns)
            <= index
            < len(columns)
        ):
            return [
                columns[index]
            ]

        return []

    if isinstance(
        selector,
        slice,
    ):
        return columns[
            selector
        ]

    if hasattr(
        selector,
        "tolist",
    ):
        selector = selector.tolist()

    if not isinstance(
        selector,
        (
            list,
            tuple,
        ),
    ):
        return []

    if (
        len(selector)
        == len(columns)
        and all(
            isinstance(
                item,
                (bool, np.bool_),
            )
            for item in selector
        )
    ):
        return [
            column
            for column, keep in zip(
                columns,
                selector,
            )
            if keep
        ]

    resolved = []

    for item in selector:
        if isinstance(
            item,
            str,
        ):
            resolved.append(
                item
            )

        elif (
            isinstance(
                item,
                (int, np.integer),
            )
            and not isinstance(
                item,
                (bool, np.bool_),
            )
        ):
            index = int(
                item
            )

            if not (
                -len(columns)
                <= index
                < len(columns)
            ):
                return []

            resolved.append(
                columns[index]
            )

        else:
            return []

    return resolved


def _fitted_transformers(
    model,
    input_cols,
):
    """Returns fitted preprocessing steps that still map to raw input columns."""
    if not isinstance(
        model,
        Pipeline,
    ):
        return []

    out = []

    def walk_pipeline(
        steps,
        columns,
    ):
        for _, step in steps:
            if isinstance(
                step,
                ColumnTransformer,
            ):
                walk_column_transformer(
                    step,
                    columns,
                )

                # A ColumnTransformer changes the feature layout, so later
                # steps cannot safely be matched back to the raw columns.
                return False

            if isinstance(
                step,
                Pipeline,
            ):
                if not walk_pipeline(
                    step.steps,
                    columns,
                ):
                    return False

                continue

            name = type(
                step
            ).__name__

            if (
                name == "SimpleImputer"
                or name in _STAT_ATTRS
            ):
                out.append(
                    (
                        step,
                        list(
                            columns
                        ),
                    )
                )

                continue

            # An unknown transformation may change the feature layout.
            # Later statistics are no longer safe to match to raw columns.
            return False

        return True

    def walk_column_transformer(
        transformer,
        columns,
    ):
        for _, nested, selector in getattr(
            transformer,
            "transformers_",
            [],
        ):
            if (
                isinstance(
                    nested,
                    str,
                )
                and nested
                in (
                    "drop",
                    "passthrough",
                )
            ):
                continue

            selected = _resolve_columns(
                selector,
                columns,
            )

            if not selected:
                continue

            if isinstance(
                nested,
                Pipeline,
            ):
                walk_pipeline(
                    nested.steps,
                    selected,
                )

                continue

            name = type(
                nested
            ).__name__

            if (
                name == "SimpleImputer"
                or name in _STAT_ATTRS
            ):
                out.append(
                    (
                        nested,
                        list(
                            selected
                        ),
                    )
                )

    walk_pipeline(
        model.steps[:-1],
        list(
            input_cols
        ),
    )

    return out


def _stat(
    df: pd.DataFrame,
    how: str,
) -> np.ndarray:
    """Returns one simple numeric statistic for each column."""
    numeric = df.apply(
        pd.to_numeric,
        errors="coerce",
    )

    function = {
        "mean": numeric.mean,
        "median": numeric.median,
        "min": numeric.min,
    }[
        how
    ]

    return function().to_numpy(
        dtype=float
    )


@register(
    "LEAK-002",
    "Preprocessing that learns from the test data",
    "leakage",
    (
        "Checks whether fitted preprocessing statistics match train + test data "
        "instead of training data only, and checks for full-data scaling fingerprints."
    ),
)
def preprocessing_fit_on_full_data(ctx):
    if ctx.flags.get(
        "preprocessing_rebuilt"
    ):
        raise SkipCheck(
            "The reference rebuild fits preprocessing on training data only."
        )

    full = pd.concat(
        [
            ctx.train,
            ctx.test,
        ],
        ignore_index=True,
    )

    fix_text = (
        "Split the data first, then fit scaling, imputing, encoding, and other "
        "learned preprocessing on the training data only."
    )

    why_text = (
        "Preprocessing that learns from test rows makes the evaluation less independent. "
        "The size of the effect depends on the preprocessing step and the data."
    )

    matched_full = []
    matched_train = []
    compared = 0
    inspectable = 0
    compared_columns = set()

    # Path A checks fitted statistics in a saved Pipeline.
    for transformer, columns in _fitted_transformers(
        ctx.model,
        ctx.feature_cols,
    ):
        name = type(
            transformer
        ).__name__

        if (
            name == "SimpleImputer"
            and getattr(
                transformer,
                "strategy",
                "",
            )
            in (
                "mean",
                "median",
            )
        ):
            attribute = "statistics_"
            how = transformer.strategy

        elif name in _STAT_ATTRS:
            attribute, how = _STAT_ATTRS[
                name
            ]

        else:
            continue

        fitted = getattr(
            transformer,
            attribute,
            None,
        )

        columns = [
            column
            for column in columns
            if column in ctx.train.columns
        ]

        if (
            fitted is None
            or not columns
        ):
            continue

        try:
            fitted_values = np.asarray(
                fitted,
                dtype=float,
            ).reshape(-1)

        except (TypeError, ValueError):
            continue

        if len(fitted_values) != len(columns):
            continue

        inspectable += len(
            columns
        )

        train_stats = _stat(
            ctx.train[
                columns
            ],
            how,
        )

        full_stats = _stat(
            full[
                columns
            ],
            how,
        )

        for (
            column,
            fitted_value,
            train_value,
            full_value,
        ) in zip(
            columns,
            fitted_values,
            train_stats,
            full_stats,
        ):
            if not all(
                np.isfinite(value)
                for value in (
                    fitted_value,
                    train_value,
                    full_value,
                )
            ):
                continue

            difference = abs(
                train_value
                - full_value
            )

            if difference <= 1e-9 * (
                1
                + abs(train_value)
            ):
                continue

            full_match = abs(
                fitted_value
                - full_value
            ) < 0.1 * abs(
                fitted_value
                - train_value
            )

            train_match = abs(
                fitted_value
                - train_value
            ) < 0.1 * abs(
                fitted_value
                - full_value
            )

            if (
                not (
                    full_match
                    or train_match
                )
                or column in compared_columns
            ):
                continue

            compared_columns.add(
                column
            )

            compared += 1

            if full_match:
                matched_full.append(
                    column
                )

            elif train_match:
                matched_train.append(
                    column
                )

    enough_full_matches = (
        bool(matched_full)
        and len(matched_full)
        >= max(
            1,
            int(
                np.ceil(
                    0.6
                    * compared
                )
            ),
        )
        and len(matched_full)
        > len(matched_train)
    )

    if enough_full_matches:
        return [
            Finding(
                "LEAK-002",
                "leakage",
                "The fitted preprocessing uses information from the test data",
                "medium",
                0.9,
                what=(
                    "The saved Pipeline contains fitted statistics for "
                    f"{len(matched_full)} column(s) that match the combined train + test "
                    "data more closely than the training data alone."
                ),
                why=why_text,
                fix=(
                    fix_text
                    + " Refit the complete Pipeline using the training rows only."
                ),
                evidence={
                    "columns_matching_full_data": matched_full[:15],
                    "columns_matching_train_only": matched_train[:15],
                    "columns_compared": compared,
                    "source": "fitted pipeline",
                },
                fix_action={
                    "type": "rebuild_preprocessing"
                },
            )
        ]

    # Path B checks data that already carries a scaling fingerprint.
    numeric_columns = [
        column
        for column in ctx.feature_cols
        if pd.api.types.is_float_dtype(
            ctx.train[column]
        )
        and ctx.train[
            column
        ].nunique() > 20
    ]

    if len(numeric_columns) < 2:
        if (
            inspectable
            or compared
        ):
            return []

        raise SkipCheck(
            "The saved model has no fitted scaler statistics to compare, "
            "and the data has too few continuous numeric columns for a scaling fingerprint."
        )

    full_standard = [
        column
        for column in numeric_columns
        if abs(
            full[
                column
            ].mean()
        ) < 1e-6
        and abs(
            full[
                column
            ].std(
                ddof=0
            )
            - 1
        ) < 1e-4
    ]

    train_standard = [
        column
        for column in numeric_columns
        if abs(
            ctx.train[
                column
            ].mean()
        ) < 1e-6
        and abs(
            ctx.train[
                column
            ].std(
                ddof=0
            )
            - 1
        ) < 1e-4
    ]

    full_minmax = [
        column
        for column in numeric_columns
        if abs(
            full[
                column
            ].min()
        ) < 1e-9
        and abs(
            full[
                column
            ].max()
            - 1
        ) < 1e-9
    ]

    train_minmax = [
        column
        for column in numeric_columns
        if abs(
            ctx.train[
                column
            ].min()
        ) < 1e-9
        and abs(
            ctx.train[
                column
            ].max()
            - 1
        ) < 1e-9
    ]

    for (
        kind,
        full_columns,
        train_columns,
    ) in (
        (
            "standard-scaled",
            full_standard,
            train_standard,
        ),
        (
            "min-max scaled",
            full_minmax,
            train_minmax,
        ),
    ):
        if (
            len(full_columns)
            >= max(
                2,
                int(
                    np.ceil(
                        0.5
                        * len(numeric_columns)
                    )
                ),
            )
            and len(train_columns)
            < 0.5
            * len(full_columns)
        ):
            return [
                Finding(
                    "LEAK-002",
                    "leakage",
                    "The data has a full-dataset scaling fingerprint",
                    "medium",
                    0.8,
                    what=(
                        f"{len(full_columns)} numeric column(s) are almost perfectly "
                        f"{kind} only when training and test rows are combined."
                    ),
                    why=(
                        "This pattern strongly suggests that scaling is fitted before "
                        "the split. It is a fingerprint rather than direct proof, so "
                        "the original data preparation should be checked."
                    ),
                    fix=(
                        "Return to the raw, unscaled data. Split it first, fit the scaler "
                        "on training rows only, transform the test rows with that fitted "
                        "scaler, and retrain the model."
                    ),
                    evidence={
                        "kind": kind,
                        "columns": full_columns[:15],
                        "source": "data fingerprint",
                        "automatic_fix": False,
                    },
                )
            ]

    return []


# LEAK-003: leakage visible in training code
@register(
    "LEAK-003",
    "Leaky steps in the training code",
    "leakage",
    (
        "Reads the training script or notebook without running it and flags "
        "preprocessing fitted before evaluation, fitted on test data, or "
        "whole-dataset imputation."
    ),
)
def leaky_code_patterns(ctx):
    facts = ctx.code_facts

    if facts is None:
        raise SkipCheck(
            "No training code is provided."
        )

    if not facts.parsed:
        raise SkipCheck(
            facts.error
            or "The training code cannot be parsed."
        )

    findings = []

    before = [
        item
        for item in facts.fits
        if item[
            "before_split"
        ]
        and not item[
            "on_test"
        ]
    ]

    on_test = [
        item
        for item in facts.fits
        if item[
            "on_test"
        ]
    ]

    if before:
        high_impact = any(
            item[
                "high_impact"
            ]
            for item in before
        )

        steps = ", ".join(
            sorted(
                {
                    item[
                        "cls"
                    ]
                    for item in before
                }
            )
        )

        findings.append(
            Finding(
                "LEAK-003",
                "leakage",
                f"Preprocessing ({steps}) runs before the evaluation split",
                (
                    "high"
                    if high_impact
                    else "medium"
                ),
                0.9,
                what=(
                    f"In the training code, {steps} is fitted on line(s) "
                    f"{', '.join(str(item['line']) for item in before)}, "
                    "before the first train/test split or cross-validation run "
                    f"on line {facts.boundary_line}."
                ),
                why=(
                    "A learned step that runs before the evaluation boundary can use "
                    "information from rows that later act as validation or test data. "
                    + (
                        "Feature selection, target encoding, and resampling are especially "
                        "important because they can change which patterns the model learns."
                        if high_impact
                        else
                        "Scaling or imputation usually has a smaller effect, but the "
                        "evaluation is still no longer fully independent."
                    )
                ),
                fix=(
                    "Create the split first. Fit learned preprocessing on training data only, "
                    "and keep fold-specific preprocessing inside the cross-validation Pipeline."
                ),
                evidence={
                    "leaky_lines": [
                        {
                            key: item[key]
                            for key in (
                                "line",
                                "cls",
                                "code",
                            )
                        }
                        for item in before
                    ],
                    "evaluation_boundary_line": facts.boundary_line,
                },
                fix_action={
                    "type": "rebuild_preprocessing"
                },
            )
        )

    if on_test:
        findings.append(
            Finding(
                "LEAK-003",
                "leakage",
                "A learned preprocessing step is fitted on the test data",
                "high",
                0.9,
                what=(
                    "The code calls fit, fit_transform, or fit_resample on test data: "
                    + "; ".join(
                        f"line {item['line']}: {item['code']}"
                        for item in on_test[:4]
                    )
                    + "."
                ),
                why=(
                    "Test data should only pass through transformations learned from "
                    "training data. Fitting a step on the test set can use test information "
                    "or create a different representation from the one the model learns."
                ),
                fix=(
                    "Fit each learned preprocessing step on training data only. "
                    "Use transform() on test and new data, and keep supervised steps "
                    "inside the training or cross-validation workflow."
                ),
                evidence={
                    "lines": [
                        {
                            key: item[key]
                            for key in (
                                "line",
                                "cls",
                                "code",
                            )
                        }
                        for item in on_test
                    ]
                },
                fix_action={
                    "type": "rebuild_preprocessing"
                },
            )
        )

    if facts.full_data_fills:
        findings.append(
            Finding(
                "LEAK-003",
                "leakage",
                "Missing values are filled using the whole dataset",
                "low",
                0.8,
                what=(
                    "The code fills missing values with a mean, median, or mode "
                    "computed before the evaluation split "
                    f"(line {', '.join(str(item['line']) for item in facts.full_data_fills)})."
                ),
                why=(
                    "The fill value uses information from rows that later act as validation "
                    "or test data, so the evaluation is not fully independent."
                ),
                fix=(
                    "Use an imputer inside the Pipeline so its values come from "
                    "training rows only."
                ),
                evidence={
                    "lines": facts.full_data_fills
                },
                fix_action={
                    "type": "rebuild_preprocessing"
                },
            )
        )

    if (
        facts.boundary_line is None
        and not findings
    ):
        raise SkipCheck(
            "The code contains no train/test split or cross-validation run to inspect."
        )

    return findings


# LEAK-004: time-order leakage
@register(
    "LEAK-004",
    "Training on the future (time leakage)",
    "leakage",
    "Checks whether training and test periods overlap when time order matters.",
)
def temporal_leakage(ctx):
    explicit = (
        ctx.time_col
        is not None
    )

    forecast_goal = bool(
        re.search(
            r"forecast|future|next|predict.*(day|week|month|year)",
            ctx.goal,
            re.I,
        )
    )

    columns = (
        [
            ctx.time_col
        ]
        if explicit
        else detect_time_cols(
            ctx.train,
            exclude={
                ctx.target
            },
        )
    )

    if not columns:
        raise SkipCheck(
            "The data has no date/time column."
        )

    column = columns[0]

    train_time = parse_datetimes(
        ctx.train[column]
    )

    test_time = parse_datetimes(
        ctx.test[column]
    )

    if (
        train_time.notna().mean() < 0.9
        or test_time.notna().mean() < 0.9
    ):
        raise SkipCheck(
            f"Column '{column}' cannot be read reliably as dates."
        )

    first_test = test_time.min()

    share_after = float(
        (
            train_time
            > first_test
        ).mean()
    )

    if share_after < 0.02:
        return []

    time_matters = (
        explicit
        or forecast_goal
    )

    if time_matters:
        severity = (
            "high"
            if share_after > 0.05
            else "medium"
        )

        confidence = (
            0.9
            if explicit
            else 0.8
        )

        title = (
            "Training data includes dates after the test period begins"
        )

        why = (
            "When the goal depends on time order, evaluation should reproduce the "
            "information available at prediction time. Training on later dates can "
            "make the test score too optimistic."
        )

        fix = (
            f"Sort by '{column}', train on the earlier period, and test on the most "
            "recent period. Use TimeSeriesSplit for cross-validation when appropriate."
        )

        fix_action = {
            "type": "time_split",
            "time_col": column,
        }

    else:
        severity = "low"
        confidence = 0.4

        title = (
            f"Possible time-order issue in '{column}'"
        )

        why = (
            "The date ranges overlap, but the audit does not know whether this date "
            "defines the prediction timeline. The overlap is only a review signal."
        )

        fix = (
            f"Confirm whether '{column}' determines what is known at prediction time. "
            "If it does, evaluate with an earlier-training and later-test split."
        )

        fix_action = None

    return [
        Finding(
            "LEAK-004",
            "leakage",
            title,
            severity,
            confidence,
            what=(
                f"{pct(share_after)} of training rows are dated after the first test date "
                f"(train: {train_time.min().date()} to {train_time.max().date()}, "
                f"test: {first_test.date()} to {test_time.max().date()}). "
                "The training and test periods overlap."
            ),
            why=why,
            fix=fix,
            evidence={
                "time_column": column,
                "explicitly_given": explicit,
                "forecast_goal": forecast_goal,
                "train_rows_after_first_test_date": round(
                    share_after,
                    3,
                ),
            },
            fix_action=fix_action,
        )
    ]


# LEAK-005: ID-like model features
def _sequence_like_id(
    train: pd.Series,
    test: pd.Series,
) -> bool:
    """Returns True when a column looks like a nearly unique row number."""
    if (
        len(train) <= 50
        or train.nunique(
            dropna=True
        )
        / max(
            1,
            train.notna().sum(),
        )
        <= 0.95
    ):
        return False

    combined = pd.to_numeric(
        pd.concat(
            [
                train,
                test,
            ],
            ignore_index=True,
        ),
        errors="coerce",
    )

    combined = combined[
        np.isfinite(
            combined
        )
    ]

    if len(combined) <= 50:
        return False

    values = combined.to_numpy(
        dtype=float
    )

    if not np.allclose(
        values,
        np.round(values),
    ):
        return False

    distinct = combined.nunique()

    if (
        distinct
        / len(combined)
        <= 0.99
    ):
        return False

    return bool(
        (
            combined.max()
            - combined.min()
            + 1
        )
        <= 1.05
        * distinct
    )


@register(
    "LEAK-005",
    "ID columns used as features",
    "leakage",
    "Flags likely row or entity identifiers that the model uses as features.",
)
def id_as_feature(ctx):
    findings = []

    for column in ctx.feature_cols:
        train = ctx.train[
            column
        ]

        name_hit = bool(
            ID_NAME_RE.search(
                str(column)
            )
        )

        distinct = train.nunique(
            dropna=True
        )

        sequence_like = _sequence_like_id(
            train,
            ctx.test[column],
        )

        if not (
            (
                name_hit
                and distinct >= 20
            )
            or sequence_like
        ):
            continue

        obvious_row_id = (
            name_hit
            and sequence_like
        )

        confidence = (
            0.9
            if obvious_row_id
            else (
                0.75
                if name_hit
                else 0.6
            )
        )

        title = (
            f"The identifier '{column}' is used as a feature"
            if name_hit
            else f"'{column}' looks like a row identifier"
        )

        findings.append(
            Finding(
                "LEAK-005",
                "leakage",
                title,
                "medium",
                confidence,
                what=(
                    f"'{column}' has {distinct} distinct training values and is used "
                    "as a model input."
                    + (
                        " Its values also look like a nearly unique sequence."
                        if sequence_like
                        else ""
                    )
                ),
                why=(
                    "Raw identifiers often describe which row or entity this is rather "
                    "than the pattern the model should learn. They can encourage memorisation "
                    "or capture accidental ordering. Some entity identifiers are intentional, "
                    "so their meaning should be checked before removal."
                ),
                fix=(
                    f"Check how '{column}' is used at prediction time. Remove it from the "
                    "model inputs if it is only an identifier; keep it if it has a documented "
                    "predictive meaning that is available for future cases."
                ),
                evidence={
                    "column": column,
                    "distinct_values": int(
                        distinct
                    ),
                    "name_looks_like_id": name_hit,
                    "sequence_like": sequence_like,
                },
                # Only a named, nearly unique sequence is removed automatically.
                fix_action=(
                    {
                        "type": "drop_columns",
                        "columns": [
                            column
                        ],
                    }
                    if obvious_row_id
                    else None
                ),
            )
        )

    return findings
