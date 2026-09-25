"""Checks data-quality problems that can reduce model performance."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import chi2

from ..core import (
    Finding,
    SkipCheck,
    evaluate,
    feature_names_of,
    is_categorical,
    register,
)
from ._utils import pct, psi

SENTINELS = [
    -999999,
    -99999,
    -9999,
    -999,
    -99,
    -1,
    999,
    9999,
    99999,
    999999,
]


def _error_mentions_missing_values(error: str | None) -> bool:
    """Returns True when a prediction error clearly points to missing values."""
    if not error:
        return False

    text = error.lower()

    return any(
        marker in text
        for marker in (
            "nan",
            "missing value",
            "missing values",
        )
    )


def _error_mentions_categories(error: str | None) -> bool:
    """Returns True when a prediction error clearly points to unseen categories."""
    if not error:
        return False

    text = error.lower()

    markers = (
        "unknown categor",
        "unknown label",
        "unseen label",
        "previously unseen",
        "found unknown",
        "could not convert string to float",
        "invalid literal for float",
    )

    return any(marker in text for marker in markers)


def _numeric_series(series: pd.Series) -> pd.Series:
    """Returns a float series and turns non-numeric values into NaN."""
    return pd.to_numeric(
        series,
        errors="coerce",
    ).astype(float)


def _prediction_error(ctx, split: str) -> str | None:
    """Returns the prediction error for one split without mixing split errors."""
    errors = ctx.flags.setdefault(
        "prediction_errors",
        {},
    )

    if split in errors:
        return errors[split]

    prediction = ctx.predict(split)

    error = ctx.model_error if prediction is None else None

    errors[split] = error

    return error


def _looks_like_integer_codes(series: pd.Series) -> bool:
    """Returns True when a numeric column contains a small set of integer-like codes."""
    if not pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
        return False

    values = _numeric_series(series)
    values = values[np.isfinite(values)]

    if values.empty:
        return False

    unique_count = values.nunique()

    if not 3 <= unique_count <= 30:
        return False

    return bool(
        np.allclose(
            values.to_numpy(),
            np.round(values.to_numpy()),
        )
    )


@register(
    "DQ-000",
    "Model runs on the data",
    "data_quality",
    "Checks that the model can make predictions on the provided data.",
)
def model_runs(ctx):
    if ctx.model is None:
        raise SkipCheck("No model is provided.")

    train_error = _prediction_error(
        ctx,
        "train",
    )

    test_error = _prediction_error(
        ctx,
        "test",
    )

    errors = {
        split: error
        for split, error in (
            (
                "train",
                train_error,
            ),
            (
                "test",
                test_error,
            ),
        )
        if error
    }

    if not errors:
        return []

    affected = " and ".join(errors)

    details = "; ".join(f"{split}: {error[:300]}" for split, error in errors.items())

    return [
        Finding(
            "DQ-000",
            "data_quality",
            f"The model cannot predict on the {affected} data",
            "critical",
            0.95,
            what=(
                f"The model returns a prediction error on {affected} data: "
                f"{details}"
            ),
            why=(
                "The saved model and the provided data are not fully compatible. "
                "Similar inputs can also fail during real use until the cause is fixed."
            ),
            fix=(
                "Read the prediction error, check the expected columns and data types, "
                "and put any required imputing or encoding inside the saved Pipeline."
            ),
            evidence={
                "prediction_errors": {
                    split: error[:1000] for split, error in errors.items()
                }
            },
        )
    ]


@register(
    "DQ-001",
    "Missing values handled differently in train and test",
    "data_quality",
    "Compares missing-value rates per column between training and test data.",
)
def missing_values(ctx):
    rows = []

    for column in ctx.feature_cols:
        train_missing = float(ctx.train[column].isna().mean())

        test_missing = float(ctx.test[column].isna().mean())

        if train_missing > 0 or test_missing > 0:
            rows.append(
                (
                    column,
                    train_missing,
                    test_missing,
                )
            )

    if not rows:
        return []

    test_error = (
        _prediction_error(
            ctx,
            "test",
        )
        if ctx.model is not None
        else None
    )

    findings = []

    shifted = [
        (
            column,
            train_missing,
            test_missing,
        )
        for column, train_missing, test_missing in rows
        if test_missing - train_missing >= 0.05
    ]

    heavy = [
        (
            column,
            train_missing,
            test_missing,
        )
        for column, train_missing, test_missing in rows
        if train_missing >= 0.4
    ]

    if shifted:
        missing_error = _error_mentions_missing_values(test_error)

        severity = "high" if missing_error else "medium"

        confidence = 0.9 if missing_error else 0.75

        extra = (
            " The model also returns a prediction error that mentions missing values."
            if missing_error
            else ""
        )

        findings.append(
            Finding(
                "DQ-001",
                "data_quality",
                "Test data has more missing values than training data",
                severity,
                confidence,
                what=(
                    "These columns are missing more often in the test data than in training: "
                    + "; ".join(
                        f"'{column}' {pct(train_missing)} -> {pct(test_missing)}"
                        for column, train_missing, test_missing in shifted[:6]
                    )
                    + "."
                    + extra
                ),
                why=(
                    "This is a change in the input data. A model can become less reliable "
                    "when training contains few examples with the same missing-value pattern."
                ),
                fix=(
                    "Check why the values go missing in the test or newer data. "
                    "If missing values are expected, handle them inside the Pipeline "
                    "and include realistic examples during training."
                ),
                evidence={
                    "columns": {
                        column: {
                            "train_missing": round(
                                train_missing,
                                3,
                            ),
                            "test_missing": round(
                                test_missing,
                                3,
                            ),
                        }
                        for column, train_missing, test_missing in shifted
                    },
                    "prediction_error_mentions_missing_values": missing_error,
                },
                fix_action=(
                    {"type": "rebuild_preprocessing"} if missing_error else None
                ),
            )
        )

    if heavy:
        findings.append(
            Finding(
                "DQ-001",
                "data_quality",
                "Some training columns are mostly empty",
                "low",
                0.7,
                what=(
                    "These columns are missing in 40% or more of training rows: "
                    + ", ".join(
                        f"'{column}' ({pct(train_missing)})"
                        for column, train_missing, _ in heavy[:6]
                    )
                    + "."
                ),
                why=(
                    "Mostly empty columns carry limited information, and different "
                    "filling rules can change model behavior."
                ),
                fix=(
                    "Review each column and choose a documented rule: keep it with "
                    "suitable missing-value handling, add a missing-value flag when useful, "
                    "or remove it if it adds little value."
                ),
                evidence={
                    "columns": {
                        column: round(
                            train_missing,
                            3,
                        )
                        for column, train_missing, _ in heavy
                    }
                },
            )
        )

    return findings


@register(
    "DQ-002",
    "Missing values disguised as numbers",
    "data_quality",
    "Finds placeholder codes such as -999 or 9999 that can stand in for unknown values.",
)
def disguised_missing(ctx):
    hits = {}

    for column in ctx.feature_cols:
        train_raw = ctx.train[column]

        if not pd.api.types.is_numeric_dtype(train_raw) or pd.api.types.is_bool_dtype(
            train_raw
        ):
            continue

        train = _numeric_series(train_raw)

        test = _numeric_series(ctx.test[column])

        train_values = train[np.isfinite(train)]

        test_values = test[np.isfinite(test)]

        if train_values.nunique() < 10:
            continue

        candidates = []

        for value in SENTINELS:
            train_share = (
                float((train_values == value).mean()) if len(train_values) else 0.0
            )

            test_share = (
                float((test_values == value).mean()) if len(test_values) else 0.0
            )

            if (
                max(
                    train_share,
                    test_share,
                )
                < 0.01
            ):
                continue

            reference = train_values[train_values != value]

            if reference.empty:
                continue

            q1 = float(reference.quantile(0.01))

            q99 = float(reference.quantile(0.99))

            span = max(
                q99 - q1,
                1e-9,
            )

            if not (value < q1 - 0.5 * span or value > q99 + 0.5 * span):
                continue

            candidates.append(
                {
                    "value": value,
                    "train_share": train_share,
                    "test_share": test_share,
                    "normal_range": [
                        round(
                            q1,
                            3,
                        ),
                        round(
                            q99,
                            3,
                        ),
                    ],
                }
            )

        if candidates:
            best = max(
                candidates,
                key=lambda item: max(
                    item["train_share"],
                    item["test_share"],
                ),
            )

            hits[column] = {
                "value": best["value"],
                "train_share": round(
                    best["train_share"],
                    4,
                ),
                "test_share": round(
                    best["test_share"],
                    4,
                ),
                "normal_range": best["normal_range"],
            }

    if not hits:
        return []

    return [
        Finding(
            "DQ-002",
            "data_quality",
            "Placeholder numbers may represent missing values",
            "medium",
            0.75,
            what=(
                "These columns contain a repeated code far outside the usual training range: "
                + "; ".join(
                    (
                        f"'{column}' uses {info['value']} "
                        f"(train {pct(info['train_share'])}, "
                        f"test {pct(info['test_share'])}; "
                        f"usual range {info['normal_range'][0]} "
                        f"to {info['normal_range'][1]})"
                    )
                    for column, info in list(hits.items())[:5]
                )
                + "."
            ),
            why=(
                "A placeholder such as -999 can act like a real extreme measurement. "
                "This can distort scaling, averages, and model decisions when the code "
                "really means 'unknown'."
            ),
            fix=(
                "Confirm that each code means a missing value. If it does, replace it "
                "with NaN, add a missing-value flag when useful, and impute inside the Pipeline."
            ),
            evidence={
                "columns": hits,
                "automatic_fix": False,
                "note": (
                    "The values need confirmation before they are changed to missing values."
                ),
            },
        )
    ]


@register(
    "DQ-003",
    "New categories that only appear in test data",
    "data_quality",
    "Lists text categories in the test data that do not appear in the training data.",
)
def unseen_categories(ctx):
    columns = [
        column for column in ctx.feature_cols if is_categorical(ctx.train[column])
    ]

    if not columns:
        raise SkipCheck("No text/category columns.")

    test_error = (
        _prediction_error(
            ctx,
            "test",
        )
        if ctx.model is not None
        else None
    )

    hits = {}

    for column in columns:
        seen = set(ctx.train[column].dropna().astype(str))

        test_values = ctx.test[column].dropna().astype(str)

        unseen = test_values[~test_values.isin(seen)]

        share = len(unseen) / max(
            1,
            len(ctx.test),
        )

        if len(unseen) and share >= 0.005:
            hits[column] = {
                "unseen_values": sorted(unseen.unique().tolist())[:8],
                "share_of_test_rows": round(
                    share,
                    4,
                ),
            }

    if not hits:
        return []

    category_error = _error_mentions_categories(test_error)

    largest_share = max(info["share_of_test_rows"] for info in hits.values())

    if category_error:
        severity = "high"
        confidence = 0.9

        extra = (
            " The model also returns a prediction error that points to "
            "unknown or incompatible categories."
        )

        fix_action = {"type": "rebuild_preprocessing"}

    elif largest_share >= 0.05:
        severity = "medium"
        confidence = 0.75
        extra = ""
        fix_action = None

    else:
        severity = "low"
        confidence = 0.65
        extra = ""
        fix_action = None

    return [
        Finding(
            "DQ-003",
            "data_quality",
            "Test data contains categories that training does not contain",
            severity,
            confidence,
            what=(
                "; ".join(
                    (
                        f"'{column}': "
                        f"{', '.join(info['unseen_values'][:4])} "
                        f"({pct(info['share_of_test_rows'])} of test rows)"
                    )
                    for column, info in hits.items()
                )
                + "."
                + extra
            ),
            why=(
                "These values do not appear during training. Some encoders reject them, "
                "while others ignore them or place them in an unknown group, so predictions "
                "for these rows can behave differently."
            ),
            fix=(
                "Use one encoder that learns from training data and handles unknown categories, "
                "such as OneHotEncoder(handle_unknown='ignore'), or use a documented 'other' "
                "group when that fits the data."
            ),
            evidence={
                "columns": hits,
                "prediction_error_mentions_categories": category_error,
            },
            fix_action=fix_action,
        )
    ]


def _code_stats(
    codes: pd.Series,
    target: pd.Series,
) -> dict:
    """Returns counts and target statistics for each numeric category code."""
    frame = pd.DataFrame(
        {
            "code": codes,
            "target": pd.to_numeric(
                target,
                errors="coerce",
            ),
        }
    ).dropna()

    counts = frame["code"].value_counts()

    if frame.empty:
        return {
            "n": counts,
            "N": 0,
            "mean": pd.Series(dtype=float),
            "var": pd.Series(dtype=float),
        }

    grouped = frame.groupby("code")["target"]

    return {
        "n": counts,
        "N": int(len(frame)),
        "mean": grouped.mean(),
        "var": grouped.var(ddof=0).fillna(0),
    }


def _pair_cost(
    test_stats: dict,
    train_stats: dict,
    test_code,
    train_code,
) -> float:
    """Returns a squared distance based on frequency and target rate."""
    if train_code not in train_stats["n"].index:
        return 50.0

    if test_stats["N"] <= 0 or train_stats["N"] <= 0:
        return 50.0

    n_test = float(test_stats["n"][test_code])

    n_train = float(train_stats["n"][train_code])

    p_test = n_test / test_stats["N"]

    p_train = n_train / train_stats["N"]

    pooled = (n_test + n_train) / (test_stats["N"] + train_stats["N"])

    z_frequency = (p_test - p_train) / np.sqrt(
        pooled * (1 - pooled) * (1 / test_stats["N"] + 1 / train_stats["N"]) + 1e-12
    )

    standard_error = np.sqrt(
        test_stats["var"][test_code] / n_test
        + train_stats["var"][train_code] / n_train
        + 1e-6
    )

    z_target = (
        test_stats["mean"][test_code] - train_stats["mean"][train_code]
    ) / standard_error

    return float(z_frequency**2 + z_target**2)


@register(
    "DQ-004",
    "Category codes that mean different things in train and test",
    "data_quality",
    (
        "Checks whether number-coded categories behave differently between training "
        "and test data and whether a remapping explains the difference."
    ),
)
def encoding_mismatch(ctx):
    candidates = [
        column
        for column in ctx.feature_cols
        if _looks_like_integer_codes(ctx.train[column])
    ]

    if not candidates:
        raise SkipCheck("The data has no number-coded category columns.")

    y_train = ctx.train[ctx.target]

    y_test = ctx.test[ctx.target]

    if ctx.is_classification:
        target_train = (y_train == ctx.minority_class).astype(float)

        target_test = (y_test == ctx.minority_class).astype(float)

    else:
        y_train_numeric = pd.to_numeric(
            y_train,
            errors="coerce",
        ).astype(float)

        y_test_numeric = pd.to_numeric(
            y_test,
            errors="coerce",
        ).astype(float)

        mean = float(y_train_numeric.mean())

        std = float(y_train_numeric.std())

        if not np.isfinite(mean):
            raise SkipCheck("The regression target cannot be read as numeric values.")

        if not np.isfinite(std) or std <= 0:
            std = 1.0

        target_train = (y_train_numeric - mean) / std

        target_test = (y_test_numeric - mean) / std

    findings = []

    for column in candidates:
        train_stats = _code_stats(
            ctx.train[column],
            target_train,
        )

        test_stats = _code_stats(
            ctx.test[column],
            target_test,
        )

        test_codes = [
            code
            for code in sorted(test_stats["n"].index)
            if test_stats["n"][code] >= 20
        ]

        if len(test_codes) < 3:
            continue

        train_codes = list(sorted(train_stats["n"].index))

        candidate_train_codes = train_codes + [
            f"__new{index}" for index in range(len(test_codes))
        ]

        cost_matrix = np.array(
            [
                [
                    _pair_cost(
                        test_stats,
                        train_stats,
                        test_code,
                        train_code,
                    )
                    for train_code in candidate_train_codes
                ]
                for test_code in test_codes
            ],
            dtype=float,
        )

        rows, columns = linear_sum_assignment(cost_matrix)

        best_cost = float(
            cost_matrix[
                rows,
                columns,
            ].sum()
        )

        identity_cost = float(
            sum(
                _pair_cost(
                    test_stats,
                    train_stats,
                    code,
                    code,
                )
                for code in test_codes
            )
        )

        degrees_of_freedom = 2 * len(test_codes)

        p_identity = float(
            chi2.sf(
                identity_cost,
                degrees_of_freedom,
            )
        )

        p_best = float(
            chi2.sf(
                best_cost,
                degrees_of_freedom,
            )
        )

        mapping = {
            str(test_codes[row]): str(candidate_train_codes[index])
            for row, index in zip(
                rows,
                columns,
            )
        }

        moved = {
            source: destination
            for source, destination in mapping.items()
            if source != destination
        }

        if not (p_identity < 1e-6 and p_best > 1e-3 and len(moved) >= 2):
            continue

        facts = ctx.code_facts

        code_evidence = (
            facts.encodes_test_separately(column)
            if facts is not None and facts.parsed
            else False
        )

        remap = {
            test_codes[row]: candidate_train_codes[index]
            for row, index in zip(
                rows,
                columns,
            )
            if not str(candidate_train_codes[index]).startswith("__new")
        }

        impact = _remap_impact(
            ctx,
            column,
            remap,
        )

        large_impact = impact is not None and impact >= 0.05

        if code_evidence is True:
            title = (
                f"Category codes in '{column}' do not match between "
                "training and test data"
            )

            confidence = 0.9

            severity = "high" if large_impact else "medium"

        else:
            title = f"Possible category-code mismatch in '{column}'"

            confidence = (
                0.7 if code_evidence is None else (0.5 if p_best > 0.05 else 0.4)
            )

            severity = "medium" if (large_impact or impact is None) else "low"

        if impact is None:
            impact_text = ""

        else:
            metrics = ctx.metrics("test")

            metric_name = (
                metrics["primary_name"].lower() if metrics else "primary score"
            )

            impact_text = (
                " Reading the codes with the suggested mapping changes the model's test "
                f"{metric_name} by {100 * impact:+.1f} points."
            )

        if code_evidence is True:
            matching_events = [
                event for event in facts.test_encodings if column in event["columns"]
            ]

            code_text = (
                " The training code also converts this column to numbers separately "
                "on the test data"
            )

            if matching_events:
                code_text += (
                    " ("
                    + "; ".join(
                        (f"line {event['line']}: " f"{event['how']}")
                        for event in matching_events
                    )
                    + ")."
                )

            else:
                code_text += "."

        elif code_evidence is None:
            unknown_events = [
                event for event in facts.test_encodings if not event["columns"]
            ]

            code_text = (
                " The training code also contains a separate test-data encoding step, "
                "but the column name cannot be confirmed"
            )

            if unknown_events:
                code_text += (
                    " ("
                    + "; ".join(
                        (f"line {event['line']}: " f"{event['how']}")
                        for event in unknown_events
                    )
                    + ")."
                )

            else:
                code_text += "."

        else:
            code_text = ""

        drift_text = ""

        if code_evidence is not True:
            drift_text = (
                " A real change in category frequencies or target behavior can create "
                "a similar pattern, so confirm the encoding step before changing the data."
            )

        findings.append(
            Finding(
                "DQ-004",
                "data_quality",
                title,
                severity,
                confidence,
                what=(
                    f"In '{column}', the numeric codes behave differently between "
                    "training and test data. "
                    f"The two sets fit much better when {len(moved)} codes are "
                    "matched differently (test code -> training code: "
                    + ", ".join(
                        f"{source}->{destination}"
                        for source, destination in list(moved.items())[:6]
                    )
                    + ")."
                    + code_text
                    + impact_text
                ),
                why=(
                    "This pattern can appear when categories turn into numbers separately "
                    "for training and test data, such as fitting LabelEncoder twice or "
                    "using pd.factorize separately. The same number can then represent "
                    "different categories, which changes predictions." + drift_text
                ),
                fix=(
                    "Fit one encoder on the training data and reuse it everywhere, or use "
                    "OneHotEncoder(handle_unknown='ignore') inside the Pipeline. Re-create "
                    "this column from the original category values."
                ),
                evidence={
                    "column": column,
                    "suggested_mapping_test_to_train": mapping,
                    "p_value_as_is": p_identity,
                    "p_value_after_remap": round(
                        p_best,
                        4,
                    ),
                    "note": (
                        "The p-value after remapping comes from a searched mapping, "
                        "so it is a screening signal."
                    ),
                    "code_evidence": code_evidence,
                    "score_change_if_remapped": (
                        None
                        if impact is None
                        else round(
                            impact,
                            4,
                        )
                    ),
                },
                # Automatic repair excludes the column only when the code confirms the mismatch.
                fix_action=(
                    {
                        "type": "exclude_untrusted_column",
                        "columns": [column],
                    }
                    if code_evidence is True
                    else None
                ),
            )
        )

    return findings


def _remap_impact(
    ctx,
    column,
    mapping,
):
    """Returns the test-score change after translating codes to training meanings.

    This only sizes the problem; it does not change the stored data.
    """
    base = ctx.metrics("test")

    if base is None or not mapping or ctx.model is None:
        return None

    try:
        features = ctx.split("test")[ctx.feature_cols].copy()

        features[column] = (
            features[column]
            .map(
                lambda value: mapping.get(
                    value,
                    value,
                )
            )
            .astype(ctx.train[column].dtype)
        )

        model_input = features

        if feature_names_of(ctx.model) is None:
            model_input = features.to_numpy()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            prediction = ctx.model.predict(model_input)

        remapped_metrics = evaluate(
            ctx,
            ctx.y("test"),
            np.asarray(prediction),
        )

        return float(remapped_metrics["primary"] - base["primary"])

    except Exception:  # noqa: BLE001
        return None


@register(
    "DQ-005",
    "Test data looks different from training data",
    "data_quality",
    "Measures numeric distribution shift between training and test data with PSI.",
)
def distribution_shift(ctx):
    scores = {}

    for column in ctx.feature_cols:
        train = ctx.train[column]

        if (
            not pd.api.types.is_numeric_dtype(train)
            or pd.api.types.is_bool_dtype(train)
            or train.nunique() <= 10
        ):
            continue

        train_values = _numeric_series(train).to_numpy()

        test_values = _numeric_series(ctx.test[column]).to_numpy()

        value = psi(
            train_values,
            test_values,
        )

        if value >= 0.25:
            scores[column] = round(
                value,
                3,
            )

    if not scores:
        return []

    top = dict(
        sorted(
            scores.items(),
            key=lambda item: -item[1],
        )[:6]
    )

    return [
        Finding(
            "DQ-005",
            "data_quality",
            "Some numeric columns look different in the test data",
            "low",
            0.5,
            what=(
                "These numeric columns shift noticeably between training and test data "
                "(PSI at or above 0.25): "
                + ", ".join(f"'{column}' ({value})" for column, value in top.items())
                + "."
            ),
            why=(
                "A change in input distribution can reduce model quality. "
                "The change can be valid, such as a newer time period, "
                "or it can point to a data-processing problem."
            ),
            fix=(
                "Check whether the shift is expected, compare the data source and "
                "processing steps, and monitor these columns after deployment."
            ),
            evidence={"psi_by_column": top},
        )
    ]
