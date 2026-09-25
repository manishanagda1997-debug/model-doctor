"""Checks model behaviour for misleading metrics, overfitting, and class imbalance."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..core import Finding, SkipCheck, register
from ._utils import pct

ACCURACY_WORDS = {"accuracy", "acc", "accuracy_score", "score"}


def _need_predictions(ctx, split="test"):
    """Returns metrics for one split or skips when the model cannot be evaluated."""
    if ctx.model is None:
        raise SkipCheck("No model is provided.")

    metrics = ctx.metrics(split)

    if metrics is None:
        raise SkipCheck(
            f"The model cannot make predictions on the {split} data."
        )

    return metrics


def _minority_test_count(ctx) -> int:
    """Returns how many test rows belong to the training minority class."""
    if not ctx.is_classification:
        return 0

    return int(
        np.sum(
            np.asarray(ctx.y("test"))
            == ctx.minority_class
        )
    )


def _imbalance_handling_detected(ctx) -> bool:
    """Returns True when weighting or resampling is visible in the model or training code."""
    params = ctx.model_params()

    handled = bool(
        params.get("class_weight")
        not in (
            None,
            "",
            {},
        )
        or params.get("scale_pos_weight")
        not in (
            None,
            1,
            1.0,
        )
        or ctx.flags.get(
            "class_weighted",
            False,
        )
    )

    facts = ctx.code_facts

    if facts is not None and facts.parsed:
        handled = handled or bool(
            facts.names_used
            & {
                "SMOTE",
                "ADASYN",
                "RandomOverSampler",
                "RandomUnderSampler",
                "compute_sample_weight",
                "compute_class_weight",
                "sample_weight",
            }
        )

    return handled


@register(
    "METRIC-001",
    "Accuracy hiding poor results on rare cases",
    "metrics",
    "On imbalanced classification data, checks whether accuracy is used as the main "
    "reported metric and compares it with rare-class performance.",
)
def misleading_accuracy(ctx):
    if not ctx.is_classification:
        raise SkipCheck(
            "Only applies to classification."
        )

    metrics = _need_predictions(
        ctx,
        "test",
    )

    minority_share = float(
        ctx.class_shares.iloc[-1]
    )

    if minority_share >= 0.2:
        raise SkipCheck(
            f"Classes are reasonably balanced "
            f"(smallest training class = {pct(minority_share)})."
        )

    facts = ctx.code_facts

    code_known = bool(
        facts is not None
        and facts.parsed
        and (
            facts.metrics_used
            or facts.uses_score_method
        )
    )

    if (
        ctx.reported_metric is None
        and not code_known
    ):
        raise SkipCheck(
            "The reported evaluation metric is not known "
            "(no --reported-metric and no metric in the training code), "
            "so this check cannot tell whether accuracy is misused."
        )

    code_says_accuracy = bool(
        code_known
        and facts.reports_only_accuracy
    )

    reported_accuracy = (
        ctx.reported_metric in ACCURACY_WORDS
        or (
            ctx.reported_metric is None
            and code_says_accuracy
        )
    )

    if not reported_accuracy:
        return []

    source = (
        "the stated reported metric"
        if ctx.reported_metric in ACCURACY_WORDS
        else "the training code"
    )

    minority_test_count = _minority_test_count(
        ctx
    )

    evidence = {
        key: (
            round(
                value,
                4,
            )
            if isinstance(
                value,
                float,
            )
            else value
        )
        for key, value in metrics.items()
    }

    evidence.update(
        {
            "training_minority_share": round(
                minority_share,
                4,
            ),
            "minority_test_rows": minority_test_count,
        }
    )

    if minority_test_count == 0:
        return [
            Finding(
                "METRIC-001",
                "metrics",
                "Accuracy cannot evaluate the rare class in this test split",
                "high",
                0.9,
                what=(
                    f"Accuracy is the reported metric from {source}, but the test data "
                    f"contains no examples of the training minority class "
                    f"'{ctx.minority_class}'. The reported accuracy is "
                    f"{pct(metrics['accuracy'])}."
                ),
                why=(
                    "A test set with no rare-class examples cannot show whether the model "
                    "detects that class. Accuracy can therefore look strong without testing "
                    "the cases that make the training data imbalanced."
                ),
                fix=(
                    f"Create an evaluation split that contains enough examples of "
                    f"'{ctx.minority_class}', then report its recall, precision and F1. "
                    "Use PR AUC when probability scores are available."
                ),
                evidence=evidence,
            )
        ]

    lift = (
        metrics["accuracy"]
        - metrics["majority_baseline_accuracy"]
    )

    weak = (
        lift < 0.05
        or metrics["minority_recall"] < 0.5
    )

    return [
        Finding(
            "METRIC-001",
            "metrics",
            "Accuracy alone can hide poor rare-class performance",
            "high" if weak else "medium",
            0.9 if weak else 0.7,
            what=(
                f"The model is judged by accuracy from {source}: "
                f"{pct(metrics['accuracy'])}. The minority class "
                f"'{ctx.minority_class}' makes up {pct(minority_share)} of training rows, "
                f"while an always-majority prediction already scores "
                f"{pct(metrics['majority_baseline_accuracy'])} on this test set. "
                f"The model recalls {pct(metrics['minority_recall'])} of the "
                f"'{ctx.minority_class}' test cases."
            ),
            why=(
                "With imbalanced classes, accuracy can stay high even when the model "
                "performs poorly on the rare class. Using accuracy alone can therefore "
                "hide behaviour that matters for the prediction goal."
            ),
            fix=(
                f"Report recall, precision and F1 for '{ctx.minority_class}', and include "
                "PR AUC when probability scores are available. Compare accuracy with the "
                "always-majority baseline instead of using it alone."
            ),
            evidence=evidence,
            fix_action={
                "type": "report_balanced_metrics"
            },
        )
    ]


@register(
    "OVERFIT-001",
    "Much better on training data than on test data",
    "overfitting",
    "Compares the model's primary score on training and test data.",
)
def train_test_gap(ctx):
    test_metrics = _need_predictions(
        ctx,
        "test",
    )

    train_metrics = ctx.metrics(
        "train"
    )

    if train_metrics is None:
        raise SkipCheck(
            "The model cannot score the training data."
        )

    train_score = float(
        train_metrics["primary"]
    )

    test_score = float(
        test_metrics["primary"]
    )

    if not (
        np.isfinite(
            train_score
        )
        and np.isfinite(
            test_score
        )
    ):
        raise SkipCheck(
            "The primary score is not defined for both training and test data."
        )

    gap = (
        train_score
        - test_score
    )

    # Smaller test sets need a larger gap before this check treats it as meaningful.
    noise = (
        1.5
        / np.sqrt(
            max(
                1,
                len(ctx.test),
            )
        )
    )

    if gap < max(
        0.10,
        noise,
    ):
        return []

    high = gap >= max(
        0.15,
        1.5 * noise,
    )

    params = ctx.model_params()
    hints = []

    if params.get(
        "max_depth",
        "not_available",
    ) is None:
        hints.append(
            "limit tree depth (max_depth)"
        )

    if params.get(
        "min_samples_leaf",
        99,
    ) == 1:
        hints.append(
            "require more samples per leaf (min_samples_leaf)"
        )

    if "C" in params:
        hints.append(
            "try stronger regularisation (a lower C)"
        )

    if not hints:
        hints = [
            "simplify the model or add regularisation",
            "use more training data when possible",
        ]

    return [
        Finding(
            "OVERFIT-001",
            "overfitting",
            "A large train-test gap suggests overfitting",
            "high" if high else "medium",
            0.85 if high else 0.65,
            what=(
                f"{test_metrics['primary_name']} is {pct(train_score)} on training data "
                f"and {pct(test_score)} on test data, a gap of {pct(gap)}."
            ),
            why=(
                "A large gap is consistent with the model learning patterns that do not "
                "generalize to the test data. Data-quality or distribution problems can "
                "also increase this gap, so those findings should be checked before treating "
                "overfitting as the only cause."
            ),
            fix=(
                "First check whether train and test data are comparable. If they are, "
                "reduce model complexity: "
                + "; ".join(
                    hints
                )
                + ". Tune changes with appropriate cross-validation."
            ),
            evidence={
                "train_score": round(
                    train_score,
                    4,
                ),
                "test_score": round(
                    test_score,
                    4,
                ),
                "gap": round(
                    gap,
                    4,
                ),
                "metric": test_metrics[
                    "primary_name"
                ],
            },
            fix_action={
                "type": "regularize"
            },
        )
    ]


@register(
    "OVERFIT-002",
    "Suspiciously perfect test score",
    "overfitting",
    "Flags near-perfect test scores that deserve a leakage or contamination review.",
)
def suspiciously_perfect(ctx):
    metrics = _need_predictions(
        ctx,
        "test",
    )

    if ctx.is_classification:
        if np.unique(
            ctx.y(
                "test"
            )
        ).size < 2:
            raise SkipCheck(
                "The test data contains only one class, so a near-perfect "
                "classification score is not informative."
            )

        perfect = (
            metrics[
                "balanced_accuracy"
            ] >= 0.99
            or metrics.get(
                "roc_auc",
                0,
            ) >= 0.998
        )

        label = (
            f"balanced accuracy "
            f"{pct(metrics['balanced_accuracy'])}"
        )

        if "roc_auc" in metrics:
            label += (
                f", ROC AUC "
                f"{metrics['roc_auc']:.3f}"
            )

    else:
        r2 = float(
            metrics[
                "r2"
            ]
        )

        if not np.isfinite(
            r2
        ):
            raise SkipCheck(
                "R-squared is not defined for this test data."
            )

        perfect = (
            r2 >= 0.99
        )

        label = (
            f"R-squared {r2:.3f}"
        )

    if not perfect:
        return []

    return [
        Finding(
            "OVERFIT-002",
            "overfitting",
            "Near-perfect test performance needs verification",
            "high",
            0.6,
            what=(
                f"The model reaches {label} on the test data."
            ),
            why=(
                "A near-perfect score can be genuine for a simple problem, but it can "
                "also appear when target information leaks into features or evaluation "
                "rows overlap with training data. The score alone does not prove either "
                "explanation."
            ),
            fix=(
                "Review the leakage and contamination findings in this audit. "
                "If they do not explain the score, confirm the result on genuinely new data."
            ),
            evidence={
                key: (
                    round(
                        value,
                        4,
                    )
                    if isinstance(
                        value,
                        float,
                    )
                    else value
                )
                for key, value in metrics.items()
            },
        )
    ]


@register(
    "IMB-001",
    "Model ignores the rare class",
    "imbalance",
    "Checks whether the model collapses toward the most common class or misses whole classes.",
)
def majority_class_collapse(ctx):
    if not ctx.is_classification:
        raise SkipCheck(
            "Only applies to classification."
        )

    metrics = _need_predictions(
        ctx,
        "test",
    )

    y = np.asarray(
        ctx.y(
            "test"
        )
    )

    pred = np.asarray(
        ctx.predict(
            "test"
        )
    )

    findings = []

    actual_majority_share = float(
        np.mean(
            y
            == ctx.majority_class
        )
    )

    minority_test_count = _minority_test_count(
        ctx
    )

    collapsed = (
        minority_test_count > 0
        and metrics[
            "predicted_majority_share"
        ] >= 0.9
        and metrics[
            "minority_recall"
        ] < 0.2
        and metrics[
            "predicted_majority_share"
        ] >= actual_majority_share - 0.005
    )

    if collapsed:
        findings.append(
            Finding(
                "IMB-001",
                "imbalance",
                f"The model almost always predicts '{ctx.majority_class}'",
                "critical",
                0.9,
                what=(
                    f"The model predicts '{ctx.majority_class}' for "
                    f"{pct(metrics['predicted_majority_share'])} of test cases, while "
                    f"{pct(actual_majority_share)} actually belong to that class. "
                    f"It recalls only {pct(metrics['minority_recall'])} of the "
                    f"'{ctx.minority_class}' cases."
                ),
                why=(
                    "The model is strongly biased toward the common class and misses "
                    "most examples of the rare class. High overall accuracy can hide "
                    "this behaviour."
                ),
                fix=(
                    "Compare class weighting, resampling, or a different decision threshold, "
                    "and judge the result with rare-class recall, precision and F1."
                ),
                evidence={
                    "predicted_majority_share": round(
                        metrics[
                            "predicted_majority_share"
                        ],
                        4,
                    ),
                    "actual_majority_share": round(
                        actual_majority_share,
                        4,
                    ),
                    "minority_recall": round(
                        metrics[
                            "minority_recall"
                        ],
                        4,
                    ),
                    "minority_test_rows": minority_test_count,
                },
                fix_action=(
                    None
                    if _imbalance_handling_detected(
                        ctx
                    )
                    else {
                        "type": "class_weight"
                    }
                ),
            )
        )

        return findings

    counts = pd.Series(
        y
    ).value_counts(
        dropna=False
    )

    missed = []

    for label, count in counts.items():
        if count < 20:
            continue

        mask = (
            y
            == label
        )

        if float(
            np.mean(
                pred[
                    mask
                ]
                == label
            )
        ) == 0:
            missed.append(
                str(
                    label
                )
            )

    if missed:
        findings.append(
            Finding(
                "IMB-001",
                "imbalance",
                "Some classes are never predicted correctly",
                "high",
                0.8,
                what=(
                    "The model has zero recall for these test classes: "
                    + ", ".join(
                        missed
                    )
                    + "."
                ),
                why=(
                    "These classes appear often enough in the test data to evaluate, "
                    "but the model does not identify any of their examples correctly."
                ),
                fix=(
                    "Review per-class recall and compare class weighting, resampling, "
                    "features, or decision rules for the missed classes."
                ),
                evidence={
                    "zero_recall_classes": missed
                },
                fix_action=(
                    None
                    if _imbalance_handling_detected(
                        ctx
                    )
                    else {
                        "type": "class_weight"
                    }
                ),
            )
        )

    return findings


@register(
    "IMB-002",
    "Imbalanced data with no counter-measure",
    "imbalance",
    "When one training class is rare, checks whether the model or training code uses "
    "class weighting, resampling, or sample weights.",
)
def imbalance_not_handled(ctx):
    if not ctx.is_classification:
        raise SkipCheck(
            "Only applies to classification."
        )

    if ctx.model is None:
        raise SkipCheck(
            "No model is provided."
        )

    minority_share = float(
        ctx.class_shares.iloc[-1]
    )

    if minority_share >= 0.1:
        raise SkipCheck(
            f"Smallest class is {pct(minority_share)} of the training data; "
            "this check treats that as outside severe imbalance."
        )

    params = ctx.model_params()

    if _imbalance_handling_detected(
        ctx
    ):
        return []

    metrics = ctx.metrics(
        "test"
    )

    minority_test_count = _minority_test_count(
        ctx
    )

    base_what = (
        f"'{ctx.minority_class}' makes up only {pct(minority_share)} of the training data, "
        "and no class weighting, resampling, or sample-weight handling is detected."
    )

    evidence = {
        "minority_share": round(
            minority_share,
            4,
        ),
        "class_weight": str(
            params.get(
                "class_weight"
            )
        ),
        "minority_test_rows": minority_test_count,
    }

    if (
        metrics is not None
        and minority_test_count > 0
    ):
        evidence.update(
            {
                "minority_recall": round(
                    metrics[
                        "minority_recall"
                    ],
                    4,
                ),
                "minority_f1": round(
                    metrics[
                        "minority_f1"
                    ],
                    4,
                ),
            }
        )

    good = bool(
        metrics is not None
        and minority_test_count > 0
        and metrics[
            "minority_recall"
        ] >= 0.5
        and metrics[
            "minority_f1"
        ] >= 0.5
    )

    if good:
        return [
            Finding(
                "IMB-002",
                "imbalance",
                "Imbalanced data, but the rare class is handled reasonably",
                "low",
                0.5,
                what=(
                    base_what
                    + f" Even so, the model recalls "
                    f"{pct(metrics['minority_recall'])} of the "
                    f"'{ctx.minority_class}' test cases "
                    f"(F1 {pct(metrics['minority_f1'])})."
                ),
                why=(
                    "The current rare-class performance is reasonable, but it can "
                    "change quickly when class frequencies or input data change."
                ),
                fix=(
                    f"Keep tracking recall, precision and F1 for "
                    f"'{ctx.minority_class}' after deployment."
                ),
                evidence=evidence,
            )
        ]

    if minority_test_count == 0:
        performance_text = (
            " The test data contains no examples of this class, so its recall and F1 "
            "cannot be assessed from this split."
        )

        confidence = 0.5

    elif metrics is None:
        performance_text = ""
        confidence = 0.5

    else:
        performance_text = (
            f" The model recalls {pct(metrics['minority_recall'])} of the "
            f"'{ctx.minority_class}' test cases "
            f"(F1 {pct(metrics['minority_f1'])})."
        )

        confidence = 0.7

    return [
        Finding(
            "IMB-002",
            "imbalance",
            "Severe class imbalance has no explicit handling",
            "medium",
            confidence,
            what=(
                base_what
                + performance_text
            ),
            why=(
                "Without an explicit imbalance strategy, many models can favour common "
                "classes. Whether weighting or resampling helps should be checked against "
                "rare-class metrics rather than assumed."
            ),
            fix=(
                "Compare class_weight='balanced' or an appropriate resampling method, "
                "then evaluate recall, precision and F1 for the rare class."
            ),
            evidence=evidence,
            fix_action={
                "type": "class_weight"
            },
        )
    ]
    