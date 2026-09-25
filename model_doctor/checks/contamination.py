"""Checks whether training and evaluation data are kept independent."""

from __future__ import annotations

import re

import numpy as np

from ..core import Finding, SkipCheck, register
from ._utils import detect_id_cols, detect_time_cols, pct, row_hashes


GROUP_AWARE_CV = {
    "GroupKFold",
    "StratifiedGroupKFold",
    "GroupShuffleSplit",
    "LeaveOneGroupOut",
    "LeavePGroupsOut",
}

TIME_AWARE_CV = {
    "TimeSeriesSplit",
}


def _forecast_goal(goal: str) -> bool:
    """Returns True when the prediction goal clearly depends on future time."""
    return bool(
        re.search(
            r"forecast|future|next|predict.*(day|week|month|year)",
            goal or "",
            re.I,
        )
    )


def _dedupe_events(events: list) -> list:
    """Returns CV events without duplicate line/name pairs."""
    seen = set()
    out = []

    for event in events:
        key = (
            event.get("line"),
            event.get("name"),
        )

        if key in seen:
            continue

        seen.add(key)
        out.append(event)

    return out


def _matching_label_stats(ctx, train_hashes, test_hashes, in_train):
    """Returns label agreement and whether matched train rows have conflicting labels."""
    if not in_train.any():
        return None, False

    labels_by_hash = {}

    for row_hash, label in zip(
        train_hashes.to_numpy(),
        ctx.train[ctx.target].to_numpy(),
    ):
        labels_by_hash.setdefault(
            int(row_hash),
            set(),
        ).add(label)

    matches = []
    conflicting_labels = False

    for row_hash, label in zip(
        test_hashes[in_train].to_numpy(),
        ctx.test.loc[in_train, ctx.target].to_numpy(),
    ):
        train_labels = labels_by_hash.get(
            int(row_hash),
            set(),
        )

        if len(train_labels) > 1:
            conflicting_labels = True

        matches.append(
            label in train_labels
        )

    share = float(
        np.mean(matches)
    ) if matches else None

    return share, conflicting_labels


@register(
    "CONT-001",
    "Same rows in training and test data",
    "contamination",
    "Looks for test feature rows that also appear in the training data.",
)
def duplicate_rows_across_splits(ctx):
    columns = [
        column
        for column in ctx.feature_cols
        if column not in ctx.id_cols
    ]

    if not columns:
        raise SkipCheck(
            "No feature columns remain for row comparison."
        )

    train_hashes = row_hashes(
        ctx.train[columns]
    )

    test_hashes = row_hashes(
        ctx.test[columns]
    )

    train_unique_ratio = (
        train_hashes.nunique()
        / max(
            1,
            len(train_hashes),
        )
    )

    train_hash_set = set(
        train_hashes
    )

    in_train = test_hashes.isin(
        train_hash_set
    )

    overlap_share = float(
        in_train.mean()
    )

    # A large possible combination space makes accidental exact copies less likely.
    log_space = float(
        np.sum(
            np.log10(
                [
                    max(
                        1,
                        ctx.train[column].nunique(
                            dropna=False
                        ),
                    )
                    for column in columns
                ]
            )
        )
    )

    rich_space = bool(
        log_space
        > np.log10(
            max(
                1,
                len(ctx.train),
            )
        )
        + 2
    )

    if (
        train_unique_ratio < 0.5
        or (
            not rich_space
            and train_unique_ratio < 0.8
        )
    ):
        raise SkipCheck(
            "Rows are naturally repetitive "
            f"({pct(train_unique_ratio)} unique), so exact copies are not reliable evidence."
        )

    if overlap_share < 0.01:
        return []

    same_label_share, conflicting_labels = _matching_label_stats(
        ctx,
        train_hashes,
        test_hashes,
        in_train,
    )

    severity = (
        "high"
        if overlap_share >= 0.05
        else "medium"
    )

    confidence = min(
        0.95,
        0.6
        + 0.35
        * (
            1.0
            if rich_space
            else train_unique_ratio
        ),
    )

    same_label_text = ""

    if same_label_share is not None:
        same_label_text = (
            " For these matched feature rows, the test label also appears with the "
            f"training copy {pct(same_label_share, 0)} of the time."
        )

    

    return [
        Finding(
            "CONT-001",
            "contamination",
            "Test feature rows also appear in the training data",
            severity,
            confidence,
            what=(
                f"{pct(overlap_share)} of test rows "
                f"({int(in_train.sum())} of {len(test_hashes)}) have the same feature "
                "values as a training row."
                + same_label_text
            ),
            why=(
                "Repeated examples across training and evaluation reduce the independence "
                "of the test set. Flexible models can memorise these examples, which can "
                "make the reported test score look better than performance on genuinely "
                "unseen cases."
            ),
            fix=(
                "Check why the same records appear on both sides of the split. "
                "When they are duplicate records, remove duplicates before splitting and "
                "create the train/test split again."
            ),
            evidence={
                "test_rows_found_in_train": int(
                    in_train.sum()
                ),
                "share": round(
                    overlap_share,
                    4,
                ),
                "train_unique_row_ratio": round(
                    train_unique_ratio,
                    3,
                ),
                "large_possible_combination_space": rich_space,
                "matching_label_share": (
                    None
                    if same_label_share is None
                    else round(
                        same_label_share,
                        3,
                    )
                ),
                "matched_train_rows_have_conflicting_labels": conflicting_labels,
            },
            fix_action=None,
        )
    ]


@register(
    "CONT-002",
    "Same people/entities in training and test data",
    "contamination",
    (
        "Checks whether repeated customer, patient, device, or other group values "
        "appear on both sides of a split."
    ),
)
def group_overlap(ctx):
    explicit = (
        ctx.group_col is not None
    )

    if explicit:
        columns = [
            ctx.group_col
        ]

    else:
        columns = [
            column
            for column in detect_id_cols(
                ctx.train,
                exclude={
                    ctx.target
                },
            )
            if ctx.train[
                column
            ].nunique(
                dropna=True
            )
            < 0.9
            * len(ctx.train)
        ]

    if not columns:
        raise SkipCheck(
            "The data has no repeated group or ID column to compare."
        )

    findings = []

    for column in columns[:2]:
        train_groups = set(
            ctx.train[
                column
            ].dropna()
        )

        in_train = ctx.test[
            column
        ].isin(
            train_groups
        )

        overlap_share = float(
            in_train.mean()
        )

        rows_per_group = (
            len(ctx.train)
            / max(
                1,
                ctx.train[
                    column
                ].nunique(
                    dropna=True
                ),
            )
        )

        minimum_share = (
            0.01
            if explicit
            else 0.05
        )

        if overlap_share < minimum_share:
            continue

        if (
            not explicit
            and rows_per_group < 1.2
        ):
            continue

        if explicit:
            severity = "high"
            confidence = 0.9

            why = (
                f"'{column}' is configured as the grouping column, so rows from the same "
                "group should stay on one side of a group-independent evaluation. "
                "The current overlap allows information about the same entity to appear "
                "in both training and test data."
            )

            fix = (
                f"Split by '{column}' with GroupShuffleSplit or GroupKFold so one group "
                "does not appear on both sides of the evaluation."
            )

            fix_action = {
                "type": "group_split",
                "group_col": column,
            }

        else:
            severity = "medium"
            confidence = 0.6

            why = (
                f"'{column}' looks like a repeated entity identifier. If the goal is to "
                "measure performance on new entities, sharing these values across train and "
                "test can make the evaluation optimistic. If future predictions are meant "
                "for the same known entities, this overlap can be intentional."
            )

            fix = (
                f"Confirm whether evaluation should generalize to unseen '{column}' values. "
                "If it should, use a group-aware split such as GroupShuffleSplit or GroupKFold."
            )

            fix_action = None

        findings.append(
            Finding(
                "CONT-002",
                "contamination",
                f"The same '{column}' values appear in training and test data",
                severity,
                confidence,
                what=(
                    f"{pct(overlap_share)} of test rows belong to a '{column}' value "
                    "that also appears in the training data "
                    f"(about {rows_per_group:.1f} training rows per '{column}')."
                ),
                why=why,
                fix=fix,
                evidence={
                    "group_column": column,
                    "explicitly_given": explicit,
                    "test_rows_with_seen_group": round(
                        overlap_share,
                        3,
                    ),
                    "rows_per_group": round(
                        rows_per_group,
                        2,
                    ),
                },
                fix_action=fix_action,
            )
        )

    return findings


@register(
    "CONT-003",
    "Cross-validation that ignores groups or time",
    "contamination",
    (
        "Reads the training code and checks whether validation respects repeated groups "
        "or time order when those structures matter."
    ),
)
def improper_cross_validation(ctx):
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

    explicit_time = (
        ctx.time_col is not None
    )

    forecast_goal = _forecast_goal(
        ctx.goal
    )

    if explicit_time:
        time_columns = [
            ctx.time_col
        ]

    elif forecast_goal:
        time_columns = detect_time_cols(
            ctx.train,
            exclude={
                ctx.target
            },
        )

    else:
        time_columns = []

    explicit_group = (
        ctx.group_col is not None
    )

    if explicit_group:
        group_rows = (
            len(ctx.train)
            / max(
                1,
                ctx.train[
                    ctx.group_col
                ].nunique(
                    dropna=True
                ),
            )
        )

        group_columns = (
            [
                ctx.group_col
            ]
            if group_rows >= 1.2
            else []
        )

    else:
        group_columns = [
            column
            for column in detect_id_cols(
                ctx.train,
                exclude={
                    ctx.target
                },
            )
            if ctx.train[
                column
            ].nunique(
                dropna=True
            )
            < 0.9
            * len(ctx.train)
        ]

    has_time_structure = bool(
        time_columns
    )

    has_group_structure = bool(
        group_columns
    )

    if not (
        has_time_structure
        or has_group_structure
    ):
        raise SkipCheck(
            "The audit has no confirmed group or time structure that requires special validation."
        )

    naive_calls = _dedupe_events(
        list(
            facts.cv_naive
        )
    )

    aware_calls = _dedupe_events(
        list(
            facts.cv_aware
        )
    )

    mismatched_calls = []

    for event in aware_calls:
        name = str(
            event.get(
                "name",
                "",
            )
        )

        if name == "PredefinedSplit":
            continue

        is_group_aware = any(
            marker in name
            for marker in GROUP_AWARE_CV
        )

        is_time_aware = any(
            marker in name
            for marker in TIME_AWARE_CV
        )

        if (
            has_time_structure
            and not has_group_structure
            and is_group_aware
            and not is_time_aware
        ):
            mismatched_calls.append(
                event
            )

        elif (
            has_group_structure
            and not has_time_structure
            and is_time_aware
            and not is_group_aware
        ):
            mismatched_calls.append(
                event
            )

        elif (
            has_time_structure
            and has_group_structure
            and not (
                is_time_aware
                and is_group_aware
            )
        ):
            mismatched_calls.append(
                event
            )

    problematic_calls = _dedupe_events(
        naive_calls
        + mismatched_calls
    )

    findings = []

    if problematic_calls:
        if (
            has_time_structure
            and has_group_structure
        ):
            structure_text = "time order and groups"

            why = (
                "The validation method does not preserve all of the structure that the "
                "audit needs to respect. Rows can therefore be evaluated using information "
                "from a later period or from the same entity."
            )

            fix = (
                "Use a validation design that preserves time order and keeps groups separated. "
                "For example, create forward time windows and make sure an entity does not "
                "cross the training and validation sides when both constraints matter."
            )

        elif has_time_structure:
            structure_text = "time order"

            why = (
                "Ordinary or group-only cross-validation can train on rows that occur after "
                "the validation period. That makes the validation score optimistic for a "
                "future-looking prediction problem."
            )

            fix = (
                "Use TimeSeriesSplit or another forward-chaining validation design that "
                "trains on earlier periods and validates on later periods."
            )

        else:
            structure_text = "groups"

            why = (
                "Ordinary or time-only cross-validation can place rows from the same entity "
                "in both training and validation folds. That can make validation performance "
                "optimistic when the goal is group-independent generalization."
            )

            fix = (
                f"Use GroupKFold or StratifiedGroupKFold with groups=df['{group_columns[0]}']."
            )

        explicit_structure = (
            explicit_time
            or explicit_group
        )

        severity = (
            "high"
            if explicit_structure
            else "medium"
        )

        confidence = (
            0.85
            if explicit_structure
            else 0.65
        )

        call_names = ", ".join(
            sorted(
                {
                    str(
                        event.get(
                            "name",
                            "unknown CV",
                        )
                    )
                    for event in problematic_calls
                }
            )
        )

        first_line = problematic_calls[0].get(
            "line"
        )

        aware_note = ""

        if (
            naive_calls
            and aware_calls
        ):
            aware_note = (
                " The code also contains a structure-aware validation method elsewhere, "
                "but that does not make this separate naive validation run safe."
            )

        findings.append(
            Finding(
                "CONT-003",
                "contamination",
                f"Cross-validation does not respect {structure_text}",
                severity,
                confidence,
                what=(
                    f"The code uses {call_names}"
                    + (
                        f" on line {first_line}"
                        if first_line is not None
                        else ""
                    )
                    + ", while the data requires validation that respects "
                    f"{structure_text}."
                    + aware_note
                ),
                why=why,
                fix=fix,
                evidence={
                    "naive_cv_calls": naive_calls,
                    "mismatched_cv_calls": _dedupe_events(
                        mismatched_calls
                    ),
                    "aware_cv_calls": aware_calls,
                    "time_columns": time_columns[:2],
                    "group_columns": group_columns[:2],
                    "explicit_time_column": explicit_time,
                    "explicit_group_column": explicit_group,
                    "forecast_goal": forecast_goal,
                },
            )
        )

    if (
        has_time_structure
        and facts.split_line is not None
        and facts.split_shuffled is True
    ):
        time_column = time_columns[0]

        explicit_structure = explicit_time

        findings.append(
            Finding(
                "CONT-003",
                "contamination",
                "The train/test split shuffles time-ordered data",
                (
                    "high"
                    if explicit_structure
                    else "medium"
                ),
                (
                    0.85
                    if explicit_structure
                    else 0.7
                ),
                what=(
                    f"train_test_split on line {facts.split_line} shuffles rows, while "
                    f"'{time_column}' is used as the time structure for this prediction problem."
                ),
                why=(
                    "Shuffling can place later observations in training while earlier observations "
                    "remain in test data. For future-looking evaluation, that mixes past and future."
                ),
                fix=(
                    f"Sort by '{time_column}' and create an earlier-training/later-test split. "
                    "Use TimeSeriesSplit for cross-validation when appropriate."
                ),
                evidence={
                    "split_line": facts.split_line,
                    "time_column": time_column,
                    "explicitly_given": explicit_time,
                    "forecast_goal": forecast_goal,
                },
                fix_action=(
                    {
                        "type": "time_split",
                        "time_col": time_column,
                    }
                    if explicit_time
                    else None
                ),
            )
        )

    return findings
