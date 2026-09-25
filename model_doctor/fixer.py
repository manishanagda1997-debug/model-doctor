"""Auto-fix: apply supported repairs, retrain when needed, and re-audit.

The fixer never edits the user's code. Training or data repairs build a clean
reference pipeline with the same estimator type and train-only preprocessing.
Reporting-only repairs do not retrain the model. Before/after values are
diagnostic, not an untouched final performance estimate.
"""
from __future__ import annotations

import inspect
import json
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from .auditor import run_checks
from .checks._utils import parse_datetimes
from .core import AuditContext, is_categorical, unwrap_estimator


REPORT_ONLY_ACTIONS = {"report_balanced_metrics"}

SUPPORTED_ACTIONS = {
    "drop_columns",
    "exclude_untrusted_column",
    "group_split",
    "time_split",
    "class_weight",
    "regularize",
    "rebuild_preprocessing",
    *REPORT_ONLY_ACTIONS,
}


@dataclass
class FixResult:
    applied: list = field(default_factory=list)
    rounds: list = field(default_factory=list)
    not_fixable: list = field(default_factory=list)

    before_metrics: Optional[dict] = None
    after_metrics: Optional[dict] = None

    before_train_metrics: Optional[dict] = None
    after_train_metrics: Optional[dict] = None

    before_findings: list = field(default_factory=list)
    after_findings: list = field(default_factory=list)
    after_runs: list = field(default_factory=list)

    new_model: object = None
    new_ctx: Optional[AuditContext] = None

    eval_rows_changed: bool = False
    model_rebuilt: bool = False
    reporting_changed: bool = False

    error: Optional[str] = None

    @property
    def not_open_ids(self) -> list:
        """Check IDs found originally that are not open after the update.

        Some of these checks do not run after a reference rebuild (for example,
        checks that read the original training code), so this list does not
        prove that every item is fixed.
        """
        after = {
            finding.check_id
            for finding in self.after_findings
        }

        before = {
            finding.check_id
            for finding in self.before_findings
        }

        return sorted(
            before - after
        )

    def to_dict(self) -> dict:
        return {
            "applied": self.applied,
            "rounds": self.rounds,
            "eval_rows_changed": self.eval_rows_changed,
            "model_rebuilt": self.model_rebuilt,
            "reporting_changed": self.reporting_changed,
            "not_fixable": self.not_fixable,
            "before_test_metrics": self.before_metrics,
            "after_test_metrics": self.after_metrics,
            "before_train_metrics": self.before_train_metrics,
            "after_train_metrics": self.after_train_metrics,
            "issues_before": len(self.before_findings),
            "issues_after": len(self.after_findings),
            "not_open_after_update": self.not_open_ids,
            "remaining": [
                finding.to_dict()
                for finding in self.after_findings
            ],
            "error": self.error,
        }


def _preprocessor(
    df: pd.DataFrame,
    cols: list,
) -> ColumnTransformer:
    """Build train-only preprocessing for mixed tabular columns."""

    categorical = [
        column
        for column in cols
        if is_categorical(
            df[column]
        )
    ]

    numeric = [
        column
        for column in cols
        if column not in categorical
    ]

    parts = []

    if numeric:
        parts.append(
            (
                "num",
                Pipeline(
                    [
                        (
                            "impute",
                            SimpleImputer(
                                strategy="median"
                            ),
                        ),
                        (
                            "scale",
                            StandardScaler(),
                        ),
                    ]
                ),
                numeric,
            )
        )

    if categorical:
        parts.append(
            (
                "cat",
                Pipeline(
                    [
                        (
                            "impute",
                            SimpleImputer(
                                strategy="most_frequent"
                            ),
                        ),
                        (
                            "onehot",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                sparse_output=False,
                            ),
                        ),
                    ]
                ),
                categorical,
            )
        )

    # Dense output also works with estimators that do not accept sparse input.
    return ColumnTransformer(
        parts,
        remainder="drop",
        sparse_threshold=0,
    )


def _action_key(
    action: dict,
) -> str:
    """Return a stable identity for one machine-readable repair action."""

    return json.dumps(
        action,
        sort_keys=True,
        default=str,
        separators=(",", ":"),
    )


def _usable_findings(
    findings: list,
) -> list:
    """Return findings that have a sufficiently supported automatic action."""

    return [
        finding
        for finding in findings
        if finding.fix_action
        and finding.confidence >= 0.5
    ]


def _reporting_context(
    ctx: AuditContext,
    reported_metric: Optional[str],
) -> AuditContext:
    """Copy the audit setup without retraining the model."""

    new_ctx = AuditContext(
        ctx.train,
        ctx.test,
        ctx.target,
        model=ctx.model,
        pipeline_code=ctx.pipeline_code,
        task=ctx.task,
        time_col=ctx.time_col,
        group_col=ctx.group_col,
        id_cols=ctx.id_cols,
        reported_metric=reported_metric,
        goal=ctx.goal,
        feature_cols=ctx.feature_cols,
        sample_rows=ctx.sample_rows,
        random_state=ctx.random_state,
    )

    new_ctx.flags.update(
        ctx.flags
    )

    return new_ctx


def _finish_without_rebuild(
    ctx: AuditContext,
    res: FixResult,
    reporting_changed: bool,
):
    """Re-audit metadata-only changes while keeping the model unchanged."""

    reported_metric = (
        "balanced_metrics"
        if reporting_changed
        else ctx.reported_metric
    )

    new_ctx = _reporting_context(
        ctx,
        reported_metric,
    )

    res.after_findings, res.after_runs = run_checks(
        new_ctx
    )

    if reporting_changed:
        res.after_metrics = new_ctx.metrics(
            "test"
        )

        res.after_train_metrics = new_ctx.metrics(
            "train"
        )

    res.new_model = ctx.model
    res.new_ctx = new_ctx

    res.model_rebuilt = False
    res.eval_rows_changed = False


def auto_fix(
    ctx: AuditContext,
    findings: list,
    max_rounds: int = 3,
) -> FixResult:
    """Apply supported repairs, retrain only when needed, and re-audit."""

    if (
        not isinstance(
            max_rounds,
            int,
        )
        or isinstance(
            max_rounds,
            bool,
        )
        or max_rounds < 1
    ):
        raise ValueError(
            "max_rounds must be a positive integer."
        )

    res = FixResult(
        before_findings=list(
            findings
        ),
        before_metrics=ctx.metrics(
            "test"
        ),
        before_train_metrics=ctx.metrics(
            "train"
        ),
    )

    if not _usable_findings(
        findings
    ):
        res.after_findings = list(
            findings
        )

        res.new_model = ctx.model
        res.new_ctx = ctx

        return res

    todo = list(
        findings
    )

    seen_changes = set()

    try:
        for round_number in range(
            1,
            max_rounds + 1,
        ):
            res.applied = []
            res.not_fixable = []

            _apply(
                ctx,
                todo,
                res,
            )

            changes = [
                item
                for item in res.applied
                if item["change"]
                not in seen_changes
            ]

            seen_changes.update(
                item["change"]
                for item in changes
            )

            res.rounds.append(
                {
                    "round": round_number,
                    "changes": changes,
                    "reaudit": [
                        {
                            "check_id": finding.check_id,
                            "title": finding.title,
                            "severity": finding.severity,
                        }
                        for finding in res.after_findings
                    ],
                }
            )

            # Reporting-only or unsuccessful repairs must not cause
            # unrelated model changes on later rounds.
            if not res.model_rebuilt:
                break

            completed_actions = {
                _action_key(
                    finding.fix_action
                )
                for finding in todo
                if finding.fix_action
                and finding.confidence >= 0.5
            }

            new_findings = [
                finding
                for finding in res.after_findings
                if finding.fix_action
                and finding.confidence >= 0.5
                and _action_key(
                    finding.fix_action
                )
                not in completed_actions
            ]

            if not new_findings:
                break

            todo.extend(
                new_findings
            )

    except Exception as exc:  # noqa: BLE001
        res.error = (
            f"{type(exc).__name__}: {exc}"
        )

    return res


def _apply(
    ctx: AuditContext,
    findings: list,
    res: FixResult,
):
    """Apply the cumulative repair plan from the original audit context."""

    usable = _usable_findings(
        findings
    )

    actions = [
        finding.fix_action
        for finding in usable
    ]

    action_types = {
        action["type"]
        for action in actions
    }

    def log(
        change: str,
        *types: str,
    ):
        because = sorted(
            {
                finding.check_id
                for finding in usable
                if finding.fix_action[
                    "type"
                ]
                in types
            }
        )

        res.applied.append(
            {
                "change": change,
                "because": because,
            }
        )

    reporting_changed = (
        "report_balanced_metrics"
        in action_types
    )

    res.reporting_changed = (
        reporting_changed
    )

    unknown_actions = (
        action_types
        - SUPPORTED_ACTIONS
    )

    for action_type in sorted(
        unknown_actions
    ):
        res.not_fixable.append(
            f"Unknown automatic repair action '{action_type}' "
            "is not supported by this fixer."
        )

    training_types = (
        action_types
        - REPORT_ONLY_ACTIONS
        - unknown_actions
    )

    if not training_types:
        if reporting_changed:
            log(
                "Reports recall, precision, F1 and AUC for the rare class "
                "instead of accuracy alone "
                "(a reporting change only; it does not change the model).",
                "report_balanced_metrics",
            )

        _finish_without_rebuild(
            ctx,
            res,
            reporting_changed,
        )

        return

    target = ctx.target

    train = ctx.train.copy()
    test = ctx.test.copy()

    features = list(
        ctx.feature_cols
    )

    substantive_rebuild = (
        "rebuild_preprocessing"
        in training_types
    )

    # 1) Apply only column exclusions explicitly approved by checks.
    drop = {
        column
        for action in actions
        if action["type"]
        == "drop_columns"
        for column in action[
            "columns"
        ]
    } & set(
        features
    )

    if drop:
        substantive_rebuild = True

        features = [
            column
            for column in features
            if column not in drop
        ]

        log(
            "Removes automatically excluded column(s): "
            + ", ".join(
                f"'{column}'"
                for column in sorted(
                    drop
                )
            )
            + ".",
            "drop_columns",
        )

    untrusted = {
        column
        for action in actions
        if action["type"]
        == "exclude_untrusted_column"
        for column in action[
            "columns"
        ]
    } & set(
        features
    )

    if untrusted:
        substantive_rebuild = True

        features = [
            column
            for column in features
            if column not in untrusted
        ]

        log(
            "Excludes "
            + ", ".join(
                f"'{column}'"
                for column in sorted(
                    untrusted
                )
            )
            + " from the reference rebuild because the category codes "
            "cannot be trusted. Re-encode the original category values "
            "before adding the column back.",
            "exclude_untrusted_column",
        )

    if not features:
        raise ValueError(
            "No usable feature columns remain after automatic exclusions."
        )

    # 2) Repair the evaluation split.
    # Group and time repairs do not remove duplicate rows.
    test_fraction = (
        len(test)
        / (
            len(train)
            + len(test)
        )
    )

    full = pd.concat(
        [
            train,
            test,
        ],
        ignore_index=True,
    )

    time_action = next(
        (
            action
            for action in actions
            if action["type"]
            == "time_split"
        ),
        None,
    )

    group_action = next(
        (
            action
            for action in actions
            if action["type"]
            == "group_split"
        ),
        None,
    )

    split_changed = False

    if time_action:
        column = time_action[
            "time_col"
        ]

        parsed = parse_datetimes(
            full[column]
        )

        if parsed.isna().any():
            res.not_fixable.append(
                f"Cannot re-split by '{column}' automatically because "
                "some values are missing or cannot be read as dates."
            )

        else:
            order = parsed.sort_values(
                kind="stable"
            ).index

            full = full.loc[
                order
            ].reset_index(
                drop=True
            )

            if len(full) < 2:
                raise ValueError(
                    "Too few rows remain for a time-based train/test split."
                )

            n_test = int(
                round(
                    len(full)
                    * test_fraction
                )
            )

            n_test = max(
                1,
                min(
                    len(full) - 1,
                    n_test,
                ),
            )

            train = full.iloc[
                :-n_test
            ].copy()

            test = full.iloc[
                -n_test:
            ].copy()

            split_changed = True
            substantive_rebuild = True

            log(
                f"Re-splits by date on '{column}': trains on earlier "
                "rows and tests on the most recent rows.",
                "time_split",
            )

            if group_action:
                res.not_fixable.append(
                    "Both time order and group separation are requested. "
                    "The reference rebuild applies the time split; verify "
                    "a combined group-and-time validation design manually."
                )

    elif group_action:
        column = group_action[
            "group_col"
        ]

        if full[
            column
        ].isna().any():
            res.not_fixable.append(
                f"Cannot re-split by '{column}' automatically because "
                "some group values are missing."
            )

        elif full[
            column
        ].nunique(
            dropna=True
        ) < 2:
            res.not_fixable.append(
                f"Cannot re-split by '{column}' automatically because "
                "fewer than two groups are available."
            )

        else:
            splitter = GroupShuffleSplit(
                n_splits=1,
                test_size=test_fraction,
                random_state=0,
            )

            train_index, test_index = next(
                splitter.split(
                    full,
                    groups=full[
                        column
                    ],
                )
            )

            train = full.iloc[
                train_index
            ].copy()

            test = full.iloc[
                test_index
            ].copy()

            split_changed = True
            substantive_rebuild = True

            log(
                f"Re-splits by '{column}' so each "
                f"{column.replace('_', ' ')} is only in training "
                "or only in test.",
                "group_split",
            )

    res.eval_rows_changed = (
        split_changed
    )

    # 3) Clone the estimator and apply supported parameter changes.
    base = clone(
        unwrap_estimator(
            ctx.model
        )
    )

    params = base.get_params()

    use_sample_weight = False

    if (
        "class_weight"
        in action_types
        and ctx.is_classification
    ):
        if "class_weight" in params:
            base.set_params(
                class_weight="balanced"
            )

            substantive_rebuild = True

            log(
                "Gives the rare class extra weight "
                "(class_weight='balanced').",
                "class_weight",
            )

        elif (
            "sample_weight"
            in inspect.signature(
                base.fit
            ).parameters
        ):
            use_sample_weight = True
            substantive_rebuild = True

            log(
                "Gives the rare class extra weight with "
                "balanced sample weights.",
                "class_weight",
            )

        else:
            res.not_fixable.append(
                "This estimator does not expose class_weight or "
                "sample_weight; compare an appropriate resampling "
                "method manually."
            )

    if "regularize" in action_types:
        changes = {}

        if (
            "max_depth" in params
            and params[
                "max_depth"
            ] is None
        ):
            changes[
                "max_depth"
            ] = 6

        if params.get(
            "min_samples_leaf"
        ) == 1:
            changes[
                "min_samples_leaf"
            ] = 10

        c_value = params.get(
            "C"
        )

        if (
            isinstance(
                c_value,
                (
                    int,
                    float,
                    np.integer,
                    np.floating,
                ),
            )
            and c_value > 0
        ):
            changes[
                "C"
            ] = float(
                c_value
            ) * 0.5

        if changes:
            base.set_params(
                **changes
            )

            substantive_rebuild = True

            log(
                "Makes the estimator more regularized: "
                + ", ".join(
                    f"{name}={value}"
                    for name, value in changes.items()
                )
                + ".",
                "regularize",
            )

        else:
            res.not_fixable.append(
                "This estimator has no conservative regularization "
                "setting that Model Doctor changes automatically."
            )

    # If every requested training repair failed safely,
    # keep the original model unchanged.
    if not substantive_rebuild:
        if reporting_changed:
            log(
                "Reports recall, precision, F1 and AUC for the rare class "
                "instead of accuracy alone "
                "(a reporting change only; it does not change the model).",
                "report_balanced_metrics",
            )

        _finish_without_rebuild(
            ctx,
            res,
            reporting_changed,
        )

        return

    # 4) Build the reference model with train-only preprocessing.
    model = Pipeline(
        [
            (
                "prep",
                _preprocessor(
                    train,
                    features,
                ),
            ),
            (
                "model",
                base,
            ),
        ]
    )

    # This is a standard step of the reference rebuild,
    # not a separate finding-specific repair.
    log(
        "Builds train-only preprocessing for the reference model: "
        "fills missing values, scales numeric columns, and one-hot "
        "encodes categorical columns with unknown-category handling."
    )

    fit_kwargs = {}

    if use_sample_weight:
        fit_kwargs[
            "model__sample_weight"
        ] = compute_sample_weight(
            "balanced",
            train[
                target
            ],
        )

    with warnings.catch_warnings():
        warnings.simplefilter(
            "ignore"
        )

        model.fit(
            train[
                features
            ],
            train[
                target
            ],
            **fit_kwargs,
        )

    if reporting_changed:
        log(
            "Reports recall, precision, F1 and AUC for the rare class "
            "instead of accuracy alone "
            "(a reporting change only; it does not change training).",
            "report_balanced_metrics",
        )

    # DQ-004 can exclude an untrusted coded column from the reference
    # rebuild, but restoring its real meaning still needs original labels.
    for finding in findings:
        if finding.check_id != "DQ-004":
            continue

        column = finding.evidence.get(
            "column"
        )

        if finding.fix_action:
            res.not_fixable.append(
                f"Re-encode '{column}' from the original category values "
                "with one encoder that learns from training data only, "
                "then add it back."
            )

        else:
            res.not_fixable.append(
                f"Check how '{column}' is converted to numbers. "
                "The evidence comes from the data alone, so Model Doctor "
                "does not change this column automatically."
            )

    reported_metric = (
        "balanced_metrics"
        if reporting_changed
        else ctx.reported_metric
    )

    new_ctx = AuditContext(
        train,
        test,
        target,
        model,
        task=ctx.task,
        time_col=ctx.time_col,
        group_col=ctx.group_col,
        id_cols=ctx.id_cols,
        reported_metric=reported_metric,
        goal=ctx.goal,
        feature_cols=features,
        sample_rows=ctx.sample_rows,
        random_state=ctx.random_state,
    )

    new_ctx.flags[
        "preprocessing_rebuilt"
    ] = True

    new_ctx.flags[
        "class_weighted"
    ] = bool(
        use_sample_weight
        or base.get_params().get(
            "class_weight"
        )
        == "balanced"
    )

    res.after_findings, res.after_runs = run_checks(
        new_ctx
    )

    res.after_metrics = new_ctx.metrics(
        "test"
    )

    res.after_train_metrics = new_ctx.metrics(
        "train"
    )

    res.new_model = model
    res.new_ctx = new_ctx
    res.model_rebuilt = True
    