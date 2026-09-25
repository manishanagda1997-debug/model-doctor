"""Command line interface for running Model Doctor audits."""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import joblib
import pandas as pd

from .auditor import audit
from .code_analysis import load_code
from .core import SEVERITY_RANK, describe_model

SEV_MARK = {
    "critical": "[CRITICAL]",
    "high": "[HIGH]    ",
    "medium": "[MEDIUM]  ",
    "low": "[LOW]     ",
}

DEFAULTS = {
    "test_size": 0.25,
    "task": "auto",
    "fix": False,
    "out": "reports/audit",
    "formats": "html,md,json",
    "debug": False,
    "sample_rows": 20000,
    "random_state": 0,
}

PATH_OPTIONS = {"model", "train", "test", "data", "code", "out"}

CONFIG_OPTIONS = {
    "model",
    "train",
    "test",
    "data",
    "split_col",
    "test_size",
    "target",
    "code",
    "task",
    "time_col",
    "group_col",
    "id_cols",
    "features",
    "reported_metric",
    "goal",
    "fix",
    "out",
    "formats",
    "fail_on",
    "debug",
    "sample_rows",
    "random_state",
}

ALLOWED_FORMATS = {
    "html",
    "md",
    "json",
}


def load_model(path: str):
    """Load a trusted joblib or pickle model file."""
    model_path = Path(path).expanduser()

    try:
        return joblib.load(model_path)

    except Exception as joblib_error:  # noqa: BLE001
        try:
            with model_path.open("rb") as file_handle:
                return pickle.load(file_handle)

        except Exception as pickle_error:  # noqa: BLE001
            raise ValueError(
                f"Cannot load model '{model_path}'. "
                f"joblib: {type(joblib_error).__name__}: {joblib_error}; "
                f"pickle: {type(pickle_error).__name__}: {pickle_error}"
            ) from pickle_error


def _read_table(path: str) -> pd.DataFrame:
    """Read a CSV or parquet table from disk."""
    table_path = Path(path).expanduser()

    if table_path.suffix.lower() in (
        ".parquet",
        ".pq",
    ):
        return pd.read_parquet(
            table_path
        )

    return pd.read_csv(
        table_path
    )


def split_by_column(
    df: pd.DataFrame,
    col: str,
):
    """Split rows using a column containing only 'train' and 'test' labels."""
    if col not in df.columns:
        raise SystemExit(
            f"error: split column '{col}' is not in the data"
        )

    flag = (
        df[col]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    bad = sorted(
        set(
            flag.fillna(
                "<empty>"
            ).unique()
        )
        - {
            "train",
            "test",
        }
    )

    if bad:
        raise SystemExit(
            f"error: split column '{col}' must contain only "
            f"'train' or 'test'; it contains: "
            f"{', '.join(map(str, bad[:5]))}"
        )

    train = (
        df[
            flag == "train"
        ]
        .drop(
            columns=col
        )
        .copy()
    )

    test = (
        df[
            flag == "test"
        ]
        .drop(
            columns=col
        )
        .copy()
    )

    if train.empty or test.empty:
        raise SystemExit(
            f"error: split column '{col}' needs both "
            "'train' and 'test' rows"
        )

    return train, test


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser without config-overriding defaults."""
    parser = argparse.ArgumentParser(
        prog="model_doctor",
        description="Automated ML audit toolkit",
    )

    subparsers = parser.add_subparsers(
        dest="cmd",
        required=True,
    )

    audit_parser = subparsers.add_parser(
        "audit",
        help="Audit a model and its data",
    )

    audit_parser.add_argument(
        "--config",
        help=(
            "JSON file with audit options; paths inside it "
            "are relative to the config file"
        ),
    )

    audit_parser.add_argument(
        "--model",
        help=(
            ".joblib/.pkl model or pipeline; "
            "omit for a data-only audit"
        ),
    )

    audit_parser.add_argument(
        "--train",
        help="training data CSV or parquet",
    )

    audit_parser.add_argument(
        "--test",
        help="test data CSV or parquet",
    )

    audit_parser.add_argument(
        "--data",
        help=(
            "single CSV or parquet; Model Doctor creates "
            "a holdout unless --split-col is given"
        ),
    )

    audit_parser.add_argument(
        "--split-col",
        help=(
            "column in --data containing "
            "'train' or 'test' labels"
        ),
    )

    audit_parser.add_argument(
        "--test-size",
        type=float,
        default=None,
        help=(
            "test fraction for an automatically created "
            "holdout (default: 0.25)"
        ),
    )

    audit_parser.add_argument(
        "--target",
        help="name of the column being predicted",
    )

    audit_parser.add_argument(
        "--code",
        help=(
            "training script (.py) or notebook (.ipynb) "
            "for static checks"
        ),
    )

    audit_parser.add_argument(
        "--task",
        default=None,
        choices=[
            "auto",
            "classification",
            "regression",
        ],
    )

    audit_parser.add_argument(
        "--time-col"
    )

    audit_parser.add_argument(
        "--group-col"
    )

    audit_parser.add_argument(
        "--id-cols",
        help=(
            "comma-separated ID columns "
            "that are not features"
        ),
    )

    audit_parser.add_argument(
        "--features",
        help=(
            "comma-separated feature columns "
            "when the model has no feature names"
        ),
    )

    audit_parser.add_argument(
        "--reported-metric",
        help=(
            "metric the team reports, "
            "for example accuracy"
        ),
    )

    audit_parser.add_argument(
        "--goal",
        help="prediction task in plain words",
    )

    audit_parser.add_argument(
        "--fix",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "apply supported repairs and re-audit; "
            "use --no-fix to override config"
        ),
    )

    audit_parser.add_argument(
        "--out",
        default=None,
        help=(
            "output path prefix without an extension "
            "(default: reports/audit)"
        ),
    )

    audit_parser.add_argument(
        "--formats",
        default=None,
        help=(
            "comma-separated report formats: "
            "html, md, json"
        ),
    )

    audit_parser.add_argument(
        "--fail-on",
        choices=list(
            SEVERITY_RANK
        ),
        help=(
            "exit with code 1 if a finding has "
            "this severity or a more severe one"
        ),
    )

    audit_parser.add_argument(
        "--sample-rows",
        type=int,
        default=None,
        help=(
            "maximum sample size for checks that sample "
            "large data (default: 20000)"
        ),
    )

    audit_parser.add_argument(
        "--random-state",
        type=int,
        default=None,
        help=(
            "random seed for sampling and automatic "
            "holdout splitting (default: 0)"
        ),
    )

    audit_parser.add_argument(
        "--debug",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "show a traceback for command-level errors; "
            "use --no-debug to override config"
        ),
    )

    return parser


def _normalise_columns(
    value,
    option: str,
):
    """Normalise comma-separated or JSON-list column options."""
    if value is None:
        return None

    if isinstance(
        value,
        str,
    ):
        raw = value.split(
            ","
        )

    elif isinstance(
        value,
        (
            list,
            tuple,
        ),
    ):
        raw = value

    else:
        raise SystemExit(
            f"error: --{option.replace('_', '-')} must be "
            "a comma-separated string or a JSON list"
        )

    columns = [
        str(
            column
        ).strip()
        for column in raw
        if str(
            column
        ).strip()
    ]

    if not columns:
        raise SystemExit(
            f"error: --{option.replace('_', '-')} cannot be empty"
        )

    return columns


def _normalise_formats(
    value,
) -> list[str]:
    """Return validated report formats from CLI or config input."""
    if isinstance(
        value,
        str,
    ):
        raw = value.split(
            ","
        )

    elif isinstance(
        value,
        (
            list,
            tuple,
        ),
    ):
        raw = value

    else:
        raise SystemExit(
            "error: formats must be a comma-separated "
            "string or a JSON list"
        )

    formats = []

    for item in raw:
        name = str(
            item
        ).strip().lower()

        if (
            name
            and name not in formats
        ):
            formats.append(
                name
            )

    if not formats:
        raise SystemExit(
            "error: at least one report format is required"
        )

    unknown = sorted(
        set(
            formats
        )
        - ALLOWED_FORMATS
    )

    if unknown:
        raise SystemExit(
            f"error: unsupported report format(s): {unknown}; "
            "choose from html, md, json"
        )

    return formats


def _merge_config(
    args,
) -> dict:
    """Merge config values with explicit CLI options; explicit CLI values win."""
    opts = {}

    if args.config:
        config_path = Path(
            args.config
        ).expanduser()

        try:
            raw_config = json.loads(
                config_path.read_text(
                    encoding="utf-8"
                )
            )

        except FileNotFoundError as exc:
            raise SystemExit(
                f"error: config file not found: {config_path}"
            ) from exc

        except json.JSONDecodeError as exc:
            raise SystemExit(
                f"error: config file is not valid JSON: {exc}"
            ) from exc

        if not isinstance(
            raw_config,
            dict,
        ):
            raise SystemExit(
                "error: config JSON must contain "
                "one object of audit options"
            )

        config = {}

        for raw_key, value in raw_config.items():
            key = str(
                raw_key
            ).replace(
                "-",
                "_",
            )

            if key in config:
                raise SystemExit(
                    f"error: config contains duplicate option '{key}'"
                )

            if key not in CONFIG_OPTIONS:
                raise SystemExit(
                    f"error: unknown config option '{raw_key}'"
                )

            config[
                key
            ] = value

        base = (
            config_path
            .resolve()
            .parent
        )

        for key in PATH_OPTIONS:
            value = config.get(
                key
            )

            if not value:
                continue

            path = Path(
                str(
                    value
                )
            ).expanduser()

            if not path.is_absolute():
                path = (
                    base
                    / path
                )

            config[
                key
            ] = str(
                path
            )

        opts.update(
            config
        )

    for key, value in vars(
        args
    ).items():
        if (
            key not in (
                "config",
                "cmd",
            )
            and value is not None
        ):
            opts[
                key
            ] = value

    for key, value in DEFAULTS.items():
        opts.setdefault(
            key,
            value,
        )

    for key in (
        "fix",
        "debug",
    ):
        if not isinstance(
            opts[
                key
            ],
            bool,
        ):
            raise SystemExit(
                f"error: config option '{key}' "
                "must be true or false"
            )

    task = str(
        opts.get(
            "task",
            "auto",
        )
    ).strip().lower()

    if task not in {
        "auto",
        "classification",
        "regression",
    }:
        raise SystemExit(
            "error: task must be 'auto', "
            "'classification', or 'regression'"
        )

    opts[
        "task"
    ] = task

    try:
        test_size = float(
            opts[
                "test_size"
            ]
        )

    except (
        TypeError,
        ValueError,
    ) as exc:
        raise SystemExit(
            "error: test_size must be "
            "a number between 0 and 1"
        ) from exc

    if not (
        0
        < test_size
        < 1
    ):
        raise SystemExit(
            "error: test_size must be "
            "greater than 0 and less than 1"
        )

    opts[
        "test_size"
    ] = test_size

    sample_rows = opts[
        "sample_rows"
    ]

    if (
        not isinstance(
            sample_rows,
            int,
        )
        or isinstance(
            sample_rows,
            bool,
        )
        or sample_rows <= 0
    ):
        raise SystemExit(
            "error: sample_rows must be "
            "a positive integer"
        )

    random_state = opts[
        "random_state"
    ]

    if (
        not isinstance(
            random_state,
            int,
        )
        or isinstance(
            random_state,
            bool,
        )
    ):
        raise SystemExit(
            "error: random_state must be an integer"
        )

    fail_on = opts.get(
        "fail_on"
    )

    if fail_on is not None:
        fail_on = str(
            fail_on
        ).strip().lower()

        if fail_on not in SEVERITY_RANK:
            raise SystemExit(
                "error: fail_on must be one of: "
                + ", ".join(
                    SEVERITY_RANK
                )
            )

        opts[
            "fail_on"
        ] = fail_on

    opts[
        "id_cols"
    ] = _normalise_columns(
        opts.get(
            "id_cols"
        ),
        "id_cols",
    )

    opts[
        "features"
    ] = _normalise_columns(
        opts.get(
            "features"
        ),
        "features",
    )

    opts[
        "formats"
    ] = _normalise_formats(
        opts[
            "formats"
        ]
    )

    return opts


def run_audit(
    opts: dict,
):
    """Load inputs and run one audit from normalised CLI options."""
    target = opts.get(
        "target"
    )

    if (
        not isinstance(
            target,
            str,
        )
        or not target
    ):
        raise SystemExit(
            "error: --target is required"
        )

    has_train = bool(
        opts.get(
            "train"
        )
    )

    has_test = bool(
        opts.get(
            "test"
        )
    )

    has_data = bool(
        opts.get(
            "data"
        )
    )

    if (
        has_data
        and (
            has_train
            or has_test
        )
    ):
        raise SystemExit(
            "error: use either --train with --test, "
            "or --data; do not mix both input modes"
        )

    if has_train != has_test:
        raise SystemExit(
            "error: --train and --test "
            "must be provided together"
        )

    if (
        opts.get(
            "split_col"
        )
        and not has_data
    ):
        raise SystemExit(
            "error: --split-col can only "
            "be used with --data"
        )

    if (
        opts.get(
            "fix"
        )
        and not opts.get(
            "model"
        )
    ):
        raise SystemExit(
            "error: --fix requires --model because "
            "automatic repair retrains a reference model"
        )

    if has_train:
        train = _read_table(
            opts[
                "train"
            ]
        )

        test = _read_table(
            opts[
                "test"
            ]
        )

    elif has_data:
        data = _read_table(
            opts[
                "data"
            ]
        )

        if opts.get(
            "split_col"
        ):
            train, test = split_by_column(
                data,
                opts[
                    "split_col"
                ],
            )

        else:
            from sklearn.model_selection import train_test_split

            print(
                "note: no existing split is provided; "
                "Model Doctor creates a random holdout "
                "split for this audit.",
                file=sys.stderr,
            )

            train, test = train_test_split(
                data,
                test_size=opts[
                    "test_size"
                ],
                random_state=opts[
                    "random_state"
                ],
            )

    else:
        raise SystemExit(
            "error: provide --train and --test, "
            "or provide --data"
        )

    model = (
        load_model(
            opts[
                "model"
            ]
        )
        if opts.get(
            "model"
        )
        else None
    )

    code = (
        load_code(
            opts[
                "code"
            ]
        )
        if opts.get(
            "code"
        )
        else None
    )

    return audit(
        train,
        test,
        target,
        model=model,
        pipeline_code=code,
        fix=opts[
            "fix"
        ],
        task=opts[
            "task"
        ],
        time_col=opts.get(
            "time_col"
        ),
        group_col=opts.get(
            "group_col"
        ),
        id_cols=opts.get(
            "id_cols"
        ),
        feature_cols=opts.get(
            "features"
        ),
        reported_metric=opts.get(
            "reported_metric"
        ),
        goal=opts.get(
            "goal"
        ),
        sample_rows=opts[
            "sample_rows"
        ],
        random_state=opts[
            "random_state"
        ],
    )


def print_summary(
    result,
    stream=sys.stdout,
):
    """Print a concise terminal summary without overstating incomplete audits."""
    coverage = result.coverage()

    if (
        coverage[
            "errors"
        ] == 0
        and coverage[
            "run"
        ] > 0
    ):
        score_text = (
            f"health score "
            f"{result.health_score}/100"
        )

    else:
        score_text = (
            "health score unavailable"
        )

    print(
        f"\nModel Doctor | "
        f"{result.verdict} | "
        f"{score_text}",
        file=stream,
    )

    print(
        f"Model: "
        f"{describe_model(result.ctx.model)}",
        file=stream,
    )

    if not result.findings:
        if coverage[
            "errors"
        ]:
            message = (
                "No findings from the completed checks; "
                "the audit is incomplete."
            )

        elif coverage[
            "run"
        ] == 0:
            message = (
                "No applicable checks completed."
            )

        else:
            message = (
                "No issues detected in the completed checks."
            )

        print(
            f"  {message}",
            file=stream,
        )

    for finding in result.findings:
        print(
            f"  {SEV_MARK[finding.severity]} "
            f"{finding.check_id:<11} "
            f"conf {finding.confidence:.0%}  "
            f"{finding.title}",
            file=stream,
        )

    errors = [
        run
        for run in result.runs
        if run.status == "error"
    ]

    for run in errors:
        print(
            f"  ! check {run.check_id} "
            f"errored: {run.note}",
            file=stream,
        )

    fix_result = result.fix_result

    if fix_result is None:
        return

    if fix_result.error:
        print(
            f"  auto-fix stops with an error: "
            f"{fix_result.error}",
            file=stream,
        )

    elif fix_result.model_rebuilt:
        print(
            "  reference rebuild: "
            f"{len(fix_result.before_findings)} "
            "original finding(s); "
            f"{len(fix_result.after_findings)} still open",
            file=stream,
        )

    elif fix_result.reporting_changed:
        print(
            "  reporting update applied; "
            "fitted model unchanged",
            file=stream,
        )

    elif fix_result.not_fixable:
        print(
            "  no safe automatic model/data repair "
            "is applied; human review is needed",
            file=stream,
        )

    else:
        print(
            "  no automatic model/data "
            "change is applied",
            file=stream,
        )


def main(
    argv=None,
):
    """Run the command-line interface and return a process exit code."""
    args = build_parser().parse_args(
        argv
    )

    opts = None

    try:
        if args.cmd != "audit":
            return 0

        opts = _merge_config(
            args
        )

        result = run_audit(
            opts
        )

        print_summary(
            result
        )

        for path in result.save(
            opts[
                "out"
            ],
            opts[
                "formats"
            ],
        ):
            print(
                f"  wrote {path}"
            )

        if result.coverage()[
            "errors"
        ]:
            return 2

        if (
            result.fix_result is not None
            and result.fix_result.error
        ):
            return 2

        fail_on = opts.get(
            "fail_on"
        )

        if (
            fail_on
            and any(
                SEVERITY_RANK[
                    finding.severity
                ]
                >= SEVERITY_RANK[
                    fail_on
                ]
                for finding in result.findings
            )
        ):
            return 1

        return 0

    except SystemExit:
        raise

    except Exception as exc:  # noqa: BLE001
        debug = bool(
            opts.get(
                "debug"
            )
            if opts is not None
            else getattr(
                args,
                "debug",
                False,
            )
        )

        if debug:
            raise

        print(
            f"error: "
            f"{type(exc).__name__}: "
            f"{exc}",
            file=sys.stderr,
        )

        return 2


if __name__ == "__main__":
    sys.exit(
        main()
    )
    