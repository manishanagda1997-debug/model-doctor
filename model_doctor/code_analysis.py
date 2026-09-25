"""Reads training code without running it and records patterns that audit checks use.

The module uses Python's ``ast`` parser to inspect scripts and notebook code.
It records train/test splits, preprocessing fits, cross-validation choices,
metrics in the code, and separate test-data encodings.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


PREPROCESSORS = {
    "StandardScaler",
    "MinMaxScaler",
    "RobustScaler",
    "MaxAbsScaler",
    "QuantileTransformer",
    "PowerTransformer",
    "SimpleImputer",
    "KNNImputer",
    "IterativeImputer",
    "OneHotEncoder",
    "OrdinalEncoder",
    "LabelEncoder",
    "TruncatedSVD",
    "PCA",
    "TfidfVectorizer",
    "CountVectorizer",
    "KBinsDiscretizer",
    "VarianceThreshold",
    "ColumnTransformer",
    "FeatureUnion",
}

# These tools use labels, so fitting them before a split creates a stronger leakage risk.
HIGH_IMPACT = {
    "SMOTE",
    "ADASYN",
    "BorderlineSMOTE",
    "SVMSMOTE",
    "RandomOverSampler",
    "RandomUnderSampler",
    "SelectKBest",
    "SelectPercentile",
    "SelectFromModel",
    "RFE",
    "RFECV",
    "TargetEncoder",
}

ALL_TRANSFORMERS = PREPROCESSORS | HIGH_IMPACT

SPLIT_FUNCS = {
    "train_test_split",
}

CV_NAIVE = {
    "KFold",
    "StratifiedKFold",
    "ShuffleSplit",
    "StratifiedShuffleSplit",
    "RepeatedKFold",
    "RepeatedStratifiedKFold",
}

CV_AWARE = {
    "GroupKFold",
    "StratifiedGroupKFold",
    "GroupShuffleSplit",
    "LeaveOneGroupOut",
    "LeavePGroupsOut",
    "TimeSeriesSplit",
    "PredefinedSplit",
}

CV_EVAL_FUNCS = {
    "cross_val_score",
    "cross_validate",
    "cross_val_predict",
}

CV_SEARCH_CLASSES = {
    "GridSearchCV",
    "RandomizedSearchCV",
    "HalvingGridSearchCV",
    "HalvingRandomSearchCV",
}

# Other checks can use this combined name.
CV_FUNCS = CV_EVAL_FUNCS | CV_SEARCH_CLASSES

IMBALANCE_AWARE_METRICS = {
    "f1_score",
    "recall_score",
    "precision_score",
    "roc_auc_score",
    "balanced_accuracy_score",
    "average_precision_score",
    "classification_report",
    "precision_recall_curve",
    "fbeta_score",
    "matthews_corrcoef",
    "confusion_matrix",
    "precision_recall_fscore_support",
}

ENCODERS = {
    "LabelEncoder",
    "OrdinalEncoder",
}

TEST_NAME_RE = re.compile(
    r"(^|_)(test|valid|validation|holdout|eval|evaluation|te|val|tst)($|_)",
    re.I,
)

TARGET_NAME_RE = re.compile(
    r"(^|_)(y|target|label|labels|outcome)($|_)",
    re.I,
)


@dataclass
class CodeFacts:
    parsed: bool = False
    error: Optional[str] = None
    split_line: Optional[int] = None
    boundary_line: Optional[int] = None

    fits: list = field(default_factory=list)
    full_data_fills: list = field(default_factory=list)
    cv_naive: list = field(default_factory=list)
    cv_aware: list = field(default_factory=list)
    cv_funcs: list = field(default_factory=list)
    metrics_used: set = field(default_factory=set)
    uses_score_method: bool = False
    split_shuffled: Optional[bool] = None
    names_used: set = field(default_factory=set)
    test_encodings: list = field(default_factory=list)

    def encodes_test_separately(self, column: str) -> Optional[bool]:
        """Returns whether the code separately encodes a test-data column.

        Returns None when separate test encoding exists but the column name
        cannot be read from the code.
        """
        if any(
            column in item["columns"]
            for item in self.test_encodings
        ):
            return True

        if any(
            not item["columns"]
            for item in self.test_encodings
        ):
            return None

        return False

    @property
    def leaky_fits(self) -> list:
        return [
            item
            for item in self.fits
            if item["before_split"] or item["on_test"]
        ]

    @property
    def reports_only_accuracy(self) -> bool:
        uses_accuracy = (
            "accuracy_score" in self.metrics_used
            or self.uses_score_method
        )

        return uses_accuracy and not (
            self.metrics_used
            & IMBALANCE_AWARE_METRICS
        )


def load_code(path: str) -> str:
    """Reads a .py file or the code cells of a .ipynb notebook."""
    file_path = Path(path)
    text = file_path.read_text(encoding="utf-8")

    if file_path.suffix.lower() == ".ipynb":
        notebook = json.loads(text)

        cells = [
            "".join(cell.get("source", []))
            for cell in notebook.get("cells", [])
            if cell.get("cell_type") == "code"
        ]

        text = "\n\n".join(cells)

    return text


def _strip_magics(src: str) -> str:
    """Replaces simple IPython magic and shell lines with valid Python."""
    lines = []

    for line in src.splitlines():
        stripped = line.lstrip()

        if stripped.startswith(("%", "!")):
            indent = line[: len(line) - len(stripped)]
            lines.append(
                f"{indent}pass  # IPython command"
            )
        else:
            lines.append(line)

    return "\n".join(lines)


def _import_aliases(
    tree: ast.AST,
) -> dict[str, str]:
    """Returns aliases from import statements."""
    aliases = {}

    for node in ast.walk(tree):
        if not isinstance(
            node,
            ast.ImportFrom,
        ):
            continue

        for item in node.names:
            if item.name == "*":
                continue

            aliases[
                item.asname or item.name
            ] = item.name

    return aliases


def _call_name(
    node: ast.Call,
    aliases: Optional[dict[str, str]] = None,
) -> Optional[str]:
    func = node.func

    if isinstance(
        func,
        ast.Name,
    ):
        if aliases:
            return aliases.get(
                func.id,
                func.id,
            )

        return func.id

    if isinstance(
        func,
        ast.Attribute,
    ):
        return func.attr

    return None


def _node_text(node) -> Optional[str]:
    try:
        return ast.unparse(node)
    except Exception:
        return None


def _columns_in(node) -> list:
    """Returns string column names that appear in subscript expressions."""
    return sorted(
        {
            item.slice.value
            for item in ast.walk(node)
            if isinstance(
                item,
                ast.Subscript,
            )
            and isinstance(
                item.slice,
                ast.Constant,
            )
            and isinstance(
                item.slice.value,
                str,
            )
        }
    )


def _normalize_name(
    name: str,
) -> str:
    """Normalizes snake_case and CamelCase names for simple name checks."""
    name = re.sub(
        r"(.)([A-Z][a-z]+)",
        r"\1_\2",
        name,
    )

    name = re.sub(
        r"([a-z0-9])([A-Z])",
        r"\1_\2",
        name,
    )

    return name.lower()


def _matches_name(
    node,
    pattern: re.Pattern,
) -> bool:
    """Returns True when a variable or attribute name matches a pattern."""
    for item in ast.walk(node):
        if isinstance(
            item,
            ast.Name,
        ):
            name = item.id

        elif isinstance(
            item,
            ast.Attribute,
        ):
            name = item.attr

        else:
            continue

        if pattern.search(
            _normalize_name(name)
        ):
            return True

    return False


def _looks_like_test_data(
    node,
) -> bool:
    return _matches_name(
        node,
        TEST_NAME_RE,
    )


def _looks_like_target_data(
    node,
) -> bool:
    if _matches_name(
        node,
        TARGET_NAME_RE,
    ):
        return True

    return any(
        isinstance(
            item,
            ast.Subscript,
        )
        and isinstance(
            item.slice,
            ast.Constant,
        )
        and isinstance(
            item.slice.value,
            str,
        )
        and TARGET_NAME_RE.search(
            _normalize_name(
                item.slice.value
            )
        )
        for item in ast.walk(node)
    )


def _record_history(
    history: dict,
    target,
    line: int,
    value,
) -> None:
    name = _node_text(target)

    if name is not None:
        history.setdefault(
            name,
            [],
        ).append(
            (
                line,
                value,
            )
        )


def _latest_value(
    history: dict,
    name: str,
    before_line: int,
):
    matches = [
        (
            line,
            value,
        )
        for line, value
        in history.get(name, [])
        if line < before_line
    ]

    if not matches:
        return None

    return max(
        matches,
        key=lambda item: item[0],
    )[1]


def _known_class(
    node,
    history: dict,
    allowed: set[str],
    before_line: int,
    aliases: dict[str, str],
) -> Optional[str]:
    """Returns a known class from a constructor call or a variable."""
    if isinstance(
        node,
        ast.Call,
    ):
        name = _call_name(
            node,
            aliases,
        )

        if name in allowed:
            return name

        return None

    text = _node_text(node)

    if text is None:
        return None

    value = _latest_value(
        history,
        text,
        before_line,
    )

    if value in allowed:
        return value

    return None


def _keyword_value(
    call: ast.Call,
    name: str,
):
    for keyword in call.keywords:
        if keyword.arg == name:
            return keyword.value

    return None


def _metric_name_from_scoring(
    value: str,
) -> str:
    """Maps a scoring name to the metric family that it represents."""
    scoring = value.lower().strip()

    if scoring == "accuracy":
        return "accuracy_score"

    if scoring == "balanced_accuracy":
        return "balanced_accuracy_score"

    if scoring.startswith("f1"):
        return "f1_score"

    if scoring.startswith("recall"):
        return "recall_score"

    if scoring.startswith("precision"):
        return "precision_score"

    if scoring.startswith("roc_auc"):
        return "roc_auc_score"

    if scoring.startswith(
        "average_precision"
    ):
        return "average_precision_score"

    return scoring


def _metric_names_in_node(
    node,
    aliases: dict[str, str],
) -> set[str]:
    """Returns known metric functions that appear inside an expression."""
    names = set()

    for item in ast.walk(node):
        if isinstance(
            item,
            ast.Name,
        ):
            name = aliases.get(
                item.id,
                item.id,
            )

        elif isinstance(
            item,
            ast.Attribute,
        ):
            name = item.attr

        else:
            continue

        if (
            name
            in IMBALANCE_AWARE_METRICS
            or name == "accuracy_score"
        ):
            names.add(name)

    return names


def _scoring_facts(
    node,
    aliases: dict[str, str],
    scoring_history: dict,
    before_line: int,
    seen: Optional[set[str]] = None,
) -> tuple[set[str], bool]:
    """Returns metric names and whether estimator score is used."""
    if node is None:
        return set(), True

    if isinstance(
        node,
        ast.Constant,
    ):
        if node.value is None:
            return set(), True

        if isinstance(
            node.value,
            str,
        ):
            return {
                _metric_name_from_scoring(
                    node.value
                )
            }, False

        return set(), False

    if isinstance(
        node,
        (
            ast.List,
            ast.Tuple,
            ast.Set,
        ),
    ):
        metrics = set()
        uses_score = False

        for item in node.elts:
            item_metrics, item_uses_score = _scoring_facts(
                item,
                aliases,
                scoring_history,
                before_line,
                seen,
            )

            metrics.update(
                item_metrics
            )

            uses_score = (
                uses_score
                or item_uses_score
            )

        return metrics, uses_score

    if isinstance(
        node,
        ast.Dict,
    ):
        metrics = set()
        uses_score = False

        for item in node.values:
            item_metrics, item_uses_score = _scoring_facts(
                item,
                aliases,
                scoring_history,
                before_line,
                seen,
            )

            metrics.update(
                item_metrics
            )

            uses_score = (
                uses_score
                or item_uses_score
            )

        return metrics, uses_score

    if isinstance(
        node,
        ast.Name,
    ):
        name = node.id

        seen = (
            set()
            if seen is None
            else set(seen)
        )

        if name not in seen:
            seen.add(name)

            saved = _latest_value(
                scoring_history,
                name,
                before_line,
            )

            if saved is not None:
                return _scoring_facts(
                    saved,
                    aliases,
                    scoring_history,
                    before_line,
                    seen,
                )

    metrics = _metric_names_in_node(
        node,
        aliases,
    )

    return metrics, False


def _cv_info(
    node,
    cv_history: dict,
    before_line: int,
    default_name: str,
    aliases: dict[str, str],
) -> Optional[tuple[str, str]]:
    """Returns the CV type and display name that a call uses."""
    if node is None:
        return (
            "naive",
            f"{default_name} (default folds)",
        )

    if isinstance(
        node,
        ast.Constant,
    ):
        if node.value is None:
            return (
                "naive",
                f"{default_name} (default folds)",
            )

        if (
            isinstance(
                node.value,
                int,
            )
            and not isinstance(
                node.value,
                bool,
            )
        ):
            return (
                "naive",
                f"default folds (cv={node.value})",
            )

    cv_name = _known_class(
        node,
        cv_history,
        CV_NAIVE | CV_AWARE,
        before_line,
        aliases,
    )

    if cv_name in CV_NAIVE:
        return (
            "naive",
            cv_name,
        )

    if cv_name in CV_AWARE:
        return (
            "aware",
            cv_name,
        )

    return None


def _record_cv(
    facts: CodeFacts,
    call: ast.Call,
    run_name: str,
    cv_node,
    cv_history: dict,
    aliases: dict[str, str],
) -> None:
    facts.cv_funcs.append(
        {
            "line": call.lineno,
            "name": run_name,
        }
    )

    info = _cv_info(
        cv_node,
        cv_history,
        call.lineno,
        run_name,
        aliases,
    )

    if info is None:
        return

    kind, display_name = info

    if kind == "naive":
        target = facts.cv_naive
    else:
        target = facts.cv_aware

    target.append(
        {
            "line": call.lineno,
            "name": display_name,
        }
    )


def _record_scoring(
    facts: CodeFacts,
    scoring_node,
    aliases: dict[str, str],
    scoring_history: dict,
    line: int,
) -> None:
    metrics, uses_score = _scoring_facts(
        scoring_node,
        aliases,
        scoring_history,
        line,
    )

    facts.metrics_used.update(
        metrics
    )

    facts.uses_score_method = (
        facts.uses_score_method
        or uses_score
    )


def _call_is_inside(
    inner: ast.Call,
    outer: ast.Call,
) -> bool:
    return (
        inner is not outer
        and any(
            node is inner
            for node in ast.walk(outer)
        )
    )


def _before_first_boundary(
    call: ast.Call,
    boundary_call: Optional[ast.Call],
) -> bool:
    """Returns whether a call runs before the first split or CV boundary."""
    if boundary_call is None:
        return False

    # Function arguments run before the outer function call.
    if _call_is_inside(
        call,
        boundary_call,
    ):
        return True

    call_position = (
        call.lineno,
        getattr(
            call,
            "col_offset",
            0,
        ),
    )

    boundary_position = (
        boundary_call.lineno,
        getattr(
            boundary_call,
            "col_offset",
            0,
        ),
    )

    return (
        call_position
        < boundary_position
    )


def analyze_code(
    src: str,
) -> CodeFacts:
    facts = CodeFacts()

    try:
        tree = ast.parse(
            _strip_magics(src)
        )

    except SyntaxError as exc:
        facts.error = (
            "Model Doctor cannot parse "
            f"the training code: {exc}"
        )

        return facts

    facts.parsed = True

    aliases = _import_aliases(
        tree
    )

    transformer_history = {}
    cv_history = {}
    search_history = {}
    scoring_history = {}

    # Records assignments so a later reassignment does not keep an old type.
    for node in ast.walk(tree):
        if isinstance(
            node,
            ast.Assign,
        ):
            value = node.value
            targets = list(
                node.targets
            )

        elif isinstance(
            node,
            ast.AnnAssign,
        ):
            value = node.value
            targets = [
                node.target
            ]

        else:
            continue

        if isinstance(
            value,
            ast.Call,
        ):
            class_name = _call_name(
                value,
                aliases,
            )
        else:
            class_name = None

        if class_name in ALL_TRANSFORMERS:
            transformer_name = (
                class_name
            )
        else:
            transformer_name = None

        if class_name in (
            CV_NAIVE | CV_AWARE
        ):
            cv_name = class_name
        else:
            cv_name = None

        search_info = None

        if (
            isinstance(
                value,
                ast.Call,
            )
            and class_name
            in CV_SEARCH_CLASSES
        ):
            search_info = {
                "class": class_name,
                "cv": _keyword_value(
                    value,
                    "cv",
                ),
                "scoring": _keyword_value(
                    value,
                    "scoring",
                ),
            }

        for target in targets:
            _record_history(
                transformer_history,
                target,
                node.lineno,
                transformer_name,
            )

            _record_history(
                cv_history,
                target,
                node.lineno,
                cv_name,
            )

            _record_history(
                search_history,
                target,
                node.lineno,
                search_info,
            )

            _record_history(
                scoring_history,
                target,
                node.lineno,
                value,
            )

    calls = sorted(
        (
            node
            for node in ast.walk(tree)
            if isinstance(
                node,
                ast.Call,
            )
        ),
        key=lambda node: (
            node.lineno,
            getattr(
                node,
                "col_offset",
                0,
            ),
        ),
    )

    for node in ast.walk(tree):
        if isinstance(
            node,
            ast.Name,
        ):
            facts.names_used.add(
                node.id
            )

        elif isinstance(
            node,
            ast.Attribute,
        ):
            facts.names_used.add(
                node.attr
            )

    boundary_calls = []

    # Records train/test splits, cross-validation runs, and metrics.
    for call in calls:
        name = _call_name(
            call,
            aliases,
        )

        if name in SPLIT_FUNCS:
            boundary_calls.append(
                call
            )

            if facts.split_line is None:
                facts.split_line = (
                    call.lineno
                )

                shuffle_node = _keyword_value(
                    call,
                    "shuffle",
                )

                if shuffle_node is None:
                    facts.split_shuffled = True

                elif (
                    isinstance(
                        shuffle_node,
                        ast.Constant,
                    )
                    and isinstance(
                        shuffle_node.value,
                        bool,
                    )
                ):
                    facts.split_shuffled = (
                        shuffle_node.value
                    )

                else:
                    facts.split_shuffled = None

        if name in CV_EVAL_FUNCS:
            boundary_calls.append(
                call
            )

            _record_cv(
                facts,
                call,
                name,
                _keyword_value(
                    call,
                    "cv",
                ),
                cv_history,
                aliases,
            )

            _record_scoring(
                facts,
                _keyword_value(
                    call,
                    "scoring",
                ),
                aliases,
                scoring_history,
                call.lineno,
            )

        if (
            name == "split"
            and isinstance(
                call.func,
                ast.Attribute,
            )
        ):
            cv_name = _known_class(
                call.func.value,
                cv_history,
                CV_NAIVE | CV_AWARE,
                call.lineno,
                aliases,
            )

            if cv_name is not None:
                boundary_calls.append(
                    call
                )

                facts.cv_funcs.append(
                    {
                        "line": call.lineno,
                        "name": (
                            f"{cv_name}.split"
                        ),
                    }
                )

                if cv_name in CV_NAIVE:
                    target = (
                        facts.cv_naive
                    )
                else:
                    target = (
                        facts.cv_aware
                    )

                target.append(
                    {
                        "line": call.lineno,
                        "name": cv_name,
                    }
                )

        if (
            name == "fit"
            and isinstance(
                call.func,
                ast.Attribute,
            )
        ):
            owner = call.func.value
            search_info = None

            if (
                isinstance(
                    owner,
                    ast.Call,
                )
                and _call_name(
                    owner,
                    aliases,
                )
                in CV_SEARCH_CLASSES
            ):
                search_info = {
                    "class": _call_name(
                        owner,
                        aliases,
                    ),
                    "cv": _keyword_value(
                        owner,
                        "cv",
                    ),
                    "scoring": _keyword_value(
                        owner,
                        "scoring",
                    ),
                }

            else:
                owner_name = _node_text(
                    owner
                )

                if owner_name is not None:
                    search_info = _latest_value(
                        search_history,
                        owner_name,
                        call.lineno,
                    )

            if search_info is not None:
                boundary_calls.append(
                    call
                )

                _record_cv(
                    facts,
                    call,
                    search_info["class"],
                    search_info["cv"],
                    cv_history,
                    aliases,
                )

                _record_scoring(
                    facts,
                    search_info["scoring"],
                    aliases,
                    scoring_history,
                    call.lineno,
                )

        if (
            name
            in IMBALANCE_AWARE_METRICS
            or name == "accuracy_score"
        ):
            facts.metrics_used.add(
                name
            )

        if (
            name == "score"
            and isinstance(
                call.func,
                ast.Attribute,
            )
        ):
            facts.uses_score_method = True

    if boundary_calls:
        first_boundary = min(
            boundary_calls,
            key=lambda call: (
                call.lineno,
                getattr(
                    call,
                    "col_offset",
                    0,
                ),
            ),
        )

        facts.boundary_line = (
            first_boundary.lineno
        )

    else:
        first_boundary = None
        facts.boundary_line = None

    # Records transformer fits before a split or directly on test data.
    for call in calls:
        func = call.func

        if not (
            isinstance(
                func,
                ast.Attribute,
            )
            and func.attr
            in {
                "fit",
                "fit_transform",
                "fit_resample",
            }
        ):
            continue

        obj = func.value

        if (
            isinstance(
                obj,
                ast.Call,
            )
            and _call_name(
                obj,
                aliases,
            )
            in ALL_TRANSFORMERS
        ):
            class_name = _call_name(
                obj,
                aliases,
            )

        else:
            object_name = _node_text(
                obj
            )

            if object_name is not None:
                class_name = _latest_value(
                    transformer_history,
                    object_name,
                    call.lineno,
                )
            else:
                class_name = None

        if class_name is None:
            continue

        if call.args:
            arg_node = call.args[0]
        else:
            arg_node = None

        if arg_node is not None:
            arg = _node_text(
                arg_node
            )
        else:
            arg = ""

        on_test = (
            arg_node is not None
            and _looks_like_test_data(
                arg_node
            )
        )

        # Encoding a clearly named target does not create feature leakage.
        if (
            class_name == "LabelEncoder"
            and arg_node is not None
            and _looks_like_target_data(
                arg_node
            )
            and not on_test
        ):
            continue

        before_split = _before_first_boundary(
            call,
            first_boundary,
        )

        record = {
            "line": call.lineno,
            "cls": class_name,
            "method": func.attr,
            "arg": arg or "",
            "before_split": before_split,
            "on_test": on_test,
            "high_impact": (
                class_name
                in HIGH_IMPACT
            ),
            "code": (
                ast.unparse(
                    call
                )[:120]
            ),
        }

        facts.fits.append(
            record
        )

        if (
            on_test
            and class_name
            in ENCODERS
        ):
            facts.test_encodings.append(
                {
                    "line": call.lineno,
                    "how": class_name,
                    "code": record["code"],
                    "columns": (
                        _columns_in(
                            arg_node
                        )
                        if arg_node
                        is not None
                        else []
                    ),
                }
            )

    # Records separate category encoding on test data.
    for call in calls:
        name = _call_name(
            call,
            aliases,
        )

        if (
            name == "factorize"
            and call.args
            and _looks_like_test_data(
                call.args[0]
            )
        ):
            facts.test_encodings.append(
                {
                    "line": call.lineno,
                    "how": "pd.factorize",
                    "code": (
                        ast.unparse(
                            call
                        )[:120]
                    ),
                    "columns": (
                        _columns_in(
                            call.args[0]
                        )
                    ),
                }
            )

        if (
            name == "get_dummies"
            and call.args
            and _looks_like_test_data(
                call.args[0]
            )
        ):
            facts.test_encodings.append(
                {
                    "line": call.lineno,
                    "how": "pd.get_dummies",
                    "code": (
                        ast.unparse(
                            call
                        )[:120]
                    ),
                    "columns": (
                        _columns_in(
                            call.args[0]
                        )
                    ),
                }
            )

    for node in ast.walk(tree):
        if not (
            isinstance(
                node,
                ast.Attribute,
            )
            and node.attr == "codes"
            and isinstance(
                node.value,
                ast.Attribute,
            )
            and node.value.attr == "cat"
        ):
            continue

        source = (
            node.value.value
        )

        if _looks_like_test_data(
            source
        ):
            facts.test_encodings.append(
                {
                    "line": node.lineno,
                    "how": ".cat.codes",
                    "code": (
                        ast.unparse(
                            node
                        )[:120]
                    ),
                    "columns": (
                        _columns_in(
                            source
                        )
                    ),
                }
            )

    # Records whole-data summary imputation before a split or CV run.
    for call in calls:
        if (
            _call_name(
                call,
                aliases,
            )
            != "fillna"
        ):
            continue

        if call.args:
            fill_value = (
                call.args[0]
            )
        else:
            fill_value = _keyword_value(
                call,
                "value",
            )

        if fill_value is None:
            continue

        inner_calls = [
            node
            for node in ast.walk(
                fill_value
            )
            if isinstance(
                node,
                ast.Call,
            )
        ]

        uses_summary = any(
            _call_name(
                node,
                aliases,
            )
            in {
                "mean",
                "median",
                "mode",
            }
            for node in inner_calls
        )

        before_split = _before_first_boundary(
            call,
            first_boundary,
        )

        if (
            uses_summary
            and before_split
        ):
            facts.full_data_fills.append(
                {
                    "line": call.lineno,
                    "code": (
                        ast.unparse(
                            call
                        )[:120]
                    ),
                }
            )

    return facts