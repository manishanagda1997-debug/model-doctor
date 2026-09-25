"""
Core objects used by Model Doctor.

This module stores audit findings, registered checks, and shared audit data.
Each check receives an AuditContext and returns a list of Finding objects.
The context caches model predictions so checks do not run the same prediction
more than once.

Model Doctor supports scikit-learn-compatible estimators that provide the
standard methods and attributes needed by each check.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
from sklearn import metrics as skm
from sklearn.base import is_classifier, is_regressor
from sklearn.pipeline import Pipeline

SEVERITY_RANK = {"critical": 4, "high": 3, "medium": 2, "low": 1}
# The health score subtracts these points and scales them by evidence confidence.
SEVERITY_WEIGHT = {"critical": 35, "high": 18, "medium": 8, "low": 3}

CATEGORIES = {
    "leakage": "Data leakage",
    "contamination": "Train/test contamination",
    "metrics": "Misleading metrics",
    "overfitting": "Overfitting",
    "data_quality": "Data quality",
    "imbalance": "Class imbalance",
}


class SkipCheck(Exception):
    """A check raises this exception when the check does not apply."""

@dataclass
class Finding:
    check_id: str
    category: str
    title: str
    severity: str  # critical | high | medium | low
    confidence: float  # 0..1 evidence confidence; this is not a probability
    what: str  # plain English: what the check detects
    why: str  # plain English: why the finding matters
    fix: str  # plain English: recommended next step
    evidence: dict = field(default_factory=dict)  # technical details
    fix_action: Optional[dict] = None  # machine-readable fix for the auto-fixer

    def __post_init__(self):
        if self.category not in CATEGORIES:
            raise ValueError(f"Unknown category '{self.category}'.")

        if self.severity not in SEVERITY_RANK:
            raise ValueError(f"Unknown severity '{self.severity}'.")

        try:
            confidence = float(self.confidence)
        except (TypeError, ValueError) as exc:
            raise ValueError("Confidence must be a number between 0 and 1.") from exc

        if not np.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("Confidence must be between 0 and 1.")

        self.confidence = confidence

    @property
    def risk(self) -> float:
        return SEVERITY_RANK[self.severity] * self.confidence

    def to_dict(self) -> dict:
        d = asdict(self)
        d["confidence"] = round(self.confidence, 2)
        d["category_label"] = CATEGORIES.get(self.category, self.category)
        return d


@dataclass
class CheckSpec:
    check_id: str
    name: str
    category: str
    description: str
    fn: Callable


REGISTRY: list[CheckSpec] = []


def register(check_id: str, name: str, category: str, description: str):
    if category not in CATEGORIES:
        raise ValueError(f"Unknown category '{category}'.")

    if any(spec.check_id == check_id for spec in REGISTRY):
        raise ValueError(f"Check ID '{check_id}' is already registered.")

    def deco(fn):
        REGISTRY.append(CheckSpec(check_id, name, category, description, fn))
        return fn

    return deco


# --------------------------------------------------------------------------- helpers
ID_NAME_RE = re.compile(
    r"(^id$|_id$|^id_|uuid|guid|(^|_)key$|"
    r"^(customer|user|patient|account|client|member|session|device|order|txn|transaction|employee|student)"
    r"(_?(id|no|number|num|code))?$)",
    re.I,
)


def is_categorical(s: pd.Series) -> bool:
    return not (pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s))


def unwrap_estimator(model):
    """Returns the final estimator from a possibly nested Pipeline."""
    while isinstance(model, Pipeline):
        model = model.steps[-1][1]
    return model


def describe_model(model) -> str:
    if model is None:
        return "No model provided (data-only audit)"
    if isinstance(model, Pipeline):
        return "Pipeline: " + " -> ".join(type(s).__name__ for _, s in model.steps)
    return type(model).__name__


def feature_names_of(model):
    if model is None:
        return None
    names = getattr(model, "feature_names_in_", None)
    if names is None and isinstance(model, Pipeline):
        names = getattr(model.steps[0][1], "feature_names_in_", None)
    return None if names is None else [str(n) for n in names]


# --------------------------------------------------------------------------- context
class AuditContext:
    """Everything a check needs: data, model, task description and cached predictions."""

    def __init__(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame,
        target: str,
        model: Any = None,
        pipeline_code: Optional[str] = None,
        task: str = "auto",
        time_col: Optional[str] = None,
        group_col: Optional[str] = None,
        id_cols: Optional[list] = None,
        reported_metric: Optional[str] = None,
        goal: Optional[str] = None,
        feature_cols: Optional[list] = None,
        sample_rows: int = 20000,
        random_state: int = 0,
    ):
        if train.empty:
            raise ValueError("Training data cannot be empty.")

        if test.empty:
            raise ValueError("Test data cannot be empty.")
        for name, df in (("train", train), ("test", test)):
            if target not in df.columns:
                raise ValueError(
                    f"Target column '{target}' is missing from the {name} data."
                )

            if df[target].isna().any():
                raise ValueError(
                    f"Target column '{target}' contains missing values in the {name} data."
                )
        for col in (time_col, group_col):
            if not col:
                continue

            for name, df in (("train", train), ("test", test)):
                if col not in df.columns:
                    raise ValueError(
                        f"Column '{col}' is missing from the {name} data."
                    )
        self.train = train.reset_index(drop=True)
        self.test = test.reset_index(drop=True)
        self.target = target
        self.model = model
        self.pipeline_code = pipeline_code
        self.time_col = time_col
        self.group_col = group_col
        requested_id_cols = list(dict.fromkeys(id_cols or []))

        for col in requested_id_cols:
            for name, df in (("train", train), ("test", test)):
                if col not in df.columns:
                    raise ValueError(
                        f"ID column '{col}' is missing from the {name} data."
                    )

        self.id_cols = requested_id_cols
        self.reported_metric = reported_metric.lower().strip() if reported_metric else None
        self.goal = goal or ""
        if not isinstance(sample_rows, int) or isinstance(sample_rows, bool) or sample_rows <= 0:
            raise ValueError("sample_rows must be a positive integer.")

        self.sample_rows = sample_rows
        self.random_state = random_state
        valid_tasks = {"auto", "classification", "regression"}
        if task not in valid_tasks:
            raise ValueError(
                "Task must be 'auto', 'classification', or 'regression'."
            )

        self.task = self._detect_task() if task == "auto" else task
        self.flags: dict = {}
        self.model_error: Optional[str] = None
        self.proba_error: Optional[str] = None
        self._cache: dict = {}
        if feature_cols is not None:
            requested_features = list(feature_cols)

            if not requested_features:
                raise ValueError("The feature list cannot be empty.")

            if len(requested_features) != len(set(requested_features)):
                raise ValueError("The feature list contains duplicate column names.")

            if target in requested_features:
                raise ValueError("The target column cannot also be a feature.")

            for name, df in (("train", self.train), ("test", self.test)):
                missing = [c for c in requested_features if c not in df.columns]
                if missing:
                    raise ValueError(
                        f"Feature columns are missing from the {name} data: {missing[:10]}"
                    )

            self.feature_cols = requested_features
        else:
            self.feature_cols = self._resolve_features()
        
        from .code_analysis import analyze_code  # local import avoids a cycle

        self.code_facts = analyze_code(pipeline_code) if pipeline_code else None

    # ---- setup
    def _detect_task(self) -> str:
        if self.model is not None:
            estimator = unwrap_estimator(self.model)

            if is_classifier(estimator):
                return "classification"

            if is_regressor(estimator):
                return "regression"

        y = self.train[self.target].dropna()

        if is_categorical(y) or pd.api.types.is_bool_dtype(y):
            return "classification"

        if pd.api.types.is_float_dtype(y) and not np.allclose(y, np.round(y)):
            return "regression"

        return "classification" if y.nunique() <= 20 else "regression"

    def _resolve_features(self) -> list:
        names = feature_names_of(self.model)
        if names is not None:
            for split_name, df in (("train", self.train), ("test", self.test)):
                missing = [n for n in names if n not in df.columns]

                if missing:
                    raise ValueError(
                        f"The model expects columns that are missing from the "
                        f"{split_name} data: {missing[:10]}"
                    )

            return names
        excluded = {self.target, *self.id_cols}
        cols = [c for c in self.train.columns if c not in excluded]
        n = getattr(self.model, "n_features_in_", None)
        if self.model is not None and n is not None and len(cols) != n:
            alt = [c for c in cols if c not in {self.time_col, self.group_col}]
            if len(alt) == n:
                return alt
            raise ValueError(
                f"The model expects {n} features but the data has {len(cols)} candidate columns. "
                "Pass the feature list explicitly (--features)."
            )
        return cols

    # ---- data access
    def split(self, name: str) -> pd.DataFrame:
        if name == "train":
            return self.train
        if name == "test":
            return self.test
        raise ValueError("Split name must be 'train' or 'test'.")

    def X(self, name: str):
        X = self.split(name)[self.feature_cols]
        if self.model is not None and feature_names_of(self.model) is None:
            return X.to_numpy()
        return X

    def y(self, name: str) -> np.ndarray:
        return self.split(name)[self.target].to_numpy()

    def sample(self, df: pd.DataFrame) -> pd.DataFrame:
        if len(df) > self.sample_rows:
            return df.sample(self.sample_rows, random_state=self.random_state)
        return df

    @property
    def is_classification(self) -> bool:
        return self.task == "classification"

    @property
    def class_shares(self) -> pd.Series:
        return self.train[self.target].value_counts(normalize=True)

    @property
    def minority_class(self):
        return self.class_shares.index[-1] if self.is_classification else None

    @property
    def majority_class(self):
        return self.class_shares.index[0] if self.is_classification else None

    # ---- model access (the only way checks touch the model)
    def predict(self, name: str):
        if self.model is None:
            return None
        key = ("pred", name)
        if key not in self._cache:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self._cache[key] = np.asarray(self.model.predict(self.X(name)))
            except Exception as e:  # noqa: BLE001 - any model failure is itself a finding
                self.model_error = f"{type(e).__name__}: {e}"
                self._cache[key] = None
        return self._cache[key]

    def proba(self, name: str):
        """Returns class probabilities or None when they are unavailable."""
        if self.model is None or not self.is_classification or not hasattr(self.model, "predict_proba"):
            return None
        key = ("proba", name)
        if key not in self._cache:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self._cache[key] = np.asarray(self.model.predict_proba(self.X(name)))
            except Exception as e:  # noqa: BLE001
                self.proba_error = f"{type(e).__name__}: {e}"
                self._cache[key] = None
        return self._cache[key]

    @property
    def classes(self):
        if self.model is None:
            return None

        c = getattr(self.model, "classes_", None)

        if c is None:
            c = getattr(unwrap_estimator(self.model), "classes_", None)

        return None if c is None else list(c)

    def model_params(self) -> dict:
        est = unwrap_estimator(self.model)
        try:
            return est.get_params()
        except Exception:  # noqa: BLE001
            return {}

    def metrics(self, name: str) -> Optional[dict]:
        pred = self.predict(name)
        if pred is None:
            return None
        key = ("metrics", name)
        if key not in self._cache:
            self._cache[key] = evaluate(self, self.y(name), pred, self.proba(name))
        return self._cache[key]


# --------------------------------------------------------------------------- metrics
def evaluate(ctx: AuditContext, y_true, y_pred, proba=None) -> dict:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if not ctx.is_classification:
            yt = y_true.astype(float)
            yp = np.asarray(y_pred, dtype=float)
            return {
                "r2": float(skm.r2_score(yt, yp)),
                "mae": float(skm.mean_absolute_error(yt, yp)),
                "rmse": float(np.sqrt(skm.mean_squared_error(yt, yp))),
                "primary": float(skm.r2_score(yt, yp)),
                "primary_name": "R-squared",
            }
        minority = ctx.minority_class
        yt_min = y_true == minority
        yp_min = y_pred == minority
        out = {
            "accuracy": float(skm.accuracy_score(y_true, y_pred)),
            "balanced_accuracy": float(skm.balanced_accuracy_score(y_true, y_pred)),
            "f1_macro": float(
                skm.f1_score(y_true, y_pred, average="macro", zero_division=0)
            ),
            "minority_class": _jsonable(minority),
            "minority_recall": float(skm.recall_score(yt_min, yp_min, zero_division=0)),
            "minority_precision": float(skm.precision_score(yt_min, yp_min, zero_division=0)),
            "minority_f1": float(skm.f1_score(yt_min, yp_min, zero_division=0)),
            "majority_baseline_accuracy": float(np.mean(y_true == ctx.majority_class)),
            "predicted_majority_share": float(np.mean(y_pred == ctx.majority_class)),
            "primary_name": "Balanced accuracy",
        }
        out["primary"] = out["balanced_accuracy"]
        classes = ctx.classes
        if proba is not None and classes is not None and len(np.unique(y_true)) > 1:
            try:
                if len(classes) == 2:
                    p = proba[:, classes.index(minority)]
                    out["roc_auc"] = float(skm.roc_auc_score(yt_min, p))
                    out["pr_auc"] = float(skm.average_precision_score(yt_min, p))
                else:
                    out["roc_auc"] = float(
                        skm.roc_auc_score(y_true, proba, multi_class="ovr", labels=classes)
                    )
            except Exception:  # noqa: BLE001
                pass
        return out


def _jsonable(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v if isinstance(v, (int, float, str, bool)) or v is None else str(v)
