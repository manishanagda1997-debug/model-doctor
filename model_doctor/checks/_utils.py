"""Small helpers shared by several checks."""

from __future__ import annotations

import re
import warnings

import numpy as np
import pandas as pd

from ..core import ID_NAME_RE, is_categorical

TIME_NAME_RE = re.compile(
    r"date|time|timestamp|datetime|_dt$|^dt$|month|period|week", re.I
)


def pct(x: float, digits: int = 1) -> str:
    return f"{100 * x:.{digits}f}%"


def encode_pair(a: pd.Series, b: pd.Series, y: pd.Series | None = None, minority=None):
    """Returns numeric arrays for one train column and one test column.

    Categories are ordered by their target rate on the training data so a shallow
    tree can use them; missing values get their own value (missingness itself can leak).
    """
    if is_categorical(a):
        a_s, b_s = a.astype("string").fillna("<NA>"), b.astype("string").fillna("<NA>")
        if y is not None and minority is not None:
            order = (y.to_numpy() == minority).astype(float)
            means = pd.Series(order, index=a_s.index).groupby(a_s).mean()
        else:
            means = a_s.value_counts(normalize=True)
        return a_s.map(means).fillna(-1).to_numpy(float), b_s.map(means).fillna(
            -1
        ).to_numpy(float)
    a_f = pd.to_numeric(a, errors="coerce").astype(float)
    b_f = pd.to_numeric(b, errors="coerce").astype(float)
    lo = np.nanmin(a_f.to_numpy()) if a_f.notna().any() else 0.0
    fill = lo - 1 - abs(lo)
    return a_f.fillna(fill).to_numpy(), b_f.fillna(fill).to_numpy()


def parse_datetimes(s: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(s):
        return s
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return pd.to_datetime(s, errors="coerce", format="mixed")
        except (TypeError, ValueError):
            return pd.to_datetime(s, errors="coerce")


def detect_time_cols(df: pd.DataFrame, exclude=()) -> list:
    found = []
    for c in df.columns:
        if c in exclude:
            continue
        s = df[c]
        if pd.api.types.is_datetime64_any_dtype(s):
            found.append(c)
            continue
        if is_categorical(s) and TIME_NAME_RE.search(str(c)):
            sample = s.dropna().astype(str).head(300)
            parsed = parse_datetimes(sample)
            if len(sample) and parsed.notna().mean() > 0.9 and parsed.nunique() >= 10:
                found.append(c)
    return found


def detect_id_cols(df: pd.DataFrame, exclude=()) -> list:
    return [
        c
        for c in df.columns
        if c not in exclude and ID_NAME_RE.search(str(c)) and df[c].nunique() >= 20
    ]


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Returns the Population Stability Index for two numeric samples."""
    if not isinstance(bins, int) or isinstance(bins, bool) or bins < 2:
        raise ValueError("bins must be an integer of at least 2.")

    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)

    expected = expected[np.isfinite(expected)]
    actual = actual[np.isfinite(actual)]

    if len(expected) < 50 or len(actual) < 50 or np.unique(expected).size < 2:
        return 0.0

    edges = np.unique(
        np.quantile(
            expected,
            np.linspace(0, 1, bins + 1),
        )
    )

    if edges.size < 3:
        return 0.0

    edges[0] = -np.inf
    edges[-1] = np.inf

    expected_share = np.histogram(expected, edges)[0] / len(expected)

    actual_share = np.histogram(actual, edges)[0] / len(actual)

    expected_share = np.clip(
        expected_share,
        1e-4,
        None,
    )

    actual_share = np.clip(
        actual_share,
        1e-4,
        None,
    )

    return float(
        np.sum((actual_share - expected_share) * np.log(actual_share / expected_share))
    )


def row_hashes(df: pd.DataFrame) -> pd.Series:
    """Hashes rows after normalizing dtypes and rounding float values."""
    data = df.copy().convert_dtypes()

    for column in data.columns:
        if pd.api.types.is_float_dtype(data[column]):
            data[column] = data[column].round(6)

    return pd.util.hash_pandas_object(
        data,
        index=False,
    )
