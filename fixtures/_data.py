"""Synthetic but realistic datasets used by the fixture pipelines.

Kept separate so each fixture script reads like a normal training notebook.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def _sigmoid(z):
    return 1 / (1 + np.exp(-z))


def make_loans(n=4000, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    income = rng.lognormal(10.8, 0.5, n)
    credit = rng.normal(660, 70, n).clip(300, 850)
    amount = rng.lognormal(9.5, 0.6, n)
    debt_ratio = rng.beta(2, 5, n)
    years = rng.gamma(2, 3, n)
    age = rng.integers(21, 70, n)
    z = (
        -2.3
        - 0.012 * (credit - 660)
        + 3.0 * (debt_ratio - 0.28)
        + 0.35 * np.log(amount / income * 10)
        - 0.05 * years
    )
    default = (rng.random(n) < _sigmoid(z)).astype(int)
    # recorded by the collections team AFTER a borrower defaults -> leaks the label
    collections_flag = np.where(
        default == 1, rng.random(n) < 0.97, rng.random(n) < 0.01
    ).astype(int)
    return pd.DataFrame(
        {
            "annual_income": income.round(2),
            "credit_score": credit.round(1),
            "loan_amount": amount.round(2),
            "debt_to_income": debt_ratio.round(4),
            "employment_years": years.round(2),
            "age": age,
            "collections_flag": collections_flag,
            "default": default,
        }
    )


def make_churn(n_customers=700, seed=1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for cid in range(10000, 10000 + n_customers):
        plan = rng.choice(["basic", "standard", "premium"], p=[0.5, 0.3, 0.2])
        tenure = int(rng.integers(1, 60))
        charges = {"basic": 25, "standard": 55, "premium": 95}[plan] + rng.normal(0, 6)
        habit = rng.normal(35, 12)  # customer-specific usage level
        propensity = rng.normal(0, 1.2)  # hidden customer trait
        for m in range(int(rng.integers(3, 9))):
            calls = rng.poisson(1 + max(0, propensity))
            session = max(1, habit + rng.normal(0, 4))
            z = (
                -1.0
                + 0.9 * propensity
                - 0.03 * tenure
                + 0.25 * calls
                - 0.02 * (session - 35)
            )
            rows.append(
                {
                    "customer_id": cid,
                    "month_index": m,
                    "plan": plan,
                    "tenure_months": tenure + m,
                    "monthly_charges": round(charges, 2),
                    "support_calls": calls,
                    "avg_session_minutes": round(session, 2),
                    "data_usage_gb": round(max(0, rng.normal(habit / 5, 2)), 2),
                    "churned": int(rng.random() < _sigmoid(z)),
                }
            )
    return pd.DataFrame(rows)


def make_transactions(n=20000, seed=2) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    amount = rng.lognormal(3.8, 1.0, n)
    hour = rng.integers(0, 24, n)
    distance = rng.gamma(1.5, 15, n)
    foreign = (rng.random(n) < 0.08).astype(int)
    tx24 = rng.poisson(3, n)
    age_days = rng.integers(20, 3000, n)
    merchant_risk = rng.beta(2, 8, n)
    z = (
        -5.6
        + 0.9 * (np.log(amount) - 3.8)
        + 1.6 * foreign
        + 4.0 * merchant_risk
        + 1.0 * (hour < 5)
        + 0.02 * distance
        - 0.0008 * age_days
    )
    fraud = (rng.random(n) < _sigmoid(z)).astype(int)
    distance = np.where(
        rng.random(n) < 0.05, -999, distance.round(2)
    )  # -999 = "GPS unavailable"
    return pd.DataFrame(
        {
            "amount": amount.round(2),
            "hour": hour,
            "distance_from_home_km": distance,
            "is_foreign": foreign,
            "tx_last_24h": tx24,
            "account_age_days": age_days,
            "merchant_risk_score": merchant_risk.round(4),
            "is_fraud": fraud,
        }
    )


def make_store_days(seed=3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    types = [
        "street",
        "hyper",
        "mall",
        "street",
        "mall",
        "street",
        "hyper",
        "mall",
        "street",
        "mall",
        "street",
        "hyper",
    ]
    base = {"hyper": 900, "mall": 600, "street": 350}
    dates = pd.date_range("2023-01-01", "2024-12-31", freq="D")
    rows = []
    for sid, st in enumerate(types):
        level = base[st] * rng.uniform(0.9, 1.1)
        ar = 0.0
        for i, d in enumerate(dates):
            ar = 0.7 * ar + rng.normal(0, 0.08)
            promo = int(rng.random() < 0.15)
            season = (
                1
                + 0.15 * np.sin(2 * np.pi * d.dayofyear / 365)
                + (0.2 if d.dayofweek >= 5 else 0)
            )
            trend = 1 + 0.25 * i / len(dates)
            sales = level * season * trend * (1.25 if promo else 1) * np.exp(ar)
            temp = (
                25
                + 10 * np.sin(2 * np.pi * (d.dayofyear - 100) / 365)
                + rng.normal(0, 3)
            )
            rows.append(
                {
                    "date": d.strftime("%Y-%m-%d"),
                    "store_id": sid,
                    "store_type": st,
                    "day_of_week": d.dayofweek,
                    "month": d.month,
                    "promo": promo,
                    "temperature": round(temp, 1) if rng.random() > 0.04 else np.nan,
                    "sales": sales,
                }
            )
    df = pd.DataFrame(rows)
    df["sales_last_week"] = df.groupby("store_id")["sales"].shift(7)
    df["high_demand"] = (df["sales"] > df["sales"].quantile(0.7)).astype(int)
    return df


def make_hr(n=3000, seed=4) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dept = rng.choice(
        ["sales", "engineering", "support", "operations"], n, p=[0.3, 0.3, 0.2, 0.2]
    )
    satisfaction = rng.uniform(0.1, 1.0, n)
    hours = rng.normal(170, 30, n)
    years = rng.integers(0, 15, n)
    salary = rng.choice(["low", "medium", "high"], n, p=[0.45, 0.4, 0.15])
    overtime = (rng.random(n) < 0.3).astype(int)
    z = (
        -1.2
        - 3.0 * (satisfaction - 0.5)
        + 0.8 * overtime
        + 0.5 * (salary == "low")
        - 0.08 * years
    )
    left = (rng.random(n) < _sigmoid(z)).astype(int)
    hours = np.where(rng.random(n) < 0.03, np.nan, hours.round(1))
    return pd.DataFrame(
        {
            "department": dept,
            "satisfaction": satisfaction.round(3),
            "monthly_hours": hours,
            "years_at_company": years,
            "salary_band": salary,
            "overtime": overtime,
            "left_company": left,
        }
    )


def save_fixture(
    script: str,
    model,
    train: pd.DataFrame,
    test: pd.DataFrame,
    config: dict,
):
    """Write a runnable fixture plus separate test-only expectations."""
    out = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path(script).parent / "output" / Path(script).stem
    )
    out.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, out / "model.joblib")
    train.to_csv(out / "train.csv", index=False)
    test.to_csv(out / "test.csv", index=False)
    shutil.copy(script, out / "train_pipeline.py")

    fixture_config = dict(config)
    expected_checks = fixture_config.pop("expected_checks", [])

    fixture_config = {
        "model": "model.joblib",
        "train": "train.csv",
        "test": "test.csv",
        "code": "train_pipeline.py",
        **fixture_config,
    }

    (out / "config.json").write_text(
        json.dumps(fixture_config, indent=2),
        encoding="utf-8",
    )

    (out / "expected_checks.json").write_text(
        json.dumps(expected_checks, indent=2),
        encoding="utf-8",
    )

    print(f"saved fixture to {out}")
    return out
