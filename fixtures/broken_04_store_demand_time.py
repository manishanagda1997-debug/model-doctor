"""BROKEN PIPELINE 4: daily store demand (HistGradientBoosting), time-series data.

Planted bugs:
  * Random train/test split on daily data -> training on the future (temporal leakage)
  * Shuffled KFold cross-validation on time-ordered data
  * Missing temperatures filled with the mean of the WHOLE dataset before splitting
  * store_type is label-encoded on train, but the test/inference code uses pd.factorize
    -> the same code means a different store type in test (encoding mismatch)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder
from _data import make_store_days, save_fixture

df = make_store_days().dropna(subset=["sales_last_week"])
df["temperature"] = df["temperature"].fillna(df["temperature"].mean())           # BUG: whole-data mean
features = ["store_type", "day_of_week", "month", "promo", "temperature", "sales_last_week"]

train, test = train_test_split(df, test_size=0.2, random_state=0)                # BUG: random split on dates
train, test = train.sort_values("date").copy(), test.sort_values("date").copy()

le = LabelEncoder()
train["store_type"] = le.fit_transform(train["store_type"])
test["store_type"] = pd.factorize(test["store_type"])[0]                         # BUG: different encoding

model = HistGradientBoostingClassifier(random_state=0)
scores = cross_val_score(model, train[features], train["high_demand"], cv=KFold(5, shuffle=True))  # BUG
model.fit(train[features], train["high_demand"])
print("CV:", scores.mean(), "Test accuracy:", model.score(test[features], test["high_demand"]))

keep = ["date", "store_id"] + features + ["high_demand"]
save_fixture(__file__, model, train[keep], test[keep],
             {"target": "high_demand", "time_col": "date", "id_cols": ["store_id"],
              "goal": "Forecast whether tomorrow will be a high-demand day for each store.",
              "expected_checks": ["LEAK-004", "CONT-003", "DQ-004", "LEAK-003"]})
