"""BROKEN PIPELINE 2: customer churn (Random Forest).

Planted bugs:
  * Two monthly exports were concatenated, so ~30% of rows are duplicated -> duplicates across splits
  * Each customer has several monthly rows but the split is random -> same customers in train and test
  * Cross-validation uses plain KFold on grouped data -> optimistic CV score
  * Fully grown trees (max_depth=None, min_samples_leaf=1) -> overfitting
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from _data import make_churn, save_fixture

df = make_churn()
df = pd.concat([df, df.sample(frac=0.3, random_state=7)], ignore_index=True)   # BUG: exports merged twice
df = pd.get_dummies(df, columns=["plan"], dtype=int)

features = [c for c in df.columns if c not in ("customer_id", "churned")]
train, test = train_test_split(df, test_size=0.25, random_state=0)             # BUG: random split, grouped data

model = RandomForestClassifier(n_estimators=200, random_state=0)                  # BUG: unlimited depth
cv = cross_val_score(model, train[features], train["churned"], cv=KFold(5, shuffle=True, random_state=0))  # BUG
model.fit(train[features], train["churned"])
print("CV accuracy:", cv.mean(), "Test accuracy:", model.score(test[features], test["churned"]))

save_fixture(__file__, model, train, test,
             {"target": "churned", "group_col": "customer_id", "reported_metric": "accuracy",
              "goal": "Predict which customers will churn next month.",
              "expected_checks": ["CONT-001", "CONT-002", "CONT-003", "OVERFIT-001"]})
