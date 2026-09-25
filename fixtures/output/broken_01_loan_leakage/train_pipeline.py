"""BROKEN PIPELINE 1: loan default prediction (Logistic Regression).

Planted bugs (what Model Doctor should catch):
  * 'collections_flag' is filled in by the collections team AFTER a default -> target leakage
  * StandardScaler is fitted on the full dataset BEFORE train_test_split -> preprocessing leakage
  * Only accuracy is reported although defaults are rare -> misleading metric
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from _data import make_loans, save_fixture

df = make_loans()
X = df.drop(columns=["default"])
y = df["default"]

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)       # BUG: fit before split

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=42)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
print("Test accuracy:", accuracy_score(y_test, model.predict(X_test)))    # BUG: accuracy only

save_fixture(__file__, model,
             X_train.assign(default=y_train.values), X_test.assign(default=y_test.values),
             {"target": "default", "reported_metric": "accuracy",
              "goal": "Predict which loan applicants will default, at application time.",
              "expected_checks": ["LEAK-001", "LEAK-002", "LEAK-003", "METRIC-001"]})
