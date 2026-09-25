"""BROKEN PIPELINE 3: card fraud detection (Gradient Boosting).

Planted bugs:
  * Only ~1-2% of transactions are fraud and nothing handles the imbalance -> the model predicts almost everything as 'not fraud'
  * The team reports 98% accuracy -> misleading metric (a do-nothing model scores the same)
  * Missing GPS distance is stored as -999 -> disguised missing values
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from _data import make_transactions, save_fixture

df = make_transactions()
X, y = df.drop(columns=["is_fraud"]), df["is_fraud"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1, stratify=y)

model = GradientBoostingClassifier(random_state=0)  # BUG: no imbalance handling
model.fit(X_train, y_train)
print("Accuracy:", accuracy_score(y_test, model.predict(X_test))) # BUG: accuracy only

save_fixture(__file__, model, X_train.assign(is_fraud=y_train.values), X_test.assign(is_fraud=y_test.values),
             {"target": "is_fraud", "reported_metric": "accuracy",
              "goal": "Flag fraudulent card transactions in real time.",
              "expected_checks": ["IMB-001", "IMB-002", "METRIC-001", "DQ-002"]})
