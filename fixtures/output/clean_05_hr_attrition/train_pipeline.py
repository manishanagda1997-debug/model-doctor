"""CONTROL PIPELINE 5: employee attrition (Random Forest), built correctly.

No bugs planted. Used to show Model Doctor does not raise false alarms:
split first, preprocessing inside a Pipeline, class weights, sensible depth,
and F1 / ROC AUC reported alongside accuracy.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from _data import make_hr, save_fixture

df = make_hr()
X, y = df.drop(columns=["left_company"]), df["left_company"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=3, stratify=y)

prep = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), ["satisfaction", "monthly_hours", "years_at_company", "overtime"]),
    ("cat", OneHotEncoder(handle_unknown="ignore"), ["department", "salary_band"]),
])
model = Pipeline([("prep", prep), ("rf", RandomForestClassifier(n_estimators=200, max_depth=5, min_samples_leaf=20,
                                                                 class_weight="balanced", random_state=0))])
cv = cross_val_score(model, X_train, y_train, cv=StratifiedKFold(5, shuffle=True, random_state=0), scoring="f1")
model.fit(X_train, y_train)
print(classification_report(y_test, model.predict(X_test)))
print("ROC AUC:", roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))

save_fixture(__file__, model, X_train.assign(left_company=y_train.values), X_test.assign(left_company=y_test.values),
             {"target": "left_company", "reported_metric": "f1",
              "goal": "Predict which employees are likely to leave in the next year.",
              "expected_checks": []})
