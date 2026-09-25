"""Unit tests for static training-code analysis.

Run with: python -m unittest tests.test_code_analysis -v
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_doctor.code_analysis import analyze_code  # noqa: E402


class TestCodeAnalysis(unittest.TestCase):
    def test_fit_before_split(self):
        facts = analyze_code(
            "s = MinMaxScaler()\n"
            "X = s.fit_transform(X)\n"
            "a, b, c, d = train_test_split(X, y)\n"
        )

        self.assertTrue(facts.parsed)
        self.assertEqual(facts.split_line, 3)
        self.assertEqual(facts.boundary_line, 3)
        self.assertTrue(facts.leaky_fits[0]["before_split"])

    def test_import_alias_and_inline_high_impact_transformer(self):
        alias_facts = analyze_code(
            "from sklearn.preprocessing import StandardScaler as SS\n"
            "sc = SS()\n"
            "X = sc.fit_transform(X)\n"
            "X_tr, X_te = train_test_split(X)\n"
        )

        self.assertEqual(
            alias_facts.leaky_fits[0]["cls"],
            "StandardScaler",
        )

        smote_facts = analyze_code(
            "X, y = SMOTE().fit_resample(X, y)\n"
            "X_tr, X_te = train_test_split(X)\n"
        )

        self.assertTrue(
            smote_facts.leaky_fits[0]["high_impact"]
        )

    def test_fit_on_test_is_detected_even_without_split(self):
        facts = analyze_code(
            "enc = OrdinalEncoder()\n"
            "enc.fit_transform(test_df[['city']])\n"
        )

        self.assertEqual(
            len(facts.leaky_fits),
            1,
        )

        self.assertTrue(
            facts.leaky_fits[0]["on_test"]
        )

        self.assertFalse(
            facts.leaky_fits[0]["before_split"]
        )

    def test_test_name_matching_does_not_treat_contest_as_test_data(self):
        facts = analyze_code(
            "enc = OrdinalEncoder()\n"
            "enc.fit_transform(contest_df[['city']])\n"
            "train, test = train_test_split(df)\n"
        )

        self.assertFalse(
            facts.fits[0]["on_test"]
        )

        self.assertTrue(
            facts.fits[0]["before_split"]
        )

    def test_label_encoder_on_target_is_not_feature_leakage(self):
        facts = analyze_code(
            "enc = LabelEncoder()\n"
            "y_encoded = enc.fit_transform(y)\n"
            "X_tr, X_te, y_tr, y_te = train_test_split(X, y_encoded)\n"
        )

        self.assertEqual(
            facts.leaky_fits,
            [],
        )

    def test_clean_train_only_preprocessing_and_metrics(self):
        facts = analyze_code(
            "X_tr, X_te = train_test_split(X)\n"
            "sc = StandardScaler().fit(X_tr)\n"
            "X_tr = sc.transform(X_tr)\n"
            "X_te = sc.transform(X_te)\n"
            "print(f1_score(y, p), roc_auc_score(y, q))\n"
        )

        self.assertEqual(
            facts.leaky_fits,
            [],
        )

        self.assertFalse(
            facts.reports_only_accuracy
        )

        self.assertIn(
            "f1_score",
            facts.metrics_used,
        )

        self.assertIn(
            "roc_auc_score",
            facts.metrics_used,
        )

    def test_accuracy_only_and_estimator_score_are_detected(self):
        direct = analyze_code(
            "print(accuracy_score(y, p))\n"
        )

        self.assertTrue(
            direct.reports_only_accuracy
        )

        estimator_score = analyze_code(
            "print(model.score(X_test, y_test))\n"
        )

        self.assertTrue(
            estimator_score.uses_score_method
        )

        self.assertTrue(
            estimator_score.reports_only_accuracy
        )

    def test_notebook_magics_keep_code_parseable_and_bad_code_is_reported(self):
        facts = analyze_code(
            "if True:\n"
            "    %time x = 1\n"
            "    y = 2\n"
            "!pip install example\n"
        )

        self.assertTrue(
            facts.parsed
        )

        broken = analyze_code(
            "def (:\n"
        )

        self.assertFalse(
            broken.parsed
        )

        self.assertIsNotNone(
            broken.error
        )

    def test_separate_test_encoding_patterns_are_detected(self):
        facts = analyze_code(
            "tr, te = train_test_split(df)\n"
            "tr['store'] = LabelEncoder().fit_transform(tr['store'])\n"
            "te['store'] = pd.factorize(te['store'])[0]\n"
            "te['city'] = te['city'].astype('category').cat.codes\n"
            "enc = LabelEncoder()\n"
            "te['zone'] = enc.fit_transform(test_df['zone'])\n"
            "dummy_test = pd.get_dummies(te)\n"
        )

        self.assertTrue(
            facts.encodes_test_separately(
                "store"
            )
        )

        self.assertTrue(
            facts.encodes_test_separately(
                "city"
            )
        )

        self.assertTrue(
            facts.encodes_test_separately(
                "zone"
            )
        )

        self.assertFalse(
            facts.encodes_test_separately(
                "other"
            )
        )

        self.assertIn(
            "pd.get_dummies",
            {
                item["how"]
                for item in facts.test_encodings
            },
        )

    def test_one_train_fitted_encoder_used_on_both_sides_is_clean(self):
        facts = analyze_code(
            "tr, te = train_test_split(df)\n"
            "le = LabelEncoder().fit(tr['store'])\n"
            "tr['store'] = le.transform(tr['store'])\n"
            "te['store'] = le.transform(te['store'])\n"
        )

        self.assertEqual(
            facts.test_encodings,
            [],
        )

        self.assertEqual(
            facts.leaky_fits,
            [],
        )

    def test_whole_data_fillna_summary_is_detected_for_positional_and_keyword_value(
        self,
    ):
        positional = analyze_code(
            "df = df.fillna(df.mean())\n"
            "tr, te = train_test_split(df)\n"
        )

        keyword = analyze_code(
            "df = df.fillna(value=df.median())\n"
            "tr, te = train_test_split(df)\n"
        )

        self.assertEqual(
            len(positional.full_data_fills),
            1,
        )

        self.assertEqual(
            len(keyword.full_data_fills),
            1,
        )

    def test_cv_detection_records_only_cv_that_is_actually_used(self):
        unused = analyze_code(
            "cv = GroupKFold(5)\n"
            "x = 1\n"
        )

        self.assertEqual(
            unused.cv_funcs,
            [],
        )

        self.assertEqual(
            unused.cv_aware,
            [],
        )

        used = analyze_code(
            "cv = GroupKFold(5)\n"
            "scores = cross_val_score(model, X, y, cv=cv, groups=g)\n"
        )

        self.assertTrue(
            used.cv_aware
        )

        self.assertFalse(
            used.cv_naive
        )

    def test_cv_assignment_history_uses_latest_value(self):
        naive = analyze_code(
            "cv = GroupKFold(5)\n"
            "cv = KFold(5)\n"
            "scores = cross_val_score(model, X, y, cv=cv)\n"
        )

        self.assertTrue(
            naive.cv_naive
        )

        self.assertFalse(
            naive.cv_aware
        )

        aware = analyze_code(
            "cv = KFold(5)\n"
            "cv = GroupKFold(5)\n"
            "scores = cross_val_score(model, X, y, cv=cv, groups=g)\n"
        )

        self.assertTrue(
            aware.cv_aware
        )

        self.assertFalse(
            aware.cv_naive
        )

    def test_dynamic_shuffle_is_unknown_not_assumed_true_or_false(self):
        facts = analyze_code(
            "shuffle_rows = choose_shuffle()\n"
            "X_tr, X_te = train_test_split(X, shuffle=shuffle_rows)\n"
        )

        self.assertIsNone(
            facts.split_shuffled
        )

    def test_scoring_strings_and_scoring_variable_are_resolved(self):
        facts = analyze_code(
            "scoring = {'f1': 'f1_macro', 'ap': 'average_precision'}\n"
            "scores = cross_validate(model, X, y, cv=5, scoring=scoring)\n"
        )

        self.assertIn(
            "f1_score",
            facts.metrics_used,
        )

        self.assertIn(
            "average_precision_score",
            facts.metrics_used,
        )

        self.assertFalse(
            facts.reports_only_accuracy
        )

    def test_search_cv_fit_is_an_evaluation_boundary(self):
        facts = analyze_code(
            "sc = StandardScaler()\n"
            "X = sc.fit_transform(X)\n"
            "search = GridSearchCV(model, {'C': [0.1, 1]}, cv=5)\n"
            "search.fit(X, y)\n"
        )

        self.assertEqual(
            facts.boundary_line,
            4,
        )

        self.assertTrue(
            facts.leaky_fits[0]["before_split"]
        )

        self.assertTrue(
            facts.cv_naive
        )

    def test_group_aware_search_cv_and_scoring_are_detected(self):
        facts = analyze_code(
            "search = GridSearchCV("
            "model, params, cv=GroupKFold(5), scoring='f1')\n"
            "search.fit(X, y, groups=groups)\n"
        )

        self.assertTrue(
            facts.cv_aware
        )

        self.assertFalse(
            facts.cv_naive
        )

        self.assertIn(
            "f1_score",
            facts.metrics_used,
        )

        self.assertFalse(
            facts.reports_only_accuracy
        )

    def test_preprocessing_inside_cross_validation_pipeline_is_not_marked_leaky(
        self,
    ):
        facts = analyze_code(
            "scores = cross_val_score(\n"
            "    make_pipeline(StandardScaler(), LogisticRegression()),\n"
            "    X, y, cv=5, scoring='f1'\n"
            ")\n"
        )

        self.assertEqual(
            facts.leaky_fits,
            [],
        )

        self.assertTrue(
            facts.cv_naive
        )

        self.assertIn(
            "f1_score",
            facts.metrics_used,
        )


if __name__ == "__main__":
    unittest.main()
    