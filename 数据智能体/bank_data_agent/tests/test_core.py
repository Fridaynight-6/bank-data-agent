import unittest

import pandas as pd

from src.cleaning import BankCleaner
from src.constants import EXPECTED_COLUMNS
from src.data import drop_exact_duplicates, missing_mask, parse_numeric
from src.quality import build_quality_report


def make_row(**overrides):
    row = {
        "age": "35",
        "job": "management",
        "marital": "married",
        "education": "tertiary",
        "default": "no",
        "balance": "100",
        "housing": "yes",
        "loan": "no",
        "contact": "cellular",
        "day": "5",
        "month": "may",
        "duration": "100",
        "campaign": "1",
        "pdays": "-1",
        "previous": "0",
        "poutcome": "unknown",
        "y": "no",
    }
    row.update(overrides)
    return row


def make_frame(rows):
    return pd.DataFrame(rows, columns=EXPECTED_COLUMNS)


class DataParsingTests(unittest.TestCase):
    def test_numeric_formats_are_recovered(self):
        parsed = parse_numeric(pd.Series(["43岁", "EUR -583", "12", "?", None]))
        self.assertEqual(parsed.iloc[0], 43)
        self.assertEqual(parsed.iloc[1], -583)
        self.assertEqual(parsed.iloc[2], 12)
        self.assertTrue(pd.isna(parsed.iloc[3]))
        self.assertTrue(pd.isna(parsed.iloc[4]))

    def test_unknown_is_not_treated_as_missing(self):
        mask = missing_mask(pd.Series(["unknown", "?", None, " "]))
        self.assertEqual(mask.tolist(), [False, True, True, True])

    def test_exact_duplicates_are_removed(self):
        df = make_frame([make_row(), make_row(), make_row(age="40")])
        result, removed = drop_exact_duplicates(df)
        self.assertEqual(removed, 1)
        self.assertEqual(len(result), 2)


class CleaningTests(unittest.TestCase):
    def setUp(self):
        self.train = make_frame(
            [
                make_row(age="20", job="unknown", housing="yes"),
                make_row(age="40", job="admin.", housing="no", balance="200", y="yes"),
            ]
        )
        self.cleaner = BankCleaner().fit(self.train)

    def test_invalid_age_uses_train_median(self):
        test = make_frame([make_row(age="202", job="unknown", housing="YES")])
        cleaned = self.cleaner.transform(test)
        self.assertEqual(cleaned.loc[0, "age"], 30.0)
        self.assertEqual(cleaned.loc[0, "age_invalid"], 1)
        self.assertEqual(cleaned.loc[0, "housing"], "yes")

    def test_unknown_is_preserved_and_currency_is_parsed(self):
        test = make_frame([make_row(job=" unknown ", balance="EUR -583")])
        cleaned = self.cleaner.transform(test)
        self.assertEqual(cleaned.loc[0, "job"], "unknown")
        self.assertEqual(cleaned.loc[0, "balance"], -583.0)

    def test_label_is_not_modified(self):
        test = make_frame([make_row(y="yes")])
        cleaned = self.cleaner.transform(test)
        self.assertEqual(cleaned.loc[0, "y"], "yes")


class QualityReportTests(unittest.TestCase):
    def test_duplicate_constant_and_business_violation_are_reported(self):
        df = make_frame([make_row(age="202"), make_row(age="202")])
        report = build_quality_report(df)
        self.assertEqual(report["exact_duplicate_count"], 1)
        self.assertIn("previous", report["constant_columns"])
        self.assertEqual(report["fields"]["age"]["business_rule_violation_count"], 2)


if __name__ == "__main__":
    unittest.main()
