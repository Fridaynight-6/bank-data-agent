import copy
import unittest

import pandas as pd

from src.agent import run_cleaning_agent
from src.constants import EXPECTED_COLUMNS
from src.plan_schema import validate_plan
from src.planner import PlannerError, extract_json_object
from src.planning import build_cleaning_plan
from src.quality import build_quality_report


def make_row(index=0, **overrides):
    row = {
        "age": str(25 + index),
        "job": "management" if index % 2 == 0 else "admin.",
        "marital": "married" if index % 2 == 0 else "single",
        "education": "tertiary" if index % 2 == 0 else "secondary",
        "default": "no",
        "balance": str(100 + index),
        "housing": "yes" if index % 2 == 0 else "no",
        "loan": "no",
        "contact": "cellular" if index % 2 == 0 else "unknown",
        "day": str((index % 28) + 1),
        "month": "may" if index % 2 == 0 else "jun",
        "duration": str(100 + index),
        "campaign": str((index % 3) + 1),
        "pdays": "-1",
        "previous": "0",
        "poutcome": "unknown",
        "y": "yes" if index % 5 == 0 else "no",
    }
    row.update(overrides)
    return row


def make_dirty_frame():
    rows = [make_row(index) for index in range(12)]
    rows[0]["job"] = " management"
    rows[0]["housing"] = "YES"
    rows[1]["balance"] = "EUR 101"
    rows[2]["age"] = "202"
    rows[3]["age"] = "?"
    rows.append(copy.deepcopy(rows[4]))
    return pd.DataFrame(rows, columns=EXPECTED_COLUMNS)


class InvalidThenValidPlanner:
    name = "test-invalid-then-valid"

    def __init__(self):
        self.calls = 0
        self.call_records = []

    def generate_plan(self, report, feedback=None, attempt=1):
        self.calls += 1
        return {} if self.calls == 1 else build_cleaning_plan(report)


class ResidualThenValidPlanner:
    name = "test-residual-then-valid"

    def __init__(self):
        self.calls = 0
        self.call_records = []

    def generate_plan(self, report, feedback=None, attempt=1):
        self.calls += 1
        plan = build_cleaning_plan(report)
        if self.calls == 1:
            plan["actions"] = [
                action
                for action in plan["actions"]
                if action["action"] != "standardize_categories"
            ]
        return plan


class AlwaysFailingPlanner:
    name = "test-always-failing"

    def __init__(self):
        self.call_records = []

    def generate_plan(self, report, feedback=None, attempt=1):
        raise PlannerError("模拟规划失败")


class PlanValidationTests(unittest.TestCase):
    def test_label_change_and_nonconstant_drop_are_rejected(self):
        raw = make_dirty_frame()
        report = build_quality_report(raw)
        plan = build_cleaning_plan(report)
        plan["label_policy"] = "允许修改"
        plan["actions"].append({"action": "drop_constant_columns", "columns": ["age"]})
        errors = validate_plan(plan, report)
        self.assertTrue(any("不修改标签" in error for error in errors))
        self.assertTrue(any("不允许" in error and "age" in error for error in errors))

    def test_json_code_block_can_be_parsed(self):
        plan = extract_json_object('```json\n{"plan_version": "test"}\n```')
        self.assertEqual(plan, {"plan_version": "test"})


class AgentLoopTests(unittest.TestCase):
    def test_invalid_plan_is_sent_back_for_replanning(self):
        planner = InvalidThenValidPlanner()
        result = run_cleaning_agent(
            make_dirty_frame(), planner, test_size=0.25, max_attempts=2
        )
        self.assertEqual(planner.calls, 2)
        self.assertEqual(result["attempts"][0]["status"], "validation_failed")
        self.assertEqual(result["attempts"][1]["status"], "success")

    def test_residual_quality_issue_triggers_replanning(self):
        planner = ResidualThenValidPlanner()
        result = run_cleaning_agent(
            make_dirty_frame(), planner, test_size=0.25, max_attempts=2
        )
        self.assertEqual(planner.calls, 2)
        self.assertEqual(result["attempts"][0]["status"], "residual_issues")
        self.assertEqual(result["attempts"][1]["status"], "success")

    def test_rule_fallback_is_explicitly_recorded(self):
        result = run_cleaning_agent(
            make_dirty_frame(),
            AlwaysFailingPlanner(),
            test_size=0.25,
            max_attempts=1,
            fallback_to_rules=True,
        )
        self.assertTrue(result["fallback_used"])
        self.assertEqual(result["attempts"][-1]["attempt"], "rule_fallback")
        self.assertEqual(result["attempts"][-1]["status"], "success")


if __name__ == "__main__":
    unittest.main()
