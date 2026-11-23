"""计划—校验—执行—复检—重规划的智能体闭环。"""

import time
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from .cleaning import BankCleaner
from .constants import TARGET
from .data import drop_exact_duplicates
from .plan_schema import find_residual_issues, validate_plan
from .planner import PlannerError, RuleBasedPlanner
from .quality import build_quality_report


class AgentExecutionError(RuntimeError):
    def __init__(
        self,
        message: str,
        attempts: list[dict[str, Any]],
        call_records: list[dict[str, Any]],
    ) -> None:
        super().__init__(message)
        self.attempts = attempts
        self.call_records = call_records


def execute_plan(
    raw: pd.DataFrame,
    plan: dict[str, Any],
    test_size: float,
    random_state: int,
) -> dict[str, Any]:
    """使用确定性工具执行计划，不在执行阶段调用大模型。"""
    should_deduplicate = any(
        action.get("action") == "drop_exact_duplicates"
        for action in plan.get("actions", [])
    )
    if should_deduplicate:
        working, removed_duplicates = drop_exact_duplicates(raw)
    else:
        working, removed_duplicates = raw.copy(), 0

    train_raw, test_raw = train_test_split(
        working,
        test_size=test_size,
        random_state=random_state,
        stratify=working[TARGET],
    )
    cleaner = BankCleaner(plan=plan)
    train_clean = cleaner.fit_transform(train_raw)
    test_clean = cleaner.transform(test_raw)
    cleaned_all = pd.concat([train_clean, test_clean]).sort_index()
    quality_after = build_quality_report(cleaned_all)
    return {
        "train_clean": train_clean,
        "test_clean": test_clean,
        "cleaned_all": cleaned_all,
        "cleaner": cleaner,
        "quality_after": quality_after,
        "deduplicated_rows": len(working),
        "removed_duplicate_rows": removed_duplicates,
    }


def run_cleaning_agent(
    raw: pd.DataFrame,
    planner: Any,
    test_size: float = 0.2,
    random_state: int = 42,
    max_attempts: int = 3,
    fallback_to_rules: bool = False,
) -> dict[str, Any]:
    if max_attempts < 1:
        raise ValueError("max_attempts 必须至少为 1")
    quality_before = build_quality_report(raw)
    attempts: list[dict[str, Any]] = []
    feedback: dict[str, Any] | None = None

    for attempt_number in range(1, max_attempts + 1):
        attempt_log: dict[str, Any] = {"attempt": attempt_number}
        try:
            plan = planner.generate_plan(
                quality_before,
                feedback=feedback,
                attempt=attempt_number,
            )
            attempt_log["plan"] = plan
        except Exception as exc:
            attempt_log.update({"status": "planner_error", "error": str(exc)})
            attempts.append(attempt_log)
            feedback = {"planner_error": str(exc)}
            if isinstance(exc, PlannerError) and exc.transient and attempt_number < max_attempts:
                delay = max(exc.retry_after, 2 ** (attempt_number - 1))
                time.sleep(min(delay, 30))
            continue

        validation_errors = validate_plan(plan, quality_before)
        if validation_errors:
            attempt_log.update(
                {"status": "validation_failed", "validation_errors": validation_errors}
            )
            attempts.append(attempt_log)
            feedback = {
                "previous_plan": plan,
                "validation_errors": validation_errors,
                "instruction": "修正计划并重新输出完整 JSON",
            }
            continue

        try:
            execution = execute_plan(raw, plan, test_size, random_state)
        except (ValueError, RuntimeError) as exc:
            attempt_log.update({"status": "execution_failed", "error": str(exc)})
            attempts.append(attempt_log)
            feedback = {"previous_plan": plan, "execution_error": str(exc)}
            continue

        residual_issues = find_residual_issues(
            quality_before, execution["quality_after"]
        )
        attempt_log["residual_issues"] = residual_issues
        if not residual_issues:
            attempt_log["status"] = "success"
            attempts.append(attempt_log)
            return {
                **execution,
                "quality_before": quality_before,
                "plan": plan,
                "attempts": attempts,
                "call_records": list(getattr(planner, "call_records", [])),
                "fallback_used": False,
                "planner_name": getattr(planner, "name", type(planner).__name__),
            }

        attempt_log["status"] = "residual_issues"
        attempts.append(attempt_log)
        feedback = {
            "previous_plan": plan,
            "residual_issues": residual_issues,
            "instruction": "根据复检结果补充或修正操作，重新输出完整 JSON",
        }

    if fallback_to_rules:
        fallback_planner = RuleBasedPlanner()
        plan = fallback_planner.generate_plan(quality_before, feedback=feedback)
        validation_errors = validate_plan(plan, quality_before)
        fallback_log: dict[str, Any] = {
            "attempt": "rule_fallback",
            "plan": plan,
            "validation_errors": validation_errors,
        }
        if not validation_errors:
            execution = execute_plan(raw, plan, test_size, random_state)
            residual_issues = find_residual_issues(
                quality_before, execution["quality_after"]
            )
            fallback_log["residual_issues"] = residual_issues
            if not residual_issues:
                fallback_log["status"] = "success"
                attempts.append(fallback_log)
                return {
                    **execution,
                    "quality_before": quality_before,
                    "plan": plan,
                    "attempts": attempts,
                    "call_records": list(getattr(planner, "call_records", [])),
                    "fallback_used": True,
                    "planner_name": getattr(planner, "name", type(planner).__name__),
                }
        fallback_log["status"] = "failed"
        attempts.append(fallback_log)

    raise AgentExecutionError(
        "智能体在限定次数内未生成通过复检的计划",
        attempts,
        list(getattr(planner, "call_records", [])),
    )
