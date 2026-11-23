"""清洗计划的结构约束、业务安全约束与质量复检规则。"""

from typing import Any

from .constants import CATEGORICAL_COLUMNS, EXPECTED_COLUMNS, NUMERIC_COLUMNS, TARGET

ALLOWED_ACTIONS = {
    "drop_exact_duplicates",
    "parse_numeric_formats",
    "standardize_categories",
    "impute_missing",
    "invalidate_business_rule_violations",
    "drop_constant_columns",
}

CLEANING_PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["plan_version", "target", "label_policy", "outlier_policy", "actions"],
    "properties": {
        "plan_version": {"type": "string"},
        "target": {"const": TARGET},
        "label_policy": {"const": "不修改标签"},
        "outlier_policy": {"type": "string"},
        "actions": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["action"],
                "properties": {
                    "action": {"type": "string", "enum": sorted(ALLOWED_ACTIONS)},
                    "columns": {"type": "array", "items": {"type": "string"}},
                    "column": {"type": "string"},
                    "valid_range": {
                        "type": "array",
                        "prefixItems": [{"type": "number"}, {"type": "number"}],
                        "minItems": 2,
                        "maxItems": 2,
                    },
                },
            },
        },
    },
}


def _validate_columns(
    columns: Any,
    allowed: set[str],
    action_name: str,
    errors: list[str],
) -> None:
    if not isinstance(columns, list) or not all(isinstance(item, str) for item in columns):
        errors.append(f"{action_name}.columns 必须是字符串数组")
        return
    invalid = sorted(set(columns) - allowed)
    if invalid:
        errors.append(f"{action_name} 包含不允许的字段：{invalid}")


def validate_plan(plan: Any, report: dict[str, Any]) -> list[str]:
    """返回全部错误；空列表表示计划可进入白名单执行阶段。"""
    errors: list[str] = []
    if not isinstance(plan, dict):
        return ["计划顶层必须是 JSON 对象"]

    required = {"plan_version", "target", "label_policy", "outlier_policy", "actions"}
    missing = sorted(required - set(plan))
    extra = sorted(set(plan) - required)
    if missing:
        errors.append(f"缺少顶层字段：{missing}")
    if extra:
        errors.append(f"包含未知顶层字段：{extra}")
    if not isinstance(plan.get("plan_version"), str):
        errors.append("plan_version 必须是字符串")
    if plan.get("target") != TARGET:
        errors.append(f"target 必须固定为 {TARGET}")
    if plan.get("label_policy") != "不修改标签":
        errors.append("label_policy 必须明确为“不修改标签”")
    if not isinstance(plan.get("outlier_policy"), str):
        errors.append("outlier_policy 必须是字符串")

    actions = plan.get("actions")
    if not isinstance(actions, list):
        errors.append("actions 必须是数组")
        return errors

    feature_columns = set(EXPECTED_COLUMNS) - {TARGET}
    action_names: set[str] = set()
    for index, action in enumerate(actions):
        prefix = f"actions[{index}]"
        if not isinstance(action, dict):
            errors.append(f"{prefix} 必须是对象")
            continue
        name = action.get("action")
        if name not in ALLOWED_ACTIONS:
            errors.append(f"{prefix} 含不支持的操作：{name!r}")
            continue
        action_names.add(name)

        if name == "parse_numeric_formats":
            _validate_columns(action.get("columns"), set(NUMERIC_COLUMNS), prefix, errors)
        elif name == "standardize_categories":
            _validate_columns(action.get("columns"), set(CATEGORICAL_COLUMNS), prefix, errors)
        elif name == "impute_missing":
            _validate_columns(action.get("columns"), feature_columns, prefix, errors)
        elif name == "drop_constant_columns":
            _validate_columns(
                action.get("columns"),
                set(report.get("constant_columns", [])),
                prefix,
                errors,
            )
        elif name == "invalidate_business_rule_violations":
            if action.get("column") != "age":
                errors.append(f"{prefix}.column 当前只允许 age")
            if action.get("valid_range") != [18, 100]:
                errors.append(f"{prefix}.valid_range 必须固定为 [18, 100]")

    if report.get("exact_duplicate_count", 0) > 0 and "drop_exact_duplicates" not in action_names:
        errors.append("原始数据存在完全重复记录，计划必须包含 drop_exact_duplicates")
    return errors


def find_residual_issues(
    before: dict[str, Any], after: dict[str, Any]
) -> list[dict[str, Any]]:
    """只要求修复安全性与规则覆盖问题，不把合法极值误判为失败。"""
    issues: list[dict[str, Any]] = []
    if after.get("exact_duplicate_count", 0):
        issues.append({"issue": "exact_duplicates", "count": after["exact_duplicate_count"]})

    checks = [
        ("missing_count", "missing_values"),
        ("nonstandard_format_count", "nonstandard_numeric_formats"),
        ("whitespace_or_case_issue_count", "category_format_issues"),
        ("business_rule_violation_count", "business_rule_violations"),
    ]
    for key, issue_name in checks:
        columns = [
            column
            for column, info in after.get("fields", {}).items()
            if info.get(key, 0) > 0
        ]
        if columns:
            issues.append({"issue": issue_name, "columns": columns})

    remaining_constants = [
        column
        for column in before.get("constant_columns", [])
        if column in after.get("constant_columns", [])
    ]
    if remaining_constants:
        issues.append({"issue": "constant_columns", "columns": remaining_constants})
    return issues
