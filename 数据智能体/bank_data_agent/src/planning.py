"""根据质量画像生成可审计的规则型清洗计划。"""

from typing import Any

from .constants import CATEGORICAL_COLUMNS, TARGET


def build_cleaning_plan(report: dict[str, Any]) -> dict[str, Any]:
    """生成基线计划；后续可替换为经过 JSON 校验的大模型计划。"""
    actions: list[dict[str, Any]] = []
    fields = report["fields"]

    if report["exact_duplicate_count"]:
        actions.append(
            {
                "action": "drop_exact_duplicates",
                "count": report["exact_duplicate_count"],
                "reason": "完全重复记录会造成训练/测试泄漏",
            }
        )

    format_columns = [
        column
        for column, info in fields.items()
        if info.get("nonstandard_format_count", 0) > 0
    ]
    if format_columns:
        actions.append(
            {
                "action": "parse_numeric_formats",
                "columns": format_columns,
                "rules": ["移除 EUR 前缀", "移除岁后缀", "转换为浮点数"],
            }
        )

    category_columns = [
        column
        for column in CATEGORICAL_COLUMNS
        if fields.get(column, {}).get("whitespace_or_case_issue_count", 0) > 0
    ]
    if category_columns:
        actions.append(
            {
                "action": "standardize_categories",
                "columns": category_columns,
                "rules": ["去除首尾空格", "统一小写", "保留 unknown 类别"],
            }
        )

    missing_columns = [
        column for column, info in fields.items() if info.get("missing_count", 0) > 0
    ]
    if missing_columns:
        actions.append(
            {
                "action": "impute_missing",
                "columns": missing_columns,
                "numeric_strategy": "仅使用训练集的中位数",
                "categorical_strategy": "填为独立的 missing 类别",
            }
        )

    if fields.get("age", {}).get("business_rule_violation_count", 0) > 0:
        actions.append(
            {
                "action": "invalidate_business_rule_violations",
                "column": "age",
                "valid_range": [18, 100],
                "then": "转为缺失并添加 age_invalid 指示列",
            }
        )

    if report["constant_columns"]:
        actions.append(
            {
                "action": "drop_constant_columns",
                "columns": report["constant_columns"],
                "reason": "对当前训练数据没有区分能力",
            }
        )

    return {
        "plan_version": "rule-based-baseline-v1",
        "target": TARGET,
        "label_policy": "不修改标签",
        "outlier_policy": "负余额保留；其他极端值仅标记，不自动删除",
        "actions": actions,
    }
