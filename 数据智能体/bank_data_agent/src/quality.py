"""可序列化的数据质量画像。"""

from typing import Any

import pandas as pd

from .constants import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS, TARGET
from .data import has_nonstandard_numeric_format, missing_mask, parse_numeric


def _number(value: Any) -> float | int | None:
    if pd.isna(value):
        return None
    if isinstance(value, int):
        return value
    return float(value)


def build_quality_report(df: pd.DataFrame) -> dict[str, Any]:
    """检测缺失、重复、格式、异常值、恒定字段和标签分布。"""
    fields: dict[str, Any] = {}
    for column in df.columns:
        if column == TARGET:
            continue
        series = df[column]
        missing = missing_mask(series)
        info: dict[str, Any] = {
            "missing_count": int(missing.sum()),
            "missing_rate": round(float(missing.mean()), 6),
            "distinct_count": int(series.nunique(dropna=True)),
        }

        if column in NUMERIC_COLUMNS:
            numeric = parse_numeric(series)
            nonstandard = has_nonstandard_numeric_format(series)
            unparseable = missing | (series.notna() & numeric.isna())
            info.update(
                {
                    "nonstandard_format_count": int(nonstandard.sum()),
                    "unparseable_or_missing_count": int(unparseable.sum()),
                    "min": _number(numeric.min()),
                    "median": _number(numeric.median()),
                    "max": _number(numeric.max()),
                }
            )
            valid = numeric.dropna()
            if not valid.empty:
                q1, q3 = valid.quantile([0.25, 0.75])
                iqr = q3 - q1
                lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                info["iqr_outlier_count"] = int(((numeric < lower) | (numeric > upper)).sum())
                info["iqr_lower_bound"] = _number(lower)
                info["iqr_upper_bound"] = _number(upper)
            if column == "age":
                info["business_rule_violation_count"] = int(
                    ((numeric < 18) | (numeric > 100)).sum()
                )
            if column == "balance":
                info["negative_count"] = int((numeric < 0).sum())
        elif column in CATEGORICAL_COLUMNS:
            text = series.fillna("").astype(str)
            whitespace = text.ne(text.str.strip()) & series.notna()
            case = text.ne(text.str.lower()) & series.notna()
            info["whitespace_or_case_issue_count"] = int((whitespace | case).sum())
            info["unknown_category_count"] = int(
                text.str.strip().str.lower().eq("unknown").sum()
            )
            info["top_values"] = {
                str(key): int(value)
                for key, value in series.fillna("<NA>").value_counts().head(10).items()
            }
        fields[column] = info

    labels = df[TARGET].value_counts()
    report = {
        "row_count": int(len(df)),
        "column_count": int(df.shape[1]),
        "exact_duplicate_count": int(df.duplicated().sum()),
        "constant_columns": [
            column
            for column in df.columns
            if column != TARGET and df[column].nunique(dropna=False) <= 1
        ],
        "label_distribution": {
            str(label): {
                "count": int(count),
                "rate": round(float(count / len(df)), 6),
            }
            for label, count in labels.items()
        },
        "fields": fields,
    }
    return report
