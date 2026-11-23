"""数据读取、缺失识别与可恢复格式解析。"""

from pathlib import Path

import numpy as np
import pandas as pd

from .constants import EXPECTED_COLUMNS, MISSING_TOKENS, NEGATIVE_LABEL, POSITIVE_LABEL, TARGET


def load_bank_data(path: str | Path) -> pd.DataFrame:
    """读取原始分号分隔数据，保留字符串以便检测格式问题。"""
    path = Path(path)
    df = pd.read_csv(path, sep=";", dtype=str, encoding="utf-8-sig")
    if list(df.columns) != EXPECTED_COLUMNS:
        raise ValueError(f"字段不符合银行数据集约定：{list(df.columns)}")
    if df.empty:
        raise ValueError("输入数据为空")

    labels = df[TARGET].fillna("").str.strip().str.lower()
    invalid_labels = sorted(set(labels) - {NEGATIVE_LABEL, POSITIVE_LABEL})
    if invalid_labels:
        raise ValueError(f"标签包含非法取值：{invalid_labels}")

    df = df.copy()
    df.index.name = "source_row"
    return df


def missing_mask(series: pd.Series) -> pd.Series:
    """识别空白和常见缺失标记，但保留 unknown 业务类别。"""
    stripped = series.fillna("").astype(str).str.strip().str.lower()
    return series.isna() | stripped.isin(MISSING_TOKENS)


def normalize_missing(series: pd.Series) -> pd.Series:
    result = series.copy()
    result.loc[missing_mask(result)] = np.nan
    return result


def parse_numeric(series: pd.Series) -> pd.Series:
    """解析普通数值、年龄“岁”后缀和余额 EUR 前缀。"""
    cleaned = normalize_missing(series).astype("string").str.strip()
    cleaned = cleaned.str.replace(r"(?i)^EUR\s*", "", regex=True)
    cleaned = cleaned.str.replace(r"\s*岁$", "", regex=True)
    return pd.to_numeric(cleaned, errors="coerce").astype(float)


def has_nonstandard_numeric_format(series: pd.Series) -> pd.Series:
    """标记可通过单位剥离恢复的数值格式。"""
    normalized = normalize_missing(series).astype("string").str.strip()
    standard = normalized.str.fullmatch(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)")
    return normalized.notna() & ~standard.fillna(False)


def drop_exact_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """删除完全重复记录，保留第一次出现的记录。"""
    keep_mask = ~df.duplicated(keep="first")
    return df.loc[keep_mask].copy(), int((~keep_mask).sum())
