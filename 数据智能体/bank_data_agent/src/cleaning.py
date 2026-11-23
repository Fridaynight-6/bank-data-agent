"""训练集内拟合、由已校验清洗计划驱动的银行数据清洗器。"""

from typing import Any

import numpy as np
import pandas as pd

from .constants import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS, TARGET
from .data import missing_mask, normalize_missing, parse_numeric


class BankCleaner:
    """只执行白名单计划；统计量仅从训练集学习，不修改标签或删除行。"""

    def __init__(self, plan: dict[str, Any] | None = None) -> None:
        self.plan = plan
        actions = [] if plan is None else plan.get("actions", [])
        self.actions_by_name_ = {
            action["action"]: action for action in actions if "action" in action
        }
        all_columns = set(NUMERIC_COLUMNS + CATEGORICAL_COLUMNS)
        if plan is None:
            self.standardize_columns_ = set(CATEGORICAL_COLUMNS)
            self.impute_columns_ = all_columns
            self.requested_constant_columns_: set[str] | None = None
            self.age_valid_range_: tuple[int, int] | None = (18, 100)
        else:
            standardize = self.actions_by_name_.get("standardize_categories", {})
            impute = self.actions_by_name_.get("impute_missing", {})
            drop_constant = self.actions_by_name_.get("drop_constant_columns", {})
            age_rule = self.actions_by_name_.get("invalidate_business_rule_violations")
            self.standardize_columns_ = set(standardize.get("columns", []))
            self.impute_columns_ = set(impute.get("columns", []))
            self.requested_constant_columns_ = set(drop_constant.get("columns", []))
            self.age_valid_range_ = (
                tuple(age_rule["valid_range"]) if age_rule is not None else None
            )

        self.numeric_medians_: dict[str, float] = {}
        self.constant_columns_: list[str] = []
        self.numeric_features_: list[str] = []
        self.categorical_features_: list[str] = []
        self.is_fitted_ = False

    def _standardize(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()

        for column in CATEGORICAL_COLUMNS:
            normalized = normalize_missing(result[column]).astype("string")
            if column in self.standardize_columns_:
                normalized = normalized.str.strip().str.lower()
            result[column] = normalized

        # 数值解析是模型输入的安全前置步骤，不允许大模型关闭。
        for column in NUMERIC_COLUMNS:
            result[column] = parse_numeric(result[column])

        original_age_missing = missing_mask(df["age"])
        age_invalid = ~original_age_missing & result["age"].isna()
        if self.age_valid_range_ is not None:
            lower, upper = self.age_valid_range_
            age_invalid |= result["age"].lt(lower) | result["age"].gt(upper)
        result["age_missing"] = original_age_missing.astype(int)
        result["age_invalid"] = age_invalid.astype(int)
        result.loc[age_invalid, "age"] = np.nan
        return result

    def fit(self, train_df: pd.DataFrame) -> "BankCleaner":
        if TARGET not in train_df:
            raise ValueError(f"训练数据缺少标签列 {TARGET}")
        prepared = self._standardize(train_df)
        self.numeric_medians_ = {}
        for column in NUMERIC_COLUMNS:
            median = prepared[column].median()
            if pd.isna(median):
                raise ValueError(f"训练字段 {column} 没有可用于填充的有效数值")
            self.numeric_medians_[column] = float(median)

        detected_constants = {
            column
            for column in NUMERIC_COLUMNS + CATEGORICAL_COLUMNS
            if prepared[column].nunique(dropna=False) <= 1
        }
        if self.requested_constant_columns_ is None:
            self.constant_columns_ = sorted(detected_constants)
        else:
            # 即使计划已通过原始报告校验，也再次用训练集证据约束删除范围。
            self.constant_columns_ = sorted(
                detected_constants & self.requested_constant_columns_
            )
        self.numeric_features_ = [
            column for column in NUMERIC_COLUMNS if column not in self.constant_columns_
        ]
        if "age" in self.numeric_features_:
            self.numeric_features_ += ["age_missing", "age_invalid"]
        self.categorical_features_ = [
            column
            for column in CATEGORICAL_COLUMNS
            if column not in self.constant_columns_
        ]
        self.is_fitted_ = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted_:
            raise RuntimeError("BankCleaner 必须先在训练集上 fit")
        result = self._standardize(df)
        numeric_to_fill = set(NUMERIC_COLUMNS) & self.impute_columns_
        if self.age_valid_range_ is not None:
            numeric_to_fill.add("age")
        for column in numeric_to_fill:
            result[column] = result[column].fillna(self.numeric_medians_[column]).astype(float)
        for column in set(CATEGORICAL_COLUMNS) & self.impute_columns_:
            result[column] = result[column].fillna("missing").astype(str)

        return result.drop(columns=self.constant_columns_)

    def fit_transform(self, train_df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(train_df).transform(train_df)

    def metadata(self) -> dict[str, Any]:
        return {
            "train_numeric_medians": self.numeric_medians_,
            "dropped_constant_columns": self.constant_columns_,
            "numeric_features": self.numeric_features_,
            "categorical_features": self.categorical_features_,
            "standardized_columns": sorted(self.standardize_columns_),
            "imputed_columns": sorted(self.impute_columns_),
            "age_valid_range": self.age_valid_range_,
        }
