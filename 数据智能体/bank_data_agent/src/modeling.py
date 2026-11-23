"""分类基线与类别不平衡评估。"""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .constants import NEGATIVE_LABEL, POSITIVE_LABEL, TARGET


def build_logistic_baseline(
    numeric_features: list[str],
    categorical_features: list[str],
    random_state: int = 42,
) -> Pipeline:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )
    classifier = LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        random_state=random_state,
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])


def evaluate_classifier(
    model: Pipeline,
    test_df: pd.DataFrame,
    threshold: float = 0.5,
) -> tuple[dict[str, Any], pd.DataFrame]:
    labels = test_df[TARGET].map({NEGATIVE_LABEL: 0, POSITIVE_LABEL: 1})
    if labels.isna().any():
        raise ValueError("测试集包含无法识别的标签")
    features = test_df.drop(columns=[TARGET])
    probabilities = model.predict_proba(features)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()

    metrics = {
        "model": "LogisticRegression(class_weight=balanced)",
        "threshold": threshold,
        "positive_label": POSITIVE_LABEL,
        "test_rows": int(len(test_df)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "pr_auc": float(average_precision_score(labels, probabilities)),
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
    }
    prediction_output = pd.DataFrame(
        {
            "actual": test_df[TARGET].to_numpy(),
            "predicted": np.where(predictions == 1, POSITIVE_LABEL, NEGATIVE_LABEL),
            "positive_probability": probabilities,
        },
        index=test_df.index,
    )
    return metrics, prediction_output
