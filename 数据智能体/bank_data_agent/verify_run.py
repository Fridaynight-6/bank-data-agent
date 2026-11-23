"""核验基线或智能体输出的一致性、清洗约束与模型可复现性。"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import load

from src.constants import TARGET
from src.data import drop_exact_duplicates, load_bank_data
from src.modeling import evaluate_classifier


def main() -> None:
    parser = argparse.ArgumentParser(description="核验一次基线或智能体运行")
    parser.add_argument("--input", default="银行数据集（已添加问题）.csv")
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)

    raw = load_bank_data(args.input)
    deduplicated, _ = drop_exact_duplicates(raw)
    cleaned = pd.read_csv(output_dir / "cleaned_data.csv", index_col="source_row")
    test = pd.read_csv(output_dir / "cleaned_test.csv", index_col="source_row")
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))

    assert len(cleaned) == len(deduplicated)
    assert cleaned.index.equals(deduplicated.index)
    pd.testing.assert_series_equal(
        cleaned[TARGET].sort_index(),
        deduplicated[TARGET].sort_index(),
        check_dtype=False,
        check_names=False,
    )
    assert cleaned.duplicated().sum() == 0
    assert cleaned.isna().sum().sum() == 0
    assert cleaned["age"].between(18, 100).all()
    assert set(cleaned["housing"]) <= {"yes", "no"}
    assert cleaned["job"].equals(cleaned["job"].str.strip())
    assert not {"pdays", "previous", "poutcome"}.intersection(cleaned.columns)

    agent_artifacts = output_dir / "agent_artifacts.joblib"
    artifacts_path = (
        agent_artifacts if agent_artifacts.exists() else output_dir / "baseline_artifacts.joblib"
    )
    artifacts = load(artifacts_path)
    recomputed, predictions = evaluate_classifier(artifacts["model"], test)
    saved_predictions = pd.read_csv(output_dir / "test_predictions.csv", index_col="source_row")
    assert np.allclose(
        predictions["positive_probability"],
        saved_predictions["positive_probability"],
    )
    for name in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]:
        assert np.isclose(metrics[name], recomputed[name]), name

    if agent_artifacts.exists():
        attempts = json.loads((output_dir / "agent_attempts.json").read_text(encoding="utf-8"))
        run_log = json.loads((output_dir / "run_log.json").read_text(encoding="utf-8"))
        assert attempts["attempts"][-1]["status"] == "success"
        assert run_log["status"] == "success"
        assert (output_dir / "llm_call_log.json").exists()

    print(
        json.dumps(
            {
                "verification": "passed",
                "output_dir": str(output_dir.resolve()),
                "rows": len(cleaned),
                "remaining_missing_cells": int(cleaned.isna().sum().sum()),
                "remaining_exact_duplicates": int(cleaned.duplicated().sum()),
                "labels_unchanged": True,
                "saved_model_metrics_reproduced": True,
                "agent_audit_log_checked": agent_artifacts.exists(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
