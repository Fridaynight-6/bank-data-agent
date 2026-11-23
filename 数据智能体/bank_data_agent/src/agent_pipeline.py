"""智能体模式的结果、模型与审计日志导出。"""

from datetime import datetime
from pathlib import Path
import platform
from typing import Any

import pandas as pd
import sklearn
from joblib import dump

from .agent import AgentExecutionError, run_cleaning_agent
from .constants import TARGET
from .data import load_bank_data
from .modeling import build_logistic_baseline, evaluate_classifier
from .pipeline import file_sha256, save_json


def run_agent_pipeline(
    input_path: str | Path,
    output_dir: str | Path,
    planner: Any,
    test_size: float = 0.2,
    random_state: int = 42,
    max_attempts: int = 3,
    fallback_to_rules: bool = True,
) -> dict[str, Any]:
    started_at = datetime.now().astimezone()
    input_path = Path(input_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = load_bank_data(input_path)

    try:
        result = run_cleaning_agent(
            raw,
            planner=planner,
            test_size=test_size,
            random_state=random_state,
            max_attempts=max_attempts,
            fallback_to_rules=fallback_to_rules,
        )
    except AgentExecutionError as exc:
        save_json({"attempts": exc.attempts}, output_dir / "agent_attempts.json")
        save_json({"calls": exc.call_records}, output_dir / "llm_call_log.json")
        raise

    save_json(result["quality_before"], output_dir / "quality_report_before.json")
    save_json(result["plan"], output_dir / "cleaning_plan.json")
    save_json(result["quality_after"], output_dir / "quality_report_after.json")
    save_json(result["cleaner"].metadata(), output_dir / "cleaner_metadata.json")
    save_json({"attempts": result["attempts"]}, output_dir / "agent_attempts.json")
    save_json({"calls": result["call_records"]}, output_dir / "llm_call_log.json")

    train_clean = result["train_clean"]
    test_clean = result["test_clean"]
    result["cleaned_all"].to_csv(
        output_dir / "cleaned_data.csv", encoding="utf-8-sig", index=True
    )
    train_clean.to_csv(output_dir / "cleaned_train.csv", encoding="utf-8-sig", index=True)
    test_clean.to_csv(output_dir / "cleaned_test.csv", encoding="utf-8-sig", index=True)

    model = build_logistic_baseline(
        result["cleaner"].numeric_features_,
        result["cleaner"].categorical_features_,
        random_state=random_state,
    )
    model.fit(train_clean.drop(columns=[TARGET]), train_clean[TARGET])
    metrics, predictions = evaluate_classifier(model, test_clean)
    metrics.update(
        {
            "raw_rows": int(len(raw)),
            "deduplicated_rows": int(result["deduplicated_rows"]),
            "removed_duplicate_rows": int(result["removed_duplicate_rows"]),
            "train_rows": int(len(train_clean)),
            "test_size": test_size,
            "random_state": random_state,
            "dropped_constant_columns": result["cleaner"].constant_columns_,
            "planner": result["planner_name"],
            "agent_attempts": len(result["attempts"]),
            "fallback_used": result["fallback_used"],
        }
    )
    save_json(metrics, output_dir / "metrics.json")
    predictions.to_csv(output_dir / "test_predictions.csv", encoding="utf-8-sig", index=True)
    dump(
        {"cleaner": result["cleaner"], "model": model, "plan": result["plan"]},
        output_dir / "agent_artifacts.joblib",
    )

    costs = [
        call["estimated_cost_usd"]
        for call in result["call_records"]
        if call.get("estimated_cost_usd") is not None
    ]
    finished_at = datetime.now().astimezone()
    run_log = {
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "duration_seconds": (finished_at - started_at).total_seconds(),
        "input_path": str(input_path),
        "input_sha256": file_sha256(input_path),
        "parameters": {
            "test_size": test_size,
            "random_state": random_state,
            "max_attempts": max_attempts,
            "fallback_to_rules": fallback_to_rules,
        },
        "environment": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "planner": result["planner_name"],
        "llm_calls": len(result["call_records"]),
        "estimated_cost_usd": sum(costs) if costs else None,
        "cost_note": "None 表示未配置价格或 API 未返回 token 用量",
        "fallback_used": result["fallback_used"],
        "status": "success",
    }
    save_json(run_log, output_dir / "run_log.json")
    return metrics
