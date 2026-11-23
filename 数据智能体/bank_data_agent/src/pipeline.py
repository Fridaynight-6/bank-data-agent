"""端到端基线：质量画像、去重、分层切分、清洗、训练与评估。"""

from datetime import datetime
from hashlib import sha256
from pathlib import Path
import json
import platform
from typing import Any

import pandas as pd
import sklearn
from joblib import dump
from sklearn.model_selection import train_test_split

from .cleaning import BankCleaner
from .constants import TARGET
from .data import drop_exact_duplicates, load_bank_data
from .modeling import build_logistic_baseline, evaluate_classifier
from .planning import build_cleaning_plan
from .quality import build_quality_report


def save_json(data: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_baseline(
    input_path: str | Path,
    output_dir: str | Path,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, Any]:
    started_at = datetime.now().astimezone()
    input_path = Path(input_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stages: list[dict[str, str]] = []

    raw = load_bank_data(input_path)
    quality_before = build_quality_report(raw)
    save_json(quality_before, output_dir / "quality_report_before.json")
    stages.append({"stage": "quality_profile", "status": "success"})

    plan = build_cleaning_plan(quality_before)
    save_json(plan, output_dir / "cleaning_plan.json")
    stages.append({"stage": "cleaning_plan", "status": "success"})

    deduplicated, removed_duplicates = drop_exact_duplicates(raw)
    train_raw, test_raw = train_test_split(
        deduplicated,
        test_size=test_size,
        random_state=random_state,
        stratify=deduplicated[TARGET],
    )
    cleaner = BankCleaner()
    train_clean = cleaner.fit_transform(train_raw)
    test_clean = cleaner.transform(test_raw)
    cleaned_all = pd.concat([train_clean, test_clean]).sort_index()
    quality_after = build_quality_report(cleaned_all)

    cleaned_all.to_csv(output_dir / "cleaned_data.csv", encoding="utf-8-sig", index=True)
    train_clean.to_csv(output_dir / "cleaned_train.csv", encoding="utf-8-sig", index=True)
    test_clean.to_csv(output_dir / "cleaned_test.csv", encoding="utf-8-sig", index=True)
    save_json(quality_after, output_dir / "quality_report_after.json")
    save_json(cleaner.metadata(), output_dir / "cleaner_metadata.json")
    stages.append({"stage": "cleaning", "status": "success"})

    model = build_logistic_baseline(
        cleaner.numeric_features_,
        cleaner.categorical_features_,
        random_state=random_state,
    )
    model.fit(train_clean.drop(columns=[TARGET]), train_clean[TARGET])
    metrics, predictions = evaluate_classifier(model, test_clean)
    metrics.update(
        {
            "raw_rows": int(len(raw)),
            "deduplicated_rows": int(len(deduplicated)),
            "removed_duplicate_rows": removed_duplicates,
            "train_rows": int(len(train_clean)),
            "test_size": test_size,
            "random_state": random_state,
            "dropped_constant_columns": cleaner.constant_columns_,
        }
    )
    save_json(metrics, output_dir / "metrics.json")
    predictions.to_csv(output_dir / "test_predictions.csv", encoding="utf-8-sig", index=True)
    dump(
        {"cleaner": cleaner, "model": model},
        output_dir / "baseline_artifacts.joblib",
    )
    stages.append({"stage": "train_and_evaluate", "status": "success"})

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
            "class_weight": "balanced",
            "threshold": 0.5,
        },
        "environment": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "llm_calls": 0,
        "stages": stages,
    }
    save_json(run_log, output_dir / "run_log.json")
    return metrics
