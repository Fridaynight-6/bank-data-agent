"""命令行入口：python run_pipeline.py --input <csv> --output-dir outputs"""

import argparse
import json
from pathlib import Path

from src.pipeline import run_baseline


def main() -> None:
    parser = argparse.ArgumentParser(description="银行数据质量处理与分类基线")
    parser.add_argument("--input", default="银行数据集（已添加问题）.csv", help="原始 CSV 路径")
    parser.add_argument("--output-dir", default="outputs", help="结果目录")
    parser.add_argument("--test-size", type=float, default=0.2, help="测试集比例")
    parser.add_argument("--random-state", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    if not 0 < args.test_size < 1:
        parser.error("--test-size 必须在 0 和 1 之间")
    metrics = run_baseline(
        input_path=Path(args.input),
        output_dir=Path(args.output_dir),
        test_size=args.test_size,
        random_state=args.random_state,
    )
    summary = {
        "raw_rows": metrics["raw_rows"],
        "cleaned_rows": metrics["deduplicated_rows"],
        "train_rows": metrics["train_rows"],
        "test_rows": metrics["test_rows"],
        "pr_auc": round(metrics["pr_auc"], 4),
        "roc_auc": round(metrics["roc_auc"], 4),
        "precision": round(metrics["precision"], 4),
        "recall": round(metrics["recall"], 4),
        "f1": round(metrics["f1"], 4),
        "confusion_matrix": metrics["confusion_matrix"],
        "output_dir": str(Path(args.output_dir).resolve()),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
