"""智能体入口：规则模式可直接运行，LLM 模式使用环境变量配置。"""

import argparse
import json
from pathlib import Path

from src.agent_pipeline import run_agent_pipeline
from src.planner import OpenAICompatiblePlanner, PlannerConfigurationError, RuleBasedPlanner


def main() -> None:
    parser = argparse.ArgumentParser(description="银行数据质量处理智能体")
    parser.add_argument("--input", default="银行数据集（已添加问题）.csv")
    parser.add_argument("--output-dir", default="agent_outputs")
    parser.add_argument("--planner", choices=["rule", "llm"], default="rule")
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--no-fallback", action="store_true", help="禁用规则回退")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--model", help="覆盖 LLM_MODEL")
    parser.add_argument("--base-url", help="覆盖 LLM_BASE_URL")
    parser.add_argument("--timeout", type=float, default=60)
    args = parser.parse_args()

    if not 0 < args.test_size < 1:
        parser.error("--test-size 必须在 0 和 1 之间")
    if args.max_attempts < 1:
        parser.error("--max-attempts 必须至少为 1")

    if args.planner == "llm":
        try:
            planner = OpenAICompatiblePlanner.from_env(
                model=args.model,
                base_url=args.base_url,
                timeout=args.timeout,
            )
        except PlannerConfigurationError as exc:
            parser.error(str(exc))
    else:
        planner = RuleBasedPlanner()

    metrics = run_agent_pipeline(
        input_path=Path(args.input),
        output_dir=Path(args.output_dir),
        planner=planner,
        test_size=args.test_size,
        random_state=args.random_state,
        max_attempts=args.max_attempts,
        fallback_to_rules=not args.no_fallback,
    )
    summary = {
        "planner": metrics["planner"],
        "attempts": metrics["agent_attempts"],
        "fallback_used": metrics["fallback_used"],
        "cleaned_rows": metrics["deduplicated_rows"],
        "pr_auc": round(metrics["pr_auc"], 4),
        "roc_auc": round(metrics["roc_auc"], 4),
        "precision": round(metrics["precision"], 4),
        "recall": round(metrics["recall"], 4),
        "f1": round(metrics["f1"], 4),
        "output_dir": str(Path(args.output_dir).resolve()),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
