"""规则规划器与 OpenAI 兼容的大模型规划器。"""

from datetime import datetime
import json
import os
import time
from typing import Any
from urllib import error, request

from .plan_schema import CLEANING_PLAN_SCHEMA
from .planning import build_cleaning_plan


class PlannerError(RuntimeError):
    def __init__(
        self,
        message: str,
        retry_after: float = 0,
        transient: bool = False,
    ) -> None:
        super().__init__(message)
        self.retry_after = retry_after
        self.transient = transient


class PlannerConfigurationError(PlannerError):
    pass


class RuleBasedPlanner:
    name = "rule"

    def __init__(self) -> None:
        self.call_records: list[dict[str, Any]] = []

    def generate_plan(
        self,
        report: dict[str, Any],
        feedback: dict[str, Any] | None = None,
        attempt: int = 1,
    ) -> dict[str, Any]:
        return build_cleaning_plan(report)


def extract_json_object(content: str) -> dict[str, Any]:
    """兼容纯 JSON 和 Markdown 代码块响应。"""
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("大模型响应中没有 JSON 对象")
        value = json.loads(text[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("大模型响应的 JSON 顶层不是对象")
    return value


def compact_quality_report(report: dict[str, Any]) -> dict[str, Any]:
    """减少提示词体积，保留规划所需的检测证据。"""
    return {
        **{key: value for key, value in report.items() if key != "fields"},
        "fields": {
            column: {
                key: value
                for key, value in info.items()
                if key != "top_values"
            }
            for column, info in report.get("fields", {}).items()
        },
    }


class OpenAICompatiblePlanner:
    name = "llm"

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str,
        timeout: float = 60,
        input_cost_per_1m: float | None = None,
        output_cost_per_1m: float | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.input_cost_per_1m = input_cost_per_1m
        self.output_cost_per_1m = output_cost_per_1m
        self.call_records: list[dict[str, Any]] = []

    @classmethod
    def from_env(
        cls,
        model: str | None = None,
        base_url: str | None = None,
        timeout: float = 60,
    ) -> "OpenAICompatiblePlanner":
        resolved_model = model or os.getenv("LLM_MODEL")
        resolved_base_url = base_url or os.getenv("LLM_BASE_URL")
        api_key = os.getenv("LLM_API_KEY")
        missing = [
            name
            for name, value in [
                ("LLM_MODEL", resolved_model),
                ("LLM_BASE_URL", resolved_base_url),
                ("LLM_API_KEY", api_key),
            ]
            if not value
        ]
        if missing:
            raise PlannerConfigurationError(f"缺少大模型配置：{', '.join(missing)}")

        def optional_float(name: str) -> float | None:
            value = os.getenv(name)
            return float(value) if value else None

        return cls(
            model=resolved_model,
            base_url=resolved_base_url,
            api_key=api_key,
            timeout=timeout,
            input_cost_per_1m=optional_float("LLM_INPUT_COST_PER_1M"),
            output_cost_per_1m=optional_float("LLM_OUTPUT_COST_PER_1M"),
        )

    def _messages(
        self,
        report: dict[str, Any],
        feedback: dict[str, Any] | None,
    ) -> list[dict[str, str]]:
        system = (
            "你是银行营销数据的质量处理规划器。你只生成清洗计划，不直接修改数据。"
            "必须输出一个 JSON 对象，不要输出 Markdown 或解释。"
            "只能选择 schema 允许的操作；标签 y 绝不修改；unknown 是业务类别而非缺失；"
            "负余额保留，极值不自动删除；有完全重复记录必须先去重；"
            "数值填充只能使用训练集中位数；类别缺失填为 missing。"
            "年龄有效范围固定为 [18, 100]。若收到复检反馈，必须逐项修正。"
        )
        user_payload = {
            "task": "根据质量画像生成完整清洗计划",
            "json_schema": CLEANING_PLAN_SCHEMA,
            "quality_report": compact_quality_report(report),
            "previous_feedback": feedback,
        }
        return [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": json.dumps(user_payload, ensure_ascii=False),
            },
        ]

    def _calculate_cost(self, usage: dict[str, Any]) -> float | None:
        if self.input_cost_per_1m is None or self.output_cost_per_1m is None:
            return None
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        if prompt_tokens is None or completion_tokens is None:
            return None
        return (
            float(prompt_tokens) * self.input_cost_per_1m / 1_000_000
            + float(completion_tokens) * self.output_cost_per_1m / 1_000_000
        )

    def generate_plan(
        self,
        report: dict[str, Any],
        feedback: dict[str, Any] | None = None,
        attempt: int = 1,
    ) -> dict[str, Any]:
        messages = self._messages(report, feedback)
        parameters = {"temperature": 1}
        body = {
            "model": self.model,
            "messages": messages,
            **parameters,
        }
        record: dict[str, Any] = {
            "attempt": attempt,
            "started_at": datetime.now().astimezone().isoformat(),
            "model": self.model,
            "base_url": self.base_url,
            "request": {"messages": messages, "parameters": parameters},
            "status": "pending",
        }
        started = time.monotonic()
        endpoint = self.base_url
        if not endpoint.endswith("/chat/completions"):
            endpoint += "/chat/completions"
        http_request = request.Request(
            endpoint,
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with request.urlopen(http_request, timeout=self.timeout) as response:
                response_body = json.loads(response.read().decode("utf-8"))
            content = response_body["choices"][0]["message"]["content"]
            if isinstance(content, list):
                content = "".join(
                    item.get("text", "") for item in content if isinstance(item, dict)
                )
            plan = extract_json_object(content)
            usage = response_body.get("usage", {})
            record.update(
                {
                    "status": "success",
                    "response_text": content,
                    "usage": usage,
                    "estimated_cost_usd": self._calculate_cost(usage),
                    "cost_note": "仅在配置每百万 token 单价时估算",
                }
            )
            return plan
        except error.HTTPError as exc:
            response_text = exc.read().decode("utf-8", errors="replace")[:4000]
            retry_after = float(exc.headers.get("Retry-After", "0") or 0)
            transient = exc.code == 429 or 500 <= exc.code < 600
            record.update(
                {
                    "status": "http_error",
                    "http_status": exc.code,
                    "error": response_text,
                    "retry_after_seconds": retry_after,
                }
            )
            raise PlannerError(
                f"大模型 HTTP {exc.code}：{response_text}",
                retry_after=retry_after,
                transient=transient,
            ) from exc
        except (error.URLError, TimeoutError) as exc:
            record.update({"status": "network_error", "error": str(exc)})
            raise PlannerError(f"大模型网络错误：{exc}", transient=True) from exc
        except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
            record.update({"status": "response_parse_error", "error": str(exc)})
            raise PlannerError(f"大模型响应解析失败：{exc}") from exc
        finally:
            record["duration_seconds"] = time.monotonic() - started
            record["finished_at"] = datetime.now().astimezone().isoformat()
            self.call_records.append(record)
