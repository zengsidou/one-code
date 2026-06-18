# -*- coding: utf-8 -*-
"""根因分析模块 — 用 LLM 分析 Agent 失败的根本原因"""
import json

from agent.models import Message


ROOT_CAUSE_SYSTEM_PROMPT = (
    "你是一个 Agent 诊断专家。分析以下 Agent 执行失败 case，判断根本原因。"
    "只从以下类型中选择一个："
    "prompt_unclear(提示词不够清晰), "
    "tool_schema_vague(工具描述不够具体), "
    "context_overflow(上下文过长导致截断), "
    "model_limitation(模型能力不足), "
    "tool_error(工具本身有bug), "
    "incorrect_reasoning(推理链逻辑错误)。"
    "同时给出修复建议，必须是以下之一："
    "adjust_prompt, enrich_tool_description, trim_context, "
    "switch_model, fix_tool_code, add_reasoning_hint。"
    "输出严格 JSON 格式。"
)

ROOT_CAUSE_USER_TEMPLATE = (
    "执行以下任务时 Agent 失败：\n\n"
    "任务描述: {task_desc}\n"
    "失败步骤: 第 {failed_step} 步\n"
    "错误类型标记: {error_type}\n"
    "错误信息: {error_msg}\n\n"
    "失败时的上下文消息（最后几条）:\n{context_text}\n\n"
    "请分析根本原因，输出 JSON。"
)


class RootCauseAnalyzer:
    """Agent 失败根因分析器

    将 FailureDiagnosis 捕获的 case 交给 LLM 分析，
    输出标准化的根因类型和修复建议。
    """

    def __init__(self, llm_adapter):
        """Args:
            llm_adapter: BaseLLM 子类实例，用于调用 LLM 分析
        """
        self.llm = llm_adapter

    def analyze(self, failure_case: dict) -> dict:
        """分析单个失败 case 的根因

        Args:
            failure_case: FailureDiagnosis.capture_failure() 返回的 dict

        Returns:
            {root_cause_type, confidence, detail, suggested_fix_type, fix_description}
        """
        context_text = ""
        snapshot = failure_case.get("context_snapshot", [])
        if snapshot:
            recent = snapshot[-6:]
            lines = []
            for m in recent:
                role = m.get("role", "?")
                content = str(m.get("content", ""))[:300]
                lines.append(f"[{role}] {content}")
            context_text = "\n".join(lines)

        prompt_text = ROOT_CAUSE_USER_TEMPLATE.format(
            task_desc=failure_case.get("task_desc", ""),
            failed_step=failure_case.get("failed_step", "?"),
            error_type=failure_case.get("error_type", "other"),
            error_msg=failure_case.get("error_msg", ""),
            context_text=context_text or "(无上下文)",
        )

        messages = [
            Message(role="system", content=ROOT_CAUSE_SYSTEM_PROMPT),
            Message(role="user", content=prompt_text),
        ]

        try:
            response = self.llm.generate(messages, tools=None)
            return self._parse_response(response.content or "", failure_case)
        except Exception as e:
            return {
                "root_cause_type": "tool_error",
                "confidence": 0.3,
                "detail": f"LLM 调用失败，无法自动分析: {e}",
                "suggested_fix_type": "fix_tool_code",
                "fix_description": "自动分析失败，需人工排查",
            }

    @staticmethod
    def _parse_response(text: str, failure_case: dict) -> dict:
        """解析 LLM 返回的 JSON 根因分析结果"""
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        try:
            result = json.loads(text.strip())
        except json.JSONDecodeError:
            result = {}

        valid_causes = {
            "prompt_unclear", "tool_schema_vague", "context_overflow",
            "model_limitation", "tool_error", "incorrect_reasoning",
        }
        valid_fixes = {
            "adjust_prompt", "enrich_tool_description", "trim_context",
            "switch_model", "fix_tool_code", "add_reasoning_hint",
        }

        cause = result.get("root_cause_type", "tool_error")
        fix = result.get("suggested_fix_type", "fix_tool_code")

        return {
            "root_cause_type": cause if cause in valid_causes else "tool_error",
            "confidence": float(result.get("confidence", 0.5)),
            "detail": result.get("detail", result.get("fix_description", "无详细说明")),
            "suggested_fix_type": fix if fix in valid_fixes else "fix_tool_code",
            "fix_description": result.get("fix_description", "无修复描述"),
        }

    def batch_analyze(self, failure_cases: list[dict]) -> list[dict]:
        """批量分析多个失败 cases

        Args:
            failure_cases: FailureDiagnosis.capture_failure() 返回的 dict 列表

        Returns:
            根因分析结果列表
        """
        results = []
        for case in failure_cases:
            result = self.analyze(case)
            result["case_id"] = case.get("id", "?")
            results.append(result)
        return results
