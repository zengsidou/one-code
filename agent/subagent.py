# -*- coding: utf-8 -*-
"""SubAgent — 上下文隔离子代理，带返回格式契约和生命周期管理"""
from agent.models import Message
from llm.base import BaseLLM
from tools.registry import ToolRegistry


RETURN_FORMAT_INSTRUCTION = (
    "返回格式（必须遵守）:\n"
    "你的最终回复必须以以下格式开头:\n"
    "**Status**: success | partial | failed | blocked\n"
    "**Summary**: <一句话描述>\n\n"
    "然后才是正文内容。\n"
    "不要跳过这个格式头。"
)


class SubAgent:
    def __init__(
        self,
        llm: BaseLLM,
        registry: ToolRegistry,
        prompt: str = "",
        max_steps: int = 8,
        context: list[Message] | None = None,
    ):
        self.llm = llm
        self.registry = registry
        self.max_steps = max_steps
        self._context: list[Message] = context or []
        base_prompt = prompt or (
            "你是 One-Code 子代理，专注于执行单个子任务。"
            "使用可用工具完成任务，给出简洁结果。"
        )
        self.prompt = base_prompt + "\n\n" + RETURN_FORMAT_INSTRUCTION

    def run(self, task: str) -> str:
        messages = list(self._context)
        messages.append(Message(role="system", content=self.prompt))
        messages.append(Message(role="user", content=task))

        for step in range(self.max_steps):
            response = self.llm.generate(list(messages), tools=self.registry.get_schemas())
            tool_calls = response.tool_calls or []

            if not tool_calls:
                content = response.content or ""
                messages.append(Message(role="assistant", content=content, reasoning_content=response.reasoning_content))
                return self._parse_result(content)

            import json
            for tc in tool_calls:
                messages.append(Message(
                    role="assistant",
                    content=f"调用工具: {tc.name}",
                    tool_calls=[tc],
                    reasoning_content=response.reasoning_content,
                ))
                result = self.registry.execute(tc.name, tc.arguments)
                messages.append(Message(
                    role="tool", content=result,
                    tool_call_id=tc.id, tool_name=tc.name,
                ))
                last_tool_result = result

        messages.append(Message(role="system", content="已达到最大步骤限制。请基于已有信息给出最终结果（带 Status/Summary 头）。"))
        final = self.llm.generate(messages, tools=None)
        return self._parse_result(final.content or "子任务未能完成。")

    @staticmethod
    def _parse_result(text: str) -> str:
        """解析子代理返回内容，提取 Status/Summary"""
        status = "success"
        summary = ""
        body = text
        for line in text.split("\n"):
            line_stripped = line.strip()
            if line_stripped.startswith("**Status**:"):
                status = line_stripped.split(":", 1)[1].strip().lower().split()[0]
            elif line_stripped.startswith("**Summary**:"):
                summary = line_stripped.split(":", 1)[1].strip()
        if summary:
            return f"[{status}] {summary}"
        return body[:500]
