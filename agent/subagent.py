# -*- coding: utf-8 -*-
"""SubAgent — 轻量子代理，被主 Agent 委派执行子任务"""
from agent.models import Message
from llm.base import BaseLLM
from tools.registry import ToolRegistry


DEFAULT_SUBAGENT_PROMPT = (
    "你是一个专注于特定子任务的 AI Agent。\n"
    "请严格遵循任务要求，使用可用工具完成任务，然后给出简洁结果。\n"
    "不要在结果中添加无关解释。"
)


class SubAgent:
    def __init__(
        self,
        llm: BaseLLM,
        registry: ToolRegistry,
        prompt: str = "",
        max_steps: int = 5,
    ):
        self.llm = llm
        self.registry = registry
        self.prompt = prompt or DEFAULT_SUBAGENT_PROMPT
        self.max_steps = max_steps

    def run(self, task: str) -> str:
        messages: list[Message] = [
            Message(role="system", content=self.prompt),
            Message(role="user", content=task),
        ]
        last_tool_result = ""

        for step in range(self.max_steps):
            # Nudge if we have tool results from previous step
            context = list(messages)
            if last_tool_result and step > 0:
                context.append(Message(
                    role="system",
                    content=(
                        "工具已返回结果。请直接引用以上结果给出简洁的最终中文回复，不要再调用工具。"
                        f"\n工具返回的实际数据:\n{last_tool_result[:2000]}"
                    ),
                ))

            response = self.llm.generate(context, tools=self.registry.get_schemas())

            tool_calls = response.tool_calls or []

            if not tool_calls:
                content = response.content or ""
                messages.append(Message(role="assistant", content=content))
                return content

            for tc in tool_calls:
                import json
                messages.append(Message(
                    role="assistant",
                    content=f"调用工具: {tc.name}({json.dumps(tc.arguments, ensure_ascii=False)})",
                    tool_calls=[tc],
                ))

                result = self.registry.execute(tc.name, tc.arguments)
                messages.append(Message(
                    role="tool",
                    content=result,
                    tool_call_id=tc.id,
                    tool_name=tc.name,
                ))
                last_tool_result = result

        messages.append(Message(
            role="system",
            content="已达到最大步骤限制。请基于以上所有信息给出最终结果。",
        ))
        final = self.llm.generate(messages, tools=None)
        return final.content or "子任务未能完成。"
