# -*- coding: utf-8 -*-
"""Agent Loop — ReAct 循环 + 熔断 + 循环检测"""
import hashlib
import json

from agent.models import Message, AgentState, StepResult
from llm.base import BaseLLM
from tools.registry import ToolRegistry
from memory import MemoryManager


DEFAULT_SYSTEM_PROMPT = (
    "你是一个智能 AI Agent，能够使用工具完成用户任务。\n\n"
    "行为准则:\n"
    "1. 分析用户需求，选择合适的工具\n"
    "2. 每次只调用必要的工具，同一个工具不要反复调用\n"
    "3. 收到工具执行结果后，直接基于结果给出中文总结，不要再次调用相同工具\n"
    "4. 工具调用格式: {\"tool\": \"工具名\", \"arguments\": {...}}\n"
    "5. 完成任务后给出简洁的最终回答，不要继续调用工具"
)


class AgentLoop:
    def __init__(
        self,
        llm: BaseLLM,
        registry: ToolRegistry,
        memory: MemoryManager,
        system_prompt: str = "",
        max_steps: int = 20,
    ):
        self.llm = llm
        self.registry = registry
        self.memory = memory
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self.max_steps = max_steps
        self._tool_fingerprints: list[str] = []
        self._error_count = 0
        self._max_errors = 5

    def run(self, user_input: str, debug: bool = False) -> str:
        self._tool_fingerprints.clear()
        self._error_count = 0
        self.memory.add_message(Message(role="user", content=user_input))

        for step in range(self.max_steps):
            if debug:
                print(f"  [DEBUG step={step}] fingerprints={self._tool_fingerprints[-5:]}")
            context = self.memory.get_context(query=user_input)
            context.insert(0, Message(role="system", content=self._build_system_prompt()))

            # After tool results, nudge model to summarize
            has_tool_results = any(
                m.role == "tool" for m in self.memory.short_term.get_messages()
            )
            if has_tool_results and step > 0:
                # Inject the most recent tool result into the nudge so small models don't hallucinate
                last_tool = next(
                    (m.content for m in reversed(self.memory.short_term.get_messages()) if m.role == "tool"),
                    ""
                )
                context.append(Message(
                    role="system",
                    content=(
                        "工具已返回结果。请直接引用以上结果给用户一个简洁的中文回复。"
                        "不要说'根据之前的工具调用'、'可能包括'之类的模糊措辞，要引用实际数据。"
                        f"\n工具返回的实际数据:\n{last_tool[:2000]}"
                    ),
                ))

            response = self.llm.generate(context, tools=self.registry.get_schemas())

            tool_calls = response.tool_calls or []

            if tool_calls:
                if self._detect_tool_loop(tool_calls, debug):
                    content = response.content or ""
                    self.memory.add_message(Message(role="assistant", content=content))
                    summary = self._force_summarize()
                    return summary if summary else f"[STOPPED] 检测到重复工具调用回路，已中断。"

                for tc in tool_calls:
                    tool_msg_content = f"调用工具: {tc.name}({json.dumps(tc.arguments, ensure_ascii=False)})"
                    self.memory.add_message(Message(role="assistant", content=tool_msg_content))

                    result = self.registry.execute(tc.name, tc.arguments)
                    self.memory.add_message(Message(
                        role="tool",
                        content=result,
                        tool_call_id=tc.id,
                        tool_name=tc.name,
                    ))

                    if result.startswith("[ERROR]"):
                        self._error_count += 1
                        if self._error_count >= self._max_errors:
                            return f"[STOPPED] 连续错误达到上限 ({self._max_errors})，已熔断。最后错误: {result}"

                continue

            content = response.content or ""
            self.memory.add_message(Message(role="assistant", content=content))
            return content

        return f"[STOPPED] 达到最大迭代次数 ({self.max_steps})。"

    def _build_system_prompt(self) -> str:
        tool_desc = self.registry.get_tools_description()
        return (
            f"{self.system_prompt}\n\n"
            "---\n"
            f"可用工具:\n{tool_desc}\n\n"
            "工具调用时输出严格 JSON: "
            '{"tool": "工具名", "arguments": {"参数名": "参数值"}}'
        )

    def _detect_tool_loop(self, tool_calls: list, debug: bool = False) -> bool:
        for tc in tool_calls:
            raw = f"{tc.name}:{json.dumps(tc.arguments, ensure_ascii=False, sort_keys=True)}"
            fp = hashlib.md5(raw.encode()).hexdigest()
            self._tool_fingerprints.append(fp)
            if debug:
                print(f"  [DEBUG detect] tool={tc.name} args={tc.arguments} fp={fp[:12]} count={self._tool_fingerprints.count(fp)}")
            if len(self._tool_fingerprints) > 20:
                self._tool_fingerprints = self._tool_fingerprints[-20:]
            if self._tool_fingerprints.count(fp) >= 2:
                return True
        return False

    def _force_summarize(self) -> str:
        msgs = self.memory.short_term.get_messages()
        # Find the most recent tool result and user query
        tool_results = []
        user_query = ""
        for i in range(len(msgs) - 1, -1, -1):
            if msgs[i].role == "tool" and msgs[i].content and not msgs[i].content.startswith("[ERROR]"):
                tool_results.insert(0, msgs[i].content)
            if msgs[i].role == "user" and not user_query:
                user_query = msgs[i].content or ""
        if not tool_results:
            return ""
        result_text = "\n\n".join(tool_results)
        # For small models, if result is short enough, return directly
        if len(result_text) < 600:
            return f"查询「{user_query}」的结果:\n{result_text}"
        # Otherwise, truncate and summarize via LLM
        return self._summarize_result(user_query, result_text[:1500])

    def _summarize_result(self, query: str, result: str) -> str:
        prompt = [
            Message(role="system", content="用户的问题是: " + query),
            Message(role="system", content="工具返回了以下结果。请用中文简洁总结给用户。"),
            Message(role="user", content=result),
        ]
        try:
            resp = self.llm.generate(prompt, tools=None)
            return resp.content or result
        except Exception:
            return result
