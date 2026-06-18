# -*- coding: utf-8 -*-
"""Agent Loop — ReAct 循环 + 熔断 + 循环检测 + 自优化闭环"""
import hashlib
import json
import os
from typing import Any

from agent.models import Message, AgentState, StepResult
from llm.base import BaseLLM
from llm.deepseek_api import DeepSeekAdapter
from llm.ollama import OllamaClient
from tools.registry import ToolRegistry
from memory import MemoryManager
from agent.diagnosis import FailureDiagnosis
from agent.root_cause import RootCauseAnalyzer
from agent.self_repair import SelfRepair
from agent.verify import VerifyRepair


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
        llm: BaseLLM | None = None,
        registry: ToolRegistry | None = None,
        memory: MemoryManager | None = None,
        system_prompt: str = "",
        max_steps: int = 20,
        enable_self_optimize: bool = False,
        llm_type: str = "deepseek",
        deepseek_api_key: str | None = None,
    ):
        if llm is not None:
            self.llm = llm
        elif llm_type == "deepseek":
            api_key = deepseek_api_key or os.environ.get("DEEPSEEK_API_KEY", "")
            self.llm = DeepSeekAdapter(api_key=api_key)
        elif llm_type == "ollama":
            self.llm = OllamaClient()
        else:
            raise ValueError(f"Unknown llm_type: {llm_type}")

        self.registry = registry or ToolRegistry()
        self.memory = memory
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self.max_steps = max_steps
        self._tool_fingerprints: list[str] = []
        self._error_count = 0
        self._max_errors = 5

        self.enable_self_optimize = enable_self_optimize
        self._diagnosis = FailureDiagnosis() if enable_self_optimize else None
        self._root_cause_analyzer = RootCauseAnalyzer(self.llm) if enable_self_optimize else None
        self._self_repair = SelfRepair(self.llm) if enable_self_optimize else None
        self._verify = VerifyRepair() if enable_self_optimize else None
        self._last_failure_cases: list[dict] = []

    def run(self, user_input: str, debug: bool = False) -> str:
        self._tool_fingerprints.clear()
        self._error_count = 0
        if self.enable_self_optimize:
            self._last_failure_cases.clear()
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

                    if self.enable_self_optimize:
                        self._capture_failure(
                            user_input, step,
                            "[STOPPED] 检测到重复工具调用回路，已中断",
                            "loop_detected",
                        )

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
                            if self.enable_self_optimize:
                                self._capture_failure(
                                    user_input, step,
                                    f"[STOPPED] 连续错误达到上限 ({self._max_errors})，已熔断。最后错误: {result}",
                                    "circuit_breaker",
                                )
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

    # ─── 自优化闭环 ─────────────────────────────────────

    def _capture_failure(self, task_desc: str, step: int, error_msg: str, error_type: str):
        """捕获执行失败信息"""
        if self._diagnosis is None:
            return
        snapshot = list(self.memory.short_term.get_messages())
        case = self._diagnosis.capture_failure(
            task_desc=task_desc,
            step=step,
            error_msg=error_msg,
            context_snapshot=snapshot,
            error_type=error_type,
        )
        self._last_failure_cases.append(case)

    def run_self_optimize(self, failure_cases: list[dict] | None = None) -> dict:
        """执行自优化闭环：根因分析 → 生成修复 → 应用修复 → 验证 → 保留或回滚

        Args:
            failure_cases: 失败 case 列表，如为 None 则使用上次 run() 捕获的 cases

        Returns:
            自优化报告 dict:
            {total_cases, analyzed, fixes_generated, fixes_applied, fixes_kept,
             fixes_rolled_back, details: [...]}
        """
        if not self.enable_self_optimize:
            return {"message": "自优化未启用 (enable_self_optimize=False)", "total_cases": 0}

        cases = failure_cases or self._last_failure_cases
        if not cases:
            return {"message": "没有失败 cases 可分析", "total_cases": 0}

        report = {
            "total_cases": len(cases),
            "analyzed": 0,
            "fixes_generated": 0,
            "fixes_applied": 0,
            "fixes_kept": 0,
            "fixes_rolled_back": 0,
            "details": [],
        }

        current_config = {
            "system_prompt": self.system_prompt,
            "tool_descriptions": {
                name: meta.get("description", "")
                for name, meta in self.registry._tool_metadata.items()
            },
            "memory_max_tokens": self.memory.short_term.max_tokens
            if hasattr(self.memory, "short_term") else 4096,
            "model_name": getattr(self.llm, "model", "unknown"),
        }

        for case in cases:
            detail = {"case_id": case.get("id", "?"), "task": case.get("task_desc", "")[:80]}

            # 1. 根因分析
            root_cause = self._root_cause_analyzer.analyze(case)
            detail["root_cause"] = root_cause.get("root_cause_type", "?")
            detail["confidence"] = root_cause.get("confidence", 0)
            report["analyzed"] += 1

            # 跳过低置信度
            if root_cause.get("confidence", 0) < 0.4:
                detail["action"] = "skipped_low_confidence"
                detail["message"] = f"置信度 {root_cause.get('confidence', 0):.1f} < 0.4，跳过"
                report["details"].append(detail)
                continue

            # 跳过需人工介入的类型
            if root_cause.get("suggested_fix_type") in ("fix_tool_code", "switch_model"):
                detail["action"] = "skipped_requires_manual"
                detail["message"] = "需人工介入"
                report["details"].append(detail)
                continue

            # 2. 生成修复
            fix = self._self_repair.generate_fix(root_cause, current_config)
            detail["fix_type"] = fix.get("fix_type", "?")
            report["fixes_generated"] += 1

            # 3. 应用修复
            applied = self._self_repair.apply_fix(fix, self)
            detail["applied"] = applied

            if not applied:
                detail["action"] = "apply_failed"
                report["details"].append(detail)
                continue

            report["fixes_applied"] += 1

            # 4. 验证
            task = case.get("task_desc", "")
            verify_result = self._verify.verify(fix, self, task)
            detail["after_success"] = verify_result.get("after_success", False)
            detail["after_message"] = verify_result.get("after_message", "")[:100]

            # 5. 保留或回滚
            if verify_result.get("improved", False):
                detail["action"] = "kept"
                report["fixes_kept"] += 1
            else:
                self._self_repair.rollback(fix, self)
                detail["action"] = "rolled_back"
                report["fixes_rolled_back"] += 1

            report["details"].append(detail)

        return report
