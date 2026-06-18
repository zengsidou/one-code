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
from agent.fix_history import FixHistory
from agent.meta_optimize import MetaOptimizer
from agent.evolve import TaskPostMortem, SkillLibrary, AbilityProfile, ChallengeGenerator
from agent.evolve import ArchitectureBottleneckDetector, ArchitectureProposalGenerator, ArchitectureApplier, ArchitectureValidator
from agent.orchestrator import AgentOrchestrator
from agent.checkpoint import AgentCheckpoint


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
        enable_evolution: bool = False,
        llm_type: str = "deepseek",
        deepseek_api_key: str | None = None,
        self_optimize_max_retries: int = 2,
        fix_history_file: str = "./fix_history.json",
        skill_library_file: str = "./skill_library.json",
        ability_profile_file: str = "./ability_profile.json",
        enable_orchestrate: bool = False,
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
        self.self_optimize_max_retries = self_optimize_max_retries
        self._diagnosis = FailureDiagnosis() if enable_self_optimize else None
        self._root_cause_analyzer = RootCauseAnalyzer(self.llm) if enable_self_optimize else None
        self._self_repair = SelfRepair(self.llm) if enable_self_optimize else None
        self._verify = VerifyRepair() if enable_self_optimize else None
        self._fix_history = FixHistory(fix_history_file) if enable_self_optimize else None
        self._meta_optimizer = MetaOptimizer(self.llm) if enable_self_optimize else None
        self._last_failure_cases: list[dict] = []
        self._self_optimize_retry_count = 0
        self._meta_optimize_count = 0
        self._meta_optimize_max = 2
        self._last_optimize_report: dict = {}

        self.enable_evolution = enable_evolution
        self._post_mortem = TaskPostMortem(self.llm) if enable_evolution else None
        self._skill_library = SkillLibrary(skill_library_file) if enable_evolution else None
        if self._skill_library:
            self._skill_library.set_llm(self.llm)
        self._ability_profile = AbilityProfile(ability_profile_file) if enable_evolution else None
        self._challenge_gen = ChallengeGenerator(self.llm) if enable_evolution else None
        self._arch_bottleneck = ArchitectureBottleneckDetector(self.llm) if enable_evolution else None
        self._arch_proposer = ArchitectureProposalGenerator(self.llm) if enable_evolution else None
        self._arch_applier = ArchitectureApplier() if enable_evolution else None
        self._arch_validator = ArchitectureValidator() if enable_evolution else None
        self._orchestrator = AgentOrchestrator(self.llm, self.registry) if enable_orchestrate else None
        self._step_trace: list[str] = []
        self._last_step_count = 0

    def run(self, user_input: str, debug: bool = False) -> str:
        # 检查是否是重启恢复
        if AgentCheckpoint.has_restart_flag():
            AgentCheckpoint.clear_restart_flag()
            ckpt = AgentCheckpoint.load()
            if ckpt:
                self._restore_from_checkpoint(ckpt)
                user_input = ckpt.get("current_task", user_input)
                if debug:
                    print(f"  [RESTART] 从检查点恢复，继续执行: {user_input[:80]}")

        self._tool_fingerprints.clear()
        self._error_count = 0
        self._self_optimize_retry_count = 0
        self._step_trace = []
        self._last_step_count = 0
        if self.enable_self_optimize:
            self._last_failure_cases.clear()
        self.memory.add_message(Message(role="user", content=user_input))

        result = self._run_loop(user_input, debug)
        if "[STOPPED]" in result and self.enable_self_optimize:
            result = self._try_self_heal(user_input, debug)

        # 检查是否触发了自重启
        if "[RESTART]" in result:
            # 自重启已由 AgentCheckpoint.trigger_restart 处理
            # 进程会在这里被替换，不会继续
            return result

        # 进化层：每次执行后复盘 + 沉淀
        if self.enable_evolution:
            self._evolve_after_run(user_input, result)

        return result

    def _run_loop(self, user_input: str, debug: bool = False) -> str:
        """内部执行循环，返回结果或 [STOPPED] 错误"""
        self._error_count = 0
        self._tool_fingerprints.clear()

        for step in range(self.max_steps):
            if debug:
                print(f"  [DEBUG step={step}] fingerprints={self._tool_fingerprints[-5:]}")
            context = self.memory.get_context(query=user_input)
            context.insert(0, Message(role="system", content=self._build_system_prompt()))

            has_tool_results = any(
                m.role == "tool" for m in self.memory.short_term.get_messages()
            )
            if has_tool_results and step > 0:
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
                    self.memory.add_message(Message(role="assistant", content=response.content or ""))
                    self._step_trace.append(f"Step{step}: LOOP_DETECTED")
                    if self.enable_self_optimize:
                        self._capture_failure(
                            user_input, step,
                            "[STOPPED] 检测到重复工具调用回路，已中断",
                            "loop_detected",
                        )
                    return "[STOPPED] 检测到重复工具调用回路，已中断。"

                for tc in tool_calls:
                    tool_msg_content = f"调用工具: {tc.name}({json.dumps(tc.arguments, ensure_ascii=False)})"
                    self._step_trace.append(f"Step{step}: call {tc.name}")
                    self.memory.add_message(Message(role="assistant", content=tool_msg_content, tool_calls=[tc]))
                    result = self.registry.execute(tc.name, tc.arguments)
                    self.memory.add_message(Message(
                        role="tool", content=result,
                        tool_call_id=tc.id, tool_name=tc.name,
                    ))
                    if result.startswith("[ERROR]"):
                        self._step_trace.append(f"Step{step}: ERROR {tc.name}")
                        self._error_count += 1
                        if self._error_count >= self._max_errors:
                            if self.enable_self_optimize:
                                self._capture_failure(
                                    user_input, step,
                                    f"[STOPPED] 连续错误达到上限 ({self._max_errors})，已熔断。最后错误: {result}",
                                    "circuit_breaker",
                                )
                            self._step_trace.append(f"Step{step}: CIRCUIT_BREAKER")
                            return f"[STOPPED] 连续错误达到上限 ({self._max_errors})，已熔断。最后错误: {result}"
                continue

            content = response.content or ""
            self._step_trace.append(f"Step{step}: final_answer")
            self._last_step_count = step + 1
            self.memory.add_message(Message(role="assistant", content=content))
            return content

        self._step_trace.append(f"MAX_STEPS reached")
        return f"[STOPPED] 达到最大迭代次数 ({self.max_steps})。"

    def _build_system_prompt(self) -> str:
        tool_desc = self.registry.get_tools_description()
        base = (
            f"{self.system_prompt}\n\n"
            "---\n"
            f"可用工具:\n{tool_desc}\n\n"
            "工具调用时输出严格 JSON: "
            '{"tool": "工具名", "arguments": {"参数名": "参数值"}}'
        )
        # 注入已学技能
        if self.enable_evolution and self._skill_library:
            skills = self._skill_library.query("", top_k=3)
            hint = self._skill_library.to_prompt_hint(skills)
            if hint:
                base += hint
        return base

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

    def _try_self_heal(self, user_input: str, debug: bool = False) -> str:
        """自动触发自优化并重试执行

        Returns:
            修复后重试的结果，或原始错误消息
        """
        if self._self_optimize_retry_count >= self.self_optimize_max_retries:
            return f"[STOPPED] 自优化重试已达上限 ({self.self_optimize_max_retries})"

        self._self_optimize_retry_count += 1

        if debug:
            print(f"  [SELF-OPTIMIZE] 自动触发自优化 (attempt {self._self_optimize_retry_count}/{self.self_optimize_max_retries})")

        # 1. 先查历史修复
        self._try_reuse_historical_fixes()

        # 2. 执行完整的自优化闭环
        report = self.run_self_optimize()
        self._last_optimize_report = report

        if report.get("fixes_kept", 0) == 0:
            # 元优化：自优化自身出问题了，尝试优化自优化组件
            if self._meta_optimize_count < self._meta_optimize_max:
                self._meta_optimize_count += 1
                if debug:
                    print(f"  [META-OPTIMIZE] 自优化0有效修复，触发元优化 (attempt {self._meta_optimize_count}/{self._meta_optimize_max})")
                meta_result = self._meta_optimizer.optimize(report, self)
                if meta_result.get("improved"):
                    return self._try_self_heal(user_input, debug)

            # 架构自进化：元优化也无效，可能是架构瓶颈
            if self.enable_evolution and self._arch_bottleneck:
                self._arch_bottleneck.record_failure(user_input, [report], self._last_failure_cases)
                if self._arch_bottleneck.is_bottleneck(min_consecutive_failures=2):
                    if debug:
                        print("  [ARCH-EVOLVE] 检测到架构瓶颈，尝试架构自进化...")
                    arch_report = self._try_architect_evolve(user_input)
                    if arch_report.get("applied") and arch_report.get("validated"):
                        return self._try_self_heal(user_input, debug)

            return f"[STOPPED] 自优化未产生有效修复 (analyzed={report.get('analyzed',0)}, kept=0)"

        # 3. 保存成功修复到历史
        self._save_fixes_to_history(report)

        # 4. 清理记忆，重新执行任务
        self.memory.clear()
        if debug:
            print("  [SELF-OPTIMIZE] 记忆已重置，重新执行任务...")

        result = self._run_loop(user_input, debug)

        # 5. 如果仍然失败，递归重试
        if "[STOPPED]" in result:
            return self._try_self_heal(user_input, debug)

        return result

    def _try_reuse_historical_fixes(self):
        """尝试复用历史修复 — 匹配当前失败模式，直接应用已知修复"""
        if self._fix_history is None or not self._last_failure_cases:
            return

        for case in self._last_failure_cases:
            error_type = case.get("error_type", "")
            task_desc = case.get("task_desc", "")
            similar = self._fix_history.find_similar(error_type, task_desc)
            if similar:
                best = similar[0]
                history_fix = best.get("fix", {})
                if history_fix.get("fix_type") in ("adjust_prompt", "add_reasoning_hint"):
                    if "system_prompt" in history_fix.get("fixed", {}):
                        self._self_repair._rollback_snapshots[best["signature"]] = {
                            "system_prompt": self.system_prompt,
                            "memory_max_tokens": self.memory.short_term.max_tokens
                            if hasattr(self.memory, "short_term") else None,
                        }
                        self.system_prompt = history_fix["fixed"]["system_prompt"]
                        self._fix_history.mark_reused(history_fix)
                elif history_fix.get("fix_type") == "trim_context":
                    if "memory_max_tokens" in history_fix.get("fixed", {}):
                        self._self_repair._rollback_snapshots[best["signature"]] = {
                            "system_prompt": self.system_prompt,
                            "memory_max_tokens": self.memory.short_term.max_tokens
                            if hasattr(self.memory, "short_term") else None,
                        }
                        if hasattr(self.memory, "short_term"):
                            self.memory.short_term.max_tokens = history_fix["fixed"]["memory_max_tokens"]
                        self._fix_history.mark_reused(history_fix)

    def _save_fixes_to_history(self, report: dict):
        """将验证通过的修复保存到历史"""
        if self._fix_history is None:
            return
        for detail in report.get("details", []):
            if detail.get("action") != "kept":
                continue
            for case in self._last_failure_cases:
                if case.get("id") == detail.get("case_id"):
                    fix = {"fix_type": detail.get("fix_type", ""), "original": {}, "fixed": {}}
                    if detail.get("fix_type") == "adjust_prompt" or detail.get("fix_type") == "add_reasoning_hint":
                        fix["fixed"]["system_prompt"] = self.system_prompt
                    self._fix_history.record_fix(
                        error_type=case.get("error_type", "other"),
                        task_desc=case.get("task_desc", ""),
                        fix=fix,
                        root_cause={"root_cause_type": detail.get("root_cause", ""), "confidence": detail.get("confidence", 0)},
                        verified=True,
                    )
                    break

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
            threshold = getattr(self._root_cause_analyzer, "confidence_threshold", 0.4)
            if root_cause.get("confidence", 0) < threshold:
                detail["action"] = "skipped_low_confidence"
                detail["message"] = f"置信度 {root_cause.get('confidence', 0):.1f} < {threshold}，跳过"
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

    # ─── 进化层 ─────────────────────────────────────────

    def _evolve_after_run(self, task_desc: str, result: str):
        """每次执行后复盘反思 + 沉淀技能 + 记录能力画像"""
        trace = "\n".join(self._step_trace[-15:]) if self._step_trace else "(无执行轨迹)"
        is_failure = "[STOPPED]" in result or "[ERROR]" in result

        # 构建能力上下文
        ability_ctx = "Agent 当前能力水平: "
        if self._ability_profile and len(self._ability_profile.records) > 0:
            stats = self._ability_profile.get_growth_summary(window=10)
            ability_ctx += (
                f"已完成 {stats['total_tasks']} 个任务，"
                f"成功率 {stats['recent_success_rate']:.0%}，"
                f"平均难度 {stats['recent_avg_diff']}/5，"
                f"平均效率 {stats['recent_avg_efficiency']}/5"
            )
        else:
            ability_ctx += "这是 Agent 的早期任务，尚在建立基准。"

        # 1. 复盘
        reflection = self._post_mortem.reflect(
            task_desc, result, trace,
            step_count=self._last_step_count,
            ability_context=ability_ctx,
        )

        # 2. 提取技能
        self._skill_library.add_from_post_mortem(reflection)

        # 3. 记录能力画像
        difficulty = reflection.get("difficulty_for_agent", 3)
        efficiency = reflection.get("efficiency_score", 3)
        self._ability_profile.record(
            task_desc=task_desc,
            success=not is_failure,
            difficulty=difficulty,
            efficiency=efficiency,
            steps=self._last_step_count,
        )

        # 4. 软失败检测：复盘弱点重复出现 → 触发进化链
        if self._arch_bottleneck:
            pattern = self._post_mortem.detect_repeating_weakness(min_occurrences=3, window=5)
            if pattern:
                self._arch_bottleneck.record_failure(
                    task_desc=f"[SOFT-FAIL] {pattern['pattern']}",
                    optimize_reports=[{"analyzed": 1, "fixes_kept": 0, "fixes_rolled_back": 1}],
                )
                if self._arch_bottleneck.is_bottleneck(min_consecutive_failures=2):
                    self._try_architect_evolve(task_desc)

    def grow(self) -> dict:
        """主动成长：生成挑战任务，推动能力边界

        Returns:
            {suggestion, challenges, skill_stats, growth_summary}
        """
        if not self.enable_evolution:
            return {"error": "进化层未启用 (enable_evolution=False)"}

        profile_summary = json.dumps(
            self._ability_profile.get_all_category_stats(),
            ensure_ascii=False, indent=2,
        )
        weak = self._ability_profile.get_weak_areas()
        suggestion = self._challenge_gen.suggest_next_level(self._ability_profile)
        current_level = suggestion.get("current_level", 2) if suggestion else 2

        challenges = self._challenge_gen.generate(
            profile_summary=profile_summary,
            weak_areas=weak,
            current_level=int(current_level),
            count=3,
        )

        return {
            "growth_summary": self._ability_profile.get_growth_summary(),
            "weak_areas": weak,
            "suggestion": suggestion,
            "challenges": challenges,
            "skill_stats": self._skill_library.get_stats(),
        }

    def get_evolution_report(self) -> dict:
        """获取进化状态报告"""
        if not self.enable_evolution:
            return {"error": "进化层未启用"}
        return {
            "growth": self._ability_profile.get_growth_summary(),
            "category_stats": self._ability_profile.get_all_category_stats(),
            "weak_areas": self._ability_profile.get_weak_areas(),
            "skill_count": self._skill_library.get_stats()["total_skills"],
            "post_mortem_count": len(self._post_mortem.history),
            "recent_insights": self._post_mortem.get_recent_insights(3),
        }

    def decompose_and_run(self, task: str, max_subtasks: int = 4) -> str:
        """将复杂任务拆解为子任务，并行执行后合并结果

        用于突破单 Agent 处理大规模任务的瓶颈。
        """
        if not self._orchestrator:
            return "[ERROR] Orchestrator 未启用 (enable_orchestrate=False)"

        # 用 LLM 拆解任务
        decompose_prompt = [
            Message(role="system", content="将以下任务拆解为几个独立的子任务。只输出 JSON 数组，每个元素是 {\"task\": \"描述\"}。"),
            Message(role="user", content=f"任务: {task}\n拆成最多 {max_subtasks} 个子任务。"),
        ]
        try:
            resp = self.llm.generate(decompose_prompt, tools=None)
            subtasks = self._parse_subtasks(resp.content or "")
        except Exception:
            return f"[ERROR] 任务拆解失败"

        if not subtasks:
            return f"[ERROR] 无法拆解任务"

        # 并行执行（传入写文件工具）
        results = self._orchestrator.fan_out(
            subtasks,
            tool_allowlist=["read_file", "write_file", "list_dir", "run_shell", "calculate"],
        )

        # 合并结果
        parts = []
        for r in results:
            if r.get("error"):
                parts.append(f"[FAIL] {r['task'][:60]}: {r['error'][:100]}")
            else:
                parts.append(f"[OK] {r['task'][:60]}: {r['result'][:200]}")

        return f"拆解为 {len(subtasks)} 个子任务并行执行:\n" + "\n".join(parts)

    @staticmethod
    def _parse_subtasks(text: str) -> list[dict]:
        import json
        text = text.strip()
        if text.startswith("```"): text = text.split("\n",1)[-1].rsplit("```",1)[0]
        try:
            data = json.loads(text)
            if isinstance(data, list):
                return data
            if isinstance(data, dict) and "subtasks" in data:
                return data["subtasks"]
        except json.JSONDecodeError:
            pass
        return []

    def _restore_from_checkpoint(self, ckpt: dict):
        """从检查点恢复状态"""
        self._self_optimize_retry_count = ckpt.get("self_optimize_retry_count", 0)
        self._meta_optimize_count = ckpt.get("meta_optimize_count", 0)
        self._last_step_count = ckpt.get("last_step_count", 0)

    def _try_architect_evolve(self, task_desc: str) -> dict:
        """尝试架构自进化：诊断 → 生成改动 → 应用+重载 → 验证 → 保留/回滚

        Returns:
            { applied, validated, bottleneck_type, target_file, rationale }
        """
        capability_gap = f"Agent 在任务「{task_desc[:100]}」上持续失败，所有 Config 层优化已穷尽。"

        # 1. 诊断
        bottleneck = self._arch_bottleneck.diagnose(self, capability_gap)
        if bottleneck.get("confidence", 0) < 0.4:
            return {"applied": False, "validated": False, "reason": "低置信度", "bottleneck": bottleneck}

        # 2. 生成改动
        proposal = self._arch_proposer.generate_proposal(bottleneck)
        if not proposal:
            return {"applied": False, "validated": False, "reason": "无法生成有效方案", "bottleneck": bottleneck}

        # 3. 应用 + 重载模块
        applied = self._arch_applier.apply_and_reload(proposal, self)
        if not applied:
            return {"applied": False, "validated": False, "reason": "应用失败", "bottleneck": bottleneck}

        # 4. 自重启以加载新代码（写入检查点→替换进程）
        try:
            AgentCheckpoint.trigger_restart(self, task_desc, bottleneck["bottleneck_type"])
            # 如果 os.execv 成功，不会执行到这里
            return {"applied": True, "validated": True, "restart": True,
                    "bottleneck_type": bottleneck["bottleneck_type"],
                    "rationale": proposal.get("rationale", "")}
        except Exception:
            # 回退：进程无法重启，尝试回滚
            self._arch_applier.rollback_last()
            return {"applied": True, "validated": False,
                    "reason": "重启失败，已回滚",
                    "bottleneck_type": bottleneck["bottleneck_type"],
                    "action": "rolled_back"}
