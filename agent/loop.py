# -*- coding: utf-8 -*-
"""Agent Loop — ReAct 循环 + 熔断 + 循环检测 + 自优化闭环"""
import hashlib
import json
import os
import re
import shutil
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
    "你是 Micro-Agent，一个智能 AI 编程助手，底层模型是 DeepSeek V4 Pro。\n"
    "你的知识截止于 2025 年，对于不确定的信息会用 search_web 工具查询。\n\n"
    "══════ 核心工作流（每次任务必须遵守）══════\n"
    "1. 规划: 任务开始时不调用工具，先在脑中列出：要改哪些文件、顺序、怎么验证\n"
    "2. 探索: 用 grep/glob 搜索相关代码，用 read_file 读取关键文件。不要凭记忆猜测代码内容\n"
    "3. 修复: 优先用 edit_file 做精准替换，只在创建新文件时用 write_file\n"
    "4. 验证: 用 run_shell 运行验证（Python 用 'python xxx.py'，JS 用 'node xxx.js'）\n"
    "5. 迭代: 如果验证失败，仔细阅读错误输出，找出根因后回到步骤 2，不要盲目重试\n"
    "6. 完成: 验证通过后，用简短中文总结做了什么，停止调用工具\n"
    "══════════════════════════════════════════\n\n"
    "行为准则 — 必须遵守:\n"
    "- 优先用 edit_file，不要用 write_file 重写整个文件（除非创建新文件）\n"
    "- 收到 [ERROR] 后必须阅读内容、分析原因，换方法而不是重试相同操作\n"
    "- 同一工具相同参数连续 3 次无进展，换个思路\n"
    "- 不要猜测代码内容，先 read_file 确认再修改\n"
    "- 用户要求搜索/查资料时，必须调用 search_web，禁止凭记忆编造\n"
    "- 搜索后如需深读，用 fetch_url 抓取完整页面\n"
    "- 不确定的 API、版本号、命令语法，必须用 search_web 查证后再回答\n"
    "- 不要编造不存在的函数、库、配置。不知道就老实说不知道\n"
    "- 每个回答必须有依据：工具返回结果或搜索到的真实网页\n\n"
    "代码规范:\n"
    "- 不添加不必要的注释，让代码自解释\n"
    "- 不引入超出任务范围的抽象或重构\n"
    "- 不要为不可能发生的场景添加错误处理\n"
    "- 修改代码前先理解文件的代码风格，模仿现有模式\n\n"
    "Git 安全:\n"
    "- 绝对不要主动提交代码，除非用户明确要求\n"
    "- 不要修改 .gitignore 之外的 git 配置\n"
    "- 不要运行 git push --force 或 git reset --hard\n\n"
    "工具使用策略:\n"
    "- grep: 搜索代码内容，支持正则和文件过滤(include)\n"
    "- glob: 按模式查找文件，如 '**/*.py'\n"
    "- read_file: 读文件，支持 offset/limit 分页\n"
    "- edit_file: 精确替换字符串，old_string 必须唯一\n"
    "- write_file: 写新文件，会覆盖已有文件\n"
    "- run_shell: 执行命令，Python 用 'python file.py'，JS 用 'node file.js'\n"
    "- search_web: Bing 搜索，找到信息后用 fetch_url 深读\n"
    "- fetch_url: 抓取网页全文（Markdown 格式）\n"
    "- list_dir: 列出目录内容\n"
    "- delegate_task: 分配子任务给子 Agent 并行处理\n\n"
    "══════ 对话与协作能力 ══════\n"
    "你不只是代码修理工，更是用户的软件开发搭档。除了修 bug 和写代码，你还要：\n\n"
    "讨论与建议:\n"
    "- 用户问\"怎么实现X\"时，先分析需求，提出 1-3 种方案，说明各自优劣，再推荐\n"
    "- 用户问\"你觉得这样行吗\"时，给出诚实的判断，好的说好，有问题说问题\n"
    "- 有更好的思路时主动提出来，不要只是执行指令\n"
    "- 讨论架构时用具体的文件路径和代码结构说话，不要空谈\n\n"
    "解释与教学:\n"
    "- 用户问\"这段代码什么意思\"时，逐行解释逻辑，讲清楚为什么这样写\n"
    "- 解释概念时从实际代码出发，用类比帮助理解\n"
    "- 用户说\"教我X\"时，给出从简单到复杂的渐进式示例\n\n"
    "规划与设计:\n"
    "- 用户要开始一个大任务时，先帮他拆成子任务，估计每个子任务的工作量\n"
    "- 设计功能时考虑扩展性和维护性，但不要过度设计\n"
    "- 涉及多文件改动时，先列出影响范围再动手\n\n"
    "代码审查:\n"
    "- 用户让你审查代码时，按以下顺序检查：逻辑正确性 → 安全性 → 性能 → 可读性\n"
    "- 指出具体问题所在的文件和行号，给出修改建议\n"
    "- 区分\"必须修的 bug\"和\"可以优化的风格\"\n\n"
    "技术选型:\n"
    "- 用户问\"A 和 B 哪个好\"时，列出两者的实际对比（性能、生态、学习成本、适用场景）\n"
    "- 推荐方案时说明理由，不要只说结论\n"
    "- 不确定的信息用 search_web 查证后回答\n\n"
    "输出规范:\n"
    "- 回答要简洁，直接给出结论，避免冗长解释\n"
    "- 修复 bug 后只说修了什么、验证结果，不要叙述思考过程\n"
    "- 引用文件时用格式: `文件路径:行号`\n"
    "- 工具调用格式: {\"tool\": \"工具名\", \"arguments\": {...}}"
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
        loop_detect_threshold: int = 3,
        plan_first: bool = False,
        observability=None,
        rules=None,
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
        if self.memory and hasattr(self.memory, "short_term"):
            self.memory.short_term.set_llm(self.llm)
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self.max_steps = max_steps
        self._tool_fingerprints: list[str] = []
        self._loop_detect_threshold = loop_detect_threshold
        self._plan_first = plan_first
        self._current_plan: list[dict] = []
        self.observability = observability
        self._rules = rules
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

    def run(self, user_input: str, debug: bool = False, boot_context: list[Message] | None = None) -> str:
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

        # ━━━ 注入启动上下文（首次运行） ━━━
        if boot_context and not self._step_trace:
            for msg in boot_context:
                self.memory.add_message(msg)

        self.memory.add_message(Message(role="user", content=user_input))

        mode = "plan_execute" if self._plan_first else "react"
        if self.observability:
            self.observability.start_run(user_input, mode)

        # ━━━ 硬编码搜索触发 ━━━
        search_triggers = ["搜索", "查一下", "查查", "查", "search", "最新",
                          "教程", "文档", "官方", "是什么", "什么是", "介绍一下",
                          "介绍一下", "有哪些", "fetch"]
        is_code_task = any(kw in user_input for kw in ["修复", "fix", "修改", "改", "写", "创建", "实现", "debug"])
        if any(kw in user_input for kw in search_triggers) and not is_code_task:
            self.memory.add_message(Message(
                role="system",
                content="[前置指令] 先调用 search_web 搜索相关信息，找到可靠来源后用 fetch_url 读取详细内容，然后基于实际搜索结果回答。不要凭记忆编造。",
            ))

        if self._plan_first:
            result = self._plan_and_execute(user_input, debug)
        else:
            result = self._run_loop(user_input, debug)
        if "[STOPPED]" in result and self.enable_self_optimize:
            result = self._try_self_heal(user_input, debug)

        # 检查是否触发了自重启
        if "[RESTART]" in result:
            return result

        # 验证 Oracle：检查任务是否真正完成（软失败检测）
        if "[STOPPED]" not in result and self.enable_evolution:
            vf = self._verify_task(user_input)
            if vf and not vf.get("passed"):
                result = f"[SOFT-FAIL] {vf.get('reason', '验证失败')}"
                if debug:
                    print(f"  [VERIFY] 软失败: {vf['reason']}")

        # 进化层：每次执行后复盘 + 沉淀
        if self.enable_evolution:
            self._evolve_after_run(user_input, result)

        if self.observability:
            failed = "[STOPPED]" in result or "[SOFT-FAIL]" in result or "[LLM error]" in result
            self.observability.end_run(
                success=not failed,
                failure_type=("stopped" if "[STOPPED]" in result else
                              "soft_fail" if "[SOFT-FAIL]" in result else
                              "llm_error" if "[LLM error]" in result else ""),
            )
            self.observability.record_step(self._last_step_count)
            self.observability.estimate_tokens(self.memory.short_term.get_messages())

        return result

    def _run_loop(self, user_input: str, debug: bool = False) -> str:
        """内部执行循环，返回结果或 [STOPPED] 错误"""
        self._error_count = 0
        self._tool_fingerprints.clear()

        step = 0
        idle_steps = 0
        while True:
            step += 1
            if debug:
                print(f"  [DEBUG step={step}] fingerprints={self._tool_fingerprints[-5:]}")

            context = self.memory.get_context(query=user_input)
            context.insert(0, Message(role="system", content=self._build_system_prompt()))

            # ━━━ 上下文压力感知 ━━━
            token_count = self.memory.short_term.get_token_count()
            pressure = self._pressure_level(token_count)
            if pressure >= 2 and step > 3:
                context.append(Message(
                    role="system",
                    content=f"[上下文压力 {pressure}/3] 当前上下文 {token_count} tokens。请尽快完成当前任务并给出最终回答，不要再调用非必要工具。"
                ))
            if pressure >= 3:
                # Level 3 → Context Reset（对标 Anthropic 40% 阈值重启策略）
                if self.observability:
                    self.observability.record_step(step, pressure=3)
                if debug:
                    print(f"  [CONTEXT RESET] 压力等级 {pressure}，触发上下文重置")
                context = self._context_reset(user_input, context)

            # ━━━ 步数上限软提醒 ━━━
            if step >= self.max_steps - 3:
                context.append(Message(
                    role="system",
                    content=f"[步数提醒] 已执行 {step} 步，请在当前轮给出最终回答。"
                ))

            has_tool_results = any(
                m.role == "tool" for m in self.memory.short_term.get_messages()
            )
            if has_tool_results and step > 1:
                last_tool = next(
                    (m.content for m in reversed(self.memory.short_term.get_messages()) if m.role == "tool"),
                    ""
                )
                context.append(Message(
                    role="system",
                    content=(
                        "工具已返回结果。直接引用结果给用户简洁中文回复，不要模糊措辞。"
                        f"\n工具返回:\n{last_tool[:300]}"
                    ),
                ))

            response = self.llm.generate(context, tools=self.registry.get_schemas())
            tool_calls = response.tool_calls or []

            if tool_calls:
                if self._detect_tool_loop(tool_calls, debug):
                    if self.observability:
                        self.observability.record_loop_detection()
                    self.memory.add_message(Message(role="assistant", content=response.content or "", reasoning_content=response.reasoning_content))
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
                    self.memory.add_message(Message(role="assistant", content=tool_msg_content, tool_calls=[tc], reasoning_content=response.reasoning_content))
                    if self.observability:
                        self.observability.trace_tool_start(step, tc.name, tc.arguments)
                    result = self.registry.execute(tc.name, tc.arguments)
                    is_error = result.startswith("[ERROR]")
                    if self.observability:
                        self.observability.trace_tool_end(step, tc.name, result, error=is_error)
                    self.memory.add_message(Message(
                        role="tool", content=result,
                        tool_call_id=tc.id, tool_name=tc.name,
                    ))
                    if result.startswith("[ERROR]"):
                        self._step_trace.append(f"Step{step}: ERROR {tc.name}")
                        is_tool_crash = (
                            result.startswith("[ERROR] Unknown tool")
                            or result.startswith("[ERROR] Tool '")
                        )
                        if is_tool_crash:
                            self._error_count += 1
                            if self._error_count >= self._max_errors:
                                if self.enable_self_optimize:
                                    self._capture_failure(
                                        user_input, step,
                                        f"[STOPPED] 连续工具执行错误达到上限 ({self._max_errors})，已熔断。最后错误: {result}",
                                        "circuit_breaker",
                                    )
                                self._step_trace.append(f"Step{step}: CIRCUIT_BREAKER")
                                if self.observability:
                                    self.observability.record_circuit_breaker()
                                return f"[STOPPED] 连续工具执行错误达到上限 ({self._max_errors})，已熔断。最后错误: {result}"
                idle_steps = 0
                continue

            # 无工具调用
            idle_steps += 1
            content = response.content or ""

            # 连续 2 步无工具调用 → 最终回答
            if idle_steps >= 2 or step >= self.max_steps:
                self._step_trace.append(f"Step{step}: final_answer")
                self._last_step_count = step
                self.memory.add_message(Message(role="assistant", content=content, reasoning_content=response.reasoning_content))
                return content

            # 空回复 → 提示继续
            if not content.strip():
                self.memory.add_message(Message(role="assistant", content="", reasoning_content=response.reasoning_content))
                continue

            self._step_trace.append(f"Step{step}: final_answer")
            self._last_step_count = step
            self.memory.add_message(Message(role="assistant", content=content, reasoning_content=response.reasoning_content))
            return content

    @staticmethod
    def _pressure_level(token_count: int) -> int:
        """上下文压力等级: 0=智能区(<40%), 1=关注(40-60%), 2=警告(60-80%), 3=紧急(>80%)"""
        max_tokens = 65536
        ratio = token_count / max_tokens
        if ratio > 0.80: return 3
        if ratio > 0.60: return 2
        if ratio > 0.40: return 1
        return 0

    def _context_reset(self, current_task: str, context: list) -> list:
        """Context Reset: 清理上下文窗口，通过结构化交接文档保留关键状态。

        当压力达到 3 级时触发。流程:
        1. 提取当前任务状态、已完成工作、待办事项
        2. 清空短期记忆
        3. 注入精简的交接文档 → 新 Agent 从干净状态继续
        
        对标 Anthropic 的 context reset 策略。
        """
        msgs = self.memory.short_term.get_messages()
        if len(msgs) < 6:
            return context

        tool_results = [m.content for m in msgs if m.role == "tool" and m.content]
        user_query = next((m.content for m in msgs if m.role == "user" and m.content), current_task)

        handoff = (
            "[上下文重置] 为保持推理质量，上下文已清空。以下是之前工作的交接文档:\n\n"
            f"# 当前任务\n{user_query[:500]}\n\n"
            f"# 最近工具执行结果\n" +
            "\n".join(r[:200] for r in tool_results[-3:]) +
            "\n\n"
            f"# 步骤统计\n已执行约 {self._last_step_count} 步。\n\n"
            "请基于以上信息继续完成任务。如果需要之前的具体代码，请重新读取相关文件。"
        )

        self.memory.short_term._messages.clear()
        return [
            Message(role="system", content=handoff),
            Message(role="user", content=current_task),
        ]

    def _garbage_collect(self, workspace: str | None = None) -> None:
        """清理 Agent 运行时生成的冗余产物（对标 OpenAI 的后台清理 Agent）。

        清理内容: 临时文件、.pyc 缓存、超过 10 次运行的观测数据。
        """
        import glob as _glob
        ws = workspace or "."
        patterns = ["*.tmp", "*.bak", "*.py-generated", "__pycache__/**"]
        removed = 0
        for pattern in patterns:
            for f in _glob.glob(f"{ws}/{pattern}", recursive=True):
                try:
                    if os.path.isfile(f):
                        os.remove(f)
                        removed += 1
                    elif os.path.isdir(f):
                        shutil.rmtree(f, ignore_errors=True)
                        removed += 1
                except Exception:
                    pass

        if self.observability:
            history = self.observability.load_history()
            if len(history) > 10:
                # Keep only last 10
                for old in history[:-10]:
                    path = self.observability.save_dir / f"{old.get('run_id', 'old')}.json"
                    if path.exists():
                        try:
                            os.remove(path)
                        except Exception:
                            pass

        if removed and self.observability:
            print(f"  [GC] 清理了 {removed} 个冗余文件")

    def _build_system_prompt(self) -> str:
        if not hasattr(self, "_cached_tool_desc"):
            self._cached_tool_desc = self.registry.get_tools_description()
        tool_desc = self._cached_tool_desc
        base = (
            f"{self.system_prompt}\n\n"
            "---\n"
            f"可用工具:\n{tool_desc}\n\n"
            "工具调用时输出严格 JSON: "
            '{"tool": "工具名", "arguments": {"参数名": "参数值"}}'
        )
        # 注入已学技能 — 总是注入最强的3个（embedding为0时query无效）
        if self.enable_evolution and self._skill_library:
            top_skills = self._skill_library.get_top_skills(3)
            hint = self._skill_library.to_prompt_hint(top_skills)
            if hint:
                base += hint
        if self._rules:
            rules_hint = self._rules.inject_rules()
            if rules_hint:
                base += rules_hint
        return base

    def _plan_and_execute(self, user_input: str, debug: bool = False) -> str:
        """Plan-then-Execute 模式：先规划再执行。

        1. Plan phase：LLM 生成结构化执行计划
        2. Execute phase：按计划逐步执行，执行后验证
        3. 不偏离计划；遇到失败可触发 re-plan
        """
        plan = self._generate_plan(user_input, debug)
        if not plan:
            return "[STOPPED] 无法生成执行计划"

        if debug:
            print(f"  [PLAN] {len(plan)} 步: {[s['goal'][:40] for s in plan]}")

        results = []
        for i, step in enumerate(plan):
            if debug:
                print(f"  [EXECUTE {i+1}/{len(plan)}] {step['goal'][:60]}")

            sub_prompt = (
                f"整体任务: {user_input}\n\n"
                f"执行计划 ({len(plan)} 步):\n" +
                "\n".join(f"{j+1}. [{step['action']}] {step['goal']}"
                          for j, step in enumerate(plan)) +
                f"\n\n当前执行第 {i+1} 步: {step['goal']}\n"
                f"工具提示: {step.get('tools', '所有可用工具')}\n\n"
                f"只完成当前这一步，完成后报告结果。"
            )

            self.memory.add_message(Message(role="user", content=sub_prompt))
            step_result = self._run_loop(sub_prompt, debug)
            results.append({"step": i+1, "goal": step["goal"], "result": step_result[:500]})

            if "[STOPPED]" in step_result or "[ERROR]" in step_result:
                # Step failed — try re-plan remaining steps
                remaining = plan[i+1:]
                if remaining and i < len(plan) - 1:
                    if debug:
                        print(f"  [REPLAN] 第{i+1}步失败，重新规划剩余步骤...")
                    new_plan = self._generate_plan(
                        f"{user_input}\n已完成: {step['goal']} (失败: {step_result[:200]})",
                        debug,
                    )
                    if new_plan:
                        plan[i+1:] = new_plan

        all_outputs = "\n---\n".join(
            f"步骤{r['step']}: {r['goal']}\n{r['result']}" for r in results
        )

        summary_prompt = (
            f"原始任务: {user_input}\n\n"
            f"执行结果:\n{all_outputs[:4000]}\n\n"
            f"请给出最终回答（中文），总结完成情况和关键发现。"
        )
        self.memory.add_message(Message(role="user", content=summary_prompt))
        final = self._run_loop(summary_prompt, debug)
        return final

    def _generate_plan(self, task: str, debug: bool = False) -> list[dict] | None:
        """用 LLM 生成结构化执行计划。"""
        prompt = (
            "你是一个任务规划专家。将以下任务分解为可执行的步骤列表。\n\n"
            f"任务: {task}\n\n"
            "输出 JSON 数组，每个元素包含:\n"
            '- action: 动作类型 (read/search/edit/verify/shell)\n'
            '- goal: 这一步要完成什么（中文，一句话）\n'
            '- tools: 建议使用的工具（可选）\n\n'
            "规则:\n"
            "- 每个步骤应该只做一件事\n"
            "- 信息收集（read/search）必须在修改（edit）之前\n"
            "- 修改后必须有验证步骤\n"
            "- 步骤数控制在 3-8 步\n"
            "- 第一个步骤应该是了解项目结构\n\n"
            "输出格式: [{\"action\": \"...\", \"goal\": \"...\", \"tools\": \"...\"}, ...]\n"
            "只输出 JSON 数组，不要有其他文字。"
        )
        try:
            messages = [
                Message(role="system", content="你是一个任务规划专家。只输出 JSON。"),
                Message(role="user", content=prompt),
            ]
            resp = self.llm.generate(messages, tools=None)
            text = (resp.content or "").strip()
            import re
            m = re.search(r"\[[\s\S]*\]", text)
            if m:
                plan = json.loads(m.group())
                if isinstance(plan, list) and len(plan) > 0:
                    return plan
        except Exception as e:
            if debug:
                print(f"  [PLAN FAIL] {e}")
        return None

    def _detect_tool_loop(self, tool_calls: list, debug: bool = False) -> bool:
        for tc in tool_calls:
            raw = f"{tc.name}:{json.dumps(tc.arguments, ensure_ascii=False, sort_keys=True)}"
            fp = hashlib.md5(raw.encode()).hexdigest()
            self._tool_fingerprints.append(fp)
            if debug:
                print(f"  [DEBUG detect] tool={tc.name} args={tc.arguments} fp={fp[:12]}")
            # 滑动窗口：同一指纹出现次数达到阈值才触发
            window = self._tool_fingerprints[-max(8, self._loop_detect_threshold * 2):]
            if window.count(fp) >= self._loop_detect_threshold:
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

        # ━━━ AGENTS.md 规则自动积累 ━━━
        if self._rules:
            trace = "\n".join(self._step_trace[-8:]) if self._step_trace else ""
            self._rules.learn_from_failure(
                task=task_desc, error=error_msg, trace=trace,
            )

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
        step_count = self._last_step_count

        # Token优化：仅1-2步成功任务跳过 LLM 复盘
        if not is_failure and step_count <= 2:
            reflection = {
                "outcome": "success",
                "difficulty_for_agent": 1,
                "difficulty_evidence": "1-2步完成，轻量任务",
                "what_worked": ["路径直接，无冗余操作"],
                "what_could_be_better": [],
                "strategy_used": "直接定位并修复",
                "new_skill_gained": {"name": "", "description": "", "reusable": False, "trigger": "", "steps": ""},
                "efficiency_score": 5,
                "efficiency_evidence": "无冗余步数",
                "growth_insight": f"快速定位并修复: {task_desc[:60]}",
                "task_desc": task_desc[:200],
                "timestamp": "",
                "result_preview": result[:100],
            }
            self._post_mortem.history.append(reflection)
        else:
            # 构建能力上下文
            ability_ctx = "Agent 当前能力水平: "
            if self._ability_profile and len(self._ability_profile.records) > 0:
                stats = self._ability_profile.get_growth_summary(window=10)
                ability_ctx += (
                    f"已完成 {stats['total_tasks']} 个任务，"
                    f"成功率 {stats['recent_success_rate']:.0%}，"
                    f"平均难度 {stats['recent_avg_diff']}/5"
                )
            else:
                ability_ctx += "这是 Agent 的早期任务，尚在建立基准。"

            reflection = self._post_mortem.reflect(
                task_desc, result, trace,
                step_count=step_count,
                ability_context=ability_ctx,
            )
        # 1. 复盘

        # 2. 提取技能
        self._skill_library.add_from_post_mortem(reflection)

        # ━━━ AGENTS.md 规则自动积累（非完美运行） ━━━
        if self._rules and (is_failure or reflection.get("efficiency_score", 5) < 4):
            trace = "\n".join(self._step_trace[-8:]) if self._step_trace else ""
            self._rules.learn_from_failure(
                task=task_desc,
                error=result[:300] if is_failure else reflection.get("what_could_be_better", [""])[0],
                trace=trace,
            )

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
            pattern = self._post_mortem.detect_repeating_weakness(min_occurrences=2, window=8)
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
        current_level = suggestion.get("suggested_level", suggestion.get("current_level", 2)) if suggestion else 2

        challenges = self._challenge_gen.generate(
            profile_summary=profile_summary,
            weak_areas=weak,
            current_level=int(current_level),
            count=3,
        )

        # 为每个挑战生成 buggy fixture 文件
        for ch in challenges:
            fixture = self._challenge_gen.create_fixture(
                ch.get("task", ""), ch.get("difficulty", 2)
            )
            if fixture:
                ch["fixture"] = fixture

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
        """尝试架构自进化：诊断 → 生成改动 → 应用+自测试 → 保留/回滚+重试

        自测试闭环：apply 后自动跑 pytest，失败则回滚，最多重试 3 次。
        每次重试将失败信息反馈给 LLM 以改进方案。

        Returns:
            { applied, validated, bottleneck_type, target_file, rationale, self_test }
        """
        capability_gap = f"Agent 在任务「{task_desc[:100]}」上持续失败，所有 Config 层优化已穷尽。"

        # 1. 诊断
        bottleneck = self._arch_bottleneck.diagnose(self, capability_gap)
        if bottleneck.get("confidence", 0) < 0.4:
            return {"applied": False, "validated": False, "reason": "低置信度", "bottleneck": bottleneck}

        # ━━━ 系统提示自优化（特殊路径：不改结构，只改 prompt 字符串）━━━
        if bottleneck.get("bottleneck_type") == "prompt_inadequate":
            proposal = self._arch_proposer.generate_proposal(bottleneck)
            if not proposal:
                return {"applied": False, "validated": False, "reason": "无法生成提示优化方案"}
            new_code = proposal.get("new_code", "")
            if not new_code:
                return {"applied": False, "validated": False, "reason": "生成的提示为空"}
            # 提取新提示文本（去掉 DEFAULT_SYSTEM_PROMPT = ( 和末尾的 )）
            import re
            m = re.search(r'DEFAULT_SYSTEM_PROMPT = \((.*)\)', new_code, re.DOTALL)
            if m:
                new_prompt_text = m.group(1).strip()
                # 直接更新运行时 prompt（无需重启）
                global DEFAULT_SYSTEM_PROMPT
                DEFAULT_SYSTEM_PROMPT = new_prompt_text
                self.system_prompt = new_prompt_text
                # 清除缓存以使用新提示
                if hasattr(self, "_cached_tool_desc"):
                    delattr(self, "_cached_tool_desc")
                print(f"  [L4-PROMPT] 系统提示已自动优化 ✓")
                return {"applied": True, "validated": True, "self_test": "skipped",
                        "bottleneck_type": "prompt_inadequate",
                        "rationale": proposal.get("rationale", "提示优化")}
            return {"applied": False, "validated": False, "reason": "无法解析新提示文本"}

        # 2. 生成 + 应用 + 自测试（最多重试 3 次）
        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            proposal = self._arch_proposer.generate_proposal(bottleneck)
            if not proposal:
                return {"applied": False, "validated": False, "reason": "无法生成有效方案", "bottleneck": bottleneck}

            result = self._arch_applier.apply_and_reload(proposal, self, run_tests=True)

            if result["success"] and result["test_result"] == "passed":
                print(f"  [L4-SELF-TEST] 第{attempt}次尝试通过 ✓")
                # 自重启以加载新代码（写入检查点→替换进程）
                try:
                    AgentCheckpoint.trigger_restart(self, task_desc, bottleneck["bottleneck_type"])
                    return {"applied": True, "validated": True, "restart": True,
                            "bottleneck_type": bottleneck["bottleneck_type"],
                            "rationale": proposal.get("rationale", ""),
                            "self_test": "passed", "attempts": attempt}
                except Exception:
                    self._arch_applier.rollback_last()
                    return {"applied": True, "validated": False,
                            "reason": "重启失败，已回滚",
                            "self_test": "passed"}

            # 测试失败 — 反馈给 LLM 后重试
            test_output = result.get("test_output", "")
            print(f"  [L4-SELF-TEST] 第{attempt}次尝试失败，回滚并重试...")
            if attempt < max_attempts:
                bottleneck["retry_hint"] = (
                    f"上次改动（第{attempt}次）导致 pytest 测试失败。"
                    f"失败摘要:\n{test_output[:300]}\n"
                    f"请换一个方案或修复引入的问题。"
                )

        return {"applied": False, "validated": False,
                "reason": f"自测试失败 {max_attempts} 次后放弃",
                "bottleneck_type": bottleneck["bottleneck_type"]}

    def _verify_task(self, user_input: str) -> dict | None:
        """验证任务是否真正完成。运行修改过的 Python/JS 文件，检查是否有运行时错误。

        只在任务涉及代码修复/编写时验证。搜索/分析类任务返回 None 跳过。

        Returns:
            {"passed": True} 或 {"passed": False, "reason": "..."} 或 None (无法/无需验证)
        """
        # 判断任务类型 — 只看代码修复/编写类任务
        code_keywords = ["修复", "fix", "修改", "编写", "实现", "创建", "添加", "debug", "调试"]
        if not any(kw in user_input.lower() for kw in code_keywords):
            return None

        from tools.registry import run_shell

        msgs = self.memory.short_term.get_messages()
        modified_files = set()
        import re

        for msg in msgs:
            if msg.role != "tool":
                continue
            content = msg.content or ""
            for line in content.split("\n"):
                if "File written:" in line:
                    fpath = line.split("File written:")[1].strip().split(" ")[0].strip()
                    modified_files.add(fpath)
                elif "已替换" in line:
                    m = re.search(r'(?:已替换|replaced)\s+(\S+)', line)
                    if m:
                        modified_files.add(m.group(1))

        if not modified_files:
            return None

        for fpath in list(modified_files)[:5]:
            ext = os.path.splitext(fpath)[1].lower()
            try:
                if ext == ".py":
                    result = run_shell(f"python \"{fpath}\"", timeout=10)
                    if "[ERROR]" in result or "Traceback" in result or "SyntaxError" in result or "Error: " in result:
                        return {"passed": False, "reason": f"运行 {fpath} 失败: {result[:200]}"}
                elif ext == ".js":
                    result = run_shell(f"node \"{fpath}\"", timeout=10)
                    if "[ERROR]" in result or "Error: " in result or "SyntaxError" in result:
                        return {"passed": False, "reason": f"运行 {fpath} 失败: {result[:200]}"}
                elif ext == ".json":
                    with open(fpath, encoding="utf-8") as f:
                        json.loads(f.read())
                elif ext in (".html", ".htm"):
                    with open(fpath, encoding="utf-8", errors="replace") as f:
                        html = f.read()
                        if not any(tag in html.lower() for tag in ("<!doctype", "<html", "<head", "<body")):
                            return {"passed": False, "reason": f"{fpath} 不像是有效的 HTML 文件"}
                elif ext in (".yaml", ".yml"):
                    import yaml
                    with open(fpath, encoding="utf-8") as f:
                        yaml.safe_load(f.read())
            except json.JSONDecodeError as e:
                return {"passed": False, "reason": f"{fpath} JSON 语法错误: {e}"}
            except Exception as e:
                if ext in (".yaml", ".yml"):
                    return {"passed": False, "reason": f"{fpath} YAML 语法错误: {e}"}

        return {"passed": True}
