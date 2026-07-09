# -*- coding: utf-8 -*-
"""Agent Loop — ReAct 循环 + 熔断 + 循环检测"""
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
from agent.orchestrator import AgentOrchestrator
from agent.checkpoint import AgentCheckpoint
from agent.contract_first import ContractFirstOrchestrator
from agent.goal_verifier import GoalVerifier
from agent.constraints import ConstraintEnforcer
from agent.hooks import get_hooks


DEFAULT_SYSTEM_PROMPT = (
    "你是 One-Code，一个智能 AI 编程助手，底层模型是 DeepSeek V4 Pro。\n"
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
    "- [硬约束] 回答任何问题前，必须先 search memory + history。用户的偏好、凭据、项目规则都在记忆里，上下文可能不完整，禁止凭当前上下文片段猜测\n"
    "- [硬约束] 用户说\"读图\"/\"看剪贴板\"/发截图时，自动 OCR：保存剪贴板到 C:\\Users\\Alice\\clipboard.png，然后 node \"D:\\MiMo-Code\\.mimocode\\tool\\ocr-worker.cjs\" 读文字。禁止说\"不支持图片\"\n"
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
        llm_type: str = "deepseek",
        deepseek_api_key: str | None = None,
        enable_orchestrate: bool = False,
        loop_detect_threshold: int = 3,
        plan_first: bool = False,
        enable_contract_first: bool = False,
        observability=None,
        rules=None,
        token_optimizer=None,
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
        self._token_opt = token_optimizer
        self._goal_verifier = GoalVerifier(self.llm)
        self._constraints = ConstraintEnforcer()
        self._error_count = 0
        self._max_errors = 5
        self._orchestrator = AgentOrchestrator(self.llm, self.registry) if enable_orchestrate else None
        self.enable_contract_first = enable_contract_first
        self._contract_orchestrator = ContractFirstOrchestrator(self.llm) if enable_contract_first else None
        self._step_trace: list[str] = []
        self._last_step_count = 0

    def run(self, user_input: str, debug: bool = False, boot_context: list[Message] | None = None) -> str:
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
        self._step_trace = []
        self._last_step_count = 0

        if boot_context and not self._step_trace:
            for msg in boot_context:
                self.memory.add_message(msg)

        self.memory.add_message(Message(role="user", content=user_input))

        if self.observability:
            self.observability.start_run(user_input, "plan_execute" if self._plan_first else "react")

        get_hooks().fire("agent.start", task=user_input)

        search_triggers = ["搜索", "查一下", "查查", "查", "search", "最新",
                          "教程", "文档", "官方", "是什么", "什么是", "有哪些",
                          "介绍一下", "fetch"]
        is_code_task = any(kw in user_input for kw in ["修复", "fix", "修改", "改", "写", "创建", "实现", "debug"])
        if any(kw in user_input for kw in search_triggers) and not is_code_task:
            self.memory.add_message(Message(
                role="system",
                content="[前置指令] 先调用 search_web 搜索相关信息，找到可靠来源后用 fetch_url 读取详细内容，然后基于实际搜索结果回答。不要凭记忆编造。",
            ))

        if self.enable_contract_first:
            result = self._contract_and_execute(user_input, debug)
        elif self._plan_first:
            result = self._plan_and_execute(user_input, debug)
        else:
            result = self._run_loop(user_input, debug)

        if "[RESTART]" in result:
            return result

        if "[STOPPED]" not in result and "[ERROR]" not in result and "[SOFT-FAIL]" not in result and len(result) > 200:
            gv = self._goal_verifier.verify(user_input, result)
            if not gv.get("passed"):
                result = f"[GOAL-FAIL] {gv.get('reason', '')}"
                if debug:
                    print(f"  [GOAL] 独立 judge 判定未完成: {gv['reason']}")

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

            query = user_input[:200] if len(user_input) > 200 else user_input
            context = self.memory.get_context(query=query)
            system_prompt = self._build_system_prompt()
            # 注入运行时约束提示
            hint = self._constraints.get_constraint_hint()
            if hint:
                system_prompt += f"\n\n[运行时约束]\n{hint}"
            context.insert(0, Message(role="system", content=system_prompt))

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
                    self.memory.add_message(Message(role="assistant", content=response.content or "", reasoning_content=getattr(response, "reasoning_content", "") or ""))
                    self._step_trace.append(f"Step{step}: LOOP_DETECTED")
                    return "[STOPPED] 检测到重复工具调用回路，已中断。"

                for tc in tool_calls:
                    tool_msg_content = f"调用工具: {tc.name}({json.dumps(tc.arguments, ensure_ascii=False)})"
                    self._step_trace.append(f"Step{step}: call {tc.name}")
                    self.memory.add_message(Message(role="assistant", content=tool_msg_content, tool_calls=[tc], reasoning_content=getattr(response, "reasoning_content", "") or ""))
                    if self.observability:
                        self.observability.trace_tool_start(step, tc.name, tc.arguments)
                    result = self.registry.execute(tc.name, tc.arguments)
                    get_hooks().fire("tool.after", name=tc.name, args=tc.arguments, result=result)
                    constraint_hint = self._constraints.after_tool_call(
                        tc.name, tc.arguments, result,
                        self.memory.short_term.get_messages(),
                    )
                    if constraint_hint:
                        result = result + "\n" + constraint_hint
                    # ━━━ Token 优化：压缩工具输出 ━━━
                    if self._token_opt:
                        result = self._token_opt.compress_tool_output(result)
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
                self.memory.add_message(Message(role="assistant", content=content, reasoning_content=getattr(response, "reasoning_content", "") or ""))
                return content

            # 空回复 → 提示继续
            if not content.strip():
                self.memory.add_message(Message(role="assistant", content="", reasoning_content=getattr(response, "reasoning_content", "") or ""))
                continue

            self._step_trace.append(f"Step{step}: final_answer")
            self._last_step_count = step
            self.memory.add_message(Message(role="assistant", content=content, reasoning_content=getattr(response, "reasoning_content", "") or ""))
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
        """Context Reset: 清理上下文窗口，通过结构化交接文档保留关键状态。"""
        msgs = self.memory.short_term.get_messages()
        if len(msgs) < 6:
            return context

        # 优先使用 token_optimizer 的结构化摘要
        if self._token_opt:
            handoff = self._token_opt.summarize_work(msgs, current_task)
            handoff += "\n\n请基于以上信息继续完成任务。如需之前的具体代码，请重新读取相关文件。"
        else:
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
        self.memory.short_term.add(Message(role="system", content=handoff))
        self.memory.short_term.add(Message(role="user", content=current_task))
        return self.memory.short_term.get_messages()

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

    def _contract_and_execute(self, user_input: str, debug: bool = False) -> str:
        """契约先行模式：契约生成→确认→逆向拆解→执行

        1. 自动检测契约类型 + 生成多模态预览
        2. 用户确认或修改方向
        3. 从契约逆向拆解执行步骤
        4. 构建契约上下文，交给 ReAct 循环执行
        """
        if self._contract_orchestrator is None:
            return self._run_loop(user_input, debug)

        # Phase 1 & 2: 生成契约 + 用户确认
        contract = self._contract_orchestrator.phase1_detect_and_generate(user_input)
        if not self._contract_orchestrator.phase2_confirm():
            return "[CANCELLED] 用户取消"

        # Phase 3: 逆向拆解
        steps = self._contract_orchestrator.phase3_decompose(user_input)

        # Phase 4: 构建契约上下文提示并执行
        execution_prompt = self._contract_orchestrator.phase4_build_execution_prompt(
            user_input, contract, steps
        )

        self.memory.add_message(Message(
            role="system",
            content="[契约先行模式] 已生成最终产物方向预览，用户确认方向正确。请按步骤执行，每步完成后对照契约验证产出。",
        ))

        print()
        print("  [契约先行] 方向已确认，开始执行...")
        print()

        self.memory.add_message(Message(role="user", content=execution_prompt))
        result = self._run_loop(execution_prompt, debug)

        return result

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

    def _build_system_prompt(self) -> str:
        base = self.system_prompt + "\n\n" + (
            "工具调用时输出严格 JSON: "
            '{"tool": "工具名", "arguments": {"参数名": "参数值"}}'
        )
        if self._rules:
            rules_hint = self._rules.to_prompt_hint()
            if rules_hint:
                base += "\n\n" + rules_hint
        return base

    def _restore_from_checkpoint(self, ckpt: dict):
        """从检查点恢复状态"""
        self._last_step_count = ckpt.get("last_step_count", 0)

