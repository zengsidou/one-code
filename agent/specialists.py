# -*- coding: utf-8 -*-
"""Specialized agent roles — multi-agent division of labor.

Maps to JD: "数字员工方案从 Demo 走向稳定运行"
Aligns with Anthropic 3-agent architecture + Carlini's 16-agent specialization.

Roles:
  Coder:    write/edit code, fix bugs
  Reviewer: review code for issues, security, style
  Tester:   write and run tests
  Doc:      generate documentation
"""
import json
import re
from agent.models import Message


CODER_SYSTEM_PROMPT = (
    "你是 CoderAgent，一个专注代码编写和修复的 Agent。\n\n"
    "你的职责:\n"
    "1. 理解需求后直接编写/修改代码\n"
    "2. 遵循项目现有的代码风格和约定\n"
    "3. 不添加不必要的注释、不引入额外的抽象\n"
    "4. 修改后输出 [DONE] 标记结束\n\n"
    "不要做代码审查、不要写测试、不要写文档。只写代码。"
)

REVIEWER_SYSTEM_PROMPT = (
    "你是 ReviewerAgent，一个独立的代码审查 Agent。你的评价不受生成 Agent 影响。\n\n"
    "审查维度:\n"
    "1. 逻辑正确性: 代码是否能解决描述的问题？\n"
    "2. 安全性: SQL 注入、硬编码密钥、路径遍历、命令注入\n"
    "3. 性能: 不必要的循环、内存泄漏、N+1 查询\n"
    "4. 可读性: 命名、结构、无冗余代码\n"
    "5. 规范遵循: 是否与项目现有风格一致？\n\n"
    "输出 JSON review 报告: {\"issues\": [{\"severity\":\"high|medium|low\", \"file\":\"\", \"line\":N, \"description\":\"\", \"suggestion\":\"\"}], \"score\": 0-5, \"verdict\": \"approve|changes_requested|reject\"}"
)

TESTER_SYSTEM_PROMPT = (
    "你是 TesterAgent，一个专注编写和运行测试的 Agent。\n\n"
    "你的职责:\n"
    "1. 阅读需要测试的代码\n"
    "2. 编写针对性的测试用例（边界情况、错误路径、正常路径）\n"
    "3. 运行测试并报告结果\n"
    "4. 输出 JSON: {\"tests_written\": N, \"tests_passed\": N, \"tests_failed\": N, \"failures\": [...]}"
)

DOC_SYSTEM_PROMPT = (
    "你是 DocAgent，一个专注生成项目文档的 Agent。\n\n"
    "你的职责:\n"
    "1. 分析代码结构和功能\n"
    "2. 生成简洁有用的文档\n"
    "3. 只写必要信息，不写废话\n\n"
    "输出中文文档。"
)


class Specialist:
    """专业化 Agent，携带特定角色提示和工具限制。

    Usage:
        coder = Specialist(llm, "coder", registry)
        result = coder.run("修复登录页面的空指针异常")
        
        reviewer = Specialist(llm, "reviewer", registry)
        review = reviewer.run(f"审查这段代码:\n{code}")
    """

    ROLES = {
        "coder": CODER_SYSTEM_PROMPT,
        "reviewer": REVIEWER_SYSTEM_PROMPT,
        "tester": TESTER_SYSTEM_PROMPT,
        "doc": DOC_SYSTEM_PROMPT,
    }

    TOOL_ALLOWLISTS = {
        "coder": ["read_file", "write_file", "edit_file", "list_dir", "grep", "run_shell"],
        "reviewer": ["read_file", "list_dir", "grep"],
        "tester": ["read_file", "write_file", "run_shell", "list_dir", "grep"],
        "doc": ["read_file", "list_dir", "grep", "write_file"],
    }

    def __init__(self, llm_adapter, role: str, base_registry=None, max_steps: int = 10):
        """
        Args:
            llm_adapter: BaseLLM 实例
            role: "coder" | "reviewer" | "tester" | "doc"
            base_registry: 基础 ToolRegistry（会被子集过滤）
            max_steps: 最大步数
        """
        self.llm = llm_adapter
        self.role = role
        self.max_steps = max_steps

        if role not in self.ROLES:
            raise ValueError(f"Unknown role: {role}. Choices: {list(self.ROLES)}")

        self.system_prompt = self.ROLES[role]
        self.tool_allowlist = self.TOOL_ALLOWLISTS.get(role, [])
        self._filtered_registry = self._build_filtered_registry(base_registry)

    def _build_filtered_registry(self, base_registry):
        """从完整 registry 中过滤出该角色允许的工具。"""
        if base_registry is None:
            return None

        from tools.registry import ToolRegistry
        filtered = ToolRegistry(safe_mode=base_registry.safe_mode,
                                permissions=base_registry.permissions,
                                audit=base_registry.audit)

        for name in self.tool_allowlist:
            if name in base_registry._tools:
                func = base_registry._tools[name]
                meta = base_registry._tool_metadata.get(name, {})
                filtered._tools[name] = func
                filtered._tool_metadata[name] = meta

        return filtered

    def run(self, task: str, context: str = "", debug: bool = False) -> str:
        """执行专业化任务。

        Args:
            task: 任务描述
            context: 前置上下文（如 pipeline 中前一步的结果）

        Returns:
            Agent 的输出
        """
        from memory.short_term import ShortTermMemory
        from memory.long_term import LongTermMemory
        from memory import MemoryManager

        short = ShortTermMemory(max_tokens=65536)
        long = LongTermMemory(llm=self.llm, collection_name=f"specialist_{self.role}")
        memory = MemoryManager(short=short, long=long)

        full_task = f"{context}\n\n{task}" if context else task
        memory.add_message(Message(role="user", content=full_task))

        from agent.loop import AgentLoop
        agent = AgentLoop(
            llm=self.llm,
            registry=self._filtered_registry,
            memory=memory,
            max_steps=self.max_steps,
            system_prompt=self.system_prompt,
            enable_self_optimize=False,
            enable_evolution=False,
        )
        return agent.run(full_task, debug=debug)


class SpecialistPipeline:
    """专业化流水线 — 依次执行编码→审查→测试→文档。

    Usage:
        pipeline = SpecialistPipeline(llm, registry)
        results = pipeline.run("实现用户登录功能")
        # results["coder"], results["reviewer"], results["tester"], results["doc"]
    """

    DEFAULT_PIPELINE = ["coder", "reviewer", "coder", "tester", "doc"]
    # 编码 → 审查 → 修审查问题 → 测试 → 文档

    def __init__(self, llm_adapter, base_registry):
        self.llm = llm_adapter
        self.base_registry = base_registry

    def run(self, task: str, roles: list[str] | None = None, debug: bool = False) -> dict[str, str]:
        """按序执行专业化角色。

        Args:
            task: 原始任务
            roles: 角色序列，默认 ["coder", "reviewer", "coder", "tester", "doc"]

        Returns:
            {role_name: output}
        """
        roles = roles or self.DEFAULT_PIPELINE
        context = ""
        results = {}

        for role in roles:
            if debug:
                print(f"  [PIPELINE] {role}...")

            specialist = Specialist(self.llm, role, self.base_registry)
            role_task = task

            if role == "reviewer":
                role_task = f"审查以下代码修复:\n{results.get('coder', context)[:4000]}"
            elif role == "tester":
                role_task = f"为以下修复编写测试:\n{results.get('coder', context)[:4000]}"
            elif role == "doc":
                role_task = f"为以下实现编写文档:\n{results.get('coder', context)[:3000]}"

            result = specialist.run(role_task, context=context, debug=debug)
            results[role] = result
            context = result[:2000]

        return results

    def run_parallel(self, tasks: dict[str, str], debug: bool = False) -> dict[str, str]:
        """并行执行不同角色的任务（fan-out）。

        Args:
            tasks: {role: task_description}

        Returns:
            {role: output}
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        results = {}
        with ThreadPoolExecutor(max_workers=min(len(tasks), 4)) as executor:
            futures = {}
            for role, task in tasks.items():
                specialist = Specialist(self.llm, role, self.base_registry)
                futures[executor.submit(specialist.run, task, debug=debug)] = role

            for future in as_completed(futures):
                role = futures[future]
                try:
                    results[role] = future.result()
                except Exception as e:
                    results[role] = f"[ERROR] {e}"

        return results
