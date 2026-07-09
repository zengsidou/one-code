# -*- coding: utf-8 -*-
"""Orchestrator — 多 Agent 编排：fan-out 并行、pipeline 串行、共享上下文"""
from concurrent.futures import ThreadPoolExecutor, as_completed

from llm.base import BaseLLM
from tools.registry import ToolRegistry
from agent.subagent import SubAgent
from agent.models import Message


# 任务类型 → 推荐工具白名单
TOOL_PRESETS = {
    "read": ["read_file", "grep", "glob", "list_dir", "lsp_def", "lsp_refs", "lsp_hover"],
    "edit": ["read_file", "edit_file", "write_file", "diff_file", "lsp_diag"],
    "full": ["read_file", "edit_file", "write_file", "grep", "glob", "list_dir",
             "run_shell", "diff_file", "git", "lsp_def", "lsp_refs", "lsp_diag", "lsp_hover"],
    "shell": ["run_shell", "read_file", "list_dir"],
}


class AgentOrchestrator:
    def __init__(self, llm: BaseLLM, full_registry: ToolRegistry):
        self.llm = llm
        self.full_registry = full_registry
        self._shared_context: list[Message] = []

    def set_shared_context(self, messages: list[Message]):
        """设置所有子 Agent 共享的上下文（项目结构、架构概览等）"""
        self._shared_context = list(messages)

    def fan_out(
        self,
        tasks: list[dict],
        tool_allowlist: list[str] | str | None = None,
        shared_context: bool = True,
        max_workers: int = 4,
    ) -> list[dict]:
        """并行派发多个子任务

        Args:
            tasks: [{"task": "..."}, ...]
            tool_allowlist: 工具名列表 或 "read"/"edit"/"full"/"shell" 预设
            shared_context: 是否注入共享上下文
            max_workers: 最大并行数

        Returns:
            [{task, result, error}, ...]
        """
        allowlist = self._resolve_allowlist(tool_allowlist)
        sub_registry = self._build_subset_registry(allowlist)
        shared = list(self._shared_context) if shared_context and self._shared_context else []

        def run_one(task: dict) -> dict:
            desc = task.get("task", task.get("description", str(task)))
            try:
                sub = SubAgent(
                    llm=self.llm, registry=sub_registry,
                    context=shared, max_steps=6,
                )
                result = sub.run(desc)
                return {"task": desc, "result": result, "error": None}
            except Exception as e:
                return {"task": desc, "result": None, "error": str(e)}

        results = []
        workers = min(len(tasks), max(1, max_workers))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(run_one, t): t for t in tasks}
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as e:
                    t = futures[future]
                    results.append({"task": t.get("task", str(t)), "result": None, "error": str(e)})
        return results

    def fan_out_and_merge(
        self,
        tasks: list[dict],
        merge_prompt: str = "",
        tool_allowlist: list[str] | str | None = None,
        shared_context: bool = True,
    ) -> dict:
        """并行执行多个子任务，然后用 LLM 合成最终结果

        Returns:
            {tasks: [...], merged: "合成后的总结", conflicts: [...]}
        """
        sub_results = self.fan_out(tasks, tool_allowlist=tool_allowlist,
                                   shared_context=shared_context)

        # 检测潜在冲突（多个 agent 修改了同一文件）
        conflicts = []
        touched_files: dict[str, list[str]] = {}
        for r in sub_results:
            task_name = r["task"][:60]
            for line in (r.get("result") or "").split("\n"):
                for kw in ["edit_file", "write_file", "delete_file", "rename_file"]:
                    if kw in line:
                        parts = line.split()
                        for p in parts:
                            if p.endswith((".py", ".js", ".ts", ".html", ".json", ".yaml", ".md")):
                                touched_files.setdefault(p, []).append(task_name)
        for f, agents in touched_files.items():
            if len(agents) >= 2:
                conflicts.append({"file": f, "agents": agents})

        # LLM 合成
        merge_input = merge_prompt or "请基于以下并行子任务的结果，给出一份综合总结。说明：完成了什么、潜在冲突、下一步建议。"
        summary_parts = []
        for i, r in enumerate(sub_results):
            status = "成功" if r["error"] is None else f"失败({r['error']})"
            summary_parts.append(f"## 子任务{i+1}: {r['task'][:80]}\n状态: {status}\n结果: {(r.get('result') or '')[:300]}")
        merge_full = merge_input + "\n\n" + "\n---\n".join(summary_parts) + "\n\n冲突警告: " + (str(conflicts) if conflicts else "无")

        try:
            resp = self.llm.generate(
                [Message(role="user", content=merge_full[:4000])],
                tools=None,
            )
            merged = (resp.content or "")[:1000]
        except Exception:
            merged = f"子任务完成: {len(sub_results)} 个, {sum(1 for r in sub_results if r['error'] is None)} 成功"

        return {"tasks": sub_results, "merged": merged, "conflicts": conflicts}

    def pipeline(self, tasks: list[dict], tool_allowlist: list[str] | str | None = None) -> list[dict]:
        """串行执行子任务，每个子任务可以看到前一个的结果"""
        allowlist = self._resolve_allowlist(tool_allowlist)
        sub_registry = self._build_subset_registry(allowlist)
        results = []
        context = ""

        for task in tasks:
            desc = task.get("task", task.get("description", str(task)))
            if context:
                desc = f"前一个子任务的结果: {context[:500]}\n\n当前任务: {desc}"
            try:
                sub = SubAgent(llm=self.llm, registry=sub_registry, max_steps=6)
                result = sub.run(desc)
                context = result
                results.append({"task": desc, "result": result, "error": None})
            except Exception as e:
                results.append({"task": desc, "result": None, "error": str(e)})

        return results

    @staticmethod
    def _resolve_allowlist(tool_allowlist: list[str] | str | None) -> list[str]:
        if tool_allowlist is None:
            return TOOL_PRESETS["full"]
        if isinstance(tool_allowlist, str):
            return TOOL_PRESETS.get(tool_allowlist, TOOL_PRESETS["read"])
        return tool_allowlist

    def _build_subset_registry(self, tool_names: list[str]) -> ToolRegistry:
        sub_registry = ToolRegistry(safe_mode=False)
        for name in tool_names:
            if name in self.full_registry._tools:
                sub_registry._tools[name] = self.full_registry._tools[name]
                if name in self.full_registry._tool_metadata:
                    sub_registry._tool_metadata[name] = self.full_registry._tool_metadata[name]
        return sub_registry
