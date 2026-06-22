# -*- coding: utf-8 -*-
"""Orchestrator — 多 Agent 编排，并行/串行派发子任务"""
from concurrent.futures import ThreadPoolExecutor, as_completed

from llm.base import BaseLLM
from tools.registry import ToolRegistry
from agent.subagent import SubAgent


class AgentOrchestrator:
    def __init__(self, llm: BaseLLM, full_registry: ToolRegistry):
        self.llm = llm
        self.full_registry = full_registry

    def fan_out(self, tasks: list[dict], tool_allowlist: list[str] | None = None) -> list[dict]:
        """并行派发多个子任务，每个子任务返回 {task, result, error}"""
        sub_registry = self._build_subset_registry(tool_allowlist or ["read_file", "list_dir", "calculate"])

        def run_one(task: dict) -> dict:
            desc = task.get("task", task.get("description", str(task)))
            try:
                sub = SubAgent(llm=self.llm, registry=sub_registry, max_steps=5)
                result = sub.run(desc)
                return {"task": desc, "result": result, "error": None}
            except Exception as e:
                return {"task": desc, "result": None, "error": str(e)}

        results = []
        with ThreadPoolExecutor(max_workers=min(len(tasks), 4)) as executor:
            pairs = [(executor.submit(run_one, t), t) for t in tasks]
            for future, task in pairs:
                try:
                    results.append(future.result())
                except Exception:
                    results.append({"task": task.get("task", str(task)), "result": None, "error": "fan_out failed"})
        return results

    def pipeline(self, tasks: list[dict], tool_allowlist: list[str] | None = None) -> list[dict]:
        """串行执行子任务，每个子任务可以看到前一个的结果"""
        sub_registry = self._build_subset_registry(tool_allowlist or ["read_file", "list_dir", "calculate"])
        results = []
        context = ""

        for task in tasks:
            desc = task.get("task", task.get("description", str(task)))
            if context:
                desc = f"前一个子任务的结果: {context}\n\n当前任务: {desc}"
            try:
                sub = SubAgent(llm=self.llm, registry=sub_registry, max_steps=5)
                result = sub.run(desc)
                context = result
                results.append({"task": desc, "result": result, "error": None})
            except Exception as e:
                results.append({"task": desc, "result": None, "error": str(e)})

        return results

    def _build_subset_registry(self, tool_names: list[str]) -> ToolRegistry:
        sub_registry = ToolRegistry(safe_mode=False)
        for name in tool_names:
            if name in self.full_registry._tools:
                sub_registry._tools[name] = self.full_registry._tools[name]
                if name in self.full_registry._tool_metadata:
                    sub_registry._tool_metadata[name] = self.full_registry._tool_metadata[name]
        return sub_registry
