# -*- coding: utf-8 -*-
"""自动验证模块 — 验证修复方案是否有效"""
from agent.models import Message


class VerifyRepair:
    """修复效果验证器

    对修复后的 Agent Loop 重新执行失败任务，
    对比修复前后的执行结果，判断修复是否有效。
    """

    def __init__(self):
        self.results: list[dict] = []

    def verify(self, fix: dict, agent_loop, original_task: str) -> dict:
        """验证单个修复方案

        Args:
            fix: 已应用的修复方案 dict
            agent_loop: 已应用修复的 AgentLoop 实例
            original_task: 原始失败任务描述（用户输入）

        Returns:
            {fix_id, task, before_success, after_success, after_message, improved}
        """
        fix_id = fix.get("fix_id", "unknown")

        try:
            result = agent_loop.run(original_task)
            after_success = not self._is_failure(result)
            after_message = result[:300]
        except Exception as e:
            after_success = False
            after_message = f"执行异常: {e}"

        report = {
            "fix_id": fix_id,
            "task": original_task[:100],
            "before_success": False,
            "after_success": after_success,
            "after_message": after_message,
            "improved": after_success,
        }
        self.results.append(report)
        return report

    @staticmethod
    def _is_failure(result: str) -> bool:
        """判断执行结果是否为失败"""
        failure_markers = ["[STOPPED]", "[ERROR]", "[LLM error]"]
        return any(marker in result for marker in failure_markers)

    def full_verify(self, fixes: list[dict], agent_loop, original_tasks: list[str]) -> list[dict]:
        """批量验证所有修复

        Args:
            fixes: 修复方案列表
            agent_loop: AgentLoop 实例
            original_tasks: 原始任务列表

        Returns:
            验证结果列表
        """
        results = []
        for i, fix in enumerate(fixes):
            if not fix.get("applied"):
                continue
            task = original_tasks[i] if i < len(original_tasks) else original_tasks[0]
            result = self.verify(fix, agent_loop, task)
            results.append(result)
        return results
