# -*- coding: utf-8 -*-
"""修复策略模块 — 根据根因生成修复方案并应用到 Agent Loop"""
from agent.models import Message


FIX_GENERATION_PROMPT = (
    "你是一个 Agent 调优专家。根据以下诊断结果生成具体的修复方案。\n\n"
    "诊断结果:\n"
    "- 根因类型: {root_cause_type}\n"
    "- 详情: {detail}\n"
    "- 建议修复类型: {suggested_fix_type}\n\n"
    "当前配置:\n{current_config}\n\n"
    "请生成一个具体的、可直接应用的修复方案。\n"
    "输出严格 JSON 格式。"
)


class SelfRepair:
    """Agent 自修复引擎

    根据根因分析结果生成修复方案，可应用到 Agent Loop 实例上，
    并支持回滚操作。
    """

    def __init__(self, llm_adapter):
        """Args:
            llm_adapter: BaseLLM 子类实例，用于生成修复内容
        """
        self.llm = llm_adapter
        self._rollback_snapshots: dict[str, dict] = {}

    def generate_fix(self, root_cause: dict, current_config: dict) -> dict:
        """根据根因类型生成具体修复方案

        Args:
            root_cause: RootCauseAnalyzer.analyze() 的返回结果
            current_config: 当前 Agent 配置快照，包含:
                - system_prompt: 当前系统提示词
                - tool_descriptions: {tool_name: description}
                - memory_max_tokens: ShortTermMemory 的 max_tokens
                - model_name: 当前模型名

        Returns:
            {fix_id, fix_type, original, fixed, applied: False}
        """
        fix_type = root_cause.get("suggested_fix_type", "fix_tool_code")

        fix = {
            "fix_id": f"fix_{root_cause.get('root_cause_type', 'unknown')}",
            "fix_type": fix_type,
            "original": {},
            "fixed": {},
            "applied": False,
        }

        if fix_type == "adjust_prompt":
            fix["original"]["system_prompt"] = current_config.get("system_prompt", "")
            improved = self._generate_prompt_fix(root_cause, current_config)
            fix["fixed"]["system_prompt"] = improved

        elif fix_type == "enrich_tool_description":
            fix["original"]["tool_descriptions"] = current_config.get("tool_descriptions", {})
            fix["fixed"]["tool_descriptions"] = fix["original"]["tool_descriptions"].copy()

        elif fix_type == "trim_context":
            old_tokens = current_config.get("memory_max_tokens", 4096)
            fix["original"]["memory_max_tokens"] = old_tokens
            fix["fixed"]["memory_max_tokens"] = max(1024, int(old_tokens * 0.6))

        elif fix_type == "add_reasoning_hint":
            fix["original"]["system_prompt"] = current_config.get("system_prompt", "")
            hint = self._generate_reasoning_hint(root_cause)
            current = fix["original"]["system_prompt"]
            fix["fixed"]["system_prompt"] = current + "\n\n" + hint

        elif fix_type == "fix_tool_code":
            fix["fixed"]["requires_manual"] = True
            fix["fixed"]["message"] = "需要人工修改工具代码 — 自动修改代码风险过大"

        elif fix_type == "switch_model":
            fix["fixed"]["requires_manual"] = True
            fix["fixed"]["message"] = "需要人工选择替代模型"

        return fix

    def _generate_prompt_fix(self, root_cause: dict, current_config: dict) -> str:
        """用 LLM 生成改进后的 system_prompt"""
        detail = root_cause.get("detail", "")
        original = current_config.get("system_prompt", "")
        prompt = (
            f"原始 system prompt:\n```\n{original}\n```\n\n"
            f"诊断出的问题: {detail}\n\n"
            "请输出改进后的 system prompt（中文）。保持原有结构，只修复问题所在部分。"
        )
        try:
            resp = self.llm.generate(
                [Message(role="user", content=prompt)], tools=None
            )
            return (resp.content or "").strip()
        except Exception:
            return original

    def _generate_reasoning_hint(self, root_cause: dict) -> str:
        """生成推理引导提示"""
        detail = root_cause.get("detail", "")
        prompt = (
            f"Agent 执行任务时出现推理错误: {detail}\n\n"
            "请用一两句中文写出应追加到 system prompt 末尾的推理引导提示，"
            "帮助模型在遇到类似情况时做出正确决策。直接输出提示文本。"
        )
        try:
            resp = self.llm.generate(
                [Message(role="user", content=prompt)], tools=None
            )
            return (resp.content or "请仔细分析问题后再做决策。").strip()
        except Exception:
            return "请逐步推理，确认每个步骤的必要性后再执行。"

    def apply_fix(self, fix: dict, agent_loop) -> bool:
        """将修复方案应用到 Agent Loop 实例

        Args:
            fix: generate_fix() 返回的修复方案 dict
            agent_loop: AgentLoop 实例

        Returns:
            是否应用成功
        """
        if fix.get("fixed", {}).get("requires_manual"):
            return False

        fix_type = fix.get("fix_type", "")
        fix_id = fix.get("fix_id", "unknown")

        try:
            self._rollback_snapshots[fix_id] = {
                "system_prompt": agent_loop.system_prompt,
                "memory_max_tokens": agent_loop.memory.short_term.max_tokens
                if hasattr(agent_loop.memory, "short_term") else None,
            }

            if fix_type == "adjust_prompt" and "system_prompt" in fix.get("fixed", {}):
                agent_loop.system_prompt = fix["fixed"]["system_prompt"]

            elif fix_type == "add_reasoning_hint" and "system_prompt" in fix.get("fixed", {}):
                agent_loop.system_prompt = fix["fixed"]["system_prompt"]

            elif fix_type == "trim_context" and "memory_max_tokens" in fix.get("fixed", {}):
                new_max = fix["fixed"]["memory_max_tokens"]
                if hasattr(agent_loop.memory, "short_term"):
                    agent_loop.memory.short_term.max_tokens = new_max

            elif fix_type == "enrich_tool_description":
                for tname, new_desc in fix.get("fixed", {}).get("tool_descriptions", {}).items():
                    if tname in agent_loop.registry._tool_metadata:
                        agent_loop.registry._tool_metadata[tname]["description"] = new_desc
                        agent_loop.registry._tool_metadata[tname]["schema"]["function"]["description"] = new_desc

            fix["applied"] = True
            return True

        except Exception:
            return False

    def rollback(self, fix: dict, agent_loop) -> bool:
        """回滚修复，恢复修改前的状态

        Args:
            fix: 已应用的修复方案 dict
            agent_loop: AgentLoop 实例

        Returns:
            是否回滚成功
        """
        fix_id = fix.get("fix_id", "unknown")
        snapshot = self._rollback_snapshots.pop(fix_id, None)
        if snapshot is None:
            return False

        try:
            if snapshot.get("system_prompt") is not None:
                agent_loop.system_prompt = snapshot["system_prompt"]

            if snapshot.get("memory_max_tokens") is not None:
                if hasattr(agent_loop.memory, "short_term"):
                    agent_loop.memory.short_term.max_tokens = snapshot["memory_max_tokens"]

            fix["applied"] = False
            return True
        except Exception:
            return False
