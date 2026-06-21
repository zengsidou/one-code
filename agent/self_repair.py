# -*- coding: utf-8 -*-
"""修复策略模块 — 根据根因生成修复方案并应用到 Agent Loop"""
import os
import re
import shutil
import py_compile
from datetime import datetime
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

TOOL_CODE_FIX_PROMPT = (
    "你是一个代码修复专家。以下工具代码存在 bug，请修复它。\n\n"
    "问题描述: {detail}\n\n"
    "当前代码 ({file_path}):\n```python\n{source}\n```\n\n"
    "请输出 JSON: {{\"old_code\": \"出问题的代码片段\", \"new_code\": \"修复后的代码片段\"}}\n"
    "old_code 必须能在源文件中精确匹配，new_code 是替换后的版本。"
)

TOOL_DESC_ENRICH_PROMPT = (
    "你是一个工具描述优化专家。以下工具的描述不够准确或完整，导致 Agent 使用不当。\n\n"
    "问题: {detail}\n\n"
    "工具名: {tool_name}\n"
    "当前描述: {description}\n"
    "工具参数: {parameters}\n\n"
    "请输出改进后的工具描述（中文，50-200 字），包含:\n"
    "1. 工具做什么\n"
    "2. 每个参数的含义和约束\n"
    "3. 何时使用（vs 其他类似工具）\n"
    "直接输出描述文本，不要 JSON 包裹。"
)

MODEL_FALLBACK_CHAIN = [
    "deepseek-v4-pro",
    "deepseek-v4-flash",
    "deepseek-reasoner",
    "deepseek-chat",
]


class SelfRepair:
    """Agent 自修复引擎

    根据根因分析结果生成修复方案，可应用到 Agent Loop 实例上，
    并支持回滚操作。

    可通过 get_tunable_params() / apply_params() 被 MetaOptimizer 调优。
    """

    def __init__(self, llm_adapter, trim_ratio: float = 0.6, backup_dir: str = "./self_repair_backups"):
        """Args:
            llm_adapter: BaseLLM 子类实例，用于生成修复内容
            trim_ratio: trim_context 时保留的内存比例 (0-1)
            backup_dir: 工具代码修复时的备份目录
        """
        self.llm = llm_adapter
        self.trim_ratio = trim_ratio
        self.backup_dir = backup_dir
        os.makedirs(backup_dir, exist_ok=True)
        self._prompt_fix_prompt = (
            "原始 system prompt:\n```\n{original}\n```\n\n"
            "诊断出的问题: {detail}\n\n"
            "请输出改进后的 system prompt（中文）。保持原有结构，只修复问题所在部分。"
        )
        self._reasoning_hint_prompt = (
            "Agent 执行任务时出现推理错误: {detail}\n\n"
            "请用一两句中文写出应追加到 system prompt 末尾的推理引导提示，"
            "帮助模型在遇到类似情况时做出正确决策。直接输出提示文本。"
        )
        self._rollback_snapshots: dict[str, dict] = {}
        self._snapshot_data: dict | None = None

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
            enriched = self._generate_tool_desc_enrichment(root_cause, current_config)
            fix["fixed"]["tool_descriptions"] = enriched

        elif fix_type == "trim_context":
            old_tokens = current_config.get("memory_max_tokens", 4096)
            fix["original"]["memory_max_tokens"] = old_tokens
            fix["fixed"]["memory_max_tokens"] = max(1024, int(old_tokens * self.trim_ratio))

        elif fix_type == "add_reasoning_hint":
            fix["original"]["system_prompt"] = current_config.get("system_prompt", "")
            hint = self._generate_reasoning_hint(root_cause)
            current = fix["original"]["system_prompt"]
            fix["fixed"]["system_prompt"] = current + "\n\n" + hint

        elif fix_type == "fix_tool_code":
            result = self._generate_tool_code_fix(root_cause, current_config)
            if result:
                fix["fixed"]["target_file"] = result["target_file"]
                fix["fixed"]["old_code"] = result["old_code"]
                fix["fixed"]["new_code"] = result["new_code"]
            else:
                fix["fixed"]["requires_manual"] = True
                fix["fixed"]["message"] = "无法生成工具代码修复方案"

        elif fix_type == "switch_model":
            current_model = current_config.get("model_name", "")
            fallback = self._pick_fallback_model(current_model)
            fix["original"]["model_name"] = current_model
            fix["fixed"]["model_name"] = fallback
            fix["fixed"]["message"] = f"切换模型: {current_model} → {fallback}"

        return fix

    def _generate_prompt_fix(self, root_cause: dict, current_config: dict) -> str:
        """用 LLM 生成改进后的 system_prompt"""
        detail = root_cause.get("detail", "")
        original = current_config.get("system_prompt", "")
        prompt = self._prompt_fix_prompt.format(original=original, detail=detail)
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
        prompt = self._reasoning_hint_prompt.format(detail=detail)
        try:
            resp = self.llm.generate(
                [Message(role="user", content=prompt)], tools=None
            )
            return (resp.content or "请仔细分析问题后再做决策。").strip()
        except Exception:
            return "请逐步推理，确认每个步骤的必要性后再执行。"

    def _generate_tool_desc_enrichment(self, root_cause: dict, current_config: dict) -> dict:
        """用 LLM 为问题工具生成更准确/完整的描述"""
        tool_name = root_cause.get("tool_name", "")
        detail = root_cause.get("detail", "")
        original_descs = current_config.get("tool_descriptions", {})

        if not tool_name or tool_name not in original_descs:
            return original_descs.copy()

        description = original_descs[tool_name]
        parameters = root_cause.get("tool_parameters", "")
        if not parameters:
            params = current_config.get("tool_schemas", {}).get(tool_name, "")
            parameters = params if params else "无"

        prompt = TOOL_DESC_ENRICH_PROMPT.format(
            detail=detail, tool_name=tool_name,
            description=description, parameters=str(parameters)[:2000],
        )
        try:
            resp = self.llm.generate(
                [Message(role="user", content=prompt)], tools=None
            )
            enriched = (resp.content or description).strip()
            if len(enriched) < 10:
                enriched = description
        except Exception:
            enriched = description

        result = original_descs.copy()
        result[tool_name] = enriched
        return result

    def get_tunable_params(self) -> dict:
        """返回可被 MetaOptimizer 调优的参数"""
        return {
            "prompt_fix_prompt": self._prompt_fix_prompt,
            "reasoning_hint_prompt": self._reasoning_hint_prompt,
            "trim_ratio": self.trim_ratio,
        }

    def apply_params(self, params: dict):
        """应用 MetaOptimizer 调优后的参数"""
        if "prompt_fix_prompt" in params:
            self._prompt_fix_prompt = params["prompt_fix_prompt"]
        if "reasoning_hint_prompt" in params:
            self._reasoning_hint_prompt = params["reasoning_hint_prompt"]
        if "trim_ratio" in params:
            self.trim_ratio = float(params["trim_ratio"])

    def snapshot(self) -> dict:
        self._snapshot_data = {
            "prompt_fix_prompt": self._prompt_fix_prompt,
            "reasoning_hint_prompt": self._reasoning_hint_prompt,
            "trim_ratio": self.trim_ratio,
        }
        return self._snapshot_data

    def restore(self, snapshot: dict | None = None):
        data = snapshot or self._snapshot_data
        if data:
            self.apply_params(data)

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

            elif fix_type == "fix_tool_code":
                if not self._apply_tool_code_fix(fix):
                    return False

            elif fix_type == "switch_model" and "model_name" in fix.get("fixed", {}):
                new_model = fix["fixed"]["model_name"]
                if hasattr(agent_loop, "llm") and hasattr(agent_loop.llm, "model"):
                    self._rollback_snapshots[fix_id]["model_name"] = agent_loop.llm.model
                    agent_loop.llm.model = new_model

            fix["applied"] = True
            return True

        except Exception:
            return False

    def _apply_tool_code_fix(self, fix: dict) -> bool:
        """安全地应用工具代码修复（备份 → 写入 → 语法检查）"""
        fixed = fix.get("fixed", {})
        target_file = fixed.get("target_file", "")
        old_code = fixed.get("old_code", "")
        new_code = fixed.get("new_code", "")

        if not target_file or not os.path.exists(target_file) or not new_code:
            return False

        try:
            backup_name = f"{os.path.basename(target_file)}.{datetime.now().strftime('%Y%m%d_%H%M%S')}.bak"
            backup_path = os.path.join(self.backup_dir, backup_name)
            shutil.copy2(target_file, backup_path)
            self._rollback_snapshots[fix.get("fix_id", "unknown")]["tool_backup"] = backup_path
            self._rollback_snapshots[fix.get("fix_id", "unknown")]["tool_file"] = target_file

            with open(target_file, "r", encoding="utf-8") as f:
                content = f.read()

            if old_code and old_code in content:
                new_content = content.replace(old_code, new_code, 1)
            else:
                return False

            tmp_path = target_file + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                f.write(new_content)
            py_compile.compile(tmp_path, doraise=True)
            os.remove(tmp_path)

            with open(target_file, "w", encoding="utf-8") as f:
                f.write(new_content)

            return True
        except Exception:
            return False

    def _generate_tool_code_fix(self, root_cause: dict, current_config: dict) -> dict | None:
        """用 LLM 生成工具代码修复方案"""
        target_file = root_cause.get("target_file", "")
        detail = root_cause.get("detail", "")

        if not target_file:
            search_dir = os.path.join(os.path.dirname(__file__), "..", "tools")
            target_file = os.path.join(search_dir, "builtin", "__init__.py")

        target_file = os.path.abspath(target_file)
        if not os.path.exists(target_file):
            return None

        try:
            with open(target_file, "r", encoding="utf-8") as f:
                source = f.read()
        except Exception:
            return None

        prompt = TOOL_CODE_FIX_PROMPT.format(detail=detail, file_path=target_file, source=source[-6000:])
        try:
            resp = self.llm.generate([Message(role="user", content=prompt)], tools=None)
            text = (resp.content or "").strip()
            m = re.search(r'\{[\s\S]*"old_code"[\s\S]*\}', text)
            if m:
                result = __import__("json").loads(m.group())
                result["target_file"] = target_file
                return result
        except Exception:
            pass
        return None

    def _pick_fallback_model(self, current_model: str) -> str:
        """选一个不同于当前的替代模型"""
        for m in MODEL_FALLBACK_CHAIN:
            if m != current_model:
                return m
        return "deepseek-v4-pro"

    def rollback(self, fix: dict, agent_loop) -> bool:
        """回滚修复，恢复修改前的状态

        Args:
            fix: 已应用的修复方案 dict
            agent_loop: AgentLoop 实例

        Returns:
            是否回滚成功
        """
        fix_id = fix.get("fix_id", "unknown")
        fix_type = fix.get("fix_type", "")
        snapshot = self._rollback_snapshots.pop(fix_id, None)
        if snapshot is None:
            return False

        try:
            if snapshot.get("system_prompt") is not None:
                agent_loop.system_prompt = snapshot["system_prompt"]

            if snapshot.get("memory_max_tokens") is not None:
                if hasattr(agent_loop.memory, "short_term"):
                    agent_loop.memory.short_term.max_tokens = snapshot["memory_max_tokens"]

            if snapshot.get("model_name") is not None:
                if hasattr(agent_loop, "llm") and hasattr(agent_loop.llm, "model"):
                    agent_loop.llm.model = snapshot["model_name"]

            if snapshot.get("tool_backup") and snapshot.get("tool_file"):
                backup = snapshot["tool_backup"]
                target = snapshot["tool_file"]
                if os.path.exists(backup):
                    shutil.copy2(backup, target)

            fix["applied"] = False
            return True
        except Exception:
            return False
