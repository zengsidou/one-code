# -*- coding: utf-8 -*-
"""AGENTS.md 自动积累 — 对标 Mitchell Hashimoto 的 Harness 工程学实践。

每次 Agent 失败后，LLM 自动分析失败原因，提取为一条简洁规则写入 AGENTS.md。
启动时自动加载注入 system prompt，形成持续进化的防错反馈闭环。

Hashimoto 原话："Agent 每犯一次错，就工程化一个方案，尽量让它以后不再犯同类错误。
Ghostty 项目里的 AGENTS.md 每一行都对应一个过去的 Agent 失败案例。"
"""
import os
import re
from datetime import datetime
from pathlib import Path

from agent.models import Message


RULE_GENERATION_PROMPT = (
    "你是一个 Agent 行为分析专家。根据以下失败信息，提取一条简洁的防错规则。\n\n"
    "## 失败信息\n"
    "任务: {task}\n"
    "错误: {error}\n"
    "上下文: {trace}\n\n"
    "## 规则要求\n"
    "用一行中文写出规则（不超过 80 字），格式: \"如果遇到 [场景]，应该 [正确做法]，不要 [错误做法]。\"\n"
    "规则应该具体到可执行，不要泛泛而谈。\n\n"
    "输出: 只输出一条规则，不要其他文字。"
)

RULE_MERGE_PROMPT = (
    "你是一个知识管理者。以下是现有的 Agent 规则列表和一条新规则，判断新规则是否可以合并到已有规则中。\n\n"
    "## 已有规则\n{existing}\n\n"
    "## 新规则\n{new_rule}\n\n"
    "如果新规则实质内容已被某条已有规则覆盖，输出被覆盖的规则编号: {{\"merge\": N}}\n"
    "如果是全新规则，输出: {{\"merge\": -1}}\n"
    "只输出 JSON。"
)


class RuleAccumulator:
    """AGENTS.md 规则自动积累器。

    Usage:
        ra = RuleAccumulator(llm_adapter, rules_file="./AGENTS.md")
        agent = AgentLoop(rules=ra, ...)
        # After each failure, the agent calls ra.learn_from_failure()
        # On startup, ra.inject_rules() adds accumulated rules to system prompt
    """

    def __init__(self, llm_adapter, rules_file: str = "./AGENTS.md"):
        self.llm = llm_adapter
        self.rules_file = Path(rules_file)
        self.rules: list[str] = []
        self._load()

    def _load(self):
        """从文件加载已有规则。"""
        if not self.rules_file.exists():
            return
        try:
            content = self.rules_file.read_text(encoding="utf-8")
            # 解析规则行（以 - 或 数字 开头的行）
            for line in content.split("\n"):
                stripped = line.strip()
                if stripped and (stripped.startswith("- ") or stripped.startswith("* ")):
                    self.rules.append(stripped[2:].strip())
                elif stripped and re.match(r"^\d+\.\s", stripped):
                    self.rules.append(re.sub(r"^\d+\.\s*", "", stripped).strip())
        except Exception:
            pass

    def _save(self):
        """将规则写入 AGENTS.md。"""
        lines = [
            "# AGENTS.md — 自动积累的规则",
            f"# 最后更新: {datetime.now().isoformat()[:19]}",
            f"# 规则数: {len(self.rules)}",
            "#",
            "# 每一条规则对应一个历史失败案例。",
            "# Agent 启动时自动加载，形成持续进化的防错反馈闭环。",
            "# 对标 Mitchell Hashimoto 的 Harness 工程学实践。",
            "",
        ]
        for i, rule in enumerate(self.rules, 1):
            lines.append(f"{i}. {rule}")
        self.rules_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def learn_from_failure(self, task: str, error: str, trace: str = "") -> str | None:
        """从失败中学习，生成新规则并存入 AGENTS.md。

        Args:
            task: 失败的任务描述
            error: 错误信息
            trace: 执行轨迹

        Returns:
            新生成的规则文本，或 None（如果规则被合并）
        """
        if not self.llm:
            return None

        prompt = RULE_GENERATION_PROMPT.format(
            task=task[:500], error=error[:500], trace=trace[-1000:],
        )
        try:
            resp = self.llm.generate(
                [Message(role="user", content=prompt)],
                tools=None,
            )
            new_rule = (resp.content or "").strip()
            if not new_rule or len(new_rule) < 10:
                return None
        except Exception:
            return None

        # 去重合并
        if self.rules:
            merged = self._try_merge(new_rule)
            if merged is not None:
                return None  # 被合并，不添加

        self.rules.append(new_rule)
        if len(self.rules) > 50:
            self.rules = self.rules[-50:]  # Keep last 50
        self._save()
        return new_rule

    def _try_merge(self, new_rule: str) -> int | None:
        """尝试将新规则合并到已有规则，返回被合并规则索引或 None。"""
        try:
            existing = "\n".join(f"{i}. {r}" for i, r in enumerate(self.rules, 1))
            prompt = RULE_MERGE_PROMPT.format(existing=existing, new_rule=new_rule)
            resp = self.llm.generate(
                [Message(role="user", content=prompt)],
                tools=None,
            )
            text = (resp.content or "").strip()
            m = re.search(r"\{[\s\S]*\}", text)
            if m:
                import json
                result = json.loads(m.group())
                idx = result.get("merge", -1)
                if isinstance(idx, int) and 1 <= idx <= len(self.rules):
                    return idx - 1  # 0-indexed
        except Exception:
            pass
        return None

    def inject_rules(self) -> str:
        """将规则注入为 system prompt 片段。"""
        if not self.rules:
            return ""

        lines = [
            "\n---\n## 历史经验规则（从失败中自动积累）\n",
            "以下是基于过往失败自动提取的规则，请严格遵守：",
        ]
        for i, rule in enumerate(self.rules[-20:], 1):  # Last 20
            lines.append(f"- {rule}")
        lines.append("---")
        return "\n".join(lines)

    def get_stats(self) -> dict:
        return {"rules": len(self.rules), "file": str(self.rules_file)}
