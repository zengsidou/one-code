# -*- coding: utf-8 -*-
"""逆向拆解器 — 从契约预览反推执行步骤"""
import json
from agent.models import Message, Contract, ContractStep

REVERSE_DECOMPOSE_PROMPT = """你是一个任务执行规划专家。以下是一个任务和它的**契约预览**（最终产物的方向预览）。
请从终点往回反推，生成执行步骤。

## 原始任务
{task}

## 契约预览（这是最终产出应该达到的方向）
{contract_content}

## 要求
从**终点（契约预览达到的状态）**出发，逆向推导每一步需要做什么。
比如：最终产物需要X → 那么上一步需要Y → 再上一步需要Z...

输出JSON数组，倒序排列（最后一个元素是第一步，第一个元素是最后一步）：
[
  {{
    "index": 1,
    "goal": "最终步骤的目标（中文，一句话）",
    "tools_hint": "建议使用的工具",
    "depends_on": [前置步骤的index列表],
    "contract_checkpoint": "这一步完成后，对照契约的哪部分来验证（具体到某段内容）"
  }},
  ...
]

规则:
- 每步只做一件事
- 信息收集必须在修改之前
- 步骤数控制在 3-8 步
- contract_checkpoint 必须引用契约中的具体内容作为验证标准
- 步骤按index递增顺序（index=1是第一步）

只输出JSON数组，不要其他文字。"""


class ReverseDecomposer:
    """逆向拆解器 — 将契约预览反推为可执行步骤序列"""

    def __init__(self, llm_adapter):
        self.llm = llm_adapter

    def decompose(self, contract: Contract, task: str) -> list[ContractStep]:
        """从契约逆向拆解执行步骤

        Args:
            contract: 已生成的契约预览
            task: 原始任务描述

        Returns:
            ContractStep 列表（正序：第一步在前）
        """
        prompt = REVERSE_DECOMPOSE_PROMPT.format(
            task=task[:800],
            contract_content=contract.content[:3000],
        )

        try:
            resp = self.llm.generate(
                [Message(role="system", content="你是一个任务规划专家。只输出JSON。"),
                 Message(role="user", content=prompt)],
                tools=None,
            )
            steps = self._parse_steps(resp.content or "")
        except Exception:
            steps = []

        if not steps:
            steps = self._fallback_steps(contract, task)

        steps.sort(key=lambda s: s.index)
        return steps

    @staticmethod
    def _parse_steps(text: str) -> list[ContractStep]:
        """解析LLM输出的JSON步骤列表"""
        import re
        text = text.strip()
        m = re.search(r"\[[\s\S]*\]", text)
        if not m:
            return []
        try:
            raw = json.loads(m.group())
            if not isinstance(raw, list):
                return []
            result = []
            for item in raw:
                result.append(ContractStep(
                    index=int(item.get("index", len(result) + 1)),
                    goal=item.get("goal", ""),
                    tools_hint=item.get("tools_hint", ""),
                    depends_on=item.get("depends_on", []),
                    contract_checkpoint=item.get("contract_checkpoint", ""),
                ))
            return result
        except (json.JSONDecodeError, ValueError):
            return []

    @staticmethod
    def _fallback_steps(contract: Contract, task: str) -> list[ContractStep]:
        """LLM拆解失败时的回退步骤"""
        ct = contract.type
        if ct == "visual":
            return [
                ContractStep(1, "了解项目结构和现有文件", "list_files, read_file", [], "确认工作目录"),
                ContractStep(2, "设计页面HTML结构", "write_file, edit_file", [1], "对照线框图的结构划分"),
                ContractStep(3, "添加CSS样式和布局", "edit_file", [2], "对照线框图的布局位置"),
                ContractStep(4, "添加内容和交互元素", "edit_file", [3], "对照线框图的各区域功能标注"),
                ContractStep(5, "验证视觉效果和响应式", "run_shell", [4], "确认最终页面与线框图方向一致"),
            ]
        return [
            ContractStep(1, "了解项目上下文和相关代码", "read_file, grep", [], "确认范围"),
            ContractStep(2, "制定实施方案", "无", [1], "对照契约确认方向"),
            ContractStep(3, "执行核心修改", "edit_file, write_file", [2], "对照契约验证产出"),
            ContractStep(4, "验证结果", "run_shell", [3], "确认与契约一致"),
        ]
