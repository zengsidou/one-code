# -*- coding: utf-8 -*-
"""契约生成器 — 用LLM为不同任务类型生成多模态契约预览"""
import json
from agent.models import Message, Contract
from agent.contract_types import ContractType, get_contract_meta

VISUAL_CONTRACT_PROMPT = """你是一个UI/UX设计专家。用户描述了一个视觉任务，你需要生成一个**方向预览**而非完整产品。

## 任务
{task}

## 要求
1. 用ASCII字符画出一个线框图，标注各区域的功能（导航、内容、按钮等）
2. 在末尾用1-2句话总结布局方向
3. ASCII线框图至少10行，使用 ├─└│┌┐┘ 等Unicode框线字符让结构清晰

输出格式:
---ASCII---
(你的ASCII线框图)
---SUMMARY---
(一句话方向总结)"""

DIALOG_CONTRACT_PROMPT = """你是一个对话设计专家。用户需要一个对话Agent，请先输出**示例对话轨迹**确认方向。

## 任务
{task}

## 要求
用3-5轮关键对话展示：
- Agent的开场方式
- 用户可能的关键问题
- Agent的回答风格和逻辑走向
- 每轮标注角色（Agent/用户）

输出JSON:
[
  {{"role": "Agent", "content": "..."}},
  {{"role": "用户", "content": "..."}},
  ...
]

末尾加一句风格总结。"""

CODE_API_CONTRACT_PROMPT = """你是一个软件架构师。用户描述了一个代码相关任务，请先输出**架构预览**确认方向。

## 任务
{task}

## 要求
1. 列出关键API/接口签名（函数名 + 参数 + 返回值简述）
2. 输出一个Mermaid图描述模块关系（用 ```mermaid 包裹）
3. 1-2句话总结架构方向

输出格式:
---API---
(接口列表)
---ARCH---
```mermaid
graph TD
  ...
```
---SUMMARY---
(架构方向总结)"""

CONFIG_CONTRACT_PROMPT = """你是一个系统配置专家。用户描述了一个配置相关任务，请先输出**预期行为表**确认逻辑。

## 任务
{task}

## 要求
输出一个表格，每行是"条件 → 结果"的映射：
| 条件 | 结果 | 说明 |

至少5行。末尾用1句话总结逻辑方向。"""

DATA_CONTRACT_PROMPT = """你是一个数据分析专家。用户描述了一个数据任务，请先输出**分析框架预览**确认方向。

## 任务
{task}

## 要求
1. 列出分析维度和指标
2. 画一个输出表格的结构（列名+含义）
3. 用1句话描述预期结论方向

输出格式:
---DIMENSIONS---
(分析维度)
---OUTPUT---
| 列1 | 列2 | ... |
(表结构)
---DIRECTION---
(结论方向)"""

NARRATIVE_CONTRACT_PROMPT = """你是一个内容策划专家。用户需要写一些内容，请先输出**大纲和风格预览**确认方向。

## 任务
{task}

## 要求
1. 输出文章/文档的大纲（层级结构）
2. 写一个50-100字的风格样本段落
3. 1-2句话总结文风特征

输出格式:
---OUTLINE---
1. 引言
  1.1 ...
2. 正文
...
---SAMPLE---
(风格样本段落)
---STYLE---
(文风特征)"""


CONTRACT_PROMPTS = {
    ContractType.VISUAL: VISUAL_CONTRACT_PROMPT,
    ContractType.DIALOG: DIALOG_CONTRACT_PROMPT,
    ContractType.CODE_API: CODE_API_CONTRACT_PROMPT,
    ContractType.CONFIG: CONFIG_CONTRACT_PROMPT,
    ContractType.DATA: DATA_CONTRACT_PROMPT,
    ContractType.NARRATIVE: NARRATIVE_CONTRACT_PROMPT,
}


class ContractGenerator:
    """多模态契约生成器 — 用LLM为每种任务类型生成方向预览"""

    def __init__(self, llm_adapter):
        self.llm = llm_adapter

    def generate(self, contract_type: ContractType, task: str) -> Contract:
        """生成契约预览

        Args:
            contract_type: 由 detect_contract_type() 确定的类型
            task: 用户原始任务描述

        Returns:
            Contract 对象，包含预览内容和元信息
        """
        meta = get_contract_meta(contract_type)
        prompt_template = CONTRACT_PROMPTS.get(contract_type)
        if prompt_template is None:
            prompt_template = CODE_API_CONTRACT_PROMPT

        prompt = prompt_template.format(task=task[:800])

        try:
            resp = self.llm.generate(
                [Message(role="system", content="你是一个技术专家。只输出要求的格式，不要额外解释。"),
                 Message(role="user", content=prompt)],
                tools=None,
            )
            content = (resp.content or "").strip()
        except Exception as e:
            content = f"[契约生成失败: {e}]"

        summary = self._extract_summary(content, contract_type)

        return Contract(
            type=contract_type.value,
            format=meta["format"],
            content=content,
            summary=summary,
        )

    @staticmethod
    def _extract_summary(content: str, ct: ContractType) -> str:
        """从契约内容中提取方向总结"""
        for marker in ["---SUMMARY---", "---DIRECTION---", "---STYLE---"]:
            if marker in content:
                parts = content.split(marker, 1)
                return parts[1].strip().split("\n")[0][:120]
        for line in content.split("\n"):
            line = line.strip()
            if line.startswith("方向") or line.startswith("总结") or line.startswith("风格"):
                return line[:120]
        return content.split("\n")[-1][:120] if content else "(无法生成总结)"
