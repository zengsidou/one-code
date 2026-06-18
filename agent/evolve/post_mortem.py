# -*- coding: utf-8 -*-
"""任务复盘模块 — 每次代码任务执行后 LLM 深度反思

无论成功还是失败，复盘都回答三个核心问题：
1. 这次学到了什么？
2. 还能做得更好吗？
3. 这个策略/技巧能否复用到其他任务？
"""
import json
from datetime import datetime
from agent.models import Message

POST_MORTEM_PROMPT = """你是一位资深代码工程 mentor。分析以下 Agent 完成代码任务的过程，
进行深度复盘。Agent 的目标是成长为更优秀的代码工程师。

## 任务描述
{task_desc}

## 执行结果
{result_status} | 最终输出: {result_preview}

## 执行过程
{execution_trace}

## 复盘要求
输出严格 JSON，只输出 JSON 对象，不要任何解释。

{
  "outcome": "success 或 failure",
  "difficulty_for_agent": 1-5 (1=毫不费力, 5=超出当前能力),
  "what_worked": ["做得好的地方"],
  "what_could_be_better": ["即使成功了，哪里还能更好？如果失败了，本质缺什么能力？"],
  "strategy_used": "这次用什么策略解决问题（如：二分搜索定位bug、先读后改、最小改动原则等）",
  "new_skill_gained": {
    "name": "新技能名称（简短）",
    "description": "技能描述",
    "reusable": true/false,
    "trigger": "什么情况下可以用这个技能",
    "steps": "这个技能的操作步骤"
  },
  "efficiency_score": 1-5 (1=冗余步骤多, 5=最简路径),
  "growth_insight": "一句话：这次经历让 Agent 成长了什么"
}
"""


class TaskPostMortem:
    """代码任务复盘器

    每次 Agent 执行完代码任务后（无论成败），用 LLM 深度反思
    执行过程，提取可复用的策略、技能和成长洞察。
    """

    def __init__(self, llm_adapter):
        self.llm = llm_adapter
        self.history: list[dict] = []

    def reflect(
        self,
        task_desc: str,
        result: str,
        execution_trace: str,
    ) -> dict:
        """复盘一次任务执行

        Args:
            task_desc: 任务描述
            result: 最终结果文本
            execution_trace: 执行过程（步骤、工具调用、输出摘要）

        Returns:
            复盘报告 dict，包含 what_worked/what_could_be_better/strategy/new_skill 等
        """
        is_failure = "[STOPPED]" in result or "[ERROR]" in result
        result_status = "FAILED" if is_failure else "SUCCESS"
        result_preview = result[:500]

        prompt = (POST_MORTEM_PROMPT
            .replace("{task_desc}", task_desc[:500])
            .replace("{result_status}", result_status)
            .replace("{result_preview}", result_preview)
            .replace("{execution_trace}", execution_trace[:2000]))

        try:
            resp = self.llm.generate(
                [Message(role="system", content="你是一个代码工程 mentor。只输出 JSON。"),
                 Message(role="user", content=prompt)],
                tools=None,
            )
            report = self._parse_json(resp.content or "")
        except Exception as e:
            report = {
                "outcome": "failure" if is_failure else "success",
                "difficulty_for_agent": 3,
                "what_worked": [],
                "what_could_be_better": [f"复盘失败: {e}"],
                "strategy_used": "未知",
                "new_skill_gained": {"name": "", "description": "", "reusable": False, "trigger": "", "steps": ""},
                "efficiency_score": 2,
                "growth_insight": "复盘系统异常",
            }

        report["task_desc"] = task_desc[:200]
        report["timestamp"] = datetime.now().isoformat()
        report["result_preview"] = result_preview[:100]
        self.history.append(report)
        return report

    @staticmethod
    def _parse_json(text: str) -> dict:
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:]) if len(lines) > 1 else text[3:]
        if text.endswith("```"):
            text = text[:-3]
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            return {
                "outcome": "success",
                "difficulty_for_agent": 3,
                "what_worked": [],
                "what_could_be_better": ["LLM 返回格式异常"],
                "strategy_used": "未知",
                "new_skill_gained": {"name": "", "description": "", "reusable": False, "trigger": "", "steps": ""},
                "efficiency_score": 2,
                "growth_insight": "",
            }

    def get_recent_insights(self, n: int = 5) -> list[str]:
        """获取最近 N 次复盘的成长洞察"""
        return [h.get("growth_insight", "") for h in self.history[-n:] if h.get("growth_insight")]

    def get_avg_difficulty(self) -> float:
        """获取近期任务平均难度"""
        if not self.history:
            return 1.0
        recent = self.history[-10:]
        diffs = [h.get("difficulty_for_agent", 3) for h in recent]
        return sum(diffs) / len(diffs)

    def get_avg_efficiency(self) -> float:
        """获取近期平均效率分"""
        if not self.history:
            return 3.0
        recent = self.history[-10:]
        scores = [h.get("efficiency_score", 3) for h in recent]
        return sum(scores) / len(scores)
