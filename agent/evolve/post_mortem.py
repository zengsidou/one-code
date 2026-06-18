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

## Agent 当前能力水平
{ ability_context }

## 任务描述
{task_desc}

## 执行结果
{result_status} | 步数: {step_count} | 最终输出: {result_preview}

## 执行过程
{execution_trace}

## 难度评分标准（基于客观指标）
- 1: 单文件，≤3步完成
- 2: 单文件，4-8步完成；或多文件但≤4步
- 3: 2-3个文件，5-12步完成
- 4: 4+个文件或多模块架构，10-18步
- 5: 需要跨系统集成或Agent无法在当前能力下完成

## 效率评分标准
- 1: 严重冗余，错误后盲重试≥3次
- 2: 有冗余操作，重复调用相同工具
- 3: 路径基本正确，偶有重复
- 4: 路径高效，无重复操作
- 5: 最优路径，每个文件一次写入成功

## 复盘要求
输出严格 JSON，只输出 JSON 对象，不要任何解释。

{
  "outcome": "success 或 failure",
  "difficulty_for_agent": 1-5 (严格按上述标准评分),
  "difficulty_evidence": "简要说明为什么给这个难度分",
  "what_worked": ["做得好的地方"],
  "what_could_be_better": ["即使成功了，哪里还能更好？如果失败了，本质缺什么能力？"],
  "strategy_used": "这次用什么策略解决问题",
  "new_skill_gained": {
    "name": "新技能名称（简短）",
    "description": "技能描述",
    "reusable": true/false,
    "trigger": "什么情况下可以用这个技能",
    "steps": "这个技能的操作步骤"
  },
  "efficiency_score": 1-5 (严格按上述标准评分),
  "efficiency_evidence": "简要说明为什么给这个效率分",
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
        step_count: int = 0,
        ability_context: str = "",
    ) -> dict:
        """复盘一次任务执行"""
        is_failure = "[STOPPED]" in result or "[ERROR]" in result
        result_status = "FAILED" if is_failure else "SUCCESS"
        result_preview = result[:500]
        ctx = ability_context or "Agent 是一个代码工程助手，正在持续成长中。"

        # 从执行轨迹粗略统计文件数和步数
        file_count = execution_trace.count("write_file") + execution_trace.count("call write_file")
        file_count = max(1, file_count)

        prompt = (POST_MORTEM_PROMPT
            .replace("{ ability_context }", ctx)
            .replace("{task_desc}", task_desc[:500])
            .replace("{step_count}", f"{step_count}步, 涉及{file_count}个文件")
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

    def detect_repeating_weakness(self, min_occurrences: int = 3, window: int = 5) -> dict | None:
        """检测近期复盘中反复出现的弱点模式

        当同一类改进建议在最近 N 次复盘中出现 ≥M 次时，
        说明这不是偶发失误而是系统性缺陷，应触发进化链。

        Returns:
            {pattern, count, insights: [...]} 或 None
        """
        if len(self.history) < min_occurrences:
            return None

        recent = self.history[-window:]
        patterns: dict[str, list[str]] = {}

        for h in recent:
            for item in h.get("what_could_be_better", []):
                key = self._normalize_pattern(item)
                if key:
                    if key not in patterns:
                        patterns[key] = []
                    patterns[key].append(item)

            insight = h.get("growth_insight", "")
            if insight:
                key = self._normalize_pattern(insight)
                if key:
                    if key not in patterns:
                        patterns[key] = []
                    patterns[key].append(insight)

        for key, items in patterns.items():
            if len(items) >= min_occurrences:
                return {
                    "pattern": key,
                    "count": len(items),
                    "insights": items[-3:],
                }

        return None

    @staticmethod
    def _normalize_pattern(text: str) -> str:
        """提取文本中的核心关键词作为模式签名"""
        keywords = [
            "缺乏", "不足", "错误", "冗余", "盲目", "重复",
            "调试", "规划", "验证", "分析", "策略", "切换",
            "上下文", "工具", "步骤", "执行", "回路", "重试",
            "debug", "错误输出", "试错", "无进展", "不读", "未读",
        ]
        parts = []
        for kw in keywords:
            if kw in text:
                parts.append(kw)
        if parts:
            return "+".join(sorted(parts))
        # Fallback: use first 5 chars... no, use a length-based hash
        return text[:30].strip()
