# -*- coding: utf-8 -*-
"""主动挑战生成器 — 基于能力画像生成递增难度任务

ChallengeGenerator 根据 Agent 的强弱项和当前能力水平，
生成针对性练习任务，推动能力边界向外扩展。
"""

from agent.models import Message

CHALLENGE_PROMPT = """你是一个代码工程 mentor。你的学生是一个 AI Agent，
当前能力画像如下：

## 能力画像
{profile_summary}

## 弱项领域
{weak_areas}

## 当前难度承受力
当前能稳定完成难度 {current_level} 的任务。

## 要求
生成 {count} 个代码工程任务，难度在 {difficulty_range} 范围内，
优先覆盖弱项领域，帮助 Agent 成长。
每个任务要具体、可执行、有明确的完成标准。

输出严格 JSON 数组，每个元素包含：
[
  {{
    "task": "任务描述（一句话，中文）",
    "category": "debug/feature/refactor/review",
    "difficulty": 1-5,
    "expected_files": 涉及文件数估计,
    "success_criteria": "完成标准",
    "target_skill": "这个任务主要锻炼什么能力"
  }}
]
"""


class ChallengeGenerator:
    """Agent 成长挑战生成器

    根据当前能力画像，生成难度递增的代码任务，
    推动 Agent 突破能力边界。
    """

    def __init__(self, llm_adapter):
        self.llm = llm_adapter

    def generate(
        self,
        profile_summary: str,
        weak_areas: list[str],
        current_level: int = 2,
        count: int = 3,
    ) -> list[dict]:
        """生成递增难度的代码任务

        Args:
            profile_summary: 能力画像摘要
            weak_areas: 弱项领域列表
            current_level: 当前能稳定完成的最高难度
            count: 生成任务数量

        Returns:
            任务列表 [{task, category, difficulty, ...}]
        """
        # 难度范围：当前级别到当前+2
        level_min = max(1, current_level)
        level_max = min(5, current_level + 2)
        difficulty_range = f"{level_min} 到 {level_max}"

        weak_text = ", ".join(weak_areas) if weak_areas else "无明显弱项，全面发展"

        prompt = (CHALLENGE_PROMPT
            .replace("{profile_summary}", profile_summary)
            .replace("{weak_areas}", weak_text)
            .replace("{current_level}", str(current_level))
            .replace("{count}", str(count))
            .replace("{difficulty_range}", difficulty_range))

        try:
            resp = self.llm.generate(
                [Message(role="system", content="你是一个代码工程 mentor。只输出 JSON 数组。"),
                 Message(role="user", content=prompt)],
                tools=None,
            )
            return self._parse_tasks(resp.content or "")
        except Exception:
            return []

    @staticmethod
    def _parse_tasks(text: str) -> list[dict]:
        """解析 LLM 返回的任务 JSON 数组"""
        import json
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:]) if len(lines) > 1 else text[3:]
        if text.endswith("```"):
            text = text[:-3]
        try:
            tasks = json.loads(text)
            if isinstance(tasks, list):
                return tasks
        except json.JSONDecodeError:
            pass
        return []

    def suggest_next_level(self, profile) -> dict | None:
        """根据能力画像建议下一步挑战

        Args:
            profile: AbilityProfile 实例

        Returns:
            建议的挑战信息，或 None（如果数据不足）
        """
        summary = profile.get_growth_summary()
        if summary["total_tasks"] < 5:
            return None

        stats = profile.get_all_category_stats()
        weak = profile.get_weak_areas()

        recent_rate = summary.get("recent_success_rate", 0)
        recent_diff = summary.get("recent_avg_diff", 2)
        overall_diff = summary.get("overall_avg_diff", 2)

        # 近期成功率 > 70% → 升级
        if recent_rate >= 0.7 and summary["total_tasks"] >= 4:
            return {
                "action": "upgrade",
                "current_level": recent_diff,
                "suggested_level": min(5, int(recent_diff + 1)),
                "reason": f"近期成功率{recent_rate:.0%}，可以挑战更高难度",
            }

        # 近期成功率 < 40% → 降级
        if recent_rate < 0.4 and summary["total_tasks"] >= 4:
            return {
                "action": "consolidate",
                "current_level": recent_diff,
                "suggested_level": max(1, int(recent_diff - 1)),
                "reason": f"近期成功率为 {recent_rate:.0%}，需要巩固当前难度",
                "focus_weak_areas": weak,
            }

        return {"action": "maintain", "current_level": summary["recent_avg_diff"]}
