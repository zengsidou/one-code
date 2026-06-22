# -*- coding: utf-8 -*-
"""能力画像模块 — 追踪 Agent 在不同任务类型上的成长曲线

AbilityProfile 持续记录每次任务的：
- 任务类型（debug / feature / refactor / review）
- 难度级别（1-5）
- 是否成功
- 效率评分
- 随时间变化的能力趋势
"""
import json
import os
from datetime import datetime


TASK_CATEGORIES = ["debug", "feature", "refactor", "review", "other"]


class AbilityProfile:
    """Agent 能力画像

    追踪 Agent 在不同任务类型上的成功率、效率、难度承受力，
    生成成长曲线，识别强弱项。
    """

    def __init__(self, filepath: str = "./ability_profile.json"):
        self.filepath = filepath
        self.records: list[dict] = []
        self._load()

    def _load(self):
        if os.path.exists(self.filepath):
            with open(self.filepath, "r", encoding="utf-8") as f:
                try: self.records = json.load(f)
                except (json.JSONDecodeError, IOError): self.records = []

    def _save(self):
        os.makedirs(os.path.dirname(self.filepath) or ".", exist_ok=True)
        with open(self.filepath, "w", encoding="utf-8") as f:
            json.dump(self.records, f, ensure_ascii=False, indent=2)

    @staticmethod
    def _classify_task(task_desc: str) -> str:
        """根据任务描述自动分类"""
        desc = task_desc.lower()
        if any(w in desc for w in ["bug", "修复", "fix", "debug", "错误", "报错", "异常"]):
            return "debug"
        if any(w in desc for w in ["重构", "refactor", "重写", "改造", "拆分"]):
            return "refactor"
        if any(w in desc for w in ["review", "审查", "检查", "审计"]):
            return "review"
        if any(w in desc for w in ["新增", "添加", "实现", "开发", "feature", "add", "编写", "创建"]):
            return "feature"
        return "other"

    def record(self, task_desc: str, success: bool, difficulty: int, efficiency: int, steps: int = 0):
        """记录一次任务执行

        Args:
            task_desc: 任务描述
            success: 是否成功
            difficulty: 难度 1-5
            efficiency: 效率 1-5
            steps: 执行步数
        """
        category = self._classify_task(task_desc)
        self.records.append({
            "timestamp": datetime.now().isoformat(),
            "category": category,
            "task_desc": task_desc[:200],
            "success": success,
            "difficulty": difficulty,
            "efficiency": efficiency,
            "steps": steps,
        })
        self._save()

    def get_category_stats(self, category: str = "all") -> dict:
        """获取某类任务的统计"""
        if category == "all":
            recs = self.records
        else:
            recs = [r for r in self.records if r.get("category") == category]

        if not recs:
            return {"count": 0, "success_rate": 0, "avg_difficulty": 0, "avg_efficiency": 0}

        successes = sum(1 for r in recs if r.get("success"))
        return {
            "count": len(recs),
            "success_rate": round(successes / len(recs), 2),
            "avg_difficulty": round(sum(r.get("difficulty", 3) for r in recs) / len(recs), 1),
            "avg_efficiency": round(sum(r.get("efficiency", 3) for r in recs) / len(recs), 1),
            "avg_steps": round(sum(r.get("steps", 0) for r in recs) / len(recs), 1),
        }

    def get_all_category_stats(self) -> dict:
        """获取所有分类统计"""
        stats = {"all": self.get_category_stats("all")}
        for cat in TASK_CATEGORIES:
            s = self.get_category_stats(cat)
            if s["count"] > 0:
                stats[cat] = s
        return stats

    def get_weak_areas(self) -> list[str]:
        """识别弱项 — 成功率 < 50% 或样本>3 且效率<3 的分类"""
        weak = []
        for cat in TASK_CATEGORIES:
            s = self.get_category_stats(cat)
            if s["count"] >= 3 and s["success_rate"] < 0.5:
                weak.append(cat)
            if s["count"] >= 3 and s["avg_efficiency"] < 3.0:
                if cat not in weak:
                    weak.append(cat)
        return weak

    def get_growth_summary(self, window: int = 10) -> dict:
        """获取近期成长摘要 — 对比最近 N 次 vs 全部"""
        recent = self.records[-window:] if len(self.records) >= window else self.records
        all_records = self.records

        def avg(recs, key, default=3):
            if not recs:
                return default
            return round(sum(r.get(key, default) for r in recs) / len(recs), 1)

        return {
            "total_tasks": len(all_records),
            "recent_window": len(recent),
            "recent_success_rate": round(
                sum(1 for r in recent if r.get("success")) / max(len(recent), 1), 2
            ),
            "overall_success_rate": round(
                sum(1 for r in all_records if r.get("success")) / max(len(all_records), 1), 2
            ),
            "recent_avg_diff": avg(recent, "difficulty"),
            "overall_avg_diff": avg(all_records, "difficulty"),
            "recent_avg_efficiency": avg(recent, "efficiency"),
            "overall_avg_efficiency": avg(all_records, "efficiency"),
            "trend": self._trend(all_records),
        }

    @staticmethod
    def _trend(records: list[dict]) -> str:
        """判断能力趋势：上升 / 平稳 / 下降"""
        if len(records) < 10:
            return "数据不足，继续执行任务"
        first_half = records[: len(records) // 2]
        second_half = records[len(records) // 2 :]
        s1 = sum(1 for r in first_half if r.get("success")) / max(len(first_half), 1)
        s2 = sum(1 for r in second_half if r.get("success")) / max(len(second_half), 1)
        diff = s2 - s1
        if diff > 0.1:
            return "上升 ↑"
        elif diff < -0.1:
            return "下降 ↓"
        return "平稳 →"
