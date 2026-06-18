# -*- coding: utf-8 -*-
"""修复历史持久化 — 记录有效修复，跨会话复用"""
import json
import os
from datetime import datetime


class FixHistory:
    """修复历史管理器

    持久化保存成功应用的修复方案，按失败模式（error_type + 关键词签名）
    分组存储。下次遇到相似失败时优先复用已验证的修复。

    可通过 get_tunable_params() / apply_params() 被 MetaOptimizer 调优。
    """

    def __init__(self, filepath: str = "./fix_history.json", similarity_threshold: float = 0.3):
        self.filepath = filepath
        self.similarity_threshold = similarity_threshold
        self.records: list[dict] = []
        self._snapshot_data: dict | None = None
        self._load()

    def _load(self):
        if os.path.exists(self.filepath):
            with open(self.filepath, "r", encoding="utf-8") as f:
                self.records = json.load(f)
        else:
            self.records = []

    def _save(self):
        os.makedirs(os.path.dirname(self.filepath) or ".", exist_ok=True)
        with open(self.filepath, "w", encoding="utf-8") as f:
            json.dump(self.records, f, ensure_ascii=False, indent=2)

    @staticmethod
    def _make_signature(error_type: str, task_desc: str) -> str:
        """生成失败模式签名 — 用于匹配相似失败"""
        keywords = task_desc[:60].lower()
        return f"{error_type}:{keywords}"

    def record_fix(
        self,
        error_type: str,
        task_desc: str,
        fix: dict,
        root_cause: dict,
        verified: bool = True,
    ):
        """记录一次成功的修复

        Args:
            error_type: 失败类型
            task_desc: 任务描述
            fix: 修复方案 dict
            root_cause: 根因分析结果
            verified: 是否验证通过
        """
        if not verified:
            return

        record = {
            "signature": self._make_signature(error_type, task_desc),
            "error_type": error_type,
            "task_desc": task_desc[:120],
            "fix": {
                "fix_type": fix.get("fix_type", ""),
                "original": fix.get("original", {}),
                "fixed": fix.get("fixed", {}),
            },
            "root_cause": {
                "type": root_cause.get("root_cause_type", ""),
                "confidence": root_cause.get("confidence", 0),
            },
            "timestamp": datetime.now().isoformat(),
            "reuse_count": 0,
        }
        self.records.append(record)
        self._save()

    def find_similar(self, error_type: str, task_desc: str, min_confidence: float = 0.5) -> list[dict]:
        """查找相似失败模式的历史修复

        Args:
            error_type: 当前失败类型
            task_desc: 当前任务描述
            min_confidence: 最低置信度阈值

        Returns:
            匹配的历史修复记录列表，按 reuse_count 降序
        """
        sig = self._make_signature(error_type, task_desc)
        matches = []
        for r in self.records:
            if r["error_type"] == error_type:
                # Simple substring match on signature
                r_sig = r.get("signature", "")
                overlap = sum(1 for c in sig if c in r_sig) / max(len(sig), 1)
                if overlap > self.similarity_threshold and r["root_cause"].get("confidence", 0) >= min_confidence:
                    matches.append((overlap, r))
        matches.sort(key=lambda x: (-x[1].get("reuse_count", 0), -x[0]))
        return [m[1] for m in matches]

    def mark_reused(self, fix: dict):
        """标记修复被复用"""
        for r in self.records:
            if r["fix"]["fix_type"] == fix.get("fix_type"):
                r["reuse_count"] = r.get("reuse_count", 0) + 1
                self._save()
                break

    def get_stats(self) -> dict:
        """获取修复历史统计"""
        by_type = {}
        for r in self.records:
            ft = r["fix"]["fix_type"]
            by_type[ft] = by_type.get(ft, 0) + 1
        return {
            "total_fixes": len(self.records),
            "by_fix_type": by_type,
            "most_reused": sorted(self.records, key=lambda r: r.get("reuse_count", 0), reverse=True)[:3],
        }

    def get_tunable_params(self) -> dict:
        """返回可被 MetaOptimizer 调优的参数"""
        return {"similarity_threshold": self.similarity_threshold}

    def apply_params(self, params: dict):
        if "similarity_threshold" in params:
            self.similarity_threshold = float(params["similarity_threshold"])

    def snapshot(self) -> dict:
        self._snapshot_data = {"similarity_threshold": self.similarity_threshold}
        return self._snapshot_data

    def restore(self, snapshot: dict | None = None):
        data = snapshot or self._snapshot_data
        if data:
            self.apply_params(data)
