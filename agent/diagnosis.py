# -*- coding: utf-8 -*-
"""诊断模块 — 捕获 Agent 执行失败信息，结构化存储"""
import json
import os
import uuid
from datetime import datetime

from agent.models import Message


class FailureDiagnosis:
    """失败诊断记录管理器

    捕获 Agent Loop 执行过程中的失败信息，以结构化字典存储。
    支持 JSON 序列化/反序列化，按「未解决」状态检索。
    """

    def __init__(self):
        self.cases: list[dict] = []

    def capture_failure(
        self,
        task_desc: str,
        step: int,
        error_msg: str,
        context_snapshot: list[Message],
        error_type: str = "other",
    ) -> dict:
        """捕获一次执行失败

        Args:
            task_desc: 用户原始任务描述
            step: 失败的步骤编号
            error_msg: 错误信息
            context_snapshot: 失败时的上下文消息列表副本
            error_type: 错误类型 (tool_error / llm_format_error / timeout / hallucination / circuit_breaker / loop_detected / other)

        Returns:
            结构化的失败 case 字典
        """
        case = {
            "id": str(uuid.uuid4())[:8],
            "task_desc": task_desc,
            "failed_step": step,
            "error_type": error_type,
            "error_msg": error_msg,
            "context_snapshot": [
                {
                    "role": m.role,
                    "content": m.content,
                }
                for m in context_snapshot
            ],
            "timestamp": datetime.now().isoformat(),
            "root_cause": None,  # 待 RootCauseAnalyzer 填充
            "fix_applied": None,  # 待 SelfRepair 填充
            "fix_verified": None,  # 待 VerifyRepair 填充
            "resolved": False,
        }
        self.cases.append(case)
        return case

    def save_to_file(self, filepath: str):
        """将全部 cases 序列化到 JSON 文件"""
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.cases, f, ensure_ascii=False, indent=2)

    def load_from_file(self, filepath: str):
        """从 JSON 文件反序列化 cases"""
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                self.cases = json.load(f)

    def get_unresolved(self) -> list[dict]:
        """返回尚未解决（无 root_cause 或 resolved=False）的 cases"""
        return [c for c in self.cases if not c.get("resolved") and c.get("root_cause") is None]
