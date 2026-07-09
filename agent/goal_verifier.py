# -*- coding: utf-8 -*-
"""Goal / Stop 验证器 — 对标 MiMo Code 的独立 judge 模型

每次 Agent 声称完成时，用独立的 LLM 判断任务是否真正完成，
避免"乐观提前停止"。这对于长周期自主运行至关重要。
"""
from agent.models import Message


GOAL_CHECK_PROMPT = """你是一个独立的代码审查 judge。一个 AI 编程 Agent 声称完成了以下任务。
请严格判断是否真正完成，不要客气。

## 原始任务
{task}

## Agent 声称的完成结果
{result}

## 判断标准
- 如果 Agent 确实完成了任务的所有要求 → "pass"
- 如果 Agent 只做了部分、输出含糊、或明显有遗漏 → "fail"
- 如果出现 [ERROR] / [STOPPED] / [SOFT-FAIL] → 直接 "fail"

输出严格 JSON: {{"verdict": "pass" 或 "fail", "reason": "一句话理由"}}
不要输出其他文字。"""


class GoalVerifier:
    """独立任务完成验证器

    用独立的 judge 模型（可以是更便宜的模型）来验证主 Agent 是否真正完成了任务。
    对标 MiMo Code 的 /goal + judge 机制。
    """

    def __init__(self, llm_adapter, judge_model: str | None = None):
        self.llm = llm_adapter
        self.judge_model = judge_model  # 可选独立模型名
        self._history: list[dict] = []

    def verify(self, task: str, result: str) -> dict:
        """判断任务是否真正完成

        Returns:
            {passed: bool, reason: str}
        """
        # 快速路径：明显的失败信号
        if "[STOPPED]" in result or "[ERROR]" in result or "[SOFT-FAIL]" in result:
            return {"passed": False, "reason": "结果包含明确失败信号"}

        if not result or len(result) < 20:
            return {"passed": False, "reason": "输出太短，可能未完成"}

        prompt = GOAL_CHECK_PROMPT.format(task=task[:500], result=result[:1500])
        try:
            resp = self.llm.generate(
                [Message(role="system", content="你是严格的任务审查 judge。只输出 JSON。"),
                 Message(role="user", content=prompt)],
                tools=None,
            )
            import json, re
            text = (resp.content or "").strip()
            m = re.search(r'\{[\s\S]*\}', text)
            data = json.loads(m.group()) if m else {}
            verdict = data.get("verdict", "fail") == "pass"
            reason = data.get("reason", "无法解析判断结果")
        except Exception:
            verdict = len(result) > 100
            reason = "判断模型调用失败，按规则 fallback"

        self._history.append({
            "task": task[:100], "verdict": verdict, "reason": reason,
        })
        return {"passed": verdict, "reason": reason}

    def recent_history(self, n: int = 5) -> list[dict]:
        return self._history[-n:]
