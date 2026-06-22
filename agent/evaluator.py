# -*- coding: utf-8 -*-
"""Independent Evaluator Agent — separate evaluation from generation.

Mirrors Anthropic's 3-agent architecture: Generator produces code, Evaluator 
independently checks it. The evaluator uses its own LLM instance and doesn't 
share context with the generator, providing unbiased quality assessment.
"""
from agent.models import Message


EVALUATOR_PROMPT = (
    "你是一个独立的代码评审 Agent。你的任务是评估另一个 Agent 生成的结果质量。\n\n"
    "## 评估维度（每项 0-3 分）\n"
    "1. 功能正确性: 是否解决了提出的问题？\n"
    "2. 代码质量: 是否清晰、无冗余、遵循规范？\n"
    "3. 安全性: 是否有注入风险、敏感信息泄露、危险操作？\n"
    "4. 完整性: 是否处理了边界情况？\n"
    "5. 可维护性: 是否容易理解和修改？\n\n"
    "## 评估规则\n"
    "- 不以原始回答者的自信程度为准，只基于实际代码质量\n"
    "- 如果看到测试结果，检查是否真的有测试 vs 只是声称通过了\n"
    "- 特别关注：文件路径是否安全、是否有硬编码密钥、SQL 是否参数化\n\n"
    "输出 JSON: {\"score\": 总均分, \"dimensions\": {\"correctness\": N, ...}, "
    "\"risks\": [\"风险1\", ...], \"verdict\": \"pass\"|\"warn\"|\"fail\"}"
)


class Evaluator:
    """独立的评估 Agent，与生成 Agent 完全隔离上下文。

    用法:
        evaluator = Evaluator(llm_adapter)
        result = evaluator.evaluate(task, generator_output, generated_files)
        if result["verdict"] == "fail":
            ...  # reject or regenerate
    """

    def __init__(self, llm_adapter, audit=None):
        """
        Args:
            llm_adapter: BaseLLM 实例，用于评估（建议与生成用不同模型实例）
            audit: 可选的 AuditLogger，记录评估结果
        """
        self.llm = llm_adapter
        self.audit = audit
        self._history: list[dict] = []

    def evaluate(
        self,
        task: str,
        output: str,
        files: list[str] | None = None,
        test_results: str = "",
    ) -> dict:
        """评估生成结果的质量。

        Args:
            task: 原始任务描述
            output: 生成 Agent 的最终输出
            files: 修改的文件列表
            test_results: 测试运行结果（如有）

        Returns:
            {score, dimensions, risks, verdict}
        """
        context = (
            f"## 任务\n{task[:1000]}\n\n"
            f"## 生成结果\n{output[:3000]}\n\n"
        )
        if files:
            context += f"## 修改文件\n" + "\n".join(files[:10]) + "\n\n"
        if test_results:
            context += f"## 测试结果\n{test_results[:500]}\n\n"

        prompt = EVALUATOR_PROMPT + "\n\n" + context

        try:
            resp = self.llm.generate(
                [Message(role="user", content=prompt)],
                tools=None,
            )
            import json
            text = (resp.content or "").strip()
            # Extract JSON
            import re
            m = re.search(r"\{[\s\S]*\}", text)
            if m:
                result = json.loads(m.group())
            else:
                result = {"score": 0, "verdict": "fail", "error": "no_json"}
        except Exception as e:
            result = {"score": 0, "verdict": "fail", "error": str(e)}

        result["task"] = task[:100]
        self._history.append(result)

        if self.audit:
            self.audit.record(
                agent_id="evaluator",
                action="evaluate",
                details=f"score={result.get('score', 0)} verdict={result.get('verdict', '?')}",
                risk_level="low",
            )

        return result

    def get_stats(self) -> dict:
        """Get evaluator statistics."""
        if not self._history:
            return {"evaluations": 0}

        scores = [e.get("score", 0) for e in self._history]
        verdicts = {}
        for e in self._history:
            v = e.get("verdict", "unknown")
            verdicts[v] = verdicts.get(v, 0) + 1

        return {
            "evaluations": len(self._history),
            "avg_score": round(sum(scores) / len(scores), 2),
            "verdicts": verdicts,
            "pass_rate": round(verdicts.get("pass", 0) / len(self._history), 2),
        }
