# -*- coding: utf-8 -*-
"""元优化层 — 让自优化组件自己优化自己

当自优化闭环持续产出低质量结果时（根因分析置信度低、
修复全部回滚等），MetaOptimizer 诊断是哪个组件出了问题，
用 LLM 生成改进后的参数/prompt，应用到组件上，然后验证效果。
"""

from agent.models import Message

META_DIAGNOSIS_PROMPT = """你是一个 Agent 元优化专家。自优化系统在尝试修复 Agent 失败时，
自身表现不佳。以下是最近几次自优化运行的质量数据：

{quality_report}

请判断哪个自优化组件最可能出了问题，只从以下选择：

- root_cause_analyzer: 根因分析器的 prompt 不够好，导致置信度持续偏低
- self_repair: 修复策略引擎的 prompt 不够好，导致生成的修复无效
- verify: 验证器的失败判断标准不够准确
- fix_history: 修复历史匹配阈值不合适

输出严格 JSON，包含 {weak_component, reason, confidence}。"""

META_FIX_PROMPT = """你是一个 Agent 元优化专家。自优化系统的组件「{component}」存在问题：
{reason}

该组件的当前可调参数：
{current_params}

请直接输出改进后的参数值。只输出一个 JSON，key 为参数名，value 为新值。
不要输出解释文字，不要包裹在 ``` 中。"""


class MetaOptimizer:
    """自优化系统的自优化器

    监控 run_self_optimize() 的输出质量，当检测到系统性失败时：
    1. 诊断哪个组件有问题（root_cause / self_repair / verify / fix_history）
    2. 用 LLM 生成改进后的参数/prompt
    3. 应用到组件
    4. 重新验证
    5. 有效保留，无效回滚
    """

    def __init__(self, llm_adapter, meta_fix_history_file: str = "./meta_fix_history.json"):
        self.llm = llm_adapter
        self.meta_history: list[dict] = []
        self._meta_fix_history_file = meta_fix_history_file

    def should_optimize(self, report: dict) -> bool:
        """判断是否需要触发元优化

        触发条件：有分析但无有效修复（全都回滚或跳过）
        """
        analyzed = report.get("analyzed", 0)
        kept = report.get("fixes_kept", 0)
        rolled = report.get("fixes_rolled_back", 0)
        generated = report.get("fixes_generated", 0)

        if analyzed == 0:
            return False
        if kept == 0 and generated > 0:
            return True
        if generated > 0 and rolled / generated > 0.7:
            return True
        return False

    def optimize(self, optimize_report: dict, agent_loop) -> dict:
        """对自优化系统执行元优化

        Args:
            optimize_report: run_self_optimize() 的返回结果
            agent_loop: AgentLoop 实例

        Returns:
            {triggered, weak_component, meta_fix, applied, improved, detail}
        """
        if not self.should_optimize(optimize_report):
            return {"triggered": False, "detail": "不需要元优化"}

        # 1. 收集质量报告
        quality_report = self._build_quality_report(optimize_report)

        # 2. 诊断弱组件
        diagnosis = self._diagnose_weak_component(quality_report)
        weak = diagnosis.get("weak_component", "root_cause_analyzer")
        reason = diagnosis.get("reason", "未知")

        # 3. 获取该组件的当前参数
        component = self._get_component(agent_loop, weak)
        if component is None:
            return {"triggered": True, "weak_component": weak, "detail": f"未找到组件 {weak}", "applied": False}

        current_params = component.get_tunable_params()
        snapshot = component.snapshot()

        # 4. 生成改进参数
        meta_fix = self._generate_meta_fix(weak, reason, current_params)
        if not meta_fix:
            return {"triggered": True, "weak_component": weak, "detail": "未能生成有效修复", "applied": False}

        # 5. 应用
        try:
            component.apply_params(meta_fix)
        except Exception:
            return {"triggered": True, "weak_component": weak, "detail": "参数应用失败", "applied": False}

        # 6. 重新运行自优化验证
        improved = False
        try:
            agent_loop._last_failure_cases.clear()
            # Cannot re-create failures, so judge by fixability
            new_report = agent_loop.run_self_optimize()
            new_kept = new_report.get("fixes_kept", 0)
            old_kept = optimize_report.get("fixes_kept", 0)
            improved = new_kept > old_kept
        except Exception:
            improved = False

        # 7. 保留或回滚
        result = {
            "triggered": True,
            "weak_component": weak,
            "reason": reason,
            "meta_fix_applied": True,
            "improved": improved,
            "params_before": snapshot,
            "params_after": meta_fix,
        }

        if improved:
            result["action"] = "kept"
        else:
            component.restore(snapshot)
            result["action"] = "rolled_back"

        self.meta_history.append(result)
        return result

    def _build_quality_report(self, report: dict) -> str:
        """构建质量报告文本"""
        lines = [
            f"- 总分析数: {report.get('analyzed', 0)}",
            f"- 生成修复数: {report.get('fixes_generated', 0)}",
            f"- 已应用: {report.get('fixes_applied', 0)}",
            f"- 已验证保留: {report.get('fixes_kept', 0)}",
            f"- 已回滚: {report.get('fixes_rolled_back', 0)}",
        ]
        for d in report.get("details", []):
            lines.append(
                f"  case[{d.get('case_id', '?')}]: root_cause={d.get('root_cause', '?')} "
                f"confidence={d.get('confidence', '?')} action={d.get('action', '?')}"
            )
        return "\n".join(lines)

    def _diagnose_weak_component(self, quality_report: str) -> dict:
        """用 LLM 诊断哪个自优化组件有问题"""
        prompt = META_DIAGNOSIS_PROMPT.replace("{quality_report}", quality_report)
        try:
            resp = self.llm.generate(
                [Message(role="system", content="输出严格 JSON，不要任何解释文字。"),
                 Message(role="user", content=prompt)],
                tools=None,
            )
            import json
            text = (resp.content or "").strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[-1].rsplit("```", 1)[0]
            return json.loads(text)
        except Exception:
            return {"weak_component": "root_cause_analyzer", "reason": "自动诊断失败，默认诊断根因分析器", "confidence": 0.3}

    def _generate_meta_fix(self, component: str, reason: str, current_params: dict) -> dict | None:
        """用 LLM 生成组件改进参数"""
        params_text = "\n".join(f"  {k}: {v}" for k, v in current_params.items())
        prompt = (META_FIX_PROMPT
            .replace("{component}", component)
            .replace("{reason}", reason)
            .replace("{current_params}", params_text))
        try:
            resp = self.llm.generate(
                [Message(role="system", content="只输出 JSON 对象，不要包裹在 ``` 中。"),
                 Message(role="user", content=prompt)],
                tools=None,
            )
            import json
            text = (resp.content or "").strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[-1].rsplit("```", 1)[0]
            result = json.loads(text)
            if isinstance(result, dict) and len(result) > 0:
                return result
        except Exception:
            pass
        return None

    @staticmethod
    def _get_component(agent_loop, name: str):
        """根据名称获取 AgentLoop 上的自优化组件"""
        mapping = {
            "root_cause_analyzer": "_root_cause_analyzer",
            "self_repair": "_self_repair",
            "verify": "_verify",
            "fix_history": "_fix_history",
        }
        attr = mapping.get(name, "")
        return getattr(agent_loop, attr, None)
