# -*- coding: utf-8 -*-
"""契约先行主编排器 — 契约生成→确认→逆向拆解→执行→校验"""

from agent.contract_types import detect_contract_type, get_contract_meta
from agent.contract_generator import ContractGenerator
from agent.reverse_decomposer import ReverseDecomposer
from agent.models import Message, Contract, ContractStep, ContractResult


class ContractFirstOrchestrator:
    """契约先行主编排器

    在Agent执行任务前：
    1. 自动检测契约类型
    2. 生成多模态契约预览
    3. 用户确认方向（CLI交互）
    4. 逆向拆解执行步骤
    5. 逐步执行 + 契约校验
    """

    def __init__(self, llm_adapter):
        self.llm = llm_adapter
        self.generator = ContractGenerator(llm_adapter)
        self.decomposer = ReverseDecomposer(llm_adapter)
        self._result: ContractResult | None = None

    def phase1_detect_and_generate(self, task: str) -> Contract:
        """阶段1: 检测类型 + 生成契约"""
        ct = detect_contract_type(task)
        meta = get_contract_meta(ct)
        contract = self.generator.generate(ct, task)
        self._result = ContractResult(contract=contract)

        print()
        print("=" * 60)
        print(f"  [契约先行] 类型: {ct.value} | 呈现: {meta['format']}")
        print(f"  [契约先行] 人判断方式: {meta['human_judge']}")
        print("=" * 60)
        print()
        print(contract.content)
        print()
        print("-" * 60)
        print(f"  方向总结: {contract.summary}")
        print("-" * 60)

        return contract

    def phase2_confirm(self) -> bool:
        """阶段2: 用户确认方向"""
        print()
        while True:
            choice = input("  [确认方向?] y=确认执行 / n=修改描述 / q=取消: ").strip().lower()
            if choice in ("y", "yes", ""):
                if self._result:
                    self._result.user_confirmed = True
                return True
            elif choice in ("n", "no", "改"):
                new_desc = input("  [修改描述] 请重新描述你的需求: ").strip()
                if new_desc and self._result:
                    ct = detect_contract_type(new_desc)
                    contract = self.generator.generate(ct, new_desc)
                    self._result.contract = contract
                    print()
                    print(contract.content)
                    print()
                    print("-" * 60)
                    print(f"  方向总结: {contract.summary}")
                    print("-" * 60)
                    print()
                continue
            elif choice in ("q", "quit", "exit"):
                return False
            print("  请输入 y / n / q")

    def phase3_decompose(self, task: str) -> list[ContractStep]:
        """阶段3: 逆向拆解"""
        if self._result is None or self._result.contract is None:
            return []
        steps = self.decomposer.decompose(self._result.contract, task)
        self._result.steps = steps

        print()
        print("=" * 60)
        print(f"  [逆向拆解] 从终点反推 {len(steps)} 个执行步骤:")
        for s in steps:
            deps = f" (依赖步骤: {s.depends_on})" if s.depends_on else ""
            print(f"  {s.index}. {s.goal}{deps}")
            if s.contract_checkpoint:
                print(f"     └ 校验点: {s.contract_checkpoint}")
        print("=" * 60)

        return steps

    def phase4_build_execution_prompt(self, task: str, contract: Contract, steps: list[ContractStep]) -> str:
        """阶段4: 构建执行上下文提示"""
        step_plan = "\n".join(
            f"{s.index}. {s.goal}"
            + (f"\n   契约校验: {s.contract_checkpoint}" if s.contract_checkpoint else "")
            for s in steps
        )

        prompt = f"""原始任务: {task}

## 契约预览（最终产物方向，执行过程中每步对照验证）
{contract.content}

## 逆向拆解执行步骤（从终点反推）
{step_plan}

执行规则:
1. 按步骤顺序执行，不要跳过
2. 每完成一步，对照契约预览检查产出是否符合方向
3. 如果偏离方向，即时修正
4. 全部完成后做最终契约一致性总结

请开始执行第一步。"""
        return prompt

    @property
    def result(self) -> ContractResult | None:
        return self._result
