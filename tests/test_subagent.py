# -*- coding: utf-8 -*-
"""SubAgent 委派系统测试"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm.ollama import OllamaClient
from tools.registry import ToolRegistry
from agent.subagent import SubAgent

llm = OllamaClient(model="deepseek-r1:8b", embedding_model="bge-m3:latest")


def test_subagent_with_calculate():
    """SubAgent uses calculate tool for simple math"""
    registry = ToolRegistry(safe_mode=False)

    @registry.register("calculate", "Execute math expression")
    def calculate(expression: str) -> str:
        import math
        allowed = {"__builtins__": {}, **{k: getattr(math, k) for k in dir(math) if not k.startswith("_")}}
        return str(eval(expression, allowed))

    sub = SubAgent(llm=llm, registry=registry, max_steps=2)
    result = sub.run("计算 10+20，直接回复结果数字")

    assert "30" in result, f"Expected 30 in result, got: {result[:200]}"
    print("  [PASS] test_subagent_with_calculate")


def test_subagent_no_tools():
    """SubAgent with no tools - direct answer"""
    registry = ToolRegistry(safe_mode=False)
    sub = SubAgent(llm=llm, registry=registry, max_steps=1)
    result = sub.run("只回复一个字: 好")
    assert len(result.strip()) > 0
    print("  [PASS] test_subagent_no_tools")


if __name__ == "__main__":
    print("Running SubAgent tests...\n")
    test_subagent_with_calculate()
    test_subagent_no_tools()
    print("\nAll SubAgent tests passed!")
