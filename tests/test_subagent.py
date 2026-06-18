# -*- coding: utf-8 -*-
"""SubAgent 委派系统测试"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm.base import BaseLLM
from tools.registry import ToolRegistry
from agent.subagent import SubAgent
from agent.models import Message


class MockLLM(BaseLLM):
    def generate(self, messages, tools=None):
        return Message(role="assistant", content="30", tool_calls=None)
    def embed(self, text): return [0.0]


def test_subagent_no_tools():
    """SubAgent with no tools - direct answer"""
    registry = ToolRegistry(safe_mode=False)
    sub = SubAgent(llm=MockLLM(), registry=registry, max_steps=1)
    result = sub.run("只回复一个字: 好")
    assert len(result.strip()) > 0
    print("  [PASS] test_subagent_no_tools")


def test_subagent_with_tool_call():
    """SubAgent with tool_calls in assistant message"""
    registry = ToolRegistry(safe_mode=False)

    @registry.register("echo", "Echo back")
    def echo(text: str) -> str:
        return text

    class ToolMockLLM(BaseLLM):
        def generate(self, messages, tools=None):
            from agent.models import ToolCall
            if tools and any("echo" in str(m.content) for m in messages if m.role == "user"):
                return Message(role="assistant", content="30", tool_calls=None)
            if tools:
                return Message(role="assistant", content="call",
                    tool_calls=[ToolCall(id="c1", name="echo", arguments={"text": "test"})])
            return Message(role="assistant", content="done", tool_calls=None)
        def embed(self, text): return [0.0]

    sub = SubAgent(llm=ToolMockLLM(), registry=registry, max_steps=2)
    result = sub.run("test")
    assert len(result.strip()) > 0
    print("  [PASS] test_subagent_with_tool_call")


if __name__ == "__main__":
    print("Running SubAgent tests...\n")
    test_subagent_no_tools()
    test_subagent_with_tool_call()
    print("\nAll SubAgent tests passed!")
