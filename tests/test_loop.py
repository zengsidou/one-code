# -*- coding: utf-8 -*-
"""Micro-Agent 框架单元测试"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.registry import ToolRegistry
from tools.schema import generate_tool_schema, python_type_to_json_type
from memory.short_term import ShortTermMemory
from memory.long_term import LongTermMemory
from agent.models import Message, ToolCall, AgentState, StepResult


def test_tool_registry():
    registry = ToolRegistry(safe_mode=False)

    @registry.register("echo", "Echo back input")
    def echo(text: str) -> str:
        return text

    assert "echo" in registry.tool_names, "Tool not registered"
    assert registry.execute("echo", {"text": "hello"}) == "hello", "Tool execution failed"
    print("  [PASS] test_tool_registry")


def test_tool_schema():
    schema = generate_tool_schema("add", "Add numbers", {"a": (int, "first"), "b": (int, "second")})
    assert schema["function"]["name"] == "add"
    assert schema["function"]["parameters"]["properties"]["a"]["type"] == "integer"
    print("  [PASS] test_tool_schema")


def test_python_type_to_json_type():
    assert python_type_to_json_type(str) == "string"
    assert python_type_to_json_type(int) == "integer"
    assert python_type_to_json_type(float) == "number"
    assert python_type_to_json_type(bool) == "boolean"
    print("  [PASS] test_python_type_to_json_type")


def test_safe_mode():
    registry = ToolRegistry(safe_mode=True)

    @registry.register("echo", "Echo")
    def echo(text: str) -> str:
        return text

    result = registry.execute("echo", {"text": "rm -rf / --no-preserve-root"})
    assert "BLOCKED" in result, "Safe mode should block dangerous commands"
    print("  [PASS] test_safe_mode")


def test_short_term_memory():
    memory = ShortTermMemory(max_tokens=1000)
    for i in range(5):
        memory.add(Message(role="user", content=f"msg{i} " + "x" * 200))
    assert len(memory) > 0
    assert len(memory) <= 5
    print("  [PASS] test_short_term_memory")


def test_models():
    tc = ToolCall(id="1", name="test_tool", arguments={"key": "val"})
    msg = Message(role="assistant", content="hi", tool_calls=[tc])
    assert msg.tool_calls[0].name == "test_tool"
    assert AgentState.IDLE.value == "idle"
    assert AgentState.THINKING.value == "thinking"
    print("  [PASS] test_models")


def test_max_steps():
    from llm.base import BaseLLM

    class MockLLM(BaseLLM):
        def generate(self, messages, tools=None):
            return Message(role="assistant", content="mock response", tool_calls=None)
        def embed(self, text):
            return [0.0]

    from agent.loop import AgentLoop
    from memory import MemoryManager

    registry = ToolRegistry()

    @registry.register("mock_tool", "Mock tool")
    def mock_tool(x: str) -> str:
        return x

    short_mem = ShortTermMemory(max_tokens=4096)
    long_mem = LongTermMemory(MockLLM(), persist_dir="./test_db")
    memory = MemoryManager(short=short_mem, long=long_mem)

    agent = AgentLoop(
        llm=MockLLM(),
        registry=registry,
        memory=memory,
        max_steps=3,
    )
    result = agent.run("test")
    assert "[STOPPED]" in result or "mock response" in result
    print("  [PASS] test_max_steps")


if __name__ == "__main__":
    print("Running Micro-Agent tests...\n")
    test_tool_registry()
    test_tool_schema()
    test_python_type_to_json_type()
    test_safe_mode()
    test_short_term_memory()
    test_models()
    test_max_steps()
    print("\nAll tests passed!")
