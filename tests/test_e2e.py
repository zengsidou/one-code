# -*- coding: utf-8 -*-
"""真正的端到端测试 — 模拟 LLM 响应验证完整 Agent Loop 行为"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from unittest.mock import MagicMock, patch
from agent.models import Message, ToolCall, Contract, ContractStep
from agent.loop import AgentLoop, DEFAULT_SYSTEM_PROMPT
from tools.registry import ToolRegistry


class FakeLLM:
    """可控的假 LLM，返回预设的 Message"""
    def __init__(self, responses: list[Message]):
        self.responses = responses
        self.call_count = 0
        self.messages_seen: list[list[Message]] = []
        self.model = "fake"

    def generate(self, messages: list[Message], tools=None) -> Message:
        self.messages_seen.append(messages)
        if self.call_count < len(self.responses):
            resp = self.responses[self.call_count]
            self.call_count += 1
            return resp
        return Message(role="assistant", content="done")

    def embed(self, text: str) -> list[float]:
        return [0.0] * 1024


def make_fake_registry():
    r = ToolRegistry(safe_mode=False)
    @r.register("hello", "say hello")
    def hello(name: str = "World"): return f"Hello, {name}!"
    @r.register("read_file", "read file")
    def read_file(path: str): return f"content of {path}"
    return r


class TestAgentLoopEndToEnd:
    def test_simple_conversation(self):
        """最基本的对话：用户说一句话，Agent 不调用工具直接回复"""
        llm = FakeLLM([
            Message(role="assistant", content="你好！我是 One-Code。"),
        ])
        agent = AgentLoop(llm=llm, max_steps=5)
        result = agent.run("你好")
        assert "你好" in result
        assert "[STOPPED]" not in result
        assert "[ERROR]" not in result
        assert "[GOAL-FAIL]" not in result

    def test_tool_calling_roundtrip(self):
        """Agent 调用工具 → 获取结果 → 返回最终回答"""
        llm = FakeLLM([
            Message(role="assistant", content=None, tool_calls=[
                ToolCall(id="1", name="hello", arguments={"name": "World"}),
            ]),
            Message(role="assistant", content="结果已获取：Hello, World!"),
        ])
        reg = make_fake_registry()
        agent = AgentLoop(llm=llm, registry=reg, max_steps=10)
        result = agent.run("say hello")
        assert "Hello" in result
        assert "[STOPPED]" not in result

    def test_multi_tool_sequence(self):
        """连续调用多个工具，最终总结"""
        llm = FakeLLM([
            Message(role="assistant", content=None, tool_calls=[
                ToolCall(id="1", name="read_file", arguments={"path": "a.py"}),
            ]),
            Message(role="assistant", content=None, tool_calls=[
                ToolCall(id="2", name="read_file", arguments={"path": "b.py"}),
            ]),
            Message(role="assistant", content="两个文件都读到了，开始修改。"),
        ])
        reg = make_fake_registry()
        agent = AgentLoop(llm=llm, registry=reg, max_steps=10)
        result = agent.run("read two files")
        assert "修改" in result or "读" in result
        assert llm.call_count == 3

    def test_max_steps_hit(self):
        """步数耗尽时返回 [STOPPED]"""
        # 永远返回工具调用，直到步数耗尽
        calls = [Message(role="assistant", content=None, tool_calls=[
            ToolCall(id=str(i), name="hello", arguments={"name": "x"}),
        ]) for i in range(20)]
        llm = FakeLLM(calls)
        reg = make_fake_registry()
        agent = AgentLoop(llm=llm, registry=reg, max_steps=3)
        result = agent.run("loop forever")
        assert "[STOPPED]" in result

    def test_tool_loop_detection(self):
        """同一工具+参数连续调用被检测为回路"""
        calls = [Message(role="assistant", content=None, tool_calls=[
            ToolCall(id=str(i), name="hello", arguments={"name": "World"}),
        ]) for i in range(10)]
        llm = FakeLLM(calls)
        reg = make_fake_registry()
        agent = AgentLoop(llm=llm, registry=reg, max_steps=10, loop_detect_threshold=3)
        result = agent.run("loop")
        assert "[STOPPED]" in result
        assert "回路" in result or "重复" in result

    def test_tool_error_handling(self):
        """工具执行失败时继续尝试其他方法"""
        class FailThenOK:
            def __init__(self):
                self.calls = 0
                self.model = "fake"
            def generate(self, messages, tools=None):
                self.calls += 1
                if self.calls == 1:
                    return Message(role="assistant", content=None, tool_calls=[
                        ToolCall(id="1", name="unknown_tool", arguments={}),
                    ])
                return Message(role="assistant", content="工具不可用，我直接回答。")
            def embed(self, text): return [0.0] * 1024

        llm = FailThenOK()
        agent = AgentLoop(llm=llm, max_steps=10)
        result = agent.run("do something")
        assert "[STOPPED]" not in result

    def test_idle_detection(self):
        """连续两轮无工具调用 → 返回最终回答"""
        llm = FakeLLM([
            Message(role="assistant", content=None, tool_calls=[
                ToolCall(id="1", name="hello", arguments={"name": "x"}),
            ]),
            Message(role="assistant", content="完成。"),
        ])
        reg = make_fake_registry()
        agent = AgentLoop(llm=llm, registry=reg, max_steps=10)
        result = agent.run("test")
        assert "完成" in result
        # 第一轮工具调用后，第二轮无工具调用 → idle 1
        # 第三轮无工具调用 → idle 2 → 返回
        assert llm.call_count == 2

    def test_empty_input(self):
        """空输入依然能处理"""
        llm = FakeLLM([
            Message(role="assistant", content="请提供一个有效的问题。"),
        ])
        agent = AgentLoop(llm=llm, max_steps=5)
        result = agent.run("")
        assert len(result) > 0
        assert "[STOPPED]" not in result


class TestContractFirstFlow:
    def test_contract_detection_and_typing(self):
        from agent.contract_types import detect_contract_type, ContractType
        assert detect_contract_type("做个网页") == ContractType.VISUAL
        assert detect_contract_type("写API") == ContractType.CODE_API
        assert detect_contract_type("配置nginx参数") == ContractType.CONFIG

    def test_contract_first_disabled_by_default(self):
        llm = FakeLLM([Message(role="assistant", content="ok")])
        agent = AgentLoop(llm=llm, max_steps=5)
        assert agent.enable_contract_first == False


class TestEdgeCases:
    def test_very_short_response(self):
        llm = FakeLLM([Message(role="assistant", content="ok")])
        agent = AgentLoop(llm=llm, max_steps=5)
        result = agent.run("hi")
        assert result == "ok"

    def test_long_response_passes(self):
        long_msg = "这是一个很长的回复。" * 30  # ~300 chars
        llm = FakeLLM([Message(role="assistant", content=long_msg)])
        agent = AgentLoop(llm=llm, max_steps=5)
        result = agent.run("explain everything")
        assert long_msg in result
        assert "[GOAL-FAIL]" not in result  # GoalVerifier must not auto-trigger


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
