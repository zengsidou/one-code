# -*- coding: utf-8 -*-
"""自优化闭环 — 单元测试"""
import sys
import os
import json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import patch, MagicMock
from agent.models import Message, ToolCall
from agent.diagnosis import FailureDiagnosis
from agent.root_cause import RootCauseAnalyzer
from agent.self_repair import SelfRepair
from agent.verify import VerifyRepair
from agent.fix_history import FixHistory
from agent.loop import AgentLoop, DEFAULT_SYSTEM_PROMPT
from tools.registry import ToolRegistry
from memory import MemoryManager
from memory.short_term import ShortTermMemory
from memory.long_term import LongTermMemory
from llm.base import BaseLLM


class MockLLM(BaseLLM):
    """Mock LLM — 返回固定响应，用于单元测试"""

    def __init__(self, response_content="mock response", model="mock-model"):
        self.response_content = response_content
        self.model = model

    def generate(self, messages, tools=None):
        return Message(role="assistant", content=self.response_content, tool_calls=None)

    def embed(self, text):
        return [0.0] * 1024


class MockLLMWithToolCall(BaseLLM):
    """Mock LLM — 返回工具调用"""

    def __init__(self, tool_name="mock_tool", tool_args=None):
        self.tool_name = tool_name
        self.tool_args = tool_args or {"x": "test"}
        self.model = "mock-model"

    def generate(self, messages, tools=None):
        if tools:
            return Message(
                role="assistant",
                content="calling tool",
                tool_calls=[ToolCall(id="call_1", name=self.tool_name, arguments=self.tool_args)],
            )
        return Message(role="assistant", content="mock response", tool_calls=None)

    def embed(self, text):
        return [0.0] * 1024


class MockLLMJSON(BaseLLM):
    """Mock LLM — 返回 JSON 响应，用于根因分析和修复生成"""

    def __init__(self, json_response=None):
        self.json_response = json_response or {}
        self.model = "mock-model"
        self.last_messages = None

    def generate(self, messages, tools=None):
        self.last_messages = messages
        return Message(role="assistant", content=json.dumps(self.json_response, ensure_ascii=False), tool_calls=None)

    def embed(self, text):
        return [0.0] * 1024


def _make_memory(llm=None):
    short = ShortTermMemory(max_tokens=4096)
    long = LongTermMemory(llm or MockLLM(), persist_dir="./test_memory_db")
    return MemoryManager(short=short, long=long)


# ─── 测试 1: DeepSeekAdapter 基本调用 ─────────────────────────────

def test_deepseek_adapter():
    from llm.deepseek_api import DeepSeekAdapter

    adapter = DeepSeekAdapter(api_key="sk-test-key")
    assert adapter.model == "deepseek-chat", "Default model should be deepseek-chat"
    assert adapter.api_key == "sk-test-key", "API key not set"

    # 测试 embed 占位
    emb = adapter.embed("test")
    assert len(emb) == 1024, "Embed should return 1024 zeros"
    assert all(v == 0.0 for v in emb), "All embed values should be 0"

    # 测试 _safe_parse_json
    assert adapter._safe_parse_json('{"a":1}') == {"a": 1}
    assert adapter._safe_parse_json("not json") == {}

    # 测试 _build_api_messages
    msgs = [
        Message(role="system", content="sys"),
        Message(role="user", content="hello"),
        Message(role="assistant", content="hi", tool_calls=[
            ToolCall(id="c1", name="test", arguments={"x": 1}),
        ]),
        Message(role="tool", content="result", tool_call_id="c1", tool_name="test"),
    ]
    api = adapter._build_api_messages(msgs)
    assert len(api) == 4
    assert api[0]["role"] == "system" and api[0]["content"] == "sys"
    assert api[2]["tool_calls"][0]["function"]["name"] == "test"
    assert api[3]["tool_call_id"] == "c1"

    # 测试真实 API 调用（mock HTTP）
    mock_response = {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": "Hello!",
                "tool_calls": None,
            }
        }]
    }
    with patch("httpx.Client.post") as mock_post:
        mock_post.return_value.raise_for_status = MagicMock()
        mock_post.return_value.json.return_value = mock_response
        result = adapter.generate([Message(role="user", content="hi")], tools=None)
        assert result.role == "assistant"
        assert result.content == "Hello!"
        assert result.tool_calls is None

    print("  [PASS] test_deepseek_adapter")


# ─── 测试 2: FailureDiagnosis — capture_failure 结构验证 ──────────

def test_capture_failure():
    diag = FailureDiagnosis()
    msgs = [
        Message(role="user", content="test task"),
        Message(role="assistant", content="working"),
    ]
    case = diag.capture_failure(
        task_desc="test task",
        step=3,
        error_msg="tool error: something broke",
        context_snapshot=msgs,
        error_type="tool_error",
    )

    assert case["task_desc"] == "test task"
    assert case["failed_step"] == 3
    assert case["error_type"] == "tool_error"
    assert case["error_msg"] == "tool error: something broke"
    assert len(case["context_snapshot"]) == 2
    assert case["context_snapshot"][0]["role"] == "user"
    assert "id" in case and len(case["id"]) == 8
    assert case["resolved"] is False
    assert case["root_cause"] is None

    # 测试 get_unresolved
    unresolved = diag.get_unresolved()
    assert len(unresolved) == 1, "Should have 1 unresolved case"

    # 测试 save/load
    diag.save_to_file("./test_diag.json")
    diag2 = FailureDiagnosis()
    diag2.load_from_file("./test_diag.json")
    assert len(diag2.cases) == 1
    assert diag2.cases[0]["id"] == case["id"]

    print("  [PASS] test_capture_failure")


# ─── 测试 3: RootCauseAnalyzer — 根因分析 ─────────────────────────

def test_root_cause_analysis():
    mock = MockLLMJSON({
        "root_cause_type": "prompt_unclear",
        "confidence": 0.85,
        "detail": "提示词中未明确要求模型在工具调用后直接给出结果",
        "suggested_fix_type": "adjust_prompt",
        "fix_description": "在提示词中加入'工具调用完成后直接回复用户，不再调用工具'的指令",
    })
    analyzer = RootCauseAnalyzer(mock)

    case = {
        "id": "test001",
        "task_desc": "读取文件内容",
        "failed_step": 5,
        "error_type": "loop_detected",
        "error_msg": "检测到重复工具调用回路",
        "context_snapshot": [
            {"role": "user", "content": "读取 test.txt"},
            {"role": "assistant", "content": "read_file tool"},
            {"role": "tool", "content": "file content here"},
        ],
    }

    result = analyzer.analyze(case)
    assert result["root_cause_type"] == "prompt_unclear"
    assert result["confidence"] == 0.85
    assert result["suggested_fix_type"] == "adjust_prompt"

    # 测试 batch_analyze
    results = analyzer.batch_analyze([case, case])
    assert len(results) == 2
    assert all("case_id" in r for r in results)

    # 测试 LLM 调用失败时的兜底
    class FailingLLM(BaseLLM):
        def generate(self, messages, tools=None):
            raise RuntimeError("LLM down")
        def embed(self, text):
            return [0.0]

    fail_analyzer = RootCauseAnalyzer(FailingLLM())
    fallback = fail_analyzer.analyze(case)
    assert fallback["root_cause_type"] == "tool_error"
    assert fallback["confidence"] == 0.3
    assert "LLM 调用失败" in fallback["detail"]

    # 测试无效 JSON 响应
    invalid_mock = MockLLM()
    invalid_mock.response_content = "not json at all"
    inv_analyzer = RootCauseAnalyzer(invalid_mock)
    inv_result = inv_analyzer.analyze(case)
    assert inv_result["root_cause_type"] == "tool_error"  # defaults

    print("  [PASS] test_root_cause_analysis")


# ─── 测试 4: SelfRepair — 修复策略生成 ─────────────────────────

def test_generate_fix():
    mock_llm = MockLLMJSON({"root_cause_type": "prompt_unclear"})
    repair = SelfRepair(mock_llm)

    config = {
        "system_prompt": "You are an agent.",
        "tool_descriptions": {"read_file": "Read a file", "write_file": "Write a file"},
        "memory_max_tokens": 4096,
        "model_name": "deepseek-chat",
    }

    # adjust_prompt
    rc_prompt = {
        "root_cause_type": "prompt_unclear",
        "confidence": 0.8,
        "detail": "提示词不够清晰",
        "suggested_fix_type": "adjust_prompt",
        "fix_description": "改进提示词",
    }
    fix = repair.generate_fix(rc_prompt, config)
    assert fix["fix_type"] == "adjust_prompt"
    assert "system_prompt" in fix["fixed"]

    # enrich_tool_description
    rc_tool = {
        "root_cause_type": "tool_schema_vague",
        "confidence": 0.8,
        "detail": "工具描述不具体",
        "suggested_fix_type": "enrich_tool_description",
        "fix_description": "补充工具描述",
    }
    fix2 = repair.generate_fix(rc_tool, config)
    assert fix2["fix_type"] == "enrich_tool_description"

    # trim_context
    rc_ctx = {
        "root_cause_type": "context_overflow",
        "confidence": 0.9,
        "detail": "上下文过长",
        "suggested_fix_type": "trim_context",
        "fix_description": "裁剪上下文",
    }
    fix3 = repair.generate_fix(rc_ctx, config)
    assert fix3["fix_type"] == "trim_context"
    assert fix3["fixed"]["memory_max_tokens"] == int(4096 * 0.6)  # 60%

    # add_reasoning_hint
    rc_reason = {
        "root_cause_type": "incorrect_reasoning",
        "confidence": 0.7,
        "detail": "推理逻辑错误",
        "suggested_fix_type": "add_reasoning_hint",
        "fix_description": "添加推理引导",
    }
    fix4 = repair.generate_fix(rc_reason, config)
    assert fix4["fix_type"] == "add_reasoning_hint"
    assert "system_prompt" in fix4["fixed"]

    # fix_tool_code
    rc_fix = {
        "root_cause_type": "tool_error",
        "confidence": 0.9,
        "detail": "工具bug",
        "suggested_fix_type": "fix_tool_code",
        "fix_description": "修代码",
    }
    fix5 = repair.generate_fix(rc_fix, config)
    assert fix5["fixed"].get("requires_manual") is True

    # switch_model
    rc_model = {
        "root_cause_type": "model_limitation",
        "confidence": 0.8,
        "detail": "模型能力不足",
        "suggested_fix_type": "switch_model",
        "fix_description": "换模型",
    }
    fix6 = repair.generate_fix(rc_model, config)
    assert fix6["fixed"].get("requires_manual") is True

    print("  [PASS] test_generate_fix")


# ─── 测试 5: apply_fix / rollback ────────────────────────────────

def test_apply_and_rollback():
    registry = ToolRegistry(safe_mode=False)

    @registry.register("test_tool", "Test tool description")
    def test_tool(x: str) -> str:
        return x

    short = ShortTermMemory(max_tokens=4096)
    long = LongTermMemory(MockLLM(), persist_dir="./test_rollback_db")
    memory = MemoryManager(short=short, long=long)
    agent = AgentLoop(
        llm=MockLLM(),
        registry=registry,
        memory=memory,
        enable_self_optimize=True,
    )

    original_prompt = agent.system_prompt
    original_tokens = agent.memory.short_term.max_tokens

    # 测试 adjust_prompt 修复
    fix_prompt = {
        "fix_id": "fix_prompt",
        "fix_type": "adjust_prompt",
        "fixed": {"system_prompt": "Improved prompt"},
        "applied": False,
    }
    result = agent._self_repair.apply_fix(fix_prompt, agent)
    assert result is True
    assert agent.system_prompt == "Improved prompt"

    # 回滚
    rollback_ok = agent._self_repair.rollback(fix_prompt, agent)
    assert rollback_ok is True
    assert agent.system_prompt == original_prompt

    # 测试 trim_context 修复
    fix_trim = {
        "fix_id": "fix_trim",
        "fix_type": "trim_context",
        "fixed": {"memory_max_tokens": 1000},
        "applied": False,
    }
    agent._self_repair.apply_fix(fix_trim, agent)
    assert agent.memory.short_term.max_tokens == 1000

    agent._self_repair.rollback(fix_trim, agent)
    assert agent.memory.short_term.max_tokens == original_tokens

    # 测试 requires_manual（不应应用）
    fix_manual = {
        "fix_id": "fix_manual",
        "fix_type": "fix_tool_code",
        "fixed": {"requires_manual": True},
        "applied": False,
    }
    manual_result = agent._self_repair.apply_fix(fix_manual, agent)
    assert manual_result is False

    # 测试无快照时回滚
    fix_no_snap = {
        "fix_id": "fix_no_snap",
        "fix_type": "add_reasoning_hint",
        "fixed": {},
        "applied": False,
    }
    no_snap_result = agent._self_repair.rollback(fix_no_snap, agent)
    assert no_snap_result is False

    print("  [PASS] test_apply_and_rollback")


# ─── 测试 6: VerifyRepair — 修复验证 ────────────────────────────

def test_verify_repair():
    verifier = VerifyRepair()

    # 模拟成功修复
    class SuccessLLM(BaseLLM):
        def generate(self, messages, tools=None):
            return Message(role="assistant", content="任务完成！", tool_calls=None)
        def embed(self, text):
            return [0.0]

    registry = ToolRegistry()
    agent = AgentLoop(
        llm=SuccessLLM(),
        registry=registry,
        memory=_make_memory(),
        max_steps=3,
    )

    fix = {"fix_id": "fix_test", "fix_type": "adjust_prompt", "applied": True}
    result = verifier.verify(fix, agent, "test task")
    assert result["fix_id"] == "fix_test"
    assert result["before_success"] is False
    assert result["after_success"] is True
    assert result["improved"] is True

    # 验证 _is_failure 方法
    assert verifier._is_failure("[STOPPED] 熔断") is True
    assert verifier._is_failure("[ERROR] tool failed") is True
    assert verifier._is_failure("All good!") is False

    # 测试 full_verify
    results = verifier.full_verify([fix], agent, ["test task"])
    assert len(results) == 1
    assert results[0]["improved"] is True

    print("  [PASS] test_verify_repair")


# ─── 测试 7: enable_self_optimize=False 时行为不变 ─────────────────

def test_self_optimize_disabled():
    registry = ToolRegistry(safe_mode=False)

    @registry.register("echo", "Echo back")
    def echo(x: str) -> str:
        return x

    agent = AgentLoop(
        llm=MockLLM(response_content="done"),
        registry=registry,
        memory=_make_memory(),
        max_steps=3,
        enable_self_optimize=False,
    )

    result = agent.run("test")
    assert "done" in result, "Should return mock response"
    assert result == "done"

    # 自优化组件应为 None
    assert agent._diagnosis is None
    assert agent._root_cause_analyzer is None
    assert agent._self_repair is None
    assert agent._verify is None

    # run_self_optimize 应返回消息
    report = agent.run_self_optimize()
    assert report["total_cases"] == 0

    print("  [PASS] test_self_optimize_disabled")


# ─── 测试 8: AgentLoop 集成 — 自优化完整流程 ─────────────────────

def test_integration_self_optimize():
    """集成测试：熔断触发 -> 捕获 -> 分析 -> 修复 -> 验证 -> 回滚"""
    registry = ToolRegistry(safe_mode=False)

    @registry.register("failing_tool", "Always fails")
    def failing_tool(x: str = "test") -> str:
        return f"[ERROR] intentional failure"

    # 每次调用使用不同参数，绕过循环检测，使熔断机制触发
    class CircuitBreakerMockLLM(BaseLLM):
        def __init__(self):
            self.model = "mock"
            self.call_idx = 0

        def generate(self, messages, tools=None):
            if tools:
                self.call_idx += 1
                if self.call_idx <= 6:
                    return Message(
                        role="assistant",
                        content="calling failing_tool",
                        tool_calls=[ToolCall(
                            id=f"c{self.call_idx}",
                            name="failing_tool",
                            arguments={"x": f"attempt_{self.call_idx}"},
                        )],
                    )
            return Message(role="assistant", content="recovered", tool_calls=None)

        def embed(self, text):
            return [0.0]

    agent = AgentLoop(
        llm=CircuitBreakerMockLLM(),
        registry=registry,
        memory=_make_memory(),
        max_steps=15,
        enable_self_optimize=True,
    )

    # 第一次 run — 会触发熔断（5次错误）
    result = agent.run("test failing task")
    assert "[STOPPED]" in result, "Should hit circuit breaker"
    assert len(agent._last_failure_cases) > 0, "Should capture failure case"

    # 验证捕获的 case — 熔断由连续5次工具错误触发
    case = agent._last_failure_cases[0]
    assert case["error_type"] == "circuit_breaker", f"Expected circuit_breaker, got {case['error_type']}"
    assert case["task_desc"] == "test failing task"

    # 手动触发自优化（因为 run() 内已在熔断时自动捕获，但不自动调用 run_self_optimize）
    report = agent.run_self_optimize()
    assert report["total_cases"] > 0
    assert "analyzed" in report
    assert "details" in report

    print("  [PASS] test_integration_self_optimize")


# ─── 测试 9: 自动触发自优化 + 重试 ──────────────────────────────

def test_auto_self_heal():
    """测试 run() 熔断时自动触发自优化并重试"""
    from unittest.mock import patch

    registry = ToolRegistry(safe_mode=False)

    call_count = [0]

    @registry.register("failer", "Always fails")
    def failer(x: str = "test") -> str:
        call_count[0] += 1
        return f"[ERROR] fail #{call_count[0]}"

    # Mock LLM：先返回工具调用导致熔断，修复后返回成功
    class SelfHealMockLLM(BaseLLM):
        def __init__(self):
            self.model = "mock"
            self.call_idx = 0
            self.mode = "fail"  # fail → success after heal

        def generate(self, messages, tools=None):
            if self.mode == "fail":
                if tools:
                    self.call_idx += 1
                    if self.call_idx <= 6:
                        return Message(
                            role="assistant", content="call failer",
                            tool_calls=[ToolCall(id=f"c{self.call_idx}", name="failer",
                                         arguments={"x": f"a{self.call_idx}"})],
                        )
            # After heal or directly
            return Message(role="assistant", content="任务成功完成！", tool_calls=None)

        def embed(self, text):
            return [0.0]

    llm = SelfHealMockLLM()

    # 用 mock root_cause_analyzer 让它返回一个可修复的诊断
    agent = AgentLoop(
        llm=llm,
        registry=registry,
        memory=_make_memory(),
        max_steps=15,
        enable_self_optimize=True,
        self_optimize_max_retries=2,
        fix_history_file="./test_fix_history_auto.json",
    )

    # Mock root_cause_analyzer.analyze to return a fixable result
    with patch.object(agent._root_cause_analyzer, "analyze", return_value={
        "root_cause_type": "prompt_unclear",
        "confidence": 0.85,
        "detail": "提示词不清晰",
        "suggested_fix_type": "adjust_prompt",
        "fix_description": "改进提示词",
    }):
        # Mock _self_repair.generate_fix to return a simple fix
        with patch.object(agent._self_repair, "generate_fix", return_value={
            "fix_id": "fix_auto",
            "fix_type": "adjust_prompt",
            "original": {"system_prompt": agent.system_prompt},
            "fixed": {"system_prompt": agent.system_prompt + "\n改进后的提示"},
            "applied": False,
        }):
            # Run — should auto-heal
            result = agent.run("test auto heal")

    # 应该重试成功
    assert "任务成功完成" in result or "[STOPPED]" not in result, f"Expected success after heal, got: {result[:80]}"

    print("  [PASS] test_auto_self_heal")


# ─── 测试 10: FixHistory 持久化与复用 ────────────────────────────

def test_fix_history_persistence():
    history_file = "./test_fh_persist.json"

    # 创建并记录修复
    fh = FixHistory(history_file)
    fh.record_fix(
        error_type="circuit_breaker",
        task_desc="读取大文件并分析内容",
        fix={"fix_type": "trim_context", "original": {}, "fixed": {"memory_max_tokens": 2000}},
        root_cause={"root_cause_type": "context_overflow", "confidence": 0.9},
        verified=True,
    )
    fh.record_fix(
        error_type="loop_detected",
        task_desc="搜索数据库记录",
        fix={"fix_type": "adjust_prompt", "original": {}, "fixed": {"system_prompt": "Improved"}},
        root_cause={"root_cause_type": "prompt_unclear", "confidence": 0.85},
        verified=True,
    )
    assert len(fh.records) == 2

    # 查找相似
    similar = fh.find_similar("circuit_breaker", "读取文件内容")
    assert len(similar) > 0, "Should find similar fix for circuit_breaker"
    assert similar[0]["fix"]["fix_type"] == "trim_context"

    similar2 = fh.find_similar("loop_detected", "搜索数据库")
    assert len(similar2) > 0, "Should find similar fix for loop_detected"

    # 不相似的查询
    no_match = fh.find_similar("tool_error", "完全不相关的任务描述")
    assert len(no_match) == 0, "Should not match unrelated error"

    # 关闭后重新加载
    del fh
    fh2 = FixHistory(history_file)
    assert len(fh2.records) == 2
    assert fh2.records[0]["fix"]["fix_type"] == "trim_context"

    # mark_reused
    fh2.mark_reused(fh2.records[0]["fix"])
    assert fh2.records[0]["reuse_count"] == 1

    # stats
    stats = fh2.get_stats()
    assert stats["total_fixes"] == 2
    assert "trim_context" in stats["by_fix_type"]

    # cleanup
    import os
    os.remove(history_file)

    print("  [PASS] test_fix_history_persistence")


# ─── 测试 11: 跨会话修复复用 ─────────────────────────────────────

def test_cross_session_reuse():
    """测试 AgentLoop 在遇到已知失败模式时优先复用历史修复"""
    history_file = "./test_fh_reuse.json"
    import os

    # 先写入历史修复
    fh = FixHistory(history_file)
    fh.record_fix(
        error_type="circuit_breaker",
        task_desc="读取复杂文件",
        fix={"fix_type": "adjust_prompt", "original": {}, "fixed": {"system_prompt": "历史优化过的提示词"}},
        root_cause={"root_cause_type": "prompt_unclear", "confidence": 0.88},
        verified=True,
    )

    registry = ToolRegistry(safe_mode=False)

    @registry.register("failer", "Always fails")
    def failer(x: str = "test") -> str:
        return "[ERROR] fail"

    class ReuseLLM(BaseLLM):
        def __init__(self):
            self.model = "mock"
            self.call_idx = 0

        def generate(self, messages, tools=None):
            if tools:
                self.call_idx += 1
                if self.call_idx <= 6:
                    return Message(
                        role="assistant", content="call failer",
                        tool_calls=[ToolCall(id=f"c{self.call_idx}", name="failer",
                                     arguments={"x": f"u{self.call_idx}"})],
                    )
            return Message(role="assistant", content="修复后成功完成！", tool_calls=None)

        def embed(self, text):
            return [0.0]

    agent = AgentLoop(
        llm=ReuseLLM(),
        registry=registry,
        memory=_make_memory(),
        max_steps=15,
        enable_self_optimize=True,
        self_optimize_max_retries=2,
        fix_history_file=history_file,
    )

    # 先用 _try_reuse_historical_fixes 验证历史修复被检测到
    agent._last_failure_cases.append({
        "id": "test_case",
        "task_desc": "读取复杂文件进行解析",
        "failed_step": 3,
        "error_type": "circuit_breaker",
        "error_msg": "circuit breaker",
        "context_snapshot": [],
    })
    agent._try_reuse_historical_fixes()
    # 验证历史修复被应用
    assert agent.system_prompt == "历史优化过的提示词", f"Expected reused prompt, got: {agent.system_prompt[:30]}"

    # cleanup
    os.remove(history_file)

    print("  [PASS] test_cross_session_reuse")


if __name__ == "__main__":
    print("Running Self-Optimize tests...\n")
    test_deepseek_adapter()
    test_capture_failure()
    test_root_cause_analysis()
    test_generate_fix()
    test_apply_and_rollback()
    test_verify_repair()
    test_self_optimize_disabled()
    test_integration_self_optimize()
    test_auto_self_heal()
    test_fix_history_persistence()
    test_cross_session_reuse()
    print("\nAll self-optimize tests passed!")
