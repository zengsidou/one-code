# -*- coding: utf-8 -*-
"""集成测试 — Agent 编排、Skills、Hooks、Memory、ToolRegistry.subset"""
import os, sys, json, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from agent.models import Message, Contract, ContractStep, ContractResult
from agent.skills import SkillLibrary, Skill
from agent.hooks import HookRegistry
from tools.registry import ToolRegistry
from agent.token_optimizer import TokenOptimizer
from memory.short_term import ShortTermMemory
from memory.long_term import LongTermMemory


class TestSkillLibrary:
    def test_load_and_match(self):
        lib = SkillLibrary()
        lib.skills = [
            Skill("react", "React 前端", "useState...", ["react", "jsx", "组件"], "."),
            Skill("flask", "Flask 后端", "route...", ["flask", "api", "路由"], "."),
        ]
        m = lib.match("帮我写一个Flask API接口")
        assert len(m) > 0
        assert "flask" in {s.name for s in m}

    def test_no_match(self):
        lib = SkillLibrary()
        lib.skills = [Skill("react", "React", "...", ["react"], ".")]
        assert len(lib.match("写一个 SQL 查询")) == 0

    def test_to_prompt(self):
        lib = SkillLibrary()
        lib.skills = [Skill("test", "测试", "内容体", ["test"], ".")]
        m = lib.match("test something")
        prompt = lib.to_prompt(m)
        assert "[Skills" in prompt
        assert "内容体" in prompt


class TestHooks:
    def test_register_and_fire(self):
        h = HookRegistry()
        fired = []
        h.register("tool.after", lambda **kw: fired.append(kw.get("name")))
        h.fire("tool.after", name="read_file", args={}, result="ok")
        assert fired == ["read_file"]

    def test_multiple_hooks(self):
        h = HookRegistry()
        results = []
        h.register("agent.start", lambda **kw: results.append(1))
        h.register("agent.start", lambda **kw: results.append(2))
        h.fire("agent.start", task="test")
        assert results == [1, 2]

    def test_hook_exception_is_silent(self):
        h = HookRegistry()
        h.register("tool.after", lambda **kw: 1 / 0)  # crash
        h.fire("tool.after", name="x", args={}, result="ok")  # must not raise


class TestToolRegistrySubset:
    def test_subset_creates_filtered_registry(self):
        r = ToolRegistry()
        @r.register("t1", "tool 1")
        def t1(): return "1"
        @r.register("t2", "tool 2")
        def t2(): return "2"

        sub = r.subset(["t1"])
        assert "t1" in sub._tools
        assert "t2" not in sub._tools

    def test_subset_execute(self):
        r = ToolRegistry()
        @r.register("hello", "say hello")
        def hello(name: str = "World"): return f"Hello {name}"
        sub = r.subset(["hello"])
        assert sub.execute("hello", {"name": "Test"}) == "Hello Test"


class TestTokenOptimizer:
    def test_error_preserved(self):
        opt = TokenOptimizer(None)
        err = "[ERROR] something broke" + "x" * 500
        result = opt.compress_tool_output(err)
        assert "[ERROR]" in result
        assert len(result) <= 500

    def test_short_output_untouched(self):
        opt = TokenOptimizer(None)
        assert opt.compress_tool_output("hi") == "hi"

    def test_summarize_work(self):
        opt = TokenOptimizer(None)
        msgs = [
            Message(role="user", content="fix bug"),
            Message(role="tool", content="File written: src/main.py", tool_name="write_file"),
            Message(role="tool", content="已替换 utils.py", tool_name="edit_file"),
        ]
        s = opt.summarize_work(msgs, "fix bug")
        assert "fix bug" in s
        assert "src/main.py" in s or "utils.py" in s


class TestShortTermMemory:
    def test_basic_add(self):
        mem = ShortTermMemory(max_tokens=10000)
        mem.add(Message(role="user", content="hello"))
        assert len(mem.get_messages()) == 1

    def test_context_get(self):
        mem = ShortTermMemory(max_tokens=10000)
        mem.add(Message(role="user", content="test"))
        mem.add(Message(role="assistant", content="response"))
        msgs = mem.get_context()
        assert len(msgs) >= 1

    def test_token_count(self):
        mem = ShortTermMemory(max_tokens=10000)
        mem.add(Message(role="user", content="hello world"))
        assert mem.get_token_count() > 0


class TestContractModels:
    def test_contract_result_defaults(self):
        cr = ContractResult()
        assert cr.steps == []
        assert cr.user_confirmed is False
        assert cr.consistency_score == 0

    def test_contract_step_create(self):
        cs = ContractStep(index=2, goal="添加CSS", tools_hint="edit_file", depends_on=[1])
        assert cs.index == 2
        assert 1 in cs.depends_on


class TestApplyPatchMore:
    def test_move_file(self):
        d = tempfile.mkdtemp()
        src = os.path.join(d, "old.py")
        dst = os.path.join(d, "new.py")
        with open(src, "w") as f: f.write("x=1")
        from tools.apply_patch import apply_patch
        patch = json.dumps([{"action":"move","file":src,"new_file":dst}])
        result = apply_patch(patch)
        assert "OK" in result
        os.unlink(src)
        os.rmdir(d)

    def test_update_file(self):
        d = tempfile.mkdtemp()
        fp = os.path.join(d, "test.py")
        with open(fp, "w") as f: f.write("old stuff here")
        from tools.apply_patch import apply_patch
        patch = json.dumps([{"action":"update","file":fp,"old_string":"old stuff","new_string":"new stuff"}])
        result = apply_patch(patch)
        assert "OK" in result
        os.unlink(fp)
        os.rmdir(d)


class TestConstraints:
    def test_constraint_encode(self):
        from agent.constraints import ConstraintEnforcer
        ce = ConstraintEnforcer()
        import os
        # write_file to existing file should warn
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
            path = f.name
            f.write(b"existing")
        try:
            result = ce._check_write_file({"path": path}, [])
            assert result is not None
            assert "edit_file" in result
        finally:
            os.unlink(path)

    def test_constraint_hint_after_shell_error(self):
        from agent.constraints import ConstraintEnforcer
        ce = ConstraintEnforcer()
        hint = ce.after_tool_call("run_shell", {"command": "bad"}, "[ERROR] failed", [])
        assert hint is not None
        assert "失败" in hint or "重试" in hint or "error" in hint.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
