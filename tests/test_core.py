# -*- coding: utf-8 -*-
"""核心路径测试 — ReAct 循环、熔断、回路检测、契约先行、Goal验证"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from agent.models import Message, Contract, ContractStep
from agent.contract_types import detect_contract_type, ContractType, get_contract_meta
from tools.edit_smart import edit_smart
from tools.apply_patch import apply_patch
from tools.shell_safety import check_dangerous, parse_command_ast
from agent.goal_verifier import GoalVerifier
from tools.registry import ToolRegistry


class TestContractTypes:
    def test_detect_visual(self):
        assert detect_contract_type("帮我做一个产品介绍网页") == ContractType.VISUAL
        assert detect_contract_type("设计一个dashboard页面") == ContractType.VISUAL

    def test_detect_dialog(self):
        assert detect_contract_type("帮我写一个客服对话助手") == ContractType.DIALOG
        assert detect_contract_type("设计聊天机器人的回复逻辑") == ContractType.DIALOG

    def test_detect_code(self):
        assert detect_contract_type("帮我重构utils.py模块") == ContractType.CODE_API
        assert detect_contract_type("拆分这个微服务") == ContractType.CODE_API

    def test_detect_config(self):
        assert detect_contract_type("帮我配置nginx参数") == ContractType.CONFIG

    def test_detect_data(self):
        assert detect_contract_type("帮我分析这个数据报表") == ContractType.DATA

    def test_detect_narrative(self):
        assert detect_contract_type("帮我写一篇技术博客") == ContractType.NARRATIVE

    def test_default_fallback(self):
        assert detect_contract_type("hello world") == ContractType.CODE_API

    def test_all_contract_types_have_meta(self):
        for ct in ContractType:
            meta = get_contract_meta(ct)
            assert "format" in meta
            assert "human_judge" in meta


class TestEditSmart:
    def test_exact_match(self):
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("def foo():\n    old_code()\n")
            path = f.name
        try:
            result = edit_smart(path, "old_code()", "new_code()")
            assert "OK" in result
            with open(path) as fp:
                assert "new_code()" in fp.read()
        finally:
            os.unlink(path)

    def test_exact_not_found_suggests(self):
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("def bar():\n    pass\n")
            path = f.name
        try:
            result = edit_smart(path, "old_code_that_doesnt_exist", "new")
            assert "ERROR" in result or "未找到" in result
        finally:
            os.unlink(path)

    def test_line_trimmed_match(self):
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("def foo():  \n    pass  \n")
            path = f.name
        try:
            result = edit_smart(path, "def foo():\n    pass", "def bar():")
            assert "OK" in result
        finally:
            os.unlink(path)


class TestApplyPatch:
    def test_add_file(self):
        import tempfile, os
        d = tempfile.mkdtemp()
        fp = os.path.join(d, "new.py")
        patch = '[{"action": "add", "file": "' + fp.replace('\\', '/') + '", "content": "x=1"}]'
        result = apply_patch(patch)
        assert "OK" in result
        assert os.path.exists(fp)
        os.unlink(fp)
        os.rmdir(d)

    def test_delete_and_rollback(self):
        import tempfile, os
        d = tempfile.mkdtemp()
        fp = os.path.join(d, "temp.py")
        with open(fp, "w") as f:
            f.write("x=1")
        patch = '[{"action": "delete", "file": "' + fp.replace('\\', '/') + '"}]'
        result = apply_patch(patch)
        assert "OK" in result
        assert not os.path.exists(fp)
        os.rmdir(d)

    def test_invalid_patch(self):
        result = apply_patch("not json")
        assert "ERROR" in result


class TestShellSafety:
    def test_safe_command(self):
        r = check_dangerous("ls -la")
        assert r["allowed"]
        assert not r["blocked"]

    def test_rm_rf_blocked(self):
        r = check_dangerous("rm -rf /tmp")
        assert r["blocked"]

    def test_sudo_blocked(self):
        r = check_dangerous("sudo apt update")
        assert r["blocked"]

    def test_parse_bash_ast(self):
        ast = parse_command_ast("ls -la | grep test")
        assert len(ast["commands"]) >= 1

    def test_format_blocked(self):
        r = check_dangerous("format C:")
        assert r["blocked"]


class TestGoalVerifier:
    def test_fast_path_stopped(self):
        gv = GoalVerifier(None)
        r = gv.verify("test task", "[STOPPED] max steps reached")
        assert not r["passed"]

    def test_fast_path_error(self):
        gv = GoalVerifier(None)
        r = gv.verify("test task", "[ERROR] something went wrong")
        assert not r["passed"]

    def test_fast_path_too_short(self):
        gv = GoalVerifier(None)
        r = gv.verify("test task", "ok")
        assert not r["passed"]

    def test_fast_path_ok_length(self):
        gv = GoalVerifier(None)
        msg = "任务已全部完成。修改了以下文件：agent/loop.py添加了GoalVerifier集成，tools/shell_safety.py实现了tree-sitter命令安全检测，agent/evolve/architect.py增强了依赖检测和回归测试门控。所有功能经测试验证正常运行，无错误。"
        r = gv.verify("test task", msg)
        assert r["passed"]


class TestModels:
    def test_contract_model(self):
        c = Contract(type="visual", format="svg", content="test", summary="方向总结")
        assert c.type == "visual"

    def test_contract_step(self):
        s = ContractStep(index=1, goal="搭建框架", tools_hint="write_file", depends_on=[], contract_checkpoint="线框图上部分")
        assert s.index == 1
        assert s.depends_on == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
