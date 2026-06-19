"""L4 自测试闭环验证 — 故意注入坏代码，验证被 intercept 并回滚"""
import os, sys, io, tempfile
sys.path.insert(0, os.path.dirname(__file__))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from agent.evolve.architect import ArchitectureApplier

tmpdir = tempfile.mkdtemp(prefix="l4_test_")

# 创建一个临时 Python 文件
tmp_file = os.path.join(tmpdir, "_test.py")
with open(tmp_file, "w", encoding="utf-8") as f:
    f.write("# test file\nx = 1\n")

original = open(tmp_file, encoding="utf-8").read()
applier = ArchitectureApplier(backup_dir=os.path.join(tmpdir, "backups"))

# Test 1: 注入语法错误代码 → 应被语法安全门拦截
print("=" * 50)
print("TEST 1: syntax error should be rejected by gate")
proposal_bad = {
    "full_path": tmp_file,
    "old_code_hint": "",
    "new_code": "if True\n    pass",
}
result = applier.apply(proposal_bad)
print(f"  apply() = {result} (expect False)")
assert not result, "Syntax error should be rejected!"
content = open(tmp_file, encoding="utf-8").read()
assert content == original, "File should not be changed!"
print("  ✓ PASS")

# Test 2: 模拟 apply_and_reload 的测试运行逻辑
print("\n" + "=" * 50)
print("TEST 2: _run_tests should detect broken code")
# Create a test file with a failing assertion
test_dir = os.path.join(tmpdir, "tests")
os.makedirs(test_dir, exist_ok=True)
test_file = os.path.join(test_dir, "test_tmp.py")
with open(test_file, "w", encoding="utf-8") as f:
    f.write("def test_will_fail():\n    assert False\n")

test_original = open(test_file, encoding="utf-8").read()

# First verify _run_tests works
old_cwd = os.getcwd()
os.chdir(tmpdir)
passed, output = ArchitectureApplier._run_tests()
print(f"  _run_tests: passed={passed}, output preview={output[:100]}")
# Should fail since the test asserts False
# Note: pytest will discover this test file and fail
assert not passed, "Should detect failing test!"
print("  ✓ _run_tests detected failure")

# Test 3: End-to-end — inject bad code, verify rollback
print("\n" + "=" * 50)
print("TEST 3: apply_and_reload should rollback on test failure")
proposal_break = {
    "full_path": test_file,
    "old_code_hint": "",
    "new_code": "def test_will_fail():\n    assert False  # injected bad code",
}
applier2 = ArchitectureApplier(backup_dir=os.path.join(tmpdir, "backups2"))
result2 = applier2.apply_and_reload(proposal_break, None, run_tests=True)
print(f"  result: {result2}")
assert not result2["success"], "Should reject change that breaks tests!"
# File should be restored after rollback
restored = open(test_file, encoding="utf-8").read()
assert restored == test_original, f"File should be rolled back! Got: {restored[:50]}"
print("  ✓ PASS: rollback confirmed")

os.chdir(old_cwd)

# Cleanup
import shutil
shutil.rmtree(tmpdir, ignore_errors=True)

print("\n" + "=" * 50)
print("  L4 SELF-TEST LOOP: 3/3 PASS")
print("=" * 50)

