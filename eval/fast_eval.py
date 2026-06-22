# -*- coding: utf-8 -*-
"""Fast self-challenge eval — uses pre-crafted buggy fixtures, no LLM generation.
~5 seconds per instance, zero token cost for generation.
"""
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.loop import AgentLoop, DEFAULT_SYSTEM_PROMPT
from tools.registry import ToolRegistry
from memory import MemoryManager
from memory.short_term import ShortTermMemory
from memory.long_term import LongTermMemory

# Pre-crafted buggy fixtures — each has 1-2 known bugs
BUGGY_FIXTURES = [
    # 1: NameError — undefined variable
    {
        "desc": "修复 NameError：变量名拼写错误",
        "code": "def greet(name):\n    return f'Hello, {nam}'  # typo: nam → name\n\nprint(greet('World'))",
        "expected": "runs without error",
    },
    # 2: SyntaxError — missing colon
    {
        "desc": "修复 SyntaxError：for 循环缺少冒号",
        "code": "for i in range(3)\n    print(i)",
        "expected": "runs without error",
    },
    # 3: Logic error — wrong operator
    {
        "desc": "修复逻辑错误：赋值 = 应为比较 ==",
        "code": "x = 5\nif x = 5:\n    print('yes')\nelse:\n    print('no')",
        "expected": "runs without error",
    },
    # 4: TypeError — int + str
    {
        "desc": "修复 TypeError：整数和字符串不能拼接",
        "code": "age = 25\nprint('I am ' + age + ' years old')",
        "expected": "runs without error",
    },
    # 5: ZeroDivisionError
    {
        "desc": "修复 ZeroDivisionError：除数为零",
        "code": "def divide(a, b):\n    return a / b  # no zero check\n\nprint(divide(10, 0))",
        "expected": "runs without error",
    },
    # 6: IndentationError
    {
        "desc": "修复 IndentationError：缩进不一致",
        "code": "def foo():\n  pass\n\nif True:\n print('ok')",
        "expected": "runs without error",
    },
    # 7: AttributeError — wrong method
    {
        "desc": "修复 AttributeError：字符串拼写方法名错误",
        "code": "s = 'hello'\nprint(s.uppper())  # typo: uppper → upper",
        "expected": "runs without error",
    },
    # 8: IndexError
    {
        "desc": "修复 IndexError：列表越界",
        "code": "items = [1, 2, 3]\nprint(items[3])  # index 3 out of range",
        "expected": "runs without error",
    },
    # 9: KeyError
    {
        "desc": "修复 KeyError：字典键不存在",
        "code": "d = {'name': 'Alice'}\nprint(d['nme'])  # typo: nme → name",
        "expected": "runs without error",
    },
    # 10: ImportError
    {
        "desc": "修复 ImportError：模块名拼写错误",
        "code": "import matthlib  # should be math\nprint(matthlib.sqrt(16))",
        "expected": "runs without error",
    },
]


@dataclass
class FastEvalResult:
    desc: str
    fixed: bool
    steps: int = 0
    elapsed_sec: float = 0
    error: str = ""


def run_fast_eval(count: int = 10, max_steps: int = 8) -> list[FastEvalResult]:
    """Run fast evaluation with pre-crafted fixtures."""
    from llm.deepseek_api import DeepSeekAdapter

    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    results = []
    fixtures = BUGGY_FIXTURES[:count]

    for i, fixture in enumerate(fixtures):
        print(f"\n[{i+1}/{len(fixtures)}] {fixture['desc']}")

        workspace = tempfile.mkdtemp(prefix="bugfix_")
        bug_path = Path(workspace) / f"bug_{i}.py"
        bug_path.write_text(fixture["code"], encoding="utf-8")

        # Simple workspace tools
        registry = ToolRegistry(safe_mode=False)

        def mk_read(ws):
            def _read(path: str) -> str:
                full = os.path.join(ws, path) if not os.path.isabs(path) else path
                try: return Path(full).read_text(encoding="utf-8", errors="replace")[:8000]
                except Exception as e: return f"ERROR: {e}"
            return _read

        def mk_write(ws):
            def _write(path: str, content: str) -> str:
                full = os.path.join(ws, path) if not os.path.isabs(path) else path
                Path(full).parent.mkdir(parents=True, exist_ok=True)
                Path(full).write_text(content, encoding="utf-8")
                return f"Written {len(content)}B"
            return _write

        def mk_shell(ws):
            def _shell(cmd: str) -> str:
                try:
                    r = subprocess.run(cmd, shell=True, capture_output=True, timeout=30,
                                     cwd=ws, encoding="utf-8", errors="replace")
                    return (r.stdout + r.stderr)[:2000] or "(no output)"
                except Exception as e: return f"ERROR: {e}"
            return _shell

        ws = str(workspace)
        registry._tools.clear()
        registry._tool_metadata.clear()
        registry.register("read_file", "Read a file from disk")(mk_read(ws))
        registry.register("write_file", "Write content to a file")(mk_write(ws))
        registry.register("run_shell", "Run a shell command")(mk_shell(ws))

        llm = DeepSeekAdapter(api_key=api_key)
        memory = MemoryManager(
            short=ShortTermMemory(max_tokens=32768),
            long=LongTermMemory(llm=llm, collection_name="fast_eval"),
        )
        agent = AgentLoop(
            llm=llm, registry=registry, memory=memory,
            max_steps=max_steps,
            enable_self_optimize=False, enable_evolution=False,
            loop_detect_threshold=8,
        )

        prompt = (
            "BUG FIX TASK: 修复这个 Python 文件中的 bug。\n\n"
            f"文件路径: {bug_path}\n"
            f"文件内容:\n```python\n{fixture['code']}\n```\n\n"
            "步骤: read_file 确认内容 → 找出所有 bug → write_file 写入修复 → "
            f"run_shell `python {bug_path}` 验证 → 通过后输出 [DONE]。\n"
            "只改 bug，不改功能。"
        )

        start = time.time()
        try:
            agent.run(prompt)
            elapsed = time.time() - start
            steps = getattr(agent, "_last_step_count", 0)

            fixed = _verify(bug_path)
            results.append(FastEvalResult(
                desc=fixture["desc"], fixed=fixed,
                steps=steps, elapsed_sec=elapsed,
            ))
            print(f"  {'PASS' if fixed else 'FAIL'} in {steps} steps, {elapsed:.1f}s")

        except Exception as e:
            elapsed = time.time() - start
            results.append(FastEvalResult(
                desc=fixture["desc"], fixed=False, error=str(e)[:100],
                elapsed_sec=elapsed,
            ))

        # Cleanup
        for f in Path(workspace).glob("*"):
            try: f.unlink()
            except: pass
        try: Path(workspace).rmdir()
        except: pass

    return results


def _verify(filepath: Path) -> bool:
    try:
        import py_compile
        py_compile.compile(str(filepath), doraise=True)
    except Exception:
        return False
    try:
        r = subprocess.run([sys.executable, str(filepath)], capture_output=True,
                          timeout=10, encoding="utf-8", errors="replace")
        return r.returncode == 0 and "Traceback" not in r.stderr
    except Exception:
        return False


if __name__ == "__main__":
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    print(f"Running {count} bug-fix challenges...")
    results = run_fast_eval(count, max_steps=8)

    resolved = sum(1 for r in results if r.fixed)
    total = len(results)
    avg_steps = sum(r.steps for r in results) / max(1, total)
    avg_time = sum(r.elapsed_sec for r in results) / max(1, total)

    print(f"\n{'='*50}")
    print(f"FAST BUG-FIX EVAL RESULTS")
    print(f"{'='*50}")
    print(f"  Resolved: {resolved}/{total} ({resolved/total*100 if total else 0:.0f}%)")
    print(f"  Avg steps: {avg_steps:.1f}")
    print(f"  Avg time: {avg_time:.1f}s")
    for r in results:
        icon = "PASS" if r.fixed else "FAIL"
        print(f"  {icon} {r.steps:>3}s {r.elapsed_sec:>5.1f}s  {r.desc[:50]}")
    print(f"{'='*50}")
