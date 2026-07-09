# -*- coding: utf-8 -*-
"""Lightweight self-challenge eval — measure agent coding ability without external repos.

Generates buggy Python fixtures, runs agent to fix them, verifies fixes.
No network, no Docker, ~10s per instance.
"""
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.loop import AgentLoop, DEFAULT_SYSTEM_PROMPT
from tools.registry import ToolRegistry
from tools.builtin import register_builtin_tools
from memory import MemoryManager
from memory.short_term import ShortTermMemory
from memory.long_term import LongTermMemory
from agent.models import Message


FIX_PROMPT = (
    DEFAULT_SYSTEM_PROMPT
    + "\n\n=== BUG FIX MODE ===\n"
    "你会收到一个有 bug 的 Python 脚本。\n"
    "步骤：1) read_file 读取 2) 找出所有 bug 3) write_file 写入修复 4) run_shell 验证\n"
    "验证命令：python <文件路径>\n"
    "验证通过（无 syntax error、无 traceback）后才输出 [DONE]。\n"
    "不通过则重新检查修复。只改 bug，不改功能。"
)


@dataclass
class ChallengeResult:
    challenge_id: str
    difficulty: int
    fixed: bool
    bug_count: int
    steps: int = 0
    elapsed_sec: float = 0
    error: str = ""


@dataclass
class ChallengeReport:
    total: int
    resolved: int
    rate: float
    avg_steps: float
    avg_time: float
    results: list[ChallengeResult]


def run_challenge_eval(
    difficulty: int = 1,
    count: int = 5,
    max_steps: int = 10,
    model: str = "deepseek-v4-pro",
) -> ChallengeReport:
    """Run self-challenge evaluation.

    Args:
        difficulty: 1-5, controls bug complexity
        count: number of challenges to generate
        max_steps: max agent steps per challenge
        model: DeepSeek model name

    Returns:
        ChallengeReport with resolution rate and per-instance details.
    """
    from llm.deepseek_api import DeepSeekAdapter
    from agent.evolve.challenge_gen import ChallengeGenerator

    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    llm = DeepSeekAdapter(api_key=api_key, model=model)
    gen = ChallengeGenerator(llm)

    challenges = gen.generate(
        profile_summary=f"Agent 执行难度 {difficulty} 的单文件代码修复任务",
        weak_areas=["code_bugs", "syntax_errors"],
        current_level=difficulty,
        count=count,
    )
    if not challenges:
        print("Failed to generate challenges")
        return ChallengeReport(0, 0, 0, 0, 0, [])

    challenges = challenges[:count]  # Respect count limit

    tmp_dir = Path(tempfile.mkdtemp(prefix="challenge_"))
    results = []

    for i, task in enumerate(challenges):
        task_desc = task.get("description", task.get("task", str(task)))
        diff = task.get("difficulty", difficulty)

        # Generate buggy fixture
        fixture = gen.create_fixture(task_desc, diff)
        if not fixture:
            continue

        fixture_path = tmp_dir / f"challenge_{i}.py"
        fixture_path.write_text(fixture, encoding="utf-8")

        # Count bugs (find actual errors in the fixture)
        bug_count = _count_bugs(fixture)

        print(f"\n[{i+1}/{len(challenges)}] Difficulty {diff}, {bug_count} bugs")
        print(f"  Task: {task_desc[:80]}")

        # Build agent with workspace tools
        registry = ToolRegistry(safe_mode=False)
        register_builtin_tools(registry)

        def mk_read(ws):
            def read_file(path: str) -> str:
                full = os.path.join(ws, path) if not os.path.isabs(path) else path
                try:
                    return Path(full).read_text(encoding="utf-8", errors="replace")[:8000]
                except Exception as e:
                    return f"ERROR: {e}"
            return read_file

        def mk_write(ws):
            def write_file(path: str, content: str) -> str:
                full = os.path.join(ws, path) if not os.path.isabs(path) else path
                Path(full).parent.mkdir(parents=True, exist_ok=True)
                Path(full).write_text(content, encoding="utf-8")
                return f"Written {len(content)} bytes"
            return write_file

        ws = str(tmp_dir)
        registry._tools.clear()
        registry._tool_metadata.clear()
        registry.register("read_file", "Read a file")(mk_read(ws))
        registry.register("write_file", "Write a file")(mk_write(ws))
        registry.register("run_shell", "Run a shell command")(
            lambda cmd: _run_shell(cmd, ws))

        llm_agent = DeepSeekAdapter(api_key=api_key, model=model)
        memory = MemoryManager(
            short=ShortTermMemory(max_tokens=32768),
            long=LongTermMemory(llm=llm_agent, collection_name="challenge_eval"),
        )
        agent = AgentLoop(
            llm=llm_agent, registry=registry, memory=memory,
            max_steps=max_steps,
            loop_detect_threshold=6,
        )
        agent.system_prompt = FIX_PROMPT

        task_prompt = (
            f"修复这个 Python 脚本中的所有 bug:\n\n"
            f"文件: {fixture_path}\n\n"
            f"```python\n{fixture}\n```\n\n"
            f"先用 read_file 读取，找出 bug，修改后用 run_shell 验证。"
        )

        start = time.time()
        try:
            output = agent.run(task_prompt)
            elapsed = time.time() - start
            steps = getattr(agent, "_last_step_count", 0)

            fixed = _verify_fix(fixture_path)
            results.append(ChallengeResult(
                challenge_id=f"challenge_{i}",
                difficulty=diff,
                fixed=fixed,
                bug_count=bug_count,
                steps=steps,
                elapsed_sec=elapsed,
            ))
            print(f"  {'✓ FIXED' if fixed else '✗ FAILED'} in {steps} steps, {elapsed:.1f}s")

        except Exception as e:
            elapsed = time.time() - start
            results.append(ChallengeResult(
                challenge_id=f"challenge_{i}", difficulty=diff,
                fixed=False, bug_count=bug_count,
                error=str(e)[:100], elapsed_sec=elapsed,
            ))

    resolved = sum(1 for r in results if r.fixed)
    total = len(results)
    report = ChallengeReport(
        total=total,
        resolved=resolved,
        rate=resolved / total if total else 0,
        avg_steps=sum(r.steps for r in results) / max(1, total),
        avg_time=sum(r.elapsed_sec for r in results) / max(1, total),
        results=results,
    )

    # Cleanup
    for f in tmp_dir.glob("*"):
        try: f.unlink()
        except: pass
    try: tmp_dir.rmdir()
    except: pass

    return report


def _count_bugs(code: str) -> int:
    """Count likely bugs in generated code."""
    import py_compile, tempfile
    bugs = 0
    # Syntax check
    try:
        f = tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w")
        f.write(code); f.close()
        py_compile.compile(f.name, doraise=True)
        os.unlink(f.name)
    except (SyntaxError, py_compile.PyCompileError):
        bugs += 1
        os.unlink(f.name)
    # Runtime check
    try:
        subprocess.run([sys.executable, "-c", code], capture_output=True, timeout=5)
    except Exception:
        bugs += 1
    return max(bugs, 1)


def _verify_fix(filepath: Path) -> bool:
    """Verify the fix: syntax OK + runs without error."""
    try:
        import py_compile
        py_compile.compile(str(filepath), doraise=True)
    except py_compile.PyCompileError:
        return False
    try:
        r = subprocess.run(
            [sys.executable, str(filepath)],
            capture_output=True, timeout=10,
            encoding="utf-8", errors="replace",
        )
        return r.returncode == 0 and "Traceback" not in r.stderr
    except Exception:
        return False


def _run_shell(cmd: str, cwd: str) -> str:
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, timeout=30,
                          cwd=cwd, encoding="utf-8", errors="replace")
        out = (r.stdout + r.stderr)[:2000]
        return out or "(no output)"
    except subprocess.TimeoutExpired:
        return "TIMEOUT (30s)"
    except Exception as e:
        return f"ERROR: {e}"


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--difficulty", type=int, default=1)
    p.add_argument("--count", type=int, default=5)
    p.add_argument("--steps", type=int, default=10)
    args = p.parse_args()

    print(f"Running {args.count} challenges at difficulty {args.difficulty}...")
    report = run_challenge_eval(
        difficulty=args.difficulty, count=args.count,
        max_steps=args.steps,
    )

    print(f"\n{'='*50}")
    print(f"CHALLENGE EVAL RESULTS")
    print(f"{'='*50}")
    print(f"  Total:    {report.total}")
    print(f"  Resolved: {report.resolved}")
    print(f"  Rate:     {report.rate:.0%}")
    print(f"  Avg steps:{report.avg_steps:.1f}")
    print(f"  Avg time: {report.avg_time:.1f}s")
    for r in report.results:
        icon = "PASS" if r.fixed else "FAIL"
        print(f"  {icon} d{r.difficulty} {r.steps:>3}steps {r.elapsed_sec:>5.1f}s {r.bug_count}bugs")
    print(f"{'='*50}")
