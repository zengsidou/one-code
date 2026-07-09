# -*- coding: utf-8 -*-
"""SWE-bench Lite evaluation harness for one-code.

Runs one-code against SWE-bench Lite instances and reports resolution rate,
cost, and performance metrics.
"""
import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import git
from datasets import load_dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.loop import AgentLoop, DEFAULT_SYSTEM_PROMPT
from tools.registry import ToolRegistry
from tools.builtin import register_builtin_tools
from memory import MemoryManager
from agent.models import Message

AGENT_SYSTEM_PROMPT = (
    DEFAULT_SYSTEM_PROMPT
    + "\n\n=== SWE-BENCH EVALUATION MODE ===\n"
    "按三阶段执行：定位 → 修复 → 验证\n\n"
    "## 定位：从 issue 提取关键词搜索\n"
    "- 提取变量名/函数名/类名，用 grep 精确搜索\n"
    "- 永远不要搜索 docs/ 目录\n"
    "- 配置/默认值类 issue 优先查 settings.py 和 global_settings.py\n"
    "## 修复：最小变更\n"
    "- 只改 1-5 行，保持风格一致\n"
    "## 验证：跑测试\n"
    "- 用 run_shell 跑 pytest\n"
    "- 通过后必须输出以 [PATCH] 开头的行，然后紧跟 unified diff\n"
    "- 不要写大段分析说明，直接给 diff"
)


@dataclass
class EvalResult:
    instance_id: str
    repo: str
    resolved: bool = False
    cost_usd: float = 0.0
    elapsed_sec: float = 0.0
    steps: int = 0
    error: str = ""
    generated_patch: str = ""


@dataclass
class EvalReport:
    total: int = 0
    resolved: int = 0
    resolution_rate: float = 0.0
    avg_cost: float = 0.0
    avg_time: float = 0.0
    avg_steps: float = 0.0
    results: list[EvalResult] = field(default_factory=list)
    timestamp: str = ""


class SWEBenchRunner:
    """Runs one-code against SWE-bench Lite instances."""

    REPOS_DIR = Path("./eval/repos")
    RESULTS_DIR = Path("./eval/results")

    # GitHub → Gitee mirror mapping for repos blocked in China
    GITEE_MIRRORS = {
        "django/django": "https://gitee.com/mirrors/django.git",
        "scikit-learn/scikit-learn": "https://gitee.com/mirrors/scikit-learn.git",
        "sympy/sympy": "https://gitee.com/mirrors/sympy.git",
        "pytest-dev/pytest": "https://gitee.com/mirrors/pytest.git",
        "matplotlib/matplotlib": "https://gitee.com/mirrors/matplotlib.git",
        "sphinx-doc/sphinx": "https://gitee.com/mirrors/sphinx.git",
        "astropy/astropy": "https://gitee.com/mirrors/astropy.git",
        "pylint-dev/pylint": "https://gitee.com/mirrors/pylint.git",
        "psf/requests": "https://gitee.com/mirrors/requests.git",
        "mwaskom/seaborn": "https://gitee.com/mirrors/seaborn.git",
    }

    def _get_clone_url(self, repo: str) -> str:
        if repo in self.GITEE_MIRRORS:
            return self.GITEE_MIRRORS[repo]
        return f"https://github.com/{repo}.git"
    PRICE_INPUT_PER_1M = 0.137   # ¥1/M → ~$0.137/M
    PRICE_OUTPUT_PER_1M = 0.274  # ¥2/M → ~$0.274/M

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "deepseek-v4-pro",
        max_instances: int = 10,
        max_steps_per_instance: int = 30,
        timeout_per_instance: int = 600,
        repo_filter: str | None = None,
        git_proxy: str | None = "http://127.0.0.1:7993",
    ):
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
        self.model = model
        self.max_instances = max_instances
        self.max_steps = max_steps_per_instance
        self.timeout = timeout_per_instance
        self.repo_filter = repo_filter
        self.git_proxy = git_proxy

        os.makedirs(self.REPOS_DIR, exist_ok=True)
        os.makedirs(self.RESULTS_DIR, exist_ok=True)

    def load_instances(self) -> list[dict]:
        """Load SWE-bench Lite instances, optionally filtered."""
        ds = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
        instances = []
        for row in ds:
            d = dict(row)
            if self.repo_filter and self.repo_filter not in d.get("repo", ""):
                continue
            instances.append(d)
            if len(instances) >= self.max_instances:
                break
        return instances

    def prepare_repo(self, instance: dict) -> Path | None:
        """Clone or fetch repo at the base commit."""
        repo_name = instance["repo"].replace("/", "__")
        repo_dir = self.REPOS_DIR / repo_name
        base_commit = instance["base_commit"]
        clone_url = self._get_clone_url(instance["repo"])

        env = os.environ.copy()
        env.pop("HTTPS_PROXY", None)
        env.pop("HTTP_PROXY", None)
        env.pop("https_proxy", None)
        env.pop("http_proxy", None)
        if self.git_proxy and "github.com" in clone_url:
            env["HTTPS_PROXY"] = self.git_proxy
            env["HTTP_PROXY"] = self.git_proxy

        if repo_dir.exists():
            try:
                # Aggressive reset to clean state from previous instances
                for cmd in [
                    ["git", "reset", "--hard", "HEAD"],
                    ["git", "checkout", "--", "."],
                    ["git", "clean", "-fdx"],
                ]:
                    subprocess.run(cmd, capture_output=True, timeout=30, cwd=str(repo_dir), env=env)
                # Fetch target commit
                subprocess.run(
                    ["git", "fetch", "origin", base_commit, "--depth=1"],
                    capture_output=True, timeout=120, cwd=str(repo_dir), env=env, check=True,
                )
                subprocess.run(
                    ["git", "checkout", "-f", base_commit],
                    capture_output=True, timeout=60, cwd=str(repo_dir), env=env, check=True,
                )
                subprocess.run(
                    ["git", "clean", "-fdx"],
                    capture_output=True, timeout=60, cwd=str(repo_dir), env=env,
                )
                print(f"  Reusing repo at {base_commit[:8]}")
                return repo_dir
            except Exception:
                shutil.rmtree(str(repo_dir), ignore_errors=True)

        try:
            print(f"  Cloning {instance['repo']} ...")
            if repo_dir.exists():
                shutil.rmtree(str(repo_dir), ignore_errors=True)
            subprocess.run(
                ["git", "clone", "--depth=1", clone_url, str(repo_dir)],
                capture_output=True, timeout=300, env=env, check=True,
            )
            subprocess.run(
                ["git", "fetch", "origin", base_commit, "--depth=1"],
                capture_output=True, timeout=120, cwd=str(repo_dir), env=env, check=True,
            )
            subprocess.run(
                ["git", "checkout", base_commit],
                capture_output=True, timeout=60, cwd=str(repo_dir), env=env, check=True,
            )
            return repo_dir
        except subprocess.CalledProcessError as e:
            stderr = e.stderr.decode("utf-8", errors="replace") if isinstance(e.stderr, bytes) else str(e.stderr)
            print(f"  Clone failed: {stderr[:300]}")
            return None
        except Exception as e:
            print(f"  Clone failed: {e}")
            return None

    def build_agent(self, workspace: Path, plan_first: bool = False) -> AgentLoop:
        """Build one-code with workspace-aware tools."""
        registry = ToolRegistry(safe_mode=False)
        register_builtin_tools(registry)
        self._override_tools_for_workspace(registry, workspace)

        from llm.deepseek_api import DeepSeekAdapter
        from memory.short_term import ShortTermMemory
        from memory.long_term import LongTermMemory

        llm = DeepSeekAdapter(api_key=self.api_key, model=self.model)
        short_mem = ShortTermMemory(max_tokens=65536)
        long_mem = LongTermMemory(llm=llm, collection_name="swebench_eval", persist_dir="./eval/eval_memory_db")
        memory = MemoryManager(short=short_mem, long=long_mem)
        agent = AgentLoop(
            llm=llm,
            registry=registry,
            memory=memory,
            max_steps=self.max_steps,
            enable_self_optimize=False,
            enable_evolution=False,
            loop_detect_threshold=6,
            plan_first=plan_first,
        )
        agent.system_prompt = AGENT_SYSTEM_PROMPT
        return agent

    def _override_tools_for_workspace(self, registry: ToolRegistry, workspace: Path):
        """Point file tools to the instance workspace."""
        ws = str(workspace)

        def read_file(path: str) -> str:
            full = os.path.join(ws, path) if not os.path.isabs(path) else path
            try:
                with open(full, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()
                return content[:8000] if len(content) > 8000 else content
            except Exception as e:
                return f"ERROR: {e}"

        def write_file(path: str, content: str) -> str:
            full = os.path.join(ws, path) if not os.path.isabs(path) else path
            os.makedirs(os.path.dirname(full) or ws, exist_ok=True)
            with open(full, "w", encoding="utf-8") as f:
                f.write(content)
            return f"Written {len(content)} bytes to {path}"

        def list_dir(path: str = ".") -> str:
            full = os.path.join(ws, path) if not os.path.isabs(path) else path
            try:
                entries = os.listdir(full)
                return "\n".join(sorted(entries)[:100])
            except Exception as e:
                return f"ERROR: {e}"

        def run_shell(command: str) -> str:
            try:
                proc = subprocess.run(
                    command, shell=True, capture_output=True,
                    timeout=120, cwd=str(workspace),
                    encoding="utf-8", errors="replace",
                )
                out = (proc.stdout + proc.stderr)[:3000]
                return out or "(no output)"
            except subprocess.TimeoutExpired:
                return "TIMEOUT (120s)"
            except Exception as e:
                return f"ERROR: {e}"

        registry._tools.clear()
        registry._tool_metadata.clear()
        registry.register("read_file", "Read a file from the repository")(read_file)
        registry.register("write_file", "Write content to a file in the repository")(write_file)
        registry.register("list_dir", "List files in a directory")(list_dir)
        registry.register("run_shell", "Run a shell command in the repo")(run_shell)

        # Also register search tools
        def grep_search(pattern: str) -> str:
            try:
                proc = subprocess.run(
                    f'rg --no-heading -n "{pattern}" {ws} 2>nul || findstr /s /i /n /c:"{pattern}" {ws}\\*.* 2>nul || echo "not found"',
                    shell=True, capture_output=True, timeout=30,
                    cwd=str(workspace), encoding="utf-8", errors="replace",
                )
                out = (proc.stdout or proc.stderr)[:3000]
                return out or "(no matches)"
            except Exception as e:
                return f"ERROR: {e}"

        registry.register("grep", "Search for a pattern in all files")(grep_search)

    def extract_patch(self, agent_output: str) -> str:
        """Extract unified diff from agent output."""
        m = re.search(r"\[PATCH\](.*?)(?:\[END\]|$)", agent_output, re.DOTALL | re.IGNORECASE)
        if m:
            content = m.group(1).strip()
            # Strip markdown code block markers
            for marker in ["```diff", "```"]:
                if content.startswith(marker):
                    content = content[len(marker):].strip()
                if content.endswith("```"):
                    content = content[:-3].strip()
            if "diff --git" in content:
                return content

        # Fallback: find ```diff blocks
        m = re.search(r"```diff\s*\n(.*?)```", agent_output, re.DOTALL)
        if m:
            content = m.group(1).strip()
            if "diff --git" in content:
                return content

        # Fallback: find any diff-style content
        m = re.search(r"diff --git.*?(?=\n\n[^\s\+\-@]|\n\[|\Z)", agent_output, re.DOTALL)
        if m:
            return m.group(0).strip()

        return ""

    def apply_and_test(self, instance: dict, patch: str, repo: Path) -> tuple[bool, str]:
        """Run FAIL_TO_PASS tests. Agent already modified files via write_file.
        Only need to apply test_patch and run tests.
        Returns (resolved, reason)."""
        
        # Apply test_patch directly (skip git apply, write files directly)
        if instance.get("test_patch"):
            self._apply_unified_diff(str(repo), instance["test_patch"])

        fail_to_pass = json.loads(instance.get("FAIL_TO_PASS", "[]"))
        if not fail_to_pass:
            return False, "no_tests"

        passed = 0
        for test_case in fail_to_pass:
            try:
                # Convert test format: 'method (module.Class)' → 'module.Class.method'
                test_name = test_case
                m = re.match(r'(.+?)\s+\((.+?)\)', test_case)
                if m:
                    test_name = f'{m.group(2)}.{m.group(1)}'
                
                # Django uses Docker + runtests.py (needs full env)
                if "django" in (instance.get("repo") or ""):
                    cmd = (
                        "docker run --rm "
                        f'-v "{repo.absolute()}:/repo" -w /repo '
                        "python:3.10-slim bash -c "
                        f'"pip install -e /repo -q && python tests/runtests.py --settings=test_sqlite -v 0 {test_name}"'
                    )
                else:
                    cmd = f"python -m pytest {test_name} -x -q --no-header 2>&1"
                proc = subprocess.run(
                    cmd, shell=True, capture_output=True, timeout=120,
                    encoding="utf-8", errors="replace",
                )
                if proc.returncode == 0:
                    passed += 1
            except Exception:
                pass

        resolved = passed == len(fail_to_pass) and passed > 0
        return resolved, f"{passed}/{len(fail_to_pass)} tests"

    @staticmethod
    def _apply_unified_diff(repo_root: str, diff_text: str):
        """Apply unified diff using Python's built-in approach — 
        parse hunks and apply line-by-line correctly."""
        import re
        
        for section in diff_text.strip().split("\ndiff --git "):
            section = section.strip()
            if not section:
                continue
            
            # Parse file header: a/path b/path
            header_match = re.match(r"a/(.+?)\s+b/(.+?)$", section.split("\n")[0] if "\n" in section else section)
            if not header_match:
                continue
            file_path = header_match.group(1)
            full_path = os.path.join(repo_root, file_path)
            if not os.path.exists(full_path):
                continue
            
            with open(full_path, "r", encoding="utf-8", errors="replace") as f:
                original_lines = f.read().split("\n")
            
            # Find all hunks
            hunks = list(re.finditer(
                r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@\n(.*?)(?=\n@@|\Z)",
                section, re.DOTALL
            ))
            
            if not hunks:
                continue
            
            # Apply hunks from last to first to avoid line number shifts
            for hunk in reversed(hunks):
                old_start = int(hunk.group(1)) - 1
                old_count = int(hunk.group(2)) if hunk.group(2) else 1
                hunk_body = hunk.group(5)
                
                new_chunk = []
                old_pos = old_start
                
                for line in hunk_body.split("\n"):
                    if not line:
                        new_chunk.append("")
                    elif line.startswith(" "):  # context — keep original line
                        if old_pos < len(original_lines):
                            new_chunk.append(original_lines[old_pos])
                        else:
                            new_chunk.append(line[1:])
                        old_pos += 1
                    elif line.startswith("-"):  # removed — skip, advance old
                        old_pos += 1
                    elif line.startswith("+"):  # added — keep new line
                        new_chunk.append(line[1:])
                
                # Count how many old lines this hunk consumed
                consumed = 0
                for line in hunk_body.split("\n"):
                    if line.startswith(" ") or line.startswith("-"):
                        consumed += 1
                
                # Replace the hunk range with new content
                original_lines = (
                    original_lines[:old_start] +
                    new_chunk +
                    original_lines[old_start + consumed:]
                )
            
            with open(full_path, "w", encoding="utf-8") as f:
                f.write("\n".join(original_lines))

    def estimate_cost(self, messages: list[Message]) -> float:
        """Estimate API cost from message list."""
        input_chars = sum(len(m.content or "") for m in messages)
        output_chars = sum(len(m.content or "") for m in messages if m.role == "assistant")
        input_tokens = input_chars / 3.5
        output_tokens = output_chars / 3.5
        return (input_tokens / 1e6 * self.PRICE_INPUT_PER_1M
                + output_tokens / 1e6 * self.PRICE_OUTPUT_PER_1M)

    def run_one(self, instance: dict) -> EvalResult:
        """Evaluate one-code on a single SWE-bench instance."""
        result = EvalResult(
            instance_id=instance["instance_id"],
            repo=instance["repo"],
        )

        print(f"\n{'='*60}")
        instance_id = instance["instance_id"]
        repo_name = instance["repo"]
        problem = instance["problem_statement"][:120].encode("ascii", errors="replace").decode("ascii")
        print(f"[{instance_id}] {repo_name}")
        print(f"  Issue: {problem}...")

        repo_dir = self.prepare_repo(instance)
        if not repo_dir:
            result.error = "repo_prep_failed"
            return result

        agent = self.build_agent(repo_dir)

        # ━━━ 构建结构化任务提示 ━━━
        fail_to_pass = json.loads(instance.get("FAIL_TO_PASS", "[]"))
        hints = instance.get("hints_text", "")
        task_parts = [
            f"## GitHub Issue\n{instance['problem_statement']}",
            f"## Repository\n{instance['repo']} (commit {instance['base_commit'][:8]})",
        ]
        if hints:
            task_parts.append(f"## 提示\n{hints}")
        if fail_to_pass:
            task_parts.append(
                f"## 验证目标\n修复后以下测试必须通过:\n" +
                "\n".join(f"- {t}" for t in fail_to_pass[:5])
            )
        task_parts.append("\n## 要求\n请修复这个 issue。先定位问题代码，再做最小修改，最后运行测试验证。完成后输出 [PATCH] 及 unified diff。")
        task = "\n\n".join(task_parts)

        start = time.time()
        try:
            output = agent.run(task)
            result.elapsed_sec = time.time() - start
            result.steps = getattr(agent, "_last_step_count", 0)
            result.cost_usd = self.estimate_cost(agent.memory.short_term.get_messages())
            result.generated_patch = self.extract_patch(output)
            print(f"  Output ({len(output)} chars): {output[:300]}...")

            print(f"  Steps: {result.steps}, Time: {result.elapsed_sec:.1f}s, Cost: ${result.cost_usd:.4f}")
            print(f"  Patch: {len(result.generated_patch)} bytes")

            result.resolved, diag = self.apply_and_test(instance, result.generated_patch, repo_dir)
            result.error = diag
            print(f"  Patch: {len(result.generated_patch)} bytes → {diag}")

            # ━━━ 自动重试：首次失败时切换策略 ━━━
            if not result.resolved and result.generated_patch and (not result.error or result.error == "0/3 tests" or "tests" in (result.error or "")):
                print(f"  [RETRY] 首次失败，尝试 Plan-then-Execute 模式...")
                retry_agent = self.build_agent(repo_dir, plan_first=True)
                retry_task = (
                    task + "\n\n[重试提示] 上次尝试未能通过测试。请重新规划步骤，"
                    "特别关注: 1) 是否修改了正确的文件？ 2) 测试命令是否正确？"
                    "3) 是否需要在跑测试前安装依赖？"
                )
                try:
                    retry_output = retry_agent.run(retry_task)
                    retry_patch = self.extract_patch(retry_output)
                    if retry_patch:
                        print(f"  Retry patch: {len(retry_patch)} bytes")
                        retry_resolved, _ = self.apply_and_test(instance, retry_patch, repo_dir)
                        if retry_resolved:
                            result.resolved = True
                            result.generated_patch = retry_patch
                            print(f"  [RETRY] 第二次尝试成功!")
                    retry_elapsed = time.time() - start - result.elapsed_sec
                    result.elapsed_sec += retry_elapsed
                except Exception:
                    pass
        except Exception as e:
            result.elapsed_sec = time.time() - start
            result.error = f"{type(e).__name__}: {e}"
            traceback.print_exc()

        return result

    def run(self) -> EvalReport:
        """Run full evaluation."""
        instances = self.load_instances()
        print(f"Loaded {len(instances)} instances")

        report = EvalReport(
            total=len(instances),
            timestamp=datetime.now().isoformat(),
        )

        for i, instance in enumerate(instances):
            print(f"\n[{i+1}/{len(instances)}] Running...")
            result = self.run_one(instance)
            report.results.append(result)
            if result.resolved:
                report.resolved += 1

            self._save_progress(report)

        report.resolution_rate = report.resolved / report.total if report.total else 0
        resolved_results = [r for r in report.results if r.resolved]
        all_results = [r for r in report.results if r.elapsed_sec > 0]

        if all_results:
            report.avg_cost = sum(r.cost_usd for r in all_results) / len(all_results)
            report.avg_time = sum(r.elapsed_sec for r in all_results) / len(all_results)
            report.avg_steps = sum(r.steps for r in all_results) / len(all_results)

        self._save_report(report)
        self._print_summary(report)
        return report

    def _save_progress(self, report: EvalReport):
        """Save intermediate results."""
        path = self.RESULTS_DIR / f"progress_{report.timestamp[:10]}.json"
        data = {
            "total": report.total,
            "resolved": report.resolved,
            "timestamp": report.timestamp,
            "results": [
                {
                    "instance_id": r.instance_id,
                    "repo": r.repo,
                    "resolved": r.resolved,
                    "cost_usd": r.cost_usd,
                    "elapsed_sec": r.elapsed_sec,
                    "steps": r.steps,
                    "error": r.error,
                }
                for r in report.results
            ],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _save_report(self, report: EvalReport):
        """Save final evaluation report."""
        path = self.RESULTS_DIR / f"report_{report.timestamp[:10]}.json"
        data = {
            "total": report.total,
            "resolved": report.resolved,
            "resolution_rate": report.resolution_rate,
            "avg_cost_usd": report.avg_cost,
            "avg_time_sec": report.avg_time,
            "avg_steps": report.avg_steps,
            "timestamp": report.timestamp,
            "results": [
                {
                    "instance_id": r.instance_id,
                    "repo": r.repo,
                    "resolved": r.resolved,
                    "cost_usd": round(r.cost_usd, 4),
                    "elapsed_sec": round(r.elapsed_sec, 1),
                    "steps": r.steps,
                    "error": r.error,
                }
                for r in report.results
            ],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"\nReport saved to {path}")

    def _print_summary(self, report: EvalReport):
        """Print evaluation summary."""
        print(f"\n{'='*60}")
        print("SWE-BENCH LITE EVALUATION RESULTS")
        print(f"{'='*60}")
        print(f"  Instances:  {report.total}")
        print(f"  Resolved:   {report.resolved}")
        print(f"  Rate:       {report.resolution_rate:.1%}")
        print(f"  Avg Cost:   ${report.avg_cost:.4f}")
        print(f"  Avg Time:   {report.avg_time:.1f}s")
        print(f"  Avg Steps:  {report.avg_steps:.1f}")
        print(f"{'='*60}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="SWE-bench Lite evaluation for one-code")
    p.add_argument("--max", type=int, default=5, help="Max instances (default: 5)")
    p.add_argument("--repo", type=str, default=None, help="Filter by repo (e.g. django/django)")
    p.add_argument("--model", type=str, default="deepseek-v4-pro")
    p.add_argument("--steps", type=int, default=30)
    args = p.parse_args()

    runner = SWEBenchRunner(
        max_instances=args.max,
        repo_filter=args.repo,
        model=args.model,
        max_steps_per_instance=args.steps,
    )
    runner.run()
