# -*- coding: utf-8 -*-
"""CI/CD pipeline — auto-test, SWE-bench validation, PR description generation.

Maps to JD: "推动关键项目形成可复用、可推广的实践"
        "代码生成、测试辅助、代码评审、研发协同等能力进入真实研发流程"
"""
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass
class PipelineResult:
    stage: str
    passed: bool
    duration_sec: float = 0
    output: str = ""
    details: dict = field(default_factory=dict)


@dataclass
class PipelineReport:
    commit: str = ""
    timestamp: str = ""
    stages: list[PipelineResult] = field(default_factory=list)
    overall: bool = False


class CIPipeline:
    """CI/CD 流水线 — 测试 → SWE-bench → 报告"""

    def __init__(self, project_root: str = "."):
        self.root = Path(project_root)
        self.report = PipelineReport()

    def run(self) -> PipelineReport:
        """运行完整流水线。"""
        self.report.commit = self._get_commit()
        self.report.timestamp = datetime.now().isoformat()

        stages = [
            ("unit-tests", self._run_unit_tests),
            ("swebench-quick", self._run_swebench_quick),
            ("syntax-check", self._run_syntax_check),
            ("generate-report", self._generate_report),
        ]

        for name, func in stages:
            result = func()
            result.stage = name
            self.report.stages.append(result)
            if not result.passed:
                self.report.overall = False

        if all(s.passed for s in self.report.stages):
            self.report.overall = True

        self._save_report()
        return self.report

    def _get_commit(self) -> str:
        try:
            r = subprocess.run(
                ["git", "log", "--oneline", "-1"],
                capture_output=True, cwd=str(self.root), timeout=10,
                encoding="utf-8", errors="replace",
            )
            return r.stdout.strip()[:80]
        except Exception:
            return "unknown"

    def _run_unit_tests(self) -> PipelineResult:
        """运行所有单元测试。"""
        start = time.time()
        try:
            r = subprocess.run(
                [sys.executable, "-m", "pytest", "tests/", "-q", "--tb=short"],
                capture_output=True, cwd=str(self.root), timeout=120,
                encoding="utf-8", errors="replace",
            )
            passed = r.returncode == 0
            output = (r.stdout + "\n" + r.stderr)[:2000]
            details = self._parse_test_output(output)
            return PipelineResult(
                stage="unit-tests", passed=passed,
                duration_sec=time.time() - start,
                output=output, details=details,
            )
        except subprocess.TimeoutExpired:
            return PipelineResult(stage="unit-tests", passed=False, duration_sec=120, output="Timeout")
        except Exception as e:
            return PipelineResult(stage="unit-tests", passed=False, output=str(e))

    def _parse_test_output(self, output: str) -> dict:
        import re
        passed = 0
        failed = 0
        m = re.search(r"(\d+)\s+passed", output)
        if m: passed = int(m.group(1))
        m = re.search(r"(\d+)\s+failed", output)
        if m: failed = int(m.group(1))
        return {"passed": passed, "failed": failed, "total": passed + failed}

    def _run_swebench_quick(self) -> PipelineResult:
        """快速 SWE-bench 验证（1 个实例）。"""
        start = time.time()
        swebench_script = self.root / "eval" / "swebench_runner.py"
        if not swebench_script.exists():
            return PipelineResult(stage="swebench-quick", passed=True, output="SWE-bench eval not configured, skipping")

        try:
            r = subprocess.run(
                [sys.executable, "-m", "eval.swebench_runner", "--max", "1", "--repo", "django/django", "--steps", "15"],
                capture_output=True, cwd=str(self.root), timeout=600,
                encoding="utf-8", errors="replace",
            )
            resolved = "Resolved:   1" in r.stdout or 'resolved": true' in r.stdout.lower()
            return PipelineResult(
                stage="swebench-quick", passed=resolved,
                duration_sec=time.time() - start,
                output=r.stdout[-500:],
            )
        except subprocess.TimeoutExpired:
            return PipelineResult(stage="swebench-quick", passed=False, duration_sec=600, output="Timeout")
        except Exception as e:
            return PipelineResult(stage="swebench-quick", passed=False, output=str(e))

    def _run_syntax_check(self) -> PipelineResult:
        """语法检查所有 Python 文件。"""
        start = time.time()
        errors = []
        for py_file in self.root.rglob("*.py"):
            if any(x in str(py_file) for x in ["__pycache__", ".git", "eval/repos", "chroma_data"]):
                continue
            try:
                import py_compile
                py_compile.compile(str(py_file), doraise=True)
            except py_compile.PyCompileError as e:
                errors.append(f"{py_file.name}: {e}")
        passed = len(errors) == 0
        return PipelineResult(
            stage="syntax-check", passed=passed,
            duration_sec=time.time() - start,
            output="\n".join(errors) if errors else "All files OK",
            details={"files_checked": sum(1 for _ in self.root.rglob("*.py")), "errors": len(errors)},
        )

    def _generate_report(self) -> PipelineResult:
        """生成 CI 报告。"""
        path = self.root / "ci" / "reports"
        os.makedirs(path, exist_ok=True)
        report_path = path / f"ci_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        data = {
            "commit": self.report.commit,
            "timestamp": self.report.timestamp,
            "overall": self.report.overall,
            "stages": [
                {
                    "stage": s.stage,
                    "passed": s.passed,
                    "duration_sec": round(s.duration_sec, 2),
                    "details": s.details,
                }
                for s in self.report.stages
            ],
        }
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        return PipelineResult(
            stage="generate-report", passed=True,
            output=f"Report saved to {report_path}",
            details={"path": str(report_path)},
        )

    def _save_report(self):
        """保存最新报告到 ci/latest.json。"""
        path = self.root / "ci"
        os.makedirs(path, exist_ok=True)
        data = {
            "commit": self.report.commit,
            "timestamp": self.report.timestamp,
            "overall": self.report.overall,
            "stages": [
                {
                    "stage": s.stage,
                    "passed": s.passed,
                    "duration_sec": round(s.duration_sec, 2),
                }
                for s in self.report.stages
            ],
        }
        with open(path / "latest.json", "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def print_report(self):
        """打印 CI 报告。"""
        print(f"\n{'='*55}")
        print(f"CI PIPELINE REPORT")
        print(f"{'='*55}")
        print(f"  Commit:   {self.report.commit}")
        print(f"  Time:     {self.report.timestamp[:19]}")
        print(f"  Result:   {'PASS' if self.report.overall else 'FAIL'}")
        print(f"{'─'*55}")
        for s in self.report.stages:
            icon = "✓" if s.passed else "✗"
            print(f"  {icon} {s.stage:<20} {s.duration_sec:>5.1f}s")
        print(f"{'='*55}\n")


if __name__ == "__main__":
    ci = CIPipeline()
    ci.run()
    ci.print_report()
