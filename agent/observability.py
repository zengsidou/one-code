# -*- coding: utf-8 -*-
"""Agent observability — metrics, tracing, evaluation dashboard.

Tracks execution stats, tool usage, cost estimates, and generates reports.
Designed for both real-time monitoring and post-hoc analysis.
"""
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass
class RunMetrics:
    """Metrics for a single agent run."""
    run_id: str = ""
    task: str = ""
    start_time: float = 0.0
    end_time: float = 0.0
    steps: int = 0
    tool_calls: int = 0
    tool_errors: int = 0
    circuit_breakers: int = 0
    loop_detections: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    success: bool = False
    failure_type: str = ""
    mode: str = "react"  # react / plan_execute
    plan_steps: int = 0
    pressure_levels: list[int] = field(default_factory=list)


@dataclass
class ToolTrace:
    """Trace of a single tool call."""
    step: int
    tool_name: str
    arguments: dict
    result_preview: str
    duration_ms: float
    error: bool = False


class Observability:
    """Agent observability collector and reporter.

    Usage:
        obs = Observability()
        agent = AgentLoop(observability=obs, ...)
        result = agent.run(task)
        obs.print_report()
        obs.save("run_001.json")
    """

    PRICE_INPUT_PER_1M = 0.137   # DeepSeek: $0.137/M input
    PRICE_OUTPUT_PER_1M = 0.274  # DeepSeek: $0.274/M output

    def __init__(self, save_dir: str = "./observability"):
        self.save_dir = Path(save_dir)
        os.makedirs(self.save_dir, exist_ok=True)
        self.current: RunMetrics = RunMetrics()
        self.traces: list[ToolTrace] = []
        self.history: list[RunMetrics] = []
        self._step_timer: dict[str, float] = {}
        self._run_history_file = self.save_dir / "run_history.jsonl"

    def start_run(self, task: str, mode: str = "react") -> str:
        """Begin tracking a new run. Returns run_id."""
        run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(task) & 0xFFFF:04x}"
        self.current = RunMetrics(run_id=run_id, task=task, start_time=time.time(), mode=mode)
        self.traces = []
        return run_id

    def end_run(self, success: bool, failure_type: str = ""):
        """Mark the current run as complete."""
        self.current.end_time = time.time()
        self.current.success = success
        self.current.failure_type = failure_type
        self.history.append(self.current)

    def trace_tool_start(self, step: int, tool_name: str, arguments: dict):
        """Record tool call start."""
        key = f"{step}_{tool_name}"
        self._step_timer[key] = time.time()

    def trace_tool_end(self, step: int, tool_name: str, result: str, error: bool = False):
        """Record tool call completion."""
        key = f"{step}_{tool_name}"
        duration = (time.time() - self._step_timer.pop(key, time.time())) * 1000
        self.traces.append(ToolTrace(
            step=step, tool_name=tool_name,
            arguments={}, result_preview=result[:200] if result else "",
            duration_ms=round(duration, 2), error=error,
        ))
        self.current.tool_calls += 1
        if error:
            self.current.tool_errors += 1

    def record_step(self, steps: int, pressure: int = 0):
        """Record current step count and pressure level."""
        self.current.steps = steps
        if pressure > 0:
            self.current.pressure_levels.append(pressure)

    def record_circuit_breaker(self):
        self.current.circuit_breakers += 1

    def record_loop_detection(self):
        self.current.loop_detections += 1

    def estimate_tokens(self, messages: list) -> tuple[int, int]:
        """Estimate input/output tokens from message list."""
        input_chars = sum(len((m.content or "") + (getattr(m, 'reasoning_content', '') or "")) for m in messages if m.role in ("user", "system"))
        output_chars = sum(len(m.content or "") for m in messages if m.role == "assistant")
        input_tokens = int(input_chars / 3.5)
        output_tokens = int(output_chars / 3.5)
        self.current.input_tokens += input_tokens
        self.current.output_tokens += output_tokens
        self.current.cost_usd += (
            input_tokens / 1e6 * self.PRICE_INPUT_PER_1M +
            output_tokens / 1e6 * self.PRICE_OUTPUT_PER_1M
        )
        return input_tokens, output_tokens

    def elapsed_sec(self) -> float:
        return time.time() - self.current.start_time

    def print_report(self):
        """Print a formatted report to console."""
        m = self.current
        elapsed = m.end_time - m.start_time if m.end_time else time.time() - m.start_time
        print(f"\n{'='*60}")
        print(f"OBSERVABILITY REPORT  {m.run_id}")
        print(f"{'='*60}")
        print(f"  Task:       {m.task[:80]}...")
        print(f"  Mode:       {m.mode}")
        print(f"  Duration:   {elapsed:.1f}s")
        print(f"  Steps:      {m.steps}")
        print(f"  Tool calls: {m.tool_calls} ({m.tool_errors} errors)")
        print(f"  Breakers:   {m.circuit_breakers} circuit / {m.loop_detections} loop")
        print(f"  Est tokens: {m.input_tokens:,} in / {m.output_tokens:,} out")
        print(f"  Est cost:   ${m.cost_usd:.4f}")
        print(f"  Result:     {'PASS' if m.success else 'FAIL'}{' (' + m.failure_type + ')' if m.failure_type else ''}")
        if m.pressure_levels:
            avg_p = sum(m.pressure_levels) / len(m.pressure_levels)
            print(f"  Pressure:   avg {avg_p:.1f}/3, max {max(m.pressure_levels)}/3")

        if self.traces:
            print(f"\n  Tool trace ({min(10, len(self.traces))} of {len(self.traces)}):")
            for t in self.traces[:10]:
                flag = " [ERR]" if t.error else ""
                print(f"    step{t.step:>3} {t.tool_name:<16} {t.duration_ms:>6.0f}ms{flag}  {t.result_preview[:60]}...")

        print(f"{'='*60}\n")

    def save(self, filename: str | None = None):
        """Save run metrics to file."""
        fname = filename or f"{self.current.run_id}.json"
        path = self.save_dir / fname
        data = {
            "run_id": self.current.run_id,
            "task": self.current.task[:200],
            "mode": self.current.mode,
            "duration_sec": round(self.current.end_time - self.current.start_time, 2),
            "steps": self.current.steps,
            "tool_calls": self.current.tool_calls,
            "tool_errors": self.current.tool_errors,
            "circuit_breakers": self.current.circuit_breakers,
            "loop_detections": self.current.loop_detections,
            "est_input_tokens": self.current.input_tokens,
            "est_output_tokens": self.current.output_tokens,
            "est_cost_usd": round(self.current.cost_usd, 4),
            "success": self.current.success,
            "failure_type": self.current.failure_type,
            "plan_steps": self.current.plan_steps,
            "avg_pressure": round(sum(self.current.pressure_levels) / max(1, len(self.current.pressure_levels)), 1),
            "traces": [
                {
                    "step": t.step, "tool": t.tool_name,
                    "duration_ms": t.duration_ms, "error": t.error,
                    "preview": t.result_preview[:100],
                } for t in self.traces[:50]
            ],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        with open(self._run_history_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
        return path

    def load_history(self) -> list[dict]:
        """Load all saved run metrics."""
        if not self._run_history_file.exists():
            return []
        history = []
        with open(self._run_history_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    history.append(json.loads(line))
        return history

    def get_summary(self) -> dict:
        """Get aggregate summary across all historical runs."""
        history = self.load_history()
        if not history:
            return {"runs": 0}

        success_count = sum(1 for h in history if h.get("success"))
        return {
            "runs": len(history),
            "success_rate": round(success_count / len(history), 3),
            "avg_duration": round(sum(h.get("duration_sec", 0) for h in history) / len(history), 1),
            "avg_steps": round(sum(h.get("steps", 0) for h in history) / len(history), 1),
            "avg_cost": round(sum(h.get("est_cost_usd", 0) for h in history) / len(history), 4),
            "total_cost": round(sum(h.get("est_cost_usd", 0) for h in history), 4),
            "modes": {},
            "tools": {},
        }
