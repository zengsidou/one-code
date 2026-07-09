# -*- coding: utf-8 -*-
"""契约先行小实验 runner

用法:
  set DEEPSEEK_API_KEY=...
  python -m eval.contract_first.run --mode contract --task-id T1
  python -m eval.contract_first.run --mode all --task-id T1   # A/B/C 各跑一遍（仍需人工打分）

说明:
  - direct / plan / contract 三种条件
  - 自动记录预览文本、步数、日志路径
  - success / consistency / rework 需人工写入 results/rows.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EVAL_DIR = Path(__file__).resolve().parent
TASKS_PATH = EVAL_DIR / "tasks.json"
RESULTS_DIR = EVAL_DIR / "results"
WORK_ROOT = EVAL_DIR / "workspaces"


def load_tasks() -> dict:
    return json.loads(TASKS_PATH.read_text(encoding="utf-8"))


def get_task(bundle: dict, task_id: str) -> dict:
    for t in bundle["tasks"]:
        if t["id"] == task_id:
            return t
    raise SystemExit(f"unknown task-id: {task_id}")


def prepare_workspace(task_id: str, mode: str) -> Path:
    ws = WORK_ROOT / task_id / mode
    if ws.exists():
        shutil.rmtree(ws)
    ws.mkdir(parents=True, exist_ok=True)
    return ws


def append_row(row: dict) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "rows.jsonl"
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return out


def build_agent(max_steps: int, workspace: Path):
    from tools.registry import ToolRegistry
    from tools.builtin import register_builtin_tools
    from memory.short_term import ShortTermMemory
    from memory import MemoryManager
    from llm.deepseek_api import DeepSeekAdapter
    from agent.loop import AgentLoop

    class _NullLongTerm:
        """Experiment runs should not touch shared ChromaDB (avoids compaction races)."""

        def store(self, *args, **kwargs):
            return None

        def retrieve(self, *args, **kwargs):
            return []

        def clear(self):
            return None

    os.chdir(workspace)
    key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not key:
        raise SystemExit("请设置 DEEPSEEK_API_KEY")

    llm = DeepSeekAdapter(api_key=key)
    reg = ToolRegistry(safe_mode=True)
    register_builtin_tools(reg, llm=llm)
    mem = MemoryManager(short=ShortTermMemory(), long=_NullLongTerm())
    return AgentLoop(llm=llm, registry=reg, memory=mem, max_steps=max_steps), llm


def run_direct(task: dict, max_steps: int, workspace: Path) -> dict:
    agent, _ = build_agent(max_steps, workspace)
    t0 = time.time()
    result = agent.run(task["prompt"], debug=False)
    return {
        "preview_text": "",
        "preview_rounds": 0,
        "plan_or_steps": "",
        "exec_result": result,
        "elapsed_sec": round(time.time() - t0, 2),
        "steps": getattr(agent, "_last_step_count", None),
    }


def run_plan(task: dict, max_steps: int, workspace: Path) -> dict:
    """文本步骤计划（自动确认），模拟 PTE 门控但不做产物预览。"""
    agent, llm = build_agent(max_steps, workspace)
    from agent.models import Message

    plan_prompt = (
        "请为以下任务生成简洁的执行步骤计划（5–8 步，只写怎么做，不要写最终产物长什么样）。"
        "只输出编号列表。\n\n任务：\n" + task["prompt"]
    )
    t0 = time.time()
    plan_msg = llm.generate(
        [Message(role="user", content=plan_prompt)],
        tools=None,
    )
    plan_text = plan_msg.content or ""
    exec_prompt = (
        f"原始任务:\n{task['prompt']}\n\n"
        f"已确认的执行计划（按步骤执行）:\n{plan_text}\n\n"
        "请开始执行。"
    )
    result = agent.run(exec_prompt, debug=False)
    return {
        "preview_text": plan_text,
        "preview_rounds": 1,
        "plan_or_steps": plan_text,
        "exec_result": result,
        "elapsed_sec": round(time.time() - t0, 2),
        "steps": getattr(agent, "_last_step_count", None),
    }


def run_contract(task: dict, max_steps: int, workspace: Path, auto_confirm: bool = True) -> dict:
    """契约预览 + 逆向拆解；默认 auto_confirm 便于无人值守试跑。"""
    agent, llm = build_agent(max_steps, workspace)
    from agent.contract_first import ContractFirstOrchestrator

    orch = ContractFirstOrchestrator(llm)
    t0 = time.time()
    contract = orch.phase1_detect_and_generate(task["prompt"])

    if not auto_confirm:
        if not orch.phase2_confirm():
            return {
                "preview_text": contract.content,
                "preview_rounds": 1,
                "plan_or_steps": "",
                "exec_result": "[cancelled by user]",
                "elapsed_sec": round(time.time() - t0, 2),
                "steps": 0,
                "cancelled": True,
            }
    else:
        if orch.result:
            orch.result.user_confirmed = True

    steps = orch.phase3_decompose(task["prompt"])
    exec_prompt = orch.phase4_build_execution_prompt(task["prompt"], contract, steps)
    result = agent.run(exec_prompt, debug=False)
    step_text = "\n".join(f"{s.index}. {s.goal}" for s in steps)
    return {
        "preview_text": contract.content,
        "preview_rounds": 1,
        "plan_or_steps": step_text,
        "contract_type": contract.type,
        "contract_summary": contract.summary,
        "exec_result": result,
        "elapsed_sec": round(time.time() - t0, 2),
        "steps": getattr(agent, "_last_step_count", None),
    }


def run_one(mode: str, task: dict, max_steps: int, auto_confirm: bool) -> dict:
    workspace = prepare_workspace(task["id"], mode)
    log_dir = RESULTS_DIR / "runs" / f"{task['id']}_{mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_dir.mkdir(parents=True, exist_ok=True)

    if mode == "direct":
        out = run_direct(task, max_steps, workspace)
    elif mode == "plan":
        out = run_plan(task, max_steps, workspace)
    elif mode == "contract":
        out = run_contract(task, max_steps, workspace, auto_confirm=auto_confirm)
    else:
        raise SystemExit(f"unknown mode: {mode}")

    (log_dir / "preview.txt").write_text(out.get("preview_text") or "", encoding="utf-8")
    (log_dir / "plan_or_steps.txt").write_text(out.get("plan_or_steps") or "", encoding="utf-8")
    (log_dir / "exec_result.txt").write_text(str(out.get("exec_result") or ""), encoding="utf-8")
    (log_dir / "meta.json").write_text(
        json.dumps({k: v for k, v in out.items() if k != "exec_result"}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    row = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "task_id": task["id"],
        "task_type": task["type"],
        "mode": mode,
        "workspace": str(workspace),
        "log_dir": str(log_dir),
        "preview_rounds": out.get("preview_rounds", 0),
        "steps": out.get("steps"),
        "elapsed_sec": out.get("elapsed_sec"),
        "contract_type": out.get("contract_type"),
        # 以下字段需人工补全：
        "direction_ok_before_exec": None if mode == "direct" else None,
        "rework_rounds": None,
        "tokens_preview": None,
        "tokens_exec": None,
        "tokens_total": None,
        "success": None,
        "consistency_1_5": None,
        "notes": "",
    }
    path = append_row(row)
    print(f"\n[ok] mode={mode} task={task['id']} workspace={workspace}")
    print(f"[ok] logs={log_dir}")
    print(f"[ok] row appended -> {path}")
    print("[next] 打开产物，按 success_criteria 人工填写 success/consistency/rework_rounds")
    return row


def main():
    # Avoid Windows GBK console crashes on emoji / CJK box-drawing from models
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

    parser = argparse.ArgumentParser(description="Contract-first mini experiment runner")
    parser.add_argument("--mode", choices=["direct", "plan", "contract", "all"], required=True)
    parser.add_argument("--task-id", required=True, help="T1..T5")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--manual-confirm", action="store_true", help="contract 模式改为人工确认预览")
    args = parser.parse_args()

    bundle = load_tasks()
    task = get_task(bundle, args.task_id)
    max_steps = args.max_steps or int(bundle.get("max_steps", 15))
    modes = ["direct", "plan", "contract"] if args.mode == "all" else [args.mode]

    original_cwd = os.getcwd()
    try:
        for mode in modes:
            os.chdir(original_cwd)
            run_one(mode, task, max_steps, auto_confirm=not args.manual_confirm)
    finally:
        os.chdir(original_cwd)


if __name__ == "__main__":
    main()
