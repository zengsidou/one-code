# -*- coding: utf-8 -*-
"""Agent 状态检查点 — 架构进化后保存/恢复状态，支持进程重启"""
import json, os, sys
from datetime import datetime

CHECKPOINT_FILE = "agent_checkpoint.json"
RESTART_FLAG = "agent_restart.flag"


class AgentCheckpoint:
    """保存和恢复 Agent 在执行过程中的关键状态

    当架构自进化修改代码后，Agent 需要重启进程以加载新代码。
    检查点确保重启后能从中断处继续。
    """

    @staticmethod
    def save(agent_loop, current_task: str, restart_reason: str = "") -> str:
        """保存检查点到文件

        Args:
            agent_loop: AgentLoop 实例
            current_task: 当前未完成的任务描述
            restart_reason: 重启原因（如 'architecture_evolution'）

        Returns:
            检查点文件路径
        """
        state = {
            "checkpoint_version": 1,
            "timestamp": datetime.now().isoformat(),
            "restart_reason": restart_reason,
            "current_task": current_task,
            "self_optimize_retry_count": agent_loop._self_optimize_retry_count,
            "meta_optimize_count": agent_loop._meta_optimize_count,
            "last_step_count": agent_loop._last_step_count if hasattr(agent_loop, "_last_step_count") else 0,
            "enable_evolution": agent_loop.enable_evolution,
            "enable_self_optimize": agent_loop.enable_self_optimize,
            "skill_library_file": agent_loop._skill_library.filepath if agent_loop._skill_library else "",
            "ability_profile_file": agent_loop._ability_profile.filepath if agent_loop._ability_profile else "",
            "fix_history_file": agent_loop._fix_history.filepath if agent_loop._fix_history else "",
            "system_prompt": agent_loop.system_prompt[:500],
        }

        with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

        # 写入重启标记
        with open(RESTART_FLAG, "w") as f:
            f.write("1")

        return CHECKPOINT_FILE

    @staticmethod
    def load() -> dict | None:
        """加载检查点，如果不存在返回 None"""
        if not os.path.exists(CHECKPOINT_FILE):
            return None
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def has_restart_flag() -> bool:
        """检查是否需要重启"""
        return os.path.exists(RESTART_FLAG)

    @staticmethod
    def clear_restart_flag():
        """清除重启标记"""
        if os.path.exists(RESTART_FLAG):
            os.remove(RESTART_FLAG)

    @staticmethod
    def clear_checkpoint():
        """清除检查点文件"""
        for f in [CHECKPOINT_FILE, RESTART_FLAG]:
            if os.path.exists(f):
                os.remove(f)

    @staticmethod
    def trigger_restart(agent_loop, task: str, reason: str = "architecture_evolution"):
        """保存检查点并触发进程重启

        保存当前状态后替换当前进程，加载新代码继续执行。
        """
        AgentCheckpoint.save(agent_loop, task, reason)
        # 用 sys.executable 重启当前脚本
        python = sys.executable
        script = sys.argv[0] if sys.argv else __file__
        args = [python, script] + sys.argv[1:] + ["--checkpoint"]
        os.execv(python, args)
