# -*- coding: utf-8 -*-
"""Agent permissions and risk audit system.

Provides fine-grained tool access control and audit logging for enterprise-grade
agent governance. Maps to JD requirement: "协作机制、权限控制、风险治理、效果评估".
"""
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path


class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class AuditEntry:
    timestamp: str = ""
    agent_id: str = ""
    action: str = ""
    tool: str = ""
    details: str = ""
    risk_level: str = "low"
    allowed: bool = True
    workspace: str = ""


class Permissions:
    """Tool-level permission system.

    Usage:
        perms = Permissions(role="swe_bench")
        perms.allow("read_file", "write_file", "list_dir", "run_shell")
        perms.deny("rm_rf", "format_disk")
        registry = ToolRegistry(permissions=perms)
    """

    # Default tool risk classifications
    RISK_CLASSIFICATION = {
        "read_file": RiskLevel.LOW,
        "list_dir": RiskLevel.LOW,
        "grep": RiskLevel.LOW,
        "search_web": RiskLevel.LOW,
        "fetch_url": RiskLevel.LOW,
        "calculate": RiskLevel.LOW,
        "write_file": RiskLevel.MEDIUM,
        "edit_file": RiskLevel.MEDIUM,
        "run_shell": RiskLevel.HIGH,
        "delegate_task": RiskLevel.MEDIUM,
        "register_tool": RiskLevel.CRITICAL,
        "delete_file": RiskLevel.HIGH,
        "format": RiskLevel.CRITICAL,
        "rm": RiskLevel.CRITICAL,
        "shutdown": RiskLevel.CRITICAL,
    }

    def __init__(self, role: str = "default"):
        self.role = role
        self._allowed: set[str] = set()
        self._denied: set[str] = set()
        self._max_risk: RiskLevel = RiskLevel.HIGH
        self._workspace_root: str = ""
        self._quota: dict[str, int] = {}  # tool_name → max calls
        self._used: dict[str, int] = {}

    def allow(self, *tool_names: str):
        self._allowed.update(tool_names)

    def deny(self, *tool_names: str):
        self._denied.update(tool_names)

    def set_max_risk(self, level: RiskLevel):
        self._max_risk = level

    def set_workspace(self, path: str):
        self._workspace_root = os.path.abspath(path)

    def set_quota(self, tool: str, max_calls: int):
        self._quota[tool] = max_calls

    def check(self, tool_name: str, arguments: dict | None = None) -> tuple[bool, str]:
        """Check if a tool call is allowed. Returns (allowed, reason)."""
        if tool_name in self._denied:
            return False, f"工具 {tool_name} 已被显式禁止"

        if self._allowed and tool_name not in self._allowed:
            return False, f"工具 {tool_name} 不在允许列表中"

        risk = self.RISK_CLASSIFICATION.get(tool_name, RiskLevel.MEDIUM)
        risk_order = {RiskLevel.LOW: 0, RiskLevel.MEDIUM: 1, RiskLevel.HIGH: 2, RiskLevel.CRITICAL: 3}
        if risk_order[risk] > risk_order[self._max_risk]:
            return False, f"工具 {tool_name} 风险等级 {risk.value} 超出允许上限 {self._max_risk.value}"

        if tool_name in self._quota:
            used = self._used.get(tool_name, 0)
            if used >= self._quota[tool_name]:
                return False, f"工具 {tool_name} 已达配额上限 ({self._quota[tool_name]} 次)"
            self._used[tool_name] = used + 1

        if tool_name in ("write_file", "edit_file") and arguments:
            filepath = arguments.get("path") or arguments.get("filePath") or ""
            if filepath and self._workspace_root:
                abs_path = os.path.abspath(os.path.join(self._workspace_root, filepath))
                if not abs_path.startswith(self._workspace_root):
                    return False, f"文件路径 {filepath} 在工作区外，拒绝写入"

        return True, ""

    def get_risk(self, tool_name: str) -> RiskLevel:
        return self.RISK_CLASSIFICATION.get(tool_name, RiskLevel.MEDIUM)


class AuditLogger:
    """Enterprise-grade audit log for all agent actions.

    Usage:
        audit = AuditLogger(save_dir="./audit_logs")
        audit.record(agent_id="agent_1", action="tool_call", tool="run_shell", details="pytest tests/")
        audit.report()  # print summary
    """

    def __init__(self, save_dir: str = "./audit_logs"):
        self.save_dir = Path(save_dir)
        os.makedirs(self.save_dir, exist_ok=True)
        self._entries: list[AuditEntry] = []
        self._log_file = self.save_dir / "audit.jsonl"

    def record(
        self,
        agent_id: str,
        action: str,
        tool: str = "",
        details: str = "",
        risk_level: str = "low",
        allowed: bool = True,
        workspace: str = "",
    ):
        entry = AuditEntry(
            timestamp=datetime.now().isoformat(),
            agent_id=agent_id,
            action=action,
            tool=tool,
            details=details[:500],
            risk_level=risk_level,
            allowed=allowed,
            workspace=workspace,
        )
        self._entries.append(entry)
        with open(self._log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "ts": entry.timestamp,
                "agent": entry.agent_id,
                "action": entry.action,
                "tool": entry.tool,
                "details": entry.details,
                "risk": entry.risk_level,
                "allowed": entry.allowed,
                "workspace": entry.workspace,
            }, ensure_ascii=False) + "\n")

    def report(self) -> dict:
        """Generate audit summary report."""
        if not self._entries:
            return {"entries": 0}

        total = len(self._entries)
        blocked = sum(1 for e in self._entries if not e.allowed)
        by_risk = {}
        by_tool = {}
        for e in self._entries:
            by_risk[e.risk_level] = by_risk.get(e.risk_level, 0) + 1
            if e.tool:
                by_tool[e.tool] = by_tool.get(e.tool, 0) + 1

        high_risk_actions = [
            {"tool": e.tool, "details": e.details, "ts": e.timestamp}
            for e in self._entries if e.risk_level in ("high", "critical")
        ][-20:]

        return {
            "total_actions": total,
            "blocked_actions": blocked,
            "block_rate": round(blocked / total, 3) if total else 0,
            "by_risk_level": by_risk,
            "by_tool": by_tool,
            "recent_high_risk": high_risk_actions,
        }

    def print_report(self):
        r = self.report()
        print(f"\n{'='*50}")
        print("AUDIT LOG SUMMARY")
        print(f"{'='*50}")
        print(f"  Actions:  {r['total_actions']} ({r['blocked_actions']} blocked)")
        print(f"  Block %:  {r['block_rate']:.1%}")
        print(f"  By risk:  {r['by_risk_level']}")
        print(f"  Top tools: {dict(sorted(r['by_tool'].items(), key=lambda x: -x[1])[:5])}")
        if r['recent_high_risk']:
            print(f"  Recent critical/high:")
            for a in r['recent_high_risk'][-5:]:
                print(f"    [{a['ts'][:19]}] {a['tool']}: {a['details'][:80]}")
        print(f"{'='*50}\n")
