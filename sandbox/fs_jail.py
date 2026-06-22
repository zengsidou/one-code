# -*- coding: utf-8 -*-
"""文件系统 Jail — 路径验证与访问控制"""
import os
import re
import fnmatch

from .policy import SandboxPolicy


class FilesystemJail:
    def __init__(self, policy: SandboxPolicy):
        self.policy = policy
        self._allowed = [os.path.abspath(p) for p in policy.allowed_paths]
        self._blocked = [os.path.abspath(p.replace("~", os.path.expanduser("~"))) for p in policy.blocked_paths]

    def check_path(self, path: str, mode: str = "read") -> tuple[bool, str]:
        abs_path = os.path.abspath(path)

        for blocked in self._blocked:
            if fnmatch.fnmatch(abs_path, blocked):
                return False, f"Path blocked by policy: {blocked}"

        for pattern in self.policy.blocked_patterns:
            if re.search(pattern, os.path.basename(abs_path)):
                return False, f"Filename blocked by pattern: {pattern}"

        if self.policy.level.value != "permissive":
            is_allowed = any(
                abs_path.startswith(allowed + os.sep) or abs_path == allowed
                for allowed in self._allowed
            )
            if not is_allowed:
                return False, f"Path not in allowed list: {abs_path}"

        if mode == "write":
            parent = os.path.dirname(abs_path)
            if os.path.exists(parent) and not os.access(parent, os.W_OK):
                return False, f"Directory not writable: {parent}"

        return True, "ok"

    def restrict_command(self, command: str) -> str | None:
        """Audit command for path-based violations, return block reason or None"""
        for blocked_cmd in self.policy.blocked_commands:
            if re.search(blocked_cmd, command, re.IGNORECASE):
                return f"Command blocked by policy: {blocked_cmd}"
        return None
