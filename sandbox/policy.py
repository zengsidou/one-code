# -*- coding: utf-8 -*-
"""沙箱安全策略配置"""
from dataclasses import dataclass, field
from enum import Enum
import re


class PolicyLevel(Enum):
    STRICT = "strict"
    NORMAL = "normal"
    PERMISSIVE = "permissive"


@dataclass
class SandboxPolicy:
    level: PolicyLevel = PolicyLevel.NORMAL

    # Time & resource limits
    max_runtime_seconds: int = 30
    max_output_chars: int = 8000
    max_memory_mb: int = 512

    # Filesystem jail
    allowed_paths: list[str] = field(default_factory=lambda: [".", "./output", "./tmp", "/tmp"])
    blocked_paths: list[str] = field(default_factory=lambda: [
        "C:\\Windows\\System32", "/etc/passwd", "/etc/shadow",
        "C:\\Users\\*\\.ssh", "~/.ssh",
    ])
    blocked_patterns: list[str] = field(default_factory=lambda: [
        r"\.env$", r"\.pem$", r"id_rsa", r"credentials", r"password",
    ])

    # Command blocking
    blocked_commands: list[str] = field(default_factory=lambda: [
        r"\brm\s+(-[rf]|--recursive|--force)",
        r"\bdel\s+/[FSQ]",
        r"\bformat\s+[A-Z]:",
        r"\bmkfs\.",
        r"\bdd\s+if=",
        r">\s*/dev/",
        r"\b(sudo|su)\s",
        r"\bchmod\s+777",
        r"\b(shutdown|reboot|halt|poweroff)\b",
        r":\(\)\s*\{",
        r"Remove-Item\s+.*-Recurse\s+.*-Force",
        r"\bnc\s+-[lL]",       # netcat listener
        r"\bwget\s+.*\|\s*sh", # pipe to shell
        r"\bcurl\s+.*\|\s*sh",
    ])

    # Network
    allow_network: bool = False

    @classmethod
    def strict(cls):
        return cls(
            level=PolicyLevel.STRICT,
            max_runtime_seconds=10,
            max_output_chars=2000,
            max_memory_mb=128,
            allowed_paths=["./output", "/tmp"],
            allow_network=False,
        )

    @classmethod
    def permissive(cls):
        return cls(
            level=PolicyLevel.PERMISSIVE,
            max_runtime_seconds=60,
            max_output_chars=32000,
            max_memory_mb=1024,
            allow_network=True,
        )
