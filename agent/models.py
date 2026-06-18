# -*- coding: utf-8 -*-
"""Agent 核心数据模型"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


@dataclass
class ToolCall:
    """工具调用"""
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class Message:
    """对话消息"""
    role: str  # system / user / assistant / tool
    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None  # role=tool 时对应
    tool_name: str | None = None     # role=tool 时工具名


@dataclass
class ToolResult:
    """工具执行结果"""
    tool_call_id: str
    content: str
    is_error: bool = False


class AgentState(Enum):
    """Agent 状态"""
    IDLE = "idle"
    THINKING = "thinking"
    ACTING = "acting"
    OBSERVING = "observing"
    FINISHED = "finished"
    ERROR = "error"


@dataclass
class StepResult:
    """单步执行结果"""
    state: AgentState
    thought: str | None = None
    action: str | None = None
    observation: str | None = None
