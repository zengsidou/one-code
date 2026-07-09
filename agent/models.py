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
    reasoning_content: str | None = None  # DeepSeek V4 Pro thinking mode


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


@dataclass
class Contract:
    """多模态契约 — Agent 执行前的产物预览"""
    type: str           # visual / dialog / code_api / config / data / narrative
    format: str         # text / svg / mermaid / json / table / ascii
    content: str        # 契约内容
    summary: str        # 一句话方向总结


@dataclass
class ContractStep:
    """逆向拆解后的执行步骤"""
    index: int
    goal: str
    tools_hint: str
    depends_on: list[int] = field(default_factory=list)
    contract_checkpoint: str = ""  # 完成后对照契约哪部分验证


@dataclass
class ContractResult:
    """契约先行执行的完整结果"""
    contract: Contract | None = None
    steps: list[ContractStep] = field(default_factory=list)
    step_results: list[dict] = field(default_factory=list)
    consistency_score: int = 0  # 0-5 分，最终产物与契约的一致性
    user_confirmed: bool = False
