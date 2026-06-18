# -*- coding: utf-8 -*-
"""MCP Protocol — JSON-RPC 2.0 类型与消息解析"""
from dataclasses import dataclass, field
from typing import Any
import json


JSONRPC_VERSION = "2.0"
MCP_PROTOCOL_VERSION = "2024-11-05"

# MCP error codes (JSON-RPC -32000 to -32099 reserved)
class ErrorCode:
    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603
    SERVER_NOT_INITIALIZED = -32002


@dataclass
class Request:
    jsonrpc: str = JSONRPC_VERSION
    id: int | str = 0
    method: str = ""
    params: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def from_json(data: dict) -> "Request":
        return Request(
            jsonrpc=data.get("jsonrpc", JSONRPC_VERSION),
            id=data.get("id", 0),
            method=data.get("method", ""),
            params=data.get("params", {}),
        )


@dataclass
class Response:
    jsonrpc: str = JSONRPC_VERSION
    id: int | str = 0
    result: Any = None
    error: dict[str, Any] | None = None

    def to_json(self) -> str:
        payload = {"jsonrpc": self.jsonrpc, "id": self.id}
        if self.error:
            payload["error"] = self.error
        else:
            payload["result"] = self.result
        return json.dumps(payload, ensure_ascii=False)


@dataclass
class Notification:
    jsonrpc: str = JSONRPC_VERSION
    method: str = ""
    params: dict[str, Any] = field(default_factory=dict)


def parse_message(line: str) -> Request | Notification | None:
    try:
        data = json.loads(line)
    except json.JSONDecodeError:
        return None
    if "method" in data and "id" in data:
        return Request.from_json(data)
    if "method" in data:
        return Notification(
            jsonrpc=data.get("jsonrpc", JSONRPC_VERSION),
            method=data["method"],
            params=data.get("params", {}),
        )
    return None


def make_response(msg_id: int | str, result: Any) -> Response:
    return Response(id=msg_id, result=result)


def make_error(msg_id: int | str, code: int, message: str) -> Response:
    return Response(id=msg_id, error={"code": code, "message": message})


def make_result(msg_id: int | str, content: list[dict]) -> Response:
    return Response(id=msg_id, result={"content": content})


def tool_schema_to_mcp(schema: dict) -> dict:
    """Convert OpenAI function-calling schema to MCP Tool schema"""
    func = schema.get("function", {})
    return {
        "name": func.get("name", ""),
        "description": func.get("description", ""),
        "inputSchema": func.get("parameters", {"type": "object", "properties": {}, "required": []}),
    }
