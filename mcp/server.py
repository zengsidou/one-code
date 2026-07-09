# -*- coding: utf-8 -*-
"""MCP Server — 将 ToolRegistry 暴露为 MCP 协议服务"""
import json
import sys
from tools.registry import ToolRegistry
from .protocol import (
    Request, Response, Notification, ErrorCode,
    make_response, make_error, make_result, tool_schema_to_mcp,
    MCP_PROTOCOL_VERSION, JSONRPC_VERSION,
)


class MCPServer:
    def __init__(self, registry: ToolRegistry, name: str = "one-code", version: str = "0.1.0"):
        self.registry = registry
        self.name = name
        self.version = version
        self._initialized = False
        self._client_info: dict = {}

    def handle_message(self, raw: str) -> str | None:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return self._error(None, ErrorCode.PARSE_ERROR, "Invalid JSON")

        msg_id = data.get("id")
        method = data.get("method", "")

        if method == "initialize":
            return self._handle_initialize(msg_id, data.get("params", {}))
        elif method == "notifications/initialized":
            self._initialized = True
            return None  # No response for notifications
        elif not self._initialized:
            return self._error(msg_id, ErrorCode.SERVER_NOT_INITIALIZED, "Server not initialized")

        if method == "tools/list":
            return self._handle_tools_list(msg_id)
        elif method == "tools/call":
            return self._handle_tools_call(msg_id, data.get("params", {}))
        elif method == "ping":
            return make_response(msg_id, {})
        elif method == "shutdown":
            return self._handle_shutdown(msg_id)
        else:
            return self._error(msg_id, ErrorCode.METHOD_NOT_FOUND, f"Unknown method: {method}")

    def run_stdio(self):
        """Run as stdio MCP server"""
        while True:
            try:
                line = sys.stdin.readline()
                if not line:
                    break
                line = line.strip()
                if not line:
                    continue
                response = self.handle_message(line)
                if response:
                    sys.stdout.write(response + "\n")
                    sys.stdout.flush()
            except (EOFError, KeyboardInterrupt, BrokenPipeError):
                break
            except Exception as e:
                err = self._error(None, ErrorCode.INTERNAL_ERROR, str(e))
                sys.stdout.write(err + "\n")
                sys.stdout.flush()

    def _handle_initialize(self, msg_id, params: dict) -> str:
        self._client_info = params.get("clientInfo", {})
        return make_response(msg_id, {
            "protocolVersion": MCP_PROTOCOL_VERSION,
            "capabilities": {
                "tools": {},
            },
            "serverInfo": {
                "name": self.name,
                "version": self.version,
            },
        }).to_json()

    def _handle_tools_list(self, msg_id) -> str:
        schemas = self.registry.get_schemas()
        tools = [tool_schema_to_mcp(s) for s in schemas]
        return make_response(msg_id, {"tools": tools}).to_json()

    def _handle_tools_call(self, msg_id, params: dict) -> str:
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})
        if not tool_name:
            return self._error(msg_id, ErrorCode.INVALID_PARAMS, "Missing tool name")

        result = self.registry.execute(tool_name, arguments)
        is_error = result.startswith("[ERROR]")
        return make_result(msg_id, [{
            "type": "text",
            "text": result,
            "isError": is_error,
        }]).to_json()

    def _handle_shutdown(self, msg_id) -> str:
        return make_response(msg_id, {}).to_json()

    def _error(self, msg_id, code: int, message: str) -> str:
        return make_error(msg_id or 0, code, message).to_json()
