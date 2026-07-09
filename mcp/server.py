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
            return None
        elif not self._initialized:
            return self._error(msg_id, ErrorCode.SERVER_NOT_INITIALIZED, "Server not initialized")

        if method == "tools/list":
            return self._handle_tools_list(msg_id)
        elif method == "tools/call":
            return self._handle_tools_call(msg_id, data.get("params", {}))
        elif method == "resources/list":
            return self._handle_resources_list(msg_id)
        elif method == "resources/read":
            return self._handle_resources_read(msg_id, data.get("params", {}))
        elif method == "prompts/list":
            return self._handle_prompts_list(msg_id)
        elif method == "prompts/get":
            return self._handle_prompts_get(msg_id, data.get("params", {}))
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
                "resources": {"listChanged": False},
                "prompts": {},
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
        }])

    def _handle_resources_list(self, msg_id) -> str:
        """列出可用资源 — 工作区文件"""
        import os
        resources = []
        try:
            for f in os.listdir("."):
                if os.path.isfile(f) and not f.startswith("."):
                    resources.append({
                        "uri": f"file:///{os.path.abspath(f)}",
                        "name": f,
                        "mimeType": "text/plain",
                    })
        except Exception:
            pass
        return make_result(msg_id, resources if resources else [])

    def _handle_resources_read(self, msg_id, params: dict) -> str:
        """读取资源内容"""
        uri = params.get("uri", "")
        path = uri.replace("file:///", "").replace("file://", "")
        if not path or not os.path.exists(path):
            return self._error(msg_id, ErrorCode.INVALID_PARAMS, f"Resource not found: {uri}")
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()[:5000]
            return make_result(msg_id, [{"type": "text", "text": content}])
        except Exception as e:
            return self._error(msg_id, ErrorCode.INTERNAL_ERROR, str(e))

    def _handle_prompts_list(self, msg_id) -> str:
        """列出可用 prompt 模板"""
        prompts = [
            {"name": "code-review", "description": "代码审查 prompt"},
            {"name": "write-tests", "description": "为指定代码生成测试"},
            {"name": "refactor", "description": "重构指定代码"},
            {"name": "explain-code", "description": "解释代码逻辑"},
        ]
        return make_result(msg_id, prompts)

    def _handle_prompts_get(self, msg_id, params: dict) -> str:
        """获取 prompt 模板内容"""
        name = params.get("name", "")
        prompts = {
            "code-review": "请审查以下代码的质量、安全性和可维护性，给出具体改进建议。",
            "write-tests": "请为以下代码编写全面的单元测试，覆盖边界情况和错误路径。",
            "refactor": "请重构以下代码，提高可读性和可维护性，不改变外部行为。",
            "explain-code": "请用中文详细解释这段代码的逻辑、数据流和关键设计决策。",
        }
        text = prompts.get(name, f"未知 prompt: {name}")
        return make_result(msg_id, [{"type": "text", "text": text}]).to_json()

    def _handle_shutdown(self, msg_id) -> str:
        return make_response(msg_id, {}).to_json()

    def _error(self, msg_id, code: int, message: str) -> str:
        return make_error(msg_id or 0, code, message).to_json()
