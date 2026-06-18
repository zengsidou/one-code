# -*- coding: utf-8 -*-
"""MCP Client — 用于测试 MCP Server 的轻量客户端"""
import json
import subprocess
import sys
from typing import Any


class MCPClient:
    def __init__(self, server_command: list[str]):
        self._proc = subprocess.Popen(
            server_command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        self._msg_id = 0

    def _send(self, method: str, params: dict | None = None) -> dict[str, Any]:
        self._msg_id += 1
        request = {
            "jsonrpc": "2.0",
            "id": self._msg_id,
            "method": method,
            "params": params or {},
        }
        self._proc.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
        self._proc.stdin.flush()
        response_line = self._proc.stdout.readline().strip()
        return json.loads(response_line)

    def _notify(self, method: str, params: dict | None = None):
        notification = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params or {},
        }
        self._proc.stdin.write(json.dumps(notification, ensure_ascii=False) + "\n")
        self._proc.stdin.flush()

    def initialize(self) -> dict:
        resp = self._send("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "mcp-test-client", "version": "0.1.0"},
        })
        self._notify("notifications/initialized")
        return resp

    def list_tools(self) -> list[dict]:
        resp = self._send("tools/list")
        return resp.get("result", {}).get("tools", [])

    def call_tool(self, name: str, arguments: dict) -> str:
        resp = self._send("tools/call", {"name": name, "arguments": arguments})
        result = resp.get("result", {})
        if result:
            content = result.get("content", [])
            if content:
                return content[0].get("text", str(resp))
        error = resp.get("error", {})
        if error:
            return f"[ERROR {error.get('code','?')}] {error.get('message','')}"
        return str(resp)

    def shutdown(self):
        self._send("shutdown")
        self._proc.terminate()
        self._proc.wait(timeout=5)

    def close(self):
        try:
            self._proc.terminate()
        except Exception:
            pass
