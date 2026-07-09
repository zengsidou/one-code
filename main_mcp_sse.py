# -*- coding: utf-8 -*-
"""
One-Code MCP SSE Server
启动方式: python main_mcp_sse.py
作为 MCP SSE 服务器运行，通过 HTTP SSE 暴露工具
"""
import sys
import os
import signal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tools.registry import ToolRegistry
from tools.builtin import register_builtin_tools
from mcp.server import MCPServer
from mcp.sse_server import SSEMCPServer
from sandbox import SandboxPolicy, SafeExecutor


def main():
    registry = ToolRegistry(safe_mode=True)
    sandbox = SafeExecutor(policy=SandboxPolicy())
    register_builtin_tools(registry, sandbox=sandbox)

    mcp_server = MCPServer(
        registry=registry,
        name="one-code-mcp",
        version="0.1.0",
    )

    sse = SSEMCPServer(mcp_server, host="127.0.0.1", port=9527)
    print(f"[MCP SSE] {len(registry.tool_names)} tools registered", file=sys.stderr)
    print(f"[MCP SSE] SSE endpoint: http://127.0.0.1:9527/sse", file=sys.stderr)
    print(f"[MCP SSE] Message endpoint: http://127.0.0.1:9527/message", file=sys.stderr)
    sse.start()


if __name__ == "__main__":
    main()
