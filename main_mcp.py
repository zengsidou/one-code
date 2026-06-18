# -*- coding: utf-8 -*-
"""
Micro-Agent MCP Server
启动方式: python main_mcp.py
作为 MCP stdio 服务器运行，暴露工具给 MCP 客户端
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tools.registry import ToolRegistry
from tools.builtin import register_builtin_tools
from mcp.server import MCPServer
from sandbox import SandboxPolicy, SafeExecutor


def main():
    registry = ToolRegistry(safe_mode=True)
    sandbox = SafeExecutor(policy=SandboxPolicy())
    register_builtin_tools(registry, sandbox=sandbox)

    server = MCPServer(
        registry=registry,
        name="micro-agent-mcp",
        version="0.1.0",
    )

    print(f"[MCP Server] micro-agent-mcp v0.1.0 — {len(registry.tool_names)} tools", file=sys.stderr)
    print(f"[MCP Server] Tools: {', '.join(registry.tool_names)}", file=sys.stderr)
    print(f"[MCP Server] Listening on stdio...", file=sys.stderr)

    server.run_stdio()


if __name__ == "__main__":
    main()
