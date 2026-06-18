# -*- coding: utf-8 -*-
"""MCP 集成测试 — Client ↔ Server 端到端"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp.client import MCPClient
from mcp.server import MCPServer
from mcp.protocol import parse_message, tool_schema_to_mcp
from tools.registry import ToolRegistry
from tools.schema import generate_tool_schema
from tools.builtin import register_builtin_tools


def test_mcp_initialize():
    """直接注入测试 — 不走 subprocess"""
    registry = ToolRegistry(safe_mode=False)

    @registry.register("echo", "Echo back")
    def echo(text: str) -> str:
        return text

    server = MCPServer(registry, name="test-server", version="1.0.0")

    init_req = '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"1.0"}}}'
    resp = server.handle_message(init_req)
    assert resp is not None
    import json
    data = json.loads(resp)
    assert data["result"]["protocolVersion"] == "2024-11-05"
    assert data["result"]["serverInfo"]["name"] == "test-server"
    print("  [PASS] test_mcp_initialize")

    notified = '{"jsonrpc":"2.0","method":"notifications/initialized"}'
    resp2 = server.handle_message(notified)
    assert resp2 is None
    print("  [PASS] test_mcp_initialized_notification")


def test_mcp_tools_list():
    registry = ToolRegistry(safe_mode=False)

    @registry.register("echo", "Echo back")
    def echo(text: str) -> str:
        return text

    server = MCPServer(registry)
    server.handle_message('{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}')
    server.handle_message('{"jsonrpc":"2.0","method":"notifications/initialized"}')

    resp = server.handle_message('{"jsonrpc":"2.0","id":2,"method":"tools/list"}')
    import json
    data = json.loads(resp)
    tools = data["result"]["tools"]
    assert len(tools) == 1
    assert tools[0]["name"] == "echo"
    print("  [PASS] test_mcp_tools_list")


def test_mcp_tools_call():
    registry = ToolRegistry(safe_mode=False)

    @registry.register("calculate", "Execute math expression")
    def calculate(expression: str) -> str:
        import math
        allowed = {"__builtins__": {}, **{k: getattr(math, k) for k in dir(math) if not k.startswith("_")}}
        return str(eval(expression, allowed))

    server = MCPServer(registry)
    server.handle_message('{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}')
    server.handle_message('{"jsonrpc":"2.0","method":"notifications/initialized"}')

    resp = server.handle_message(
        '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"calculate","arguments":{"expression":"2+2"}}}'
    )
    import json
    data = json.loads(resp)
    content = data["result"]["content"]
    assert len(content) == 1
    assert "4" in content[0]["text"]
    print("  [PASS] test_mcp_tools_call")


def test_mcp_schema_conversion():
    from tools.schema import generate_tool_schema
    schema = generate_tool_schema("search", "Search files", {
        "query": (str, "Search query"),
    })
    mcp_tool = tool_schema_to_mcp(schema)
    assert mcp_tool["name"] == "search"
    assert mcp_tool["description"] == "Search files"
    assert "inputSchema" in mcp_tool
    assert mcp_tool["inputSchema"]["type"] == "object"
    print("  [PASS] test_mcp_schema_conversion")


def test_mcp_errors():
    registry = ToolRegistry(safe_mode=False)
    server = MCPServer(registry)
    server.handle_message('{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}')
    server.handle_message('{"jsonrpc":"2.0","method":"notifications/initialized"}')

    # Unknown method
    resp = server.handle_message('{"jsonrpc":"2.0","id":9,"method":"unknown"}')
    import json
    data = json.loads(resp)
    assert data["error"]["code"] == -32601
    print("  [PASS] test_mcp_error_method_not_found")

    # Unknown tool
    resp = server.handle_message(
        '{"jsonrpc":"2.0","id":10,"method":"tools/call","params":{"name":"nonexistent","arguments":{}}}'
    )
    data = json.loads(resp)
    assert "[ERROR]" in data["result"]["content"][0]["text"]
    print("  [PASS] test_mcp_error_unknown_tool")


if __name__ == "__main__":
    print("Running MCP integration tests...\n")
    test_mcp_initialize()
    test_mcp_tools_list()
    test_mcp_tools_call()
    test_mcp_schema_conversion()
    test_mcp_errors()
    print("\nAll MCP tests passed!")
