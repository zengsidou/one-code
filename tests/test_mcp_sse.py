# -*- coding: utf-8 -*-
"""MCP SSE 端到端测试"""
import sys, os, time, threading, json, queue
import httpx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.registry import ToolRegistry
from mcp.server import MCPServer
from mcp.sse_server import SSEMCPServer

PORT = 19531


def _send(msg: dict) -> dict | None:
    resp = httpx.post(f"http://127.0.0.1:{PORT}/message", json=msg, timeout=5)
    if resp.status_code == 202:
        return None
    return resp.json()


if __name__ == "__main__":
    print("Running MCP SSE tests...\n")

    registry = ToolRegistry(safe_mode=False)

    @registry.register("ping_tool", "Simple ping tool")
    def ping_tool(msg: str) -> str:
        return f"pong: {msg}"

    mcp_server = MCPServer(registry, name="test-sse", version="1.0")
    sse_server = SSEMCPServer(mcp_server, port=PORT)

    server_thread = threading.Thread(target=sse_server.start, daemon=True)
    server_thread.start()
    time.sleep(0.5)

    try:
        # Test 1: initialize
        init = _send({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
        assert init is not None
        assert init["result"]["serverInfo"]["name"] == "test-sse"
        print("  [PASS] sse_initialize")

        # Test 2: notify initialized + list tools
        _send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        tools_resp = _send({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        assert tools_resp is not None
        tools = tools_resp["result"]["tools"]
        assert len(tools) == 1
        assert tools[0]["name"] == "ping_tool"
        print("  [PASS] sse_list_tools")

        # Test 3: call tool
        call_resp = _send({
            "jsonrpc": "2.0", "id": 3, "method": "tools/call",
            "params": {"name": "ping_tool", "arguments": {"msg": "hello_sse"}},
        })
        assert call_resp is not None
        content = call_resp["result"]["content"]
        assert "pong: hello_sse" in content[0]["text"]
        print("  [PASS] sse_call_tool")

    finally:
        sse_server.stop()

    print("\nAll SSE tests passed!")
