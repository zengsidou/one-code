# -*- coding: utf-8 -*-
"""MCP subprocess end-to-end test"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp.client import MCPClient

print("=== MCP Subprocess E2E Test ===\n")

SERVER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
client = MCPClient(["python", os.path.join(SERVER_DIR, "main_mcp.py")])

result = client.initialize()
print(f"Initialize: protocol={result.get('result',{}).get('protocolVersion','?')}")
print(f"Server: {result.get('result',{}).get('serverInfo',{})}")

tools = client.list_tools()
print(f"\nTools ({len(tools)}):")
for t in tools:
    print(f"  - {t['name']}: {t['description'][:50]}")

print("\nCall read_file:")
text = client.call_tool("read_file", {"path": os.path.join(SERVER_DIR, "main.py")})
print(f"  Result: {text[:100]}...")

print("\nCall calculate:")
text = client.call_tool("calculate", {"expression": "3**10"})
print(f"  Result: {text}")

print("\nCall safe_mode test:")
text = client.call_tool("run_shell", {"command": "rm -rf /"})
print(f"  Result: {text[:80]}")

client.shutdown()
print("\n=== E2E test passed ===")
