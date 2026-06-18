# -*- coding: utf-8 -*-
"""Debug MCP subprocess"""
import sys, os, json, subprocess, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

proc = subprocess.Popen(
    ["python", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "main_mcp.py")],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.DEVNULL,
    text=True,
    bufsize=1,
)

# Wait for server to start
time.sleep(0.5)

# Send initialize
request = json.dumps({
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "test", "version": "1.0"}},
}, ensure_ascii=False)
print(f"SEND: {request}")
proc.stdin.write(request + "\n")
proc.stdin.flush()

# Read response
time.sleep(0.5)
response = proc.stdout.readline()
print(f"RECV: {repr(response)}")

if response:
    data = json.loads(response)
    print(f"PARSED: {data.get('result', {}).get('serverInfo', {})}")

# notify initialized
proc.stdin.write('{"jsonrpc":"2.0","method":"notifications/initialized"}\n')
proc.stdin.flush()

# tools/list
proc.stdin.write('{"jsonrpc":"2.0","id":2,"method":"tools/list"}\n')
proc.stdin.flush()
time.sleep(0.3)
tools_resp = proc.stdout.readline()
print(f"TOOLS: {repr(tools_resp[:200])}")

# shutdown
proc.stdin.write('{"jsonrpc":"2.0","id":3,"method":"shutdown"}\n')
proc.stdin.flush()
proc.terminate()
print("DONE")
