import sys, threading, time
import httpx
sys.path.insert(0, "D:/micro-agent")
from tools.registry import ToolRegistry
from mcp.server import MCPServer
from mcp.sse_server import SSEMCPServer

r = ToolRegistry(safe_mode=False)
@r.register("echo", "echo tool")
def echo(text: str) -> str:
    return text

s = MCPServer(r, name="test", version="1.0")
srv = SSEMCPServer(s, port=19529)
t = threading.Thread(target=srv.start, daemon=True)
t.start()
time.sleep(0.5)

try:
    # Test POST /message for initialize (should return direct response without SSE)
    resp = httpx.post(
        "http://127.0.0.1:19529/message",
        json={"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
        timeout=5
    )
    print("Init response:", resp.status_code)
    print("Body:", resp.text[:200])

    # Test tools/list
    resp2 = httpx.post(
        "http://127.0.0.1:19529/message",
        json={"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        timeout=5
    )
    print("\nTools response:", resp2.status_code)
    print("Body:", resp2.text[:200])

    print("\nSSE basic test passed!")
except Exception as e:
    print("Error:", type(e).__name__, e)
finally:
    srv.stop()
