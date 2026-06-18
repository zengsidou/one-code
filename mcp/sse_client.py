# -*- coding: utf-8 -*-
"""MCP SSE Client — 通过 SSE 协议连接 MCP Server"""
import json
import threading
import time
import httpx


class SSEMCPClient:
    def __init__(self, sse_url: str = "http://127.0.0.1:9527/sse", message_url: str = "http://127.0.0.1:9527/message"):
        self.sse_url = sse_url
        self.message_url = message_url
        self._msg_id = 0
        self._responses: dict[int, dict] = {}
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None
        self._session_id: str | None = None

    def connect(self, timeout: float = 5):
        self._running = True
        self._thread = threading.Thread(target=self._listen_sse, daemon=True)
        self._thread.start()
        time.sleep(0.3)

    def _listen_sse(self):
        try:
            with httpx.Client(timeout=30) as client:
                with client.stream("GET", self.sse_url) as response:
                    event_type = ""
                    for line in response.iter_lines():
                        if not self._running:
                            break
                        if not line:
                            event_type = ""
                            continue
                        if line.startswith("event: "):
                            event_type = line[7:].strip()
                        elif line.startswith("data: "):
                            data = line[6:].strip()
                            if event_type == "message":
                                try:
                                    msg = json.loads(data)
                                    msg_id = msg.get("id")
                                    if msg_id is not None:
                                        with self._lock:
                                            self._responses[msg_id] = msg
                                except json.JSONDecodeError:
                                    pass
                            elif event_type == "endpoint":
                                self.message_url = data
        except Exception:
            pass

    def _send(self, method: str, params: dict | None = None) -> dict:
        self._msg_id += 1
        msg_id = self._msg_id
        request = {
            "jsonrpc": "2.0",
            "id": msg_id,
            "method": method,
            "params": params or {},
        }
        headers = {"Content-Type": "application/json"}
        if self._session_id:
            headers["X-MCP-Session"] = self._session_id

        try:
            with httpx.Client(timeout=30) as client:
                resp = client.post(self.message_url, json=request, headers=headers)
                if resp.status_code == 200:
                    # Response returned directly
                    return resp.json()
                elif resp.status_code == 202:
                    # Response will come via SSE, wait for it
                    for _ in range(50):
                        with self._lock:
                            if msg_id in self._responses:
                                result = self._responses.pop(msg_id)
                                return result
                        time.sleep(0.1)
                    return {"id": msg_id, "error": {"code": -32000, "message": "SSE response timeout"}}
                else:
                    return {"id": msg_id, "error": {"code": -32000, "message": f"HTTP {resp.status_code}"}}
        except Exception as e:
            return {"id": msg_id, "error": {"code": -32000, "message": str(e)}}

    def _notify(self, method: str, params: dict | None = None):
        request = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params or {},
        }
        try:
            with httpx.Client(timeout=10) as client:
                client.post(self.message_url, json=request)
        except Exception:
            pass

    def initialize(self) -> dict:
        resp = self._send("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "mcp-sse-client", "version": "0.1.0"},
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
            return f"[ERROR {error.get('code', '?')}] {error.get('message', '')}"
        return str(resp)

    def disconnect(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
