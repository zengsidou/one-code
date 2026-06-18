# -*- coding: utf-8 -*-
"""MCP SSE Server — HTTP SSE 传输的 MCP 服务端点"""
import json
import queue
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler

from mcp.server import MCPServer


class SSESessionManager:
    def __init__(self):
        self._queues: dict[str, queue.Queue] = {}
        self._lock = threading.Lock()

    def create(self, session_id: str) -> queue.Queue:
        q = queue.Queue()
        with self._lock:
            self._queues[session_id] = q
        return q

    def get(self, session_id: str) -> queue.Queue | None:
        with self._lock:
            return self._queues.get(session_id)

    def remove(self, session_id: str):
        with self._lock:
            self._queues.pop(session_id, None)


class SSEHandler(BaseHTTPRequestHandler):
    mcp_server: MCPServer = None
    sessions: SSESessionManager = None

    def log_message(self, format, *args):
        pass  # Suppress access logging

    def do_GET(self):
        if self.path == "/sse":
            self._handle_sse()
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path == "/message":
            self._handle_message()
        else:
            self.send_error(404)

    def _handle_sse(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        session_id = str(threading.get_ident())
        q = self.sessions.create(session_id)

        # Send endpoint event with the message POST URL
        endpoint = f"http://{self.headers.get('Host', 'localhost')}/message"
        event = f"event: endpoint\ndata: {endpoint}\n\n"
        self.wfile.write(event.encode())
        self.wfile.flush()

        try:
            while True:
                try:
                    msg = q.get(timeout=30)
                    event = f"event: message\ndata: {msg}\n\n"
                    self.wfile.write(event.encode())
                    self.wfile.flush()
                except queue.Empty:
                    # Send keepalive comment
                    self.wfile.write(": ping\n\n".encode())
                    self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            self.sessions.remove(session_id)

    def _handle_message(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode("utf-8")

        response = self.mcp_server.handle_message(body)
        if response is None:
            self.send_response(202)
            self.end_headers()
            return

        # Send response via SSE to the session
        session_id = self._get_session_from_headers()
        if session_id:
            q = self.sessions.get(session_id)
            if q:
                q.put_nowait(response)
                self.send_response(202)
                self.end_headers()
                return

        # Fallback: return response directly in HTTP body
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(response.encode())

    def _get_session_from_headers(self) -> str | None:
        session_header = self.headers.get("X-MCP-Session", "")
        if session_header:
            return session_header
        return None

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, X-MCP-Session")
        self.end_headers()


class SSEMCPServer:
    def __init__(self, mcp_server: MCPServer, host: str = "127.0.0.1", port: int = 9527):
        self._mcp_server = mcp_server
        self._host = host
        self._port = port
        self._sessions = SSESessionManager()

    def start(self):
        SSEHandler.mcp_server = self._mcp_server
        SSEHandler.sessions = self._sessions
        self._httpd = HTTPServer((self._host, self._port), SSEHandler)
        print(f"[MCP SSE Server] Listening on http://{self._host}:{self._port}/sse")
        try:
            self._httpd.serve_forever()
        except KeyboardInterrupt:
            pass

    def stop(self):
        if hasattr(self, "_httpd"):
            self._httpd.shutdown()
