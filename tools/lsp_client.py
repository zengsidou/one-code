# -*- coding: utf-8 -*-
"""LSP 客户端 — 对 python-lsp-server 的轻量 JSON-RPC 封装

提供 go_to_definition, find_references, hover, diagnostics 四种操作。
通过 subprocess stdio 与 pylsp 通信。
"""
import json
import os
import subprocess
import threading
from pathlib import Path

HEADER_PATTERN = "Content-Length: "


class LSPClient:
    """轻量 LSP 客户端，按需启动 pylsp"""

    def __init__(self, root_uri: str | None = None):
        self.root_uri = root_uri or Path.cwd().as_uri()
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self._msg_id = 0
        self._initialized = False

    def _ensure_started(self):
        with self._lock:
            if self._proc is not None:
                return
            cmd = self._find_pylsp()
            self._proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, cwd=os.getcwd(),
            )
            self._send("initialize", {
                "processId": os.getpid(),
                "rootUri": self.root_uri,
                "capabilities": {
                    "textDocument": {
                        "hover": {"contentFormat": ["plaintext"]},
                        "definition": {"linkSupport": False},
                        "references": {},
                        "publishDiagnostics": {"relatedInformation": True},
                    }
                },
            })
            self._read_response()
            self._send("initialized", {})
            self._initialized = True

    @staticmethod
    def _find_pylsp() -> list[str]:
        import shutil
        path = shutil.which("pylsp")
        if path:
            return [path]
        return [os.path.join(os.path.dirname(subprocess.__file__), "../../../Scripts/pylsp.exe")]

    def _send(self, method: str, params: dict):
        self._msg_id += 1
        body = json.dumps({
            "jsonrpc": "2.0", "id": self._msg_id,
            "method": method, "params": params,
        })
        header = f"Content-Length: {len(body)}\r\n\r\n"
        self._proc.stdin.write(header.encode() + body.encode())
        self._proc.stdin.flush()

    def _read_response(self, timeout: float = 5) -> dict | None:
        import select
        parts = []
        content_length = 0
        in_header = True
        deadline = __import__("time").time() + timeout
        while True:
            remaining = deadline - __import__("time").time()
            if remaining <= 0:
                return None
            fd = self._proc.stdout.fileno()
            r, _, _ = select.select([fd], [], [], min(remaining, 1))
            if not r:
                continue
            line = self._proc.stdout.readline().decode("utf-8", errors="replace")
            if in_header:
                if line.startswith(HEADER_PATTERN):
                    content_length = int(line[len(HEADER_PATTERN):].strip())
                elif line.strip() == "" and content_length > 0:
                    in_header = False
            elif content_length > 0:
                parts.append(line)
                content_length -= len(line)
                if content_length <= 0:
                    break
        raw = "".join(parts)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None

    def _call(self, method: str, params: dict, timeout: float = 5) -> dict | None:
        self._ensure_started()
        with self._lock:
            self._send(method, params)
            return self._read_response(timeout)

    def go_to_definition(self, file_path: str, line: int, col: int) -> list[dict]:
        """跳转到定义 — 返回 [(file, line, col), ...]"""
        uri = Path(file_path).resolve().as_uri()
        resp = self._call("textDocument/definition", {
            "textDocument": {"uri": uri},
            "position": {"line": line - 1, "character": max(0, col - 1)},
        })
        if not resp or "result" not in resp:
            return []
        results = resp["result"]
        if isinstance(results, dict):
            results = [results]
        defs = []
        for r in (results or []):
            f = r.get("uri", "")
            if f.startswith("file:///"):
                f = f[8:]
            elif f.startswith("file://"):
                f = f[7:]
            start = r.get("range", {}).get("start", {})
            defs.append({"file": f, "line": start.get("line", 0) + 1, "col": start.get("character", 0) + 1})
        return defs

    def find_references(self, file_path: str, line: int, col: int) -> list[dict]:
        """查找所有引用"""
        uri = Path(file_path).resolve().as_uri()
        resp = self._call("textDocument/references", {
            "textDocument": {"uri": uri},
            "position": {"line": line - 1, "character": max(0, col - 1)},
            "context": {"includeDeclaration": True},
        }, timeout=8)
        if not resp or "result" not in resp:
            return []
        refs = []
        for r in (resp["result"] or []):
            f = r.get("uri", "")
            if f.startswith("file:///"):
                f = f[8:]
            elif f.startswith("file://"):
                f = f[7:]
            start = r.get("range", {}).get("start", {})
            end = r.get("range", {}).get("end", {})
            refs.append({
                "file": f, "line": start.get("line", 0) + 1,
                "col": start.get("character", 0) + 1,
                "end_line": end.get("line", 0) + 1,
                "end_col": end.get("character", 0) + 1,
            })
        return refs

    def hover(self, file_path: str, line: int, col: int) -> str:
        """悬停信息 — 函数签名、类型提示、文档字符串"""
        uri = Path(file_path).resolve().as_uri()
        resp = self._call("textDocument/hover", {
            "textDocument": {"uri": uri},
            "position": {"line": line - 1, "character": max(0, col - 1)},
        })
        if not resp or "result" not in resp or resp["result"] is None:
            return "(无悬停信息)"
        result = resp["result"]
        contents = result.get("contents", {})
        if isinstance(contents, list):
            parts = []
            for c in contents:
                if isinstance(c, str):
                    parts.append(c)
                elif isinstance(c, dict):
                    parts.append(c.get("value", ""))
            return "\n---\n".join(p.strip() for p in parts if p.strip())
        if isinstance(contents, dict):
            val = contents.get("value", "")
            if isinstance(val, str):
                return val
        if isinstance(contents, str):
            return contents
        return str(contents)[:800]

    def diagnostics(self, file_path: str | None = None) -> list[dict]:
        """获取诊断信息（语法错误、类型警告等）"""
        if file_path:
            uri = Path(file_path).resolve().as_uri()
            self._call("textDocument/didOpen", {
                "textDocument": {"uri": uri, "languageId": "python", "version": 1, "text": ""}
            })
            resp = self._call("textDocument/diagnostic", {
                "textDocument": {"uri": uri}
            }, timeout=3)
            if resp and "result" in resp:
                items = resp["result"].get("items", [])
                return [{
                    "line": i.get("range", {}).get("start", {}).get("line", 0) + 1,
                    "col": i.get("range", {}).get("start", {}).get("character", 0) + 1,
                    "severity": i.get("severity", 3),
                    "message": i.get("message", "")[:200],
                } for i in items]
        return []

    def close(self):
        with self._lock:
            if self._proc:
                try:
                    self._send("shutdown", {})
                    self._proc.stdin.close()
                    self._proc.terminate()
                    self._proc.wait(timeout=3)
                except Exception:
                    self._proc.kill()
                self._proc = None


# 全局单例
_lsp: LSPClient | None = None


def get_lsp() -> LSPClient:
    global _lsp
    if _lsp is None:
        _lsp = LSPClient()
    return _lsp
