# -*- coding: utf-8 -*-
"""SafeExecutor — 沙箱化进程执行"""
import os
import platform
import subprocess
import time
import psutil

from .policy import SandboxPolicy, PolicyLevel
from .fs_jail import FilesystemJail


class SafeExecutor:
    def __init__(self, policy: SandboxPolicy | None = None):
        self.policy = policy or SandboxPolicy()
        self.jail = FilesystemJail(self.policy)

    def execute(self, command: str, cwd: str = ".") -> dict:
        """Execute a command in sandbox, return {ok, output, error, duration_ms, blocked_by}"""
        result = {
            "ok": True, "output": "", "error": "", "duration_ms": 0, "blocked_by": "",
        }

        # 优先用 tree-sitter AST 安全检测
        try:
            from tools.shell_safety import check_dangerous
            check = check_dangerous(command)
            if check.get("blocked"):
                result["ok"] = False
                result["blocked_by"] = check.get("reason", "dangerous command")
                result["error"] = f"[BLOCKED] {result['blocked_by']}"
                return result
        except ImportError:
            pass

        block_reason = self.jail.restrict_command(command)
        if block_reason:
            result["ok"] = False
            result["blocked_by"] = block_reason
            result["error"] = f"[BLOCKED] {block_reason}"
            return result

        if not self.policy.allow_network:
            has_network = any(
                kw in command.lower()
                for kw in ["curl ", "wget ", "nc ", "ncat ", "ping ", "traceroute", "ssh ", "scp ", "ftp "]
            )
            if has_network:
                result["ok"] = False
                result["blocked_by"] = "Network access disabled by policy"
                result["error"] = f"[BLOCKED] Network access disabled"
                return result

        if not self.policy.allow_network:
            has_network = any(
                kw in command.lower()
                for kw in ["curl ", "wget ", "nc ", "ncat ", "ping ", "traceroute", "ssh ", "scp ", "ftp "]
            )
            if has_network:
                result["ok"] = False
                result["blocked_by"] = "Network access disabled by policy"
                result["error"] = f"[BLOCKED] Network access disabled"
                return result

        shell_cmd = self._build_command(command)

        start = time.time()
        try:
            proc = subprocess.Popen(
                shell_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=cwd,
            )

            try:
                stdout, stderr = proc.communicate(timeout=self.policy.max_runtime_seconds)
            except subprocess.TimeoutExpired:
                self._kill_proc_tree(proc.pid)
                proc.kill()
                proc.communicate()
                result["ok"] = False
                result["error"] = f"[TIMEOUT] Command exceeded {self.policy.max_runtime_seconds}s limit"
                result["duration_ms"] = int((time.time() - start) * 1000)
                return result

            elapsed = int((time.time() - start) * 1000)
            result["duration_ms"] = elapsed

            output = stdout.strip()
            err = stderr.strip()

            if output and len(output) > self.policy.max_output_chars:
                output = output[:self.policy.max_output_chars] + f"\n\n... (truncated at {self.policy.max_output_chars} chars)"

            if proc.returncode != 0:
                result["ok"] = False

            result["output"] = output
            result["error"] = err

        except Exception as e:
            result["ok"] = False
            result["error"] = f"[ERROR] {e}"
            result["duration_ms"] = int((time.time() - start) * 1000)

        return result

    def _build_command(self, command: str) -> list[str]:
        if platform.system() == "Windows":
            return ["powershell", "-NoProfile", "-NonInteractive", "-Command", command]
        else:
            return ["bash", "-c", command]

    def _kill_proc_tree(self, pid: int):
        try:
            parent = psutil.Process(pid)
            children = parent.children(recursive=True)
            for child in children:
                try:
                    child.terminate()
                except psutil.NoSuchProcess:
                    pass
            gone, alive = psutil.wait_procs(children, timeout=3)
            for p in alive:
                try:
                    p.kill()
                except psutil.NoSuchProcess:
                    pass
            try:
                parent.terminate()
            except psutil.NoSuchProcess:
                pass
        except (psutil.NoSuchProcess, Exception):
            pass
