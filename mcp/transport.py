# -*- coding: utf-8 -*-
"""MCP Transport — stdio + SSE 传输层"""
import asyncio
import json
import sys
from abc import ABC, abstractmethod
from typing import Callable, Any


class Transport(ABC):
    @abstractmethod
    async def start(self, on_message: Callable[[str], Any]):
        ...

    @abstractmethod
    async def send(self, message: str):
        ...

    @abstractmethod
    async def close(self):
        ...


class StdioTransport(Transport):
    """stdio transport — 行分隔 JSON over stdin/stdout"""

    def __init__(self, stdin=None, stdout=None):
        self._stdin = stdin or sys.stdin
        self._stdout = stdout or sys.stdout
        self._buffer = ""

    async def start(self, on_message: Callable[[str], Any]):
        reader = asyncio.StreamReader()
        loop = asyncio.get_event_loop()

        if hasattr(self._stdin, "fileno"):
            transport_protocol = await loop.connect_read_pipe(
                lambda: asyncio.StreamReaderProtocol(reader),
                self._stdin,
            )

        while True:
            line = await reader.readline()
            if not line:
                break
            text = line.decode("utf-8").strip()
            if text:
                await on_message(text)

    async def send(self, message: str):
        self._stdout.write(message + "\n")
        self._stdout.flush()

    async def close(self):
        pass


class StdioSyncTransport:
    """同步 stdio transport — 用于非异步环境"""

    def __init__(self, stdin=None, stdout=None):
        self._stdin = stdin or sys.stdin
        self._stdout = stdout or sys.stdout

    def read_message(self) -> str | None:
        try:
            line = self._stdin.readline()
            if not line:
                return None
            return line.strip()
        except (EOFError, KeyboardInterrupt):
            return None

    def send(self, message: str):
        self._stdout.write(message + "\n")
        self._stdout.flush()

    def close(self):
        pass
