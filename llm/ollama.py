# -*- coding: utf-8 -*-
"""Ollama 适配器 — 本地调用 + Function Calling 兼容 Fallback"""
import json
import re
import time
from typing import Any

import httpx

from agent.models import Message, ToolCall
from .base import BaseLLM

OLLAMA_BASE = "http://localhost:11434"


class OllamaClient(BaseLLM):
    def __init__(
        self,
        model: str = "deepseek-r1:8b",
        embedding_model: str = "bge-m3:latest",
        timeout: float = 60.0,
        max_retries: int = 1,
    ):
        self.model = model
        self.embedding_model = embedding_model
        self.timeout = timeout
        self.max_retries = max_retries

    def _token_estimate(self, text: str) -> int:
        total = 0
        for ch in text:
            if "\u4e00" <= ch <= "\u9fff" or "\u3000" <= ch <= "\u303f":
                total += 2  # 1 汉字 ~ 1.5-2 tokens
            elif ch.isascii() and ch.isalpha():
                total += 1
            else:
                total += 1
        return total

    def generate(self, messages: list[Message], tools: list[dict] | None = None) -> Message:
        api_messages = []
        for m in messages:
            entry: dict[str, Any] = {"role": m.role}
            if m.content is not None:
                entry["content"] = m.content
            if m.tool_calls:
                entry["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": json.dumps(tc.arguments, ensure_ascii=False)},
                    }
                    for tc in m.tool_calls
                ]
            if m.tool_call_id:
                entry["tool_call_id"] = m.tool_call_id
            if m.tool_name:
                entry["tool_name"] = m.tool_name
            api_messages.append(entry)

        if tools:
            return self._generate_with_tools(api_messages, tools)
        else:
            return self._generate_simple(api_messages)

    def _generate_simple(self, api_messages: list[dict]) -> Message:
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    resp = client.post(
                        f"{OLLAMA_BASE}/api/chat",
                        json={
                            "model": self.model,
                            "messages": api_messages,
                            "stream": False,
                            "options": {"temperature": 0.3},
                        },
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    content = data.get("message", {}).get("content", "")
                    return Message(role="assistant", content=content)
            except Exception as e:
                last_error = e
                if attempt < self.max_retries:
                    time.sleep(1)
        return Message(role="assistant", content=f"[LLM error: {last_error}]")

    def _generate_with_tools(self, api_messages: list[dict], tools: list[dict]) -> Message:
        # Try native function calling first
        for attempt in range(self.max_retries + 1):
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    resp = client.post(
                        f"{OLLAMA_BASE}/api/chat",
                        json={
                            "model": self.model,
                            "messages": api_messages,
                            "tools": tools,
                            "stream": False,
                            "options": {"temperature": 0.3},
                        },
                    )
                    if resp.status_code == 400:
                        # Model doesn't support tools — go directly to fallback
                        break
                    resp.raise_for_status()
                    data = resp.json()
                    msg = data.get("message", {})
                    content = msg.get("content", "")
                    raw_tool_calls = msg.get("tool_calls")

                    if raw_tool_calls:
                        tool_calls = [
                            ToolCall(
                                id=tc.get("id", f"call_{i}"),
                                name=tc.get("function", {}).get("name", ""),
                                arguments=self._safe_parse_json(tc.get("function", {}).get("arguments", "{}")),
                            )
                            for i, tc in enumerate(raw_tool_calls)
                        ]
                        return Message(role="assistant", content=content, tool_calls=tool_calls)

                    # Model returned text without tool_calls — try parsing
                    parsed = self._fallback_parse_tool_calls(content, tools)
                    if parsed:
                        return Message(role="assistant", content=content, tool_calls=parsed)
                    return Message(role="assistant", content=content)

            except httpx.HTTPStatusError:
                break  # 4xx/5xx that isn't 400 → go fallback
            except Exception:
                if attempt < self.max_retries:
                    time.sleep(1)
                    continue
                break

        # Prompt-based fallback for models without native function calling
        return self._prompt_based_tool_call(api_messages, tools)

    def _safe_parse_json(self, raw: str) -> dict:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            try:
                return json.loads(raw.replace("\\", "\\\\"))
            except json.JSONDecodeError:
                return {}

    def _fallback_parse_tool_calls(self, content: str, tools: list[dict]) -> list[ToolCall] | None:
        tool_names = {t["function"]["name"] for t in tools}
        patterns = [
            r"tool_call:\s*(\{.*?\})",
            r"```json\s*(\{.*?\})\s*```",
            r'{"tool"\s*:\s*".*?"\s*,\s*"arguments"\s*:\s*\{.*?\}\}',
        ]
        for pattern in patterns:
            for match in re.finditer(pattern, content, re.DOTALL | re.IGNORECASE):
                try:
                    obj = json.loads(match.group(1) if match.lastindex else match.group())
                    name = obj.get("tool") or obj.get("name")
                    if name and name in tool_names:
                        args = obj.get("arguments", {}) or obj.get("args", {})
                        return [ToolCall(id=f"call_{hash(name)}", name=name, arguments=args)]
                except json.JSONDecodeError:
                    continue
        return None

    def _prompt_based_tool_call(self, api_messages: list[dict], tools: list[dict]) -> Message:
        tool_desc = "\n".join(
            f"- {t['function']['name']}: {t['function']['description']}\n"
            f"  参数: {json.dumps(t['function']['parameters']['properties'], ensure_ascii=False)}"
            for t in tools
        )
        sys_msg = {
            "role": "system",
            "content": (
                "你是一个工具调用助手。根据对话需求，选择并调用合适的工具。\n\n"
                "可用工具:\n"
                f"{tool_desc}\n\n"
                "调用工具时，严格输出以下 JSON 格式（不要包含其他文字）:\n"
                '{"tool": "工具名", "arguments": {"参数名": "参数值"}}\n\n'
                "如果不需要调用工具，直接回复文本即可。"
            ),
        }
        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(
                    f"{OLLAMA_BASE}/api/chat",
                    json={
                        "model": self.model,
                        "messages": [sys_msg] + api_messages,
                        "stream": False,
                        "options": {"temperature": 0.1},
                    },
                )
                resp.raise_for_status()
                content = resp.json().get("message", {}).get("content", "")
                parsed = self._fallback_parse_tool_calls(content, tools)
                if parsed:
                    return Message(role="assistant", content=content, tool_calls=parsed)
                return Message(role="assistant", content=content)
        except Exception as e:
            return Message(role="assistant", content=f"[LLM error: {e}]")

    def embed(self, text: str) -> list[float]:
        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(
                    f"{OLLAMA_BASE}/api/embeddings",
                    json={"model": self.embedding_model, "prompt": text},
                )
                resp.raise_for_status()
                return resp.json()["embedding"]
        except Exception:
            return [0.0] * 1024
