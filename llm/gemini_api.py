# -*- coding: utf-8 -*-
"""Google Gemini API 适配器"""
import json
import os
from typing import Any

import httpx

from agent.models import Message, ToolCall
from .base import BaseLLM

GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"


def _role_to_gemini(role: str) -> str:
    if role == "assistant":
        return "model"
    if role == "tool":
        return "function"
    return "user"


class GeminiAdapter(BaseLLM):
    """Google Gemini 适配器 — gemini-2.5-flash / gemini-2.5-pro"""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gemini-2.5-flash",
        timeout: float = 60.0,
        max_retries: int = 1,
    ):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        if not self.api_key:
            raise ValueError("Gemini API key 未设置。请设置环境变量 GEMINI_API_KEY")
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries

    def generate(self, messages: list[Message], tools: list[dict] | None = None) -> Message:
        contents, system_instruction = self._build_contents(messages)
        if system_instruction:
            contents.insert(0, {"role": "user", "parts": [{"text": f"[System] {system_instruction}"}]})

        # Convert tools to Gemini format
        declarations = []
        if tools:
            for t in tools:
                fn = t.get("function", t)
                declarations.append({
                    "name": fn.get("name", ""),
                    "description": fn.get("description", "")[:500],
                    "parameters": fn.get("parameters", fn.get("input_schema", {})),
                })

        for attempt in range(self.max_retries + 1):
            try:
                body: dict[str, Any] = {
                    "contents": contents,
                    "generationConfig": {"temperature": 0.3, "maxOutputTokens": 4096},
                }
                if declarations:
                    body["tools"] = [{"functionDeclarations": declarations}]

                url = f"{GEMINI_BASE}/models/{self.model}:generateContent?key={self.api_key}"
                resp = httpx.post(url, json=body, timeout=self.timeout)
                data = resp.json()

                if "error" in data:
                    raise RuntimeError(f"Gemini error: {data['error']}")

                candidates = data.get("candidates", [])
                if not candidates:
                    return Message(role="assistant", content="[Gemini] 无回复")

                candidate = candidates[0]
                parts = candidate.get("content", {}).get("parts", [])

                content = ""
                tool_calls = []
                for part in parts:
                    if "text" in part:
                        content += part["text"]
                    if "functionCall" in part:
                        fc = part["functionCall"]
                        tool_calls.append(ToolCall(
                            id=fc.get("name", ""),
                            name=fc.get("name", ""),
                            arguments=fc.get("args", {}),
                        ))

                return Message(role="assistant", content=content or None, tool_calls=tool_calls or None)

            except Exception as e:
                if attempt == self.max_retries:
                    return Message(role="assistant", content=f"[LLM error] Gemini: {e}")
                import time; time.sleep(1)

        return Message(role="assistant", content="[LLM error] max retries exceeded")

    def embed(self, text: str) -> list[float]:
        try:
            url = f"{GEMINI_BASE}/models/text-embedding-004:embedContent?key={self.api_key}"
            resp = httpx.post(url, json={"model": "models/text-embedding-004", "content": {"parts": [{"text": text}]}}, timeout=30)
            data = resp.json()
            return data.get("embedding", {}).get("values", [0.0] * 768)
        except Exception:
            return [0.0] * 1024

    @staticmethod
    def _build_contents(messages: list[Message]) -> tuple[list[dict], str]:
        contents = []
        system_text = ""
        pending_tool_results: dict[str, list[str]] = {}

        for m in messages:
            if m.role == "system":
                system_text += (m.content or "") + "\n"
                continue

            role = _role_to_gemini(m.role)

            if m.role == "tool" and m.tool_call_id:
                pending_tool_results.setdefault(m.tool_call_id, []).append(m.content or "")
                continue

            parts = []
            if m.content:
                parts.append({"text": m.content})

            if m.tool_calls:
                for tc in m.tool_calls:
                    parts.append({
                        "functionCall": {"name": tc.name, "args": tc.arguments}
                    })
            elif m.role == "assistant" and pending_tool_results:
                # Add function responses
                for tid, results in pending_tool_results.items():
                    for r in results:
                        parts.append({
                            "functionResponse": {"name": tid, "response": {"result": r}}
                        })
                pending_tool_results.clear()

            if parts:
                contents.append({"role": role, "parts": parts})

        return contents, system_text.strip()
