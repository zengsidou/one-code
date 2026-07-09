# -*- coding: utf-8 -*-
"""OpenAI API 适配器 — GPT-4o / GPT-4.1 等模型，原生 Function Calling"""
import json
import os
from typing import Any

import httpx

from agent.models import Message, ToolCall
from .base import BaseLLM

OPENAI_BASE = "https://api.openai.com/v1"


def _convert_tools(tools: list[dict] | None) -> list[dict] | None:
    if not tools:
        return None
    result = []
    for t in tools:
        fn = t.get("function", t)
        result.append({"type": "function", "function": {
            "name": fn.get("name", ""),
            "description": fn.get("description", "")[:500],
            "parameters": fn.get("parameters", fn.get("input_schema", {})),
        }})
    return result


class OpenAIAdapter(BaseLLM):
    """OpenAI GPT 适配器"""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o",
        base_url: str | None = None,
        timeout: float = 60.0,
        max_retries: int = 1,
    ):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not self.api_key:
            raise ValueError("OpenAI API key 未设置。请设置环境变量 OPENAI_API_KEY")
        self.model = model
        self.base = base_url or OPENAI_BASE
        self.timeout = timeout
        self.max_retries = max_retries

    def generate(self, messages: list[Message], tools: list[dict] | None = None) -> Message:
        api_messages = self._build_messages(messages)

        for attempt in range(self.max_retries + 1):
            try:
                body: dict[str, Any] = {
                    "model": self.model,
                    "messages": api_messages,
                    "temperature": 0.3,
                    "max_tokens": 4096,
                }
                converted = _convert_tools(tools)
                if converted:
                    body["tools"] = converted
                    body["tool_choice"] = "auto"

                resp = httpx.post(
                    f"{self.base}/chat/completions",
                    json=body,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    timeout=self.timeout,
                )
                if resp.status_code == 400 and "tool" in resp.text.lower():
                    retry_body = body.copy()
                    retry_body.pop("tools", None)
                    retry_body.pop("tool_choice", None)
                    resp = httpx.post(
                        f"{self.base}/chat/completions",
                        json=retry_body,
                        headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                        timeout=self.timeout,
                    )

                data = resp.json()
                if "error" in data:
                    raise RuntimeError(f"OpenAI error: {data['error']}")

                choice = data["choices"][0]
                content = choice["message"].get("content", "")
                tool_calls_raw = choice["message"].get("tool_calls", [])

                tool_calls = []
                if tool_calls_raw:
                    for tc in tool_calls_raw:
                        fn = tc.get("function", {})
                        args_str = fn.get("arguments", "{}")
                        try:
                            args = json.loads(args_str) if isinstance(args_str, str) else args_str
                        except json.JSONDecodeError:
                            args = {}
                        tool_calls.append(ToolCall(id=tc.get("id", ""), name=fn.get("name", ""), arguments=args))

                return Message(role="assistant", content=content, tool_calls=tool_calls or None)

            except Exception as e:
                if attempt == self.max_retries:
                    return Message(role="assistant", content=f"[LLM error] OpenAI: {e}")
                import time; time.sleep(1)

        return Message(role="assistant", content="[LLM error] max retries exceeded")

    def embed(self, text: str) -> list[float]:
        try:
            resp = httpx.post(
                f"{self.base}/embeddings",
                json={"model": "text-embedding-3-small", "input": text},
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                timeout=30,
            )
            data = resp.json()
            return data["data"][0]["embedding"]
        except Exception:
            return [0.0] * 1024

    @staticmethod
    def _build_messages(messages: list[Message]) -> list[dict]:
        result = []
        for m in messages:
            item: dict[str, Any] = {"role": m.role}
            if m.content:
                item["content"] = m.content
            if m.tool_calls:
                item["tool_calls"] = [{
                    "id": tc.id, "type": "function",
                    "function": {"name": tc.name, "arguments": json.dumps(tc.arguments, ensure_ascii=False)},
                } for tc in m.tool_calls]
            if m.tool_call_id:
                item["tool_call_id"] = m.tool_call_id
            result.append(item)
        return result
