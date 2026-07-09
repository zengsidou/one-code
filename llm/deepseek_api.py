# -*- coding: utf-8 -*-
"""DeepSeek API 适配器 — 调用 deepseek-reasoner (DeepSeek-R1)，支持原生 Function Calling"""
import json
import os
import time
from typing import Any

import httpx

from agent.models import Message, ToolCall
from .base import BaseLLM

DEEPSEEK_BASE = "https://api.deepseek.com"


class DeepSeekAdapter(BaseLLM):
    """DeepSeek API 适配器，继承 BaseLLM，接口与 OllamaClient 保持一致"""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "deepseek-v4-pro",
        timeout: float = 60.0,
        max_retries: int = 1,
    ):
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "DeepSeek API key 未设置。请设置环境变量 DEEPSEEK_API_KEY，"
                "或通过 api_key 参数传入。"
            )
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries

    def generate(self, messages: list[Message], tools: list[dict] | None = None) -> Message:
        """生成回复，支持原生 Function Calling"""
        api_messages = self._build_api_messages(messages)

        last_error = None
        last_error_body = ""
        for attempt in range(self.max_retries + 1):
            try:
                body: dict[str, Any] = {
                    "model": self.model,
                    "messages": api_messages,
                    "temperature": 0.3,
                    "max_tokens": 4096,
                    "stream": False,
                }
                if tools:
                    body["tools"] = tools
                    body["tool_choice"] = "auto"

                with httpx.Client(timeout=self.timeout) as client:
                    resp = client.post(
                        f"{DEEPSEEK_BASE}/chat/completions",
                        json=body,
                        headers={
                            "Authorization": f"Bearer {self.api_key}",
                            "Content-Type": "application/json",
                        },
                    )
                    resp.raise_for_status()
                    data = resp.json()

                choice = data["choices"][0]
                msg = choice.get("message", {})
                content = msg.get("content") or ""
                reasoning = msg.get("reasoning_content") or ""
                raw_tool_calls = msg.get("tool_calls")

                tool_calls = None
                if raw_tool_calls:
                    tool_calls = [
                        ToolCall(
                            id=tc.get("id", f"call_{i}"),
                            name=tc.get("function", {}).get("name", ""),
                            arguments=self._safe_parse_json(
                                tc.get("function", {}).get("arguments", "{}")
                            ),
                        )
                        for i, tc in enumerate(raw_tool_calls)
                    ]

                return Message(role="assistant", content=content, tool_calls=tool_calls, reasoning_content=reasoning)

            except httpx.HTTPStatusError as e:
                last_error = e
                try:
                    last_error_body = str(e.response.json().get("error", {}).get("message", ""))
                except Exception:
                    last_error_body = e.response.text[:300]
                # 400 = orphaned tool messages → 清理后重试
                if e.response.status_code == 400 and tools:
                    cleaned = self._clean_orphaned_tools(api_messages)
                    if len(cleaned) < len(api_messages):
                        api_messages = cleaned
                        if attempt < self.max_retries:
                            time.sleep(0.5)
                            continue
                if attempt < self.max_retries:
                    time.sleep(1)
            except Exception as e:
                last_error = e
                if attempt < self.max_retries:
                    time.sleep(1)

        err_msg = f"[LLM error: {last_error}]"
        if last_error_body:
            err_msg += f" (API: {last_error_body[:200]})"
        return Message(role="assistant", content=err_msg)

    def _build_api_messages(self, messages: list[Message]) -> list[dict]:
        """将内部 Message 列表转换为 DeepSeek API 格式"""
        api_messages = []
        for m in messages:
            entry: dict[str, Any] = {"role": m.role}
            if m.content is not None:
                entry["content"] = m.content
            if m.role == "assistant":
                entry["reasoning_content"] = m.reasoning_content or ""
            if m.tool_calls:
                entry["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                        },
                    }
                    for tc in m.tool_calls
                ]
            if m.tool_call_id:
                entry["tool_call_id"] = m.tool_call_id
            if m.tool_name:
                entry["tool_name"] = m.tool_name
            api_messages.append(entry)
        return api_messages

    @staticmethod
    def _clean_orphaned_tools(messages: list[dict]) -> list[dict]:
        """移除孤立的 tool 消息（前面没有 assistant[tool_calls]）"""
        cleaned = []
        for m in messages:
            role = m.get("role", "")
            if role == "tool":
                if cleaned and cleaned[-1].get("role") == "assistant" and cleaned[-1].get("tool_calls"):
                    cleaned.append(m)
            else:
                cleaned.append(m)
        return cleaned

    @staticmethod
    def _safe_parse_json(raw: str) -> dict:
        """安全解析 JSON，失败返回空 dict"""
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}

    def embed(self, text: str) -> list[float]:
        """DeepSeek 不提供原生 embedding API，返回零向量占位。LongTermMemory 会自动降级到 BM25 关键词检索。"""
        return [0.0] * 1024
