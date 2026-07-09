# -*- coding: utf-8 -*-
"""基础功能测试 — LLM 适配器 + Token 优化"""
import sys
import os
import json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import patch, MagicMock
from agent.models import Message, ToolCall
from llm.deepseek_api import DeepSeekAdapter
from agent.token_optimizer import TokenOptimizer


def test_deepseek_adapter():
    """Test DeepSeek adapter instantiation"""
    with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "sk-test"}):
        adapter = DeepSeekAdapter(api_key="sk-test")
        assert adapter.model == "deepseek-v4-pro"
        assert adapter.api_key == "sk-test"


def test_token_optimizer_truncation():
    opt = TokenOptimizer(None)
    assert len(opt.compress_tool_output("short")) < 200
    err = "[ERROR] something went wrong" + "x" * 500
    result = opt.compress_tool_output(err)
    assert "[ERROR]" in result
    # 错误消息会被压缩
    assert len(result) < len(err)


def test_token_optimizer_list_trim():
    opt = TokenOptimizer(None)
    # 30+ lines with list-like patterns triggers compression
    listing = "\n".join(f"  - file_{i}.py" for i in range(50))
    result = opt.compress_tool_output(listing)
    assert len(result) < len(listing)


def test_deepseek_embed():
    with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "sk-test"}):
        adapter = DeepSeekAdapter(api_key="sk-test")
        emb = adapter.embed("test")
        assert len(emb) == 1024
        assert all(v == 0.0 for v in emb)


if __name__ == "__main__":
    test_deepseek_adapter()
    test_token_optimizer_truncation()
    test_token_optimizer_list_trim()
    test_deepseek_embed()
    print("All base tests passed")
