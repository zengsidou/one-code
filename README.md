# Micro-Agent Framework

从零实现的 Agent 框架，对标 DeepSeek Agent Harness 工程师技术栈。

## 架构

```
micro-agent/
├── agent/               # Agent 核心
│   ├── models.py        # 数据模型 (Message, ToolCall, AgentState)
│   ├── loop.py          # Agent Loop (ReAct + 熔断 + 循环检测)
├── llm/                 # LLM 适配层
│   ├── base.py          # 抽象基类
│   ├── ollama.py        # Ollama 实现 + function calling fallback
├── tools/               # 工具系统
│   ├── schema.py        # 自动 Schema 生成
│   ├── registry.py      # 装饰器注册 + 危险命令拦截
│   ├── builtin/         # 内置工具 (文件/Shell/搜索/计算)
├── memory/              # 记忆系统
│   ├── short_term.py    # Token 感知滑动窗口
│   ├── long_term.py     # ChromaDB 向量检索 (bge-m3)
├── mcp/                 # MCP 协议集成
│   ├── protocol.py      # JSON-RPC 2.0 + MCP 消息类型
│   ├── transport.py     # stdio 传输层
│   ├── server.py        # MCP Server (暴露 ToolRegistry)
│   ├── client.py        # MCP Client (测试用)
├── tests/               # 21 项测试
│   ├── test_loop.py     # Agent 核心测试
│   ├── test_mcp.py      # MCP 单元测试
│   ├── test_mcp_e2e.py  # MCP 子进程端到端
├── main.py              # 交互式 CLI
├── main_mcp.py          # MCP stdio 服务器入口
```

## 核心模块

| 模块 | 能力 | 对标 JD |
|------|------|---------|
| Agent Loop | ReAct 循环、回路检测 (tool+args MD5)、熔断器 | Agent Loop / Reasoning / Planning |
| Tool Use | 装饰器注册、类型→JSON Schema、危险命令正则拦截 | Tool Use / MCP |
| MCP Transport | MCP stdio 协议、tools/list + tools/call、JSON-RPC 2.0 | MCP |
| Memory | 短期滑动窗口 (Token 感知 trim)、长期 ChromaDB (bge-m3) | Memory / Context Engineering |
| LLM Adapter | Ollama 本地调用、function calling native + prompt fallback | LLM API / Prompt Engineering |

## 快速开始

```bash
pip install chromadb httpx
python main.py          # 交互式 Agent
python main_mcp.py      # MCP stdio 服务器
```

需要本地运行 Ollama：
```bash
ollama pull deepseek-r1:8b
ollama pull bge-m3
```

## MCP 模式

```bash
# 作为 MCP Server 启动
python main_mcp.py

# 在另一个终端测试
python tests/test_mcp_e2e.py
```

## 运行测试

```bash
python tests/test_loop.py
python tests/test_mcp.py
python tests/test_mcp_e2e.py
```

## 面试叙事

"我从零实现了一个 Agent 框架，包含 ReAct 循环、工具调用系统和记忆系统，
并集成了 MCP 协议将工具暴露为标准化接口。
框架的每个模块都直接对应 Harness Engineering 的核心概念。"
