# Micro-Agent Framework

从零实现的 Agent 框架，具备**自我进化能力**——能从失败中学习、自动修复缺陷、甚至修改自己的源代码。

## 架构

```
micro-agent/
├── agent/                    # Agent 核心
│   ├── models.py             # 数据模型 (Message, ToolCall, AgentState)
│   ├── loop.py               # Agent Loop (ReAct + 熔断 + 回路检测 + 自优化)
│   ├── orchestrator.py       # 多 Agent 编排 (fan-out / pipeline)
│   ├── subagent.py           # 轻量子 Agent 委派
│   ├── diagnosis.py          # 失败诊断模块
│   ├── root_cause.py         # LLM 根因分析
│   ├── self_repair.py        # 修复策略引擎
│   ├── verify.py             # 修复效果验证
│   ├── fix_history.py        # 跨会话修复复用
│   ├── meta_optimize.py      # 自优化组件的自优化
│   └── evolve/               # 进化引擎
│       ├── post_mortem.py    # 每次执行后 LLM 深度复盘
│       ├── skill_library.py  # 可复用技能库 (强化+衰减)
│       ├── ability_profile.py# 能力画像 + 成长曲线
│       ├── challenge_gen.py  # 主动挑战生成器
│       └── architect.py      # 架构自进化 (读自己代码+改代码)
├── llm/                      # LLM 适配层
│   ├── base.py               # 抽象基类
│   ├── ollama.py             # Ollama 实现
│   ├── deepseek_api.py       # DeepSeek API 适配器
├── tools/                    # 工具系统
│   ├── schema.py             # 自动 Schema 生成
│   ├── registry.py           # 装饰器注册 + 危险命令拦截
│   ├── builtin/              # 内置工具 (可被 Agent 自行扩展)
├── memory/                   # 记忆系统
│   ├── token_counter.py      # Token 精确计数 (tiktoken + fallback)
│   ├── short_term.py         # 64K 上下文窗口 + 智能摘要压缩
│   ├── long_term.py          # ChromaDB 向量检索
├── mcp/                      # MCP 协议集成
│   ├── protocol.py           # JSON-RPC 2.0 + MCP 消息类型
│   ├── transport.py          # stdio 传输层
│   ├── server.py / client.py # MCP stdio
│   ├── sse_server.py         # MCP SSE server
│   ├── sse_client.py         # MCP SSE client
├── sandbox/                  # 安全沙箱
│   ├── policy.py             # 安全策略
│   ├── fs_jail.py            # 文件系统隔离
│   ├── executor.py           # 子进程安全执行
├── tests/                    # 33 项测试
│   ├── test_loop.py          # 核心测试
│   ├── test_self_optimize.py # 自优化全链路测试
│   ├── test_mcp.py / test_mcp_e2e.py / test_mcp_sse.py
│   ├── test_sandbox.py / test_subagent.py
├── main.py                   # 交互式 CLI
├── main_mcp.py               # MCP stdio server
├── main_mcp_sse.py           # MCP SSE server
```

## 核心能力

### 五层自我改进链

| 层 | 能力 | 触发条件 |
|------|------|----------|
| **FixHistory** | 复用已验证的修复经验 | 跨会话匹配相似失败 |
| **SelfRepair** | 自动调整 prompt/参数/阈值 | 熔断/回路检测触发 |
| **MetaOptimizer** | 优化自优化组件自身 | 连续多轮无有效修复 |
| **ArchitectureEvolution** | 读自己源码 + 生成改动方案 | 架构瓶颈检测 |
| **ToolEvolution** | 识别缺工具 → 写代码 → 自动注册 | Unknown tool 重复出现 |

### 自进化引擎

| 模块 | 功能 |
|------|------|
| **PostMortem** | 每次执行后 LLM 深度复盘：难度/效率评分、策略提取、新技能识别 |
| **SkillLibrary** | 可复用技能库：查询/强化/衰减，自动注入 system prompt |
| **AbilityProfile** | 能力画像：自动任务分类，追踪成功率/效率/难度趋势 |
| **ChallengeGenerator** | 主动成长：根据弱项生成递增难度挑战任务 |

### 模块对标

| 模块 | 能力 |
|------|------|
| Agent Loop | ReAct 循环、回路检测 (tool+args MD5)、熔断器、自优化闭环 |
| Context | 64K 窗口、tiktoken 精确计数、智能摘要压缩、tool 链保护 |
| Tool Use | 装饰器注册、自动 Schema 生成、Agent 自行扩展工具 |
| LLM Adapter | DeepSeek API 原生 Function Calling、Ollama 本地调用 |
| Memory | 短期智能压缩窗口 + 长期 ChromaDB 向量检索 |
| MCP | stdio/SSE 双传输、工具暴露为标准化接口 |
| SubAgent | 轻量子 Agent 委派、多 Agent 扇出/流水线编排 |

## 快速开始

```bash
# 设置 DeepSeek API Key
$env:DEEPSEEK_API_KEY = "your-key"

# 安装依赖
pip install chromadb httpx

# 交互式 Agent
python main.py

# 开启自进化 (代码中)
agent = AgentLoop(
    registry=registry, memory=memory,
    enable_self_optimize=True,  # 五层修复链
    enable_evolution=True,      # 复盘+技能+画像+挑战
)
```

## MCP 模式

```bash
python main_mcp.py         # MCP stdio server
python main_mcp_sse.py     # MCP SSE server (HTTP)
```

## 运行测试

```bash
python tests/test_loop.py           # 7 项核心测试
python tests/test_self_optimize.py  # 26 项自优化全链路测试
python tests/test_mcp.py
python tests/test_mcp_e2e.py
python tests/test_mcp_sse.py
python tests/test_sandbox.py
python tests/test_subagent.py
```

## 面试叙事

"我从零实现了一个能自我进化的 Agent 框架。它不仅包含 ReAct 循环、工具系统和 MCP 协议等基础能力，更有五层自我改进链——从复用修复经验、自动调参，到诊断自身架构瓶颈并修改自己的源代码。Agent 在 64K 上下文窗口下能独立完成多文件代码工程任务，并在复盘后提取可复用技能、追踪能力成长曲线、主动挑战更高难度。这在传统 Agent 框架中是一个独特的方向。"
