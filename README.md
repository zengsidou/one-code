# One-Code

[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10+-blue)](https://www.python.org/)

Python 实现的 **Coding Agent Harness**：Agent Loop、工具调用、多 Agent 编排、沙箱与记忆。默认对接 DeepSeek Function Calling。

> 原名 **micro-agent**，现已统一为 **One-Code**。本仓库曾包含自进化 / SWE-bench / 完整 MCP 等实验模块，已主动删减。文档只描述**当前可运行**的能力。

仓库：https://github.com/zengsidou/one-code

---

## 当前能力

| 模块 | 状态 | 说明 |
|------|------|------|
| Agent Loop | ✅ | ReAct；可选 Plan-then-Execute、契约先行 |
| 工具系统 | ✅ | 读写文件、grep/glob、shell、git、web、LSP、skills 等 |
| 多 Agent | ✅ | SubAgent 委派；Orchestrator fan-out / pipeline；Coder/Reviewer/Tester/Doc |
| 记忆 | ✅ | 短期上下文 + ChromaDB 长期检索 |
| 沙箱 | ✅ | 命令策略、路径 jail、输出截断 |
| Token 优化 | ✅ | 工具输出压缩 / 摘要 |
| Skills | ✅ | `~/.onecode/skills` 与项目内 skill 加载 |
| CLI | ✅ | `python main.py` |
| Web IDE | ✅ | `python ide_server.py` / 桌面入口 |
| 测试 | ✅ | `pytest`：**25 passed** |
| Checkpoint / Hooks / GoalVerifier | ⚠️ stub | 接口保留，逻辑已移除 |
| MCP 服务入口 | ❌ 已移除 | `main_mcp*.py` 依赖的 `mcp` 包不在仓库中 |
| 自进化 L0–L4 / SWE-bench runner | ❌ 已移除 | 不再维护 |

---

## 架构（现状）

```
one-code/
├── agent/                 # Loop、编排、子 Agent、契约先行、skills、token 优化
├── tools/                 # 注册表、内置工具、LSP、shell 安全、智能编辑
├── llm/                   # DeepSeek / OpenAI / Gemini / Ollama
├── memory/                # 短期 + 长期 + context boot
├── sandbox/               # 策略与安全执行
├── ide/ + ide_server.py   # Web IDE
├── tests/                 # 单元测试
└── main.py                # 交互式 CLI
```

---

## 快速开始

```bash
pip install -r requirements.txt

# 需要 DeepSeek API Key
set DEEPSEEK_API_KEY=your-key          # Windows CMD
# $env:DEEPSEEK_API_KEY = "your-key"   # PowerShell

python main.py                         # CLI
python main.py --contract-first        # 契约先行：先确认方向再执行
python ide_server.py                   # Web IDE
```

运行测试：

```bash
python -m pytest tests/ -q
```

---

## 核心入口说明

- **`agent/loop.py`**：主循环；熔断、回路检测、可选 plan-first / contract-first
- **`tools/builtin/`**：内置工具（含 `delegate_task` 子 Agent）
- **`agent/orchestrator.py` / `subagent.py`**：并行/串行多 Agent
- **`agent/specialists.py`**：角色化 Agent（Coder / Reviewer / Tester / Doc）
- **`sandbox/`**：执行策略与隔离
- **`llm/deepseek_api.py`**：DeepSeek 原生 Function Calling（默认 `deepseek-v4-pro`）

---

## 设计取舍（诚实版）

**保留**：能直接支撑「写代码 / 改文件 / 跑命令 / 委派子任务」的主路径，以及可观测的测试。

**删减**：自进化、完整评测流水线、MCP 服务端等——减少半成品与文档漂移。若重新引入，会先落地代码与测试，再写回文档。

---

## 契约先行（Contract-First）

产品原则与求职精简稿：[`docs/contract-first-job-brief.md`](docs/contract-first-job-brief.md)

5 任务对照实验协议 + runner：

```bash
# 需 DEEPSEEK_API_KEY
python -m eval.contract_first.run --mode contract --task-id T1
python -m eval.contract_first.run --mode all --task-id T1
```

详见 [`docs/contract-first-experiment.md`](docs/contract-first-experiment.md)。  
最新一次自动跑批结果：[`docs/contract-first-results.md`](docs/contract-first-results.md)。

完整论述（proposal）：https://zengsidou.github.io/papers/contract-first-paper.pdf

---

## License

Apache 2.0
