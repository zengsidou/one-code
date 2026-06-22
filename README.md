# Micro-Agent Framework

从零实现的 Agent 框架，具备**自我进化能力**——能从失败中学习、自动修复缺陷、甚至修改自己的源代码。

对标岗位：AI 研发效能工程师 (AIDP)，覆盖 AI Coding / AI Agent / Harness Engineering 三大方向。

## 架构

```
micro-agent/
├── agent/                    # Agent 核心
│   ├── loop.py               # Agent Loop (ReAct/Plan-then-Execute + 熔断 + 回路检测)
│   ├── models.py             # 数据模型
│   ├── orchestrator.py       # 多 Agent 编排 (fan-out / pipeline)
│   ├── subagent.py           # 子 Agent 委派
│   ├── specialists.py        # 专业化 Agent (Coder/Reviewer/Tester/Doc)
│   ├── diagnosis.py / root_cause.py / self_repair.py / verify.py  # 自修复链
│   ├── fix_history.py        # 跨会话修复复用
│   ├── meta_optimize.py      # 自优化组件的自优化
│   ├── rules.py              # AGENTS.md 自动积累 (Hashimoto 模式)
│   ├── observability.py      # 指标/追踪/报告
│   ├── governance.py         # 权限控制 + 审计日志
│   ├── evaluator.py          # 独立评估 Agent (5 维度评分)
│   ├── token_optimizer.py    # Token 用量优化
│   └── evolve/               # 进化引擎 (L0-L4)
│       ├── post_mortem.py / skill_library.py / ability_profile.py
│       ├── challenge_gen.py / architect.py
├── eval/                     # SWE-bench Lite 评测
│   └── swebench_runner.py    # 自动化评测框架 (Agentless 3 阶段策略)
├── ci/                       # CI/CD
│   ├── pipeline.py           # 自动化流水线 (测试/SWE-bench/语法检查)
│   └── dashboard.html        # 可观测性 Dashboard
├── llm/ memory/ tools/ mcp/ sandbox/  # 基础设施
├── tests/                    # 52 项单元测试
├── main.py / ide_server.py  # CLI + Web IDE
```

## Harness Engineering 六层架构

| 层 | 名称 | 实现 |
|----|------|------|
| L1 | 信息边界层 | AGENTS.md 规则注入 + 角色专用系统提示 |
| L2 | 工具系统层 | 装饰器注册 + 自动 Schema + 权限过滤 |
| L3 | 执行编排层 | Plan-then-Execute / ReAct / Pipeline / Fan-out |
| L4 | 记忆状态层 | 短期(64K)+长期(ChromaDB) + Context Reset |
| L5 | 评估观测层 | 独立评估器 + 指标/追踪/仪表盘 |
| L6 | 约束恢复层 | 权限控制 + 审计 + 熔断 + 回路检测 |

## 核心能力

### SWE-bench 评测 (AI Coding)

- 300 实例 SWE-bench Lite 自动化评测
- Agentless 3 阶段策略（定位→修复→验证）
- 失败自动重试 + Plan-then-Execute 模式切换
- Gitee 镜像仓库支持

### 自进化引擎

| 层 | 能力 |
|------|------|
| L0 FixHistory | 跨会话复用已验证修复 |
| L1 SelfRepair | 自动调参/prompt/工具代码/模型切换 |
| L2 MetaOptimizer | 优化自优化组件自身 |
| L3 Evolution | 复盘→技能库→画像→挑战生成 |
| L4 Architecture | 读源码+生成改动+自测试+保留/回滚 |

### 企业级能力

- **权限控制**: 工具级 allow/deny + 风险分级 + 配额 + 工作区隔离
- **审计日志**: JSONL 格式，包含 agent/tool/risk/details/time
- **可观测性**: 指标/token追踪/工具耗时/控制台报告
- **Token 优化**: prompt 缓存/输出压缩/compact/flash 模型降级

## 集成 & 对标

- **OpenHands/OpenClaw**: Agent 框架对标，支持类似的代码修复 + 多 Agent 协作模式
- **MCP Protocol**: stdio/SSE 双传输，工具可暴露为标准化 MCP 接口
- **SWE-bench Lite**: 自动化评测框架，支持 Agentless 3 阶段策略
- **DeepSeek V4 Pro**: 原生 Function Calling，支持 reasoning_content 回传

```bash
# 依赖
pip install chromadb httpx datasets unidiff GitPython

# 交互式 CLI
$env:DEEPSEEK_API_KEY = "your-key"
python main.py

# SWE-bench 评测
python -m eval.swebench_runner --max 5 --repo "django/django"

# CI 流水线
python -m ci.pipeline

# 开启 Plan-then-Execute
agent = AgentLoop(plan_first=True, ...)
```

## 设计思路

Micro-Agent 框架从零实现了一个具备自我进化能力的 Agent 系统。对标 Harness Engineering 六层架构，覆盖 AI Coding 评测（SWE-bench）、多 Agent 专业化分工、企业级权限审计、可观测性仪表盘。Agent 能从失败中自动积累 AGENTS.md 规则，通过独立评估器验证生成质量，支持 Plan-then-Execute 安全执行模式。
