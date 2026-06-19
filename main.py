# -*- coding: utf-8 -*-
"""
Micro-Agent Framework
  Agent Loop  — ReAct 循环 + 熔断 + 回路检测
  Tool Use    — 装饰器注册 + 自动 Schema 生成
  Memory      — 64K 智能上下文窗口 + ChromaDB 长期记忆
  LLM Adapter — DeepSeek API / Ollama 本地调用
  Evolution   — 复盘反思 + 技能库 + 能力画像 + 五层自我改进链
"""
import sys, os

from tools.registry import ToolRegistry
from tools.builtin import register_builtin_tools
from memory.short_term import ShortTermMemory
from memory.long_term import LongTermMemory
from memory import MemoryManager
from agent.loop import AgentLoop
from agent.checkpoint import AgentCheckpoint
from agent.models import Message
from sandbox import SandboxPolicy, SafeExecutor


class Colors:
    RESET = "\033[0m"
    CYAN = "\033[96m"; YELLOW = "\033[93m"; GREEN = "\033[92m"
    RED = "\033[91m"; BLUE = "\033[94m"; MAGENTA = "\033[95m"

    @staticmethod
    def enable():
        if os.name == "nt": os.system("")

    @staticmethod
    def c(text: str, color: str) -> str:
        return f"{color}{text}{Colors.RESET}"


def main():
    Colors.enable()

    # Choose LLM backend
    llm_type = "deepseek" if os.environ.get("DEEPSEEK_API_KEY") else "ollama"
    use_evolution = "--evo" in sys.argv or "-e" in sys.argv
    use_optimize = "--opt" in sys.argv or "-o" in sys.argv

    print(Colors.c("=" * 60, Colors.CYAN))
    print(Colors.c("  Micro-Agent Framework", Colors.CYAN))
    print(Colors.c("  Agent Loop + Tool Use + Memory + Evolution", Colors.CYAN))
    print(Colors.c("=" * 60, Colors.CYAN))

    registry = ToolRegistry(safe_mode=True)
    sandbox = SafeExecutor(policy=SandboxPolicy())

    if llm_type == "deepseek":
        from llm.deepseek_api import DeepSeekAdapter
        llm = DeepSeekAdapter()
        model_name = "deepseek-reasoner"
    else:
        from llm.ollama import OllamaClient
        llm = OllamaClient(model="deepseek-r1:8b")
        model_name = llm.model

    register_builtin_tools(registry, sandbox=sandbox, llm=llm)
    short_mem = ShortTermMemory()
    long_mem = LongTermMemory(llm, persist_dir="./chroma_data")
    memory = MemoryManager(short=short_mem, long=long_mem)

    agent = AgentLoop(
        llm=llm, registry=registry, memory=memory, max_steps=15,
        enable_evolution=use_evolution, enable_self_optimize=use_optimize,
    )

    # 检查是否是架构进化后的自重启
    if AgentCheckpoint.has_restart_flag():
        ckpt = AgentCheckpoint.load()
        if ckpt:
            AgentCheckpoint.clear_restart_flag()
            task = ckpt.get("current_task", "")
            reason = ckpt.get("restart_reason", "architecture_evolution")
            print(Colors.c(f"  [RESTART] 架构进化重启: {reason}", Colors.MAGENTA))
            print(Colors.c(f"  [RESTART] 继续执行: {task[:80]}...", Colors.MAGENTA))
            response = agent.run(task)
            print(f"{Colors.c(response, Colors.CYAN)}")
            print()

    print(f"  LLM:       {Colors.c(model_name, Colors.BLUE)}")
    print(f"  Context:   {Colors.c(f'{short_mem.max_tokens//1024}K tokens', Colors.BLUE)}")
    print(f"  Tools:     {Colors.c(str(len(registry.tool_names)), Colors.GREEN)}")
    print(f"  Evolution: {Colors.c('ON' if use_evolution else 'OFF', Colors.MAGENTA)}")
    print(f"  Self-Opt:  {Colors.c('ON' if use_optimize else 'OFF', Colors.MAGENTA)}")
    print(Colors.c("-" * 60, Colors.CYAN))
    print(f"  /exit     /tools   /memory   /clear")
    if use_evolution:
        print(f"  /grow     /report  /skills")
    print(Colors.c("-" * 60, Colors.CYAN))
    print()

    while True:
        try:
            raw = input(Colors.c(">>> ", Colors.GREEN)).strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{Colors.c('再见!', Colors.YELLOW)}"); break

        if not raw: continue

        if raw.startswith("/"):
            cmd = raw.lower()
            if cmd in ("/exit", "/quit"):
                print(f"{Colors.c('再见!', Colors.YELLOW)}"); break
            elif cmd == "/tools":
                print(f"\n{Colors.c('可用工具:', Colors.BLUE)}")
                print(registry.get_tools_description()); print()
            elif cmd == "/memory":
                msgs = short_mem.get_messages()
                print(f"\n{Colors.c(f'短期记忆 ({len(msgs)} 条, {short_mem.get_token_count()} tokens):', Colors.BLUE)}")
                for m in msgs[-10:]:
                    role = Colors.c(f"[{m.role}]", Colors.MAGENTA)
                    print(f"  {role} {(m.content or '')[:120]}")
                print()
            elif cmd == "/clear":
                memory.clear(); print(f"{Colors.c('记忆已清空', Colors.GREEN)}\n")
            elif use_evolution and cmd == "/grow":
                plan = agent.grow()
                print(f"\n{Colors.c('成长建议:', Colors.BLUE)}")
                sug = plan.get("suggestion", {})
                if sug:
                    print(f"  动作: {sug.get('action','?')} | 级别: {sug.get('current_level','?')}")
                    print(f"  理由: {sug.get('reason','?')}")
                for c in plan.get("challenges", [])[:3]:
                    print(f"  [L{c.get('difficulty','?')}] {c.get('task','?')}")
                print()
            elif use_evolution and cmd == "/report":
                rpt = agent.get_evolution_report()
                g = rpt["growth"]
                print(f"\n{Colors.c('进化报告:', Colors.BLUE)}")
                print(f"  任务: {g['total_tasks']} | 成功率: {g['recent_success_rate']}")
                print(f"  均效率: {g['recent_avg_efficiency']} | 趋势: {g['trend']}")
                print(f"  技能: {rpt['skill_count']} | 弱项: {rpt['weak_areas'] or '无'}")
                for ins in rpt.get("recent_insights", [])[:2]:
                    print(f"  {ins[:100]}")
                print()
            elif use_evolution and cmd == "/skills":
                stats = agent._skill_library.get_stats()
                print(f"\n{Colors.c('技能库:', Colors.BLUE)}")
                for s in stats.get("strongest", [])[:5]:
                    print(f"  - {s['name']} (强度:{s['strength']:.1f}, 复用:{s['reinforce_count']}次)")
                print()
            else:
                print(f"{Colors.c('未知命令', Colors.RED)}: {cmd}\n")
            continue

        print(Colors.c("  ...思考中...", Colors.YELLOW))
        response = agent.run(raw)
        print(f"{Colors.c(response, Colors.CYAN)}")
        print()


if __name__ == "__main__":
    main()
