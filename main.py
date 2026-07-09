# -*- coding: utf-8 -*-
"""
OneCode Framework
  Agent Loop  — ReAct 循环 + 熔断 + 回路检测
  Tool Use    — 装饰器注册 + 自动 Schema 生成
  Memory      — 64K 智能上下文窗口 + ChromaDB 长期记忆
  LLM Adapter — DeepSeek API / Ollama 本地调用
  Evolution   — 复盘反思 + 技能库 + 能力画像 + 五层自我改进链
"""
import sys, os

from tools.registry import ToolRegistry
from tools.builtin import register_builtin_tools
from tools.plugins import load_plugin_tools
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
    use_contract = "--contract" in sys.argv or "-c" in sys.argv
    use_stream = "--stream" in sys.argv or "-s" in sys.argv

    print(Colors.c("=" * 60, Colors.CYAN))
    print(Colors.c("  One-Code Framework", Colors.CYAN))
    print(Colors.c("  Agent Loop + Tool Use + Contract-First", Colors.CYAN))
    print(Colors.c("=" * 60, Colors.CYAN))

    registry = ToolRegistry(safe_mode=True)
    sandbox = SafeExecutor(policy=SandboxPolicy())

    if llm_type == "deepseek":
        from llm.deepseek_api import DeepSeekAdapter
        llm = DeepSeekAdapter()
        model_name = "deepseek-v4-pro"
    else:
        from llm.ollama import OllamaClient
        llm = OllamaClient(model="deepseek-r1:8b")
        model_name = llm.model

    register_builtin_tools(registry, sandbox=sandbox, llm=llm)
    load_plugin_tools(registry)
    short_mem = ShortTermMemory()
    long_mem = LongTermMemory(llm, persist_dir="./chroma_data")
    memory = MemoryManager(short=short_mem, long=long_mem)

    agent = AgentLoop(
        llm=llm, registry=registry, memory=memory, max_steps=15,
        enable_contract_first=use_contract,
    )

    print(f"  LLM:       {Colors.c(model_name, Colors.BLUE)}")
    print(f"  Context:   {Colors.c(f'{short_mem.max_tokens//1024}K tokens', Colors.BLUE)}")
    print(f"  Tools:     {Colors.c(str(len(registry.tool_names)), Colors.GREEN)}")
    print(f"  Contract:  {Colors.c('ON' if use_contract else 'OFF', Colors.MAGENTA)}")
    print(Colors.c("-" * 60, Colors.CYAN))
    print(f"  /exit     /tools   /memory   /clear{'  /stream' if not use_stream else ''}")
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
            elif cmd == "/stream":
                use_stream = not use_stream; print(f"{Colors.c('流式输出: ' + ('ON' if use_stream else 'OFF'), Colors.MAGENTA)}\n")
            else:
                print(f"{Colors.c('未知命令', Colors.RED)}: {cmd}\n")
            continue

        if use_stream:
            print(Colors.c("  ...思考中...", Colors.YELLOW))
            for token in llm.generate_stream(agent.memory.short_term.get_messages(), agent.registry.get_schemas()):
                print(token, end="", flush=True)
            print()
        else:
            print(Colors.c("  ...思考中...", Colors.YELLOW))
            response = agent.run(raw)
            print(f"{Colors.c(response, Colors.CYAN)}")
        print()


if __name__ == "__main__":
    main()
