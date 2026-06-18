# -*- coding: utf-8 -*-
"""
Micro-Agent Framework
从零实现的 Agent 框架，包含:
  Agent Loop  — ReAct 循环 + 熔断 + 循环检测
  Tool Use    — 装饰器注册 + 自动 Schema 生成 + 危险命令拦截
  Memory      — 短期滑动窗口 + 长期 ChromaDB 向量检索
  LLM Adapter — Ollama 本地调用 + Function Calling 兼容 Fallback
"""
import sys
from llm.ollama import OllamaClient
from tools.registry import ToolRegistry
from tools.builtin import register_builtin_tools
from memory.short_term import ShortTermMemory
from memory.long_term import LongTermMemory
from memory import MemoryManager
from agent.loop import AgentLoop
from agent.models import Message
from sandbox import SandboxPolicy, SafeExecutor


class Colors:
    RESET = "\033[0m"
    CYAN = "\033[96m"
    YELLOW = "\033[93m"
    GREEN = "\033[92m"
    RED = "\033[91m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"

    @staticmethod
    def enable():
        import os
        if os.name == "nt":
            os.system("")  # enable ANSI on Windows

    @staticmethod
    def c(text: str, color: str) -> str:
        return f"{color}{text}{Colors.RESET}"


def main():
    Colors.enable()

    print(Colors.c("=" * 60, Colors.CYAN))
    print(Colors.c("  Micro-Agent Framework", Colors.CYAN))
    print(Colors.c("  Agent Loop + Tool Use + Memory + LLM Adapter", Colors.CYAN))
    print(Colors.c("=" * 60, Colors.CYAN))

    llm = OllamaClient(model="deepseek-r1:8b", embedding_model="bge-m3:latest")
    registry = ToolRegistry(safe_mode=True)
    sandbox = SafeExecutor(policy=SandboxPolicy())
    register_builtin_tools(registry, sandbox=sandbox, llm=llm)
    short_mem = ShortTermMemory(max_tokens=4096)
    long_mem = LongTermMemory(llm, persist_dir="./chroma_data")
    memory = MemoryManager(short=short_mem, long=long_mem)

    agent = AgentLoop(
        llm=llm,
        registry=registry,
        memory=memory,
        max_steps=15,
    )

    print(f"  Model:     {Colors.c(llm.model, Colors.BLUE)}")
    print(f"  Embedding: {Colors.c(llm.embedding_model, Colors.BLUE)}")
    print(f"  Tools:     {Colors.c(', '.join(registry.tool_names), Colors.GREEN)}")
    print(f"  Sandbox:   {Colors.c(sandbox.policy.level.value.upper(), Colors.MAGENTA)}")
    print(f"  Safe Mode: {Colors.c('ON', Colors.MAGENTA)}")
    print(Colors.c("-" * 60, Colors.CYAN))
    print(f"  /exit  退出    /memory 查看记忆")
    print(f"  /tools 查看工具  /clear  清空记忆")
    print(Colors.c("-" * 60, Colors.CYAN))
    print()

    while True:
        try:
            raw = input(Colors.c(">>> ", Colors.GREEN)).strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{Colors.c('再见!', Colors.YELLOW)}")
            break

        if not raw:
            continue

        if raw.startswith("/"):
            cmd = raw.lower()
            if cmd in ("/exit", "/quit"):
                print(f"{Colors.c('再见!', Colors.YELLOW)}")
                break
            elif cmd == "/tools":
                print(f"\n{Colors.c('可用工具:', Colors.BLUE)}")
                print(registry.get_tools_description())
                print()
            elif cmd == "/memory":
                msgs = short_mem.get_messages()
                print(f"\n{Colors.c(f'短期记忆 ({len(msgs)} 条):', Colors.BLUE)}")
                for m in msgs[-10:]:
                    role = Colors.c(f"[{m.role}]", Colors.MAGENTA)
                    preview = (m.content or "")[:120]
                    print(f"  {role} {preview}")
                print()
            elif cmd == "/clear":
                memory.clear()
                print(f"{Colors.c('记忆已清空', Colors.GREEN)}\n")
            else:
                print(f"{Colors.c('未知命令', Colors.RED)}: {cmd}\n")
            continue

        print(Colors.c("  ...思考中...", Colors.YELLOW))
        response = agent.run(raw)
        print(f"{Colors.c(response, Colors.CYAN)}")
        print()


if __name__ == "__main__":
    main()
