# -*- coding: utf-8 -*-
"""One-Code Terminal — Rich TUI, MiMo Code 风格"""
import os, sys, json, threading, time
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.markdown import Markdown
from rich import box
from rich.live import Live
from rich.syntax import Syntax

console = Console()

def make_agent():
    key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not key:
        console.print("[red]DEEPSEEK_API_KEY not set[/red]")
        sys.exit(1)
    from llm.deepseek_api import DeepSeekAdapter
    from tools.registry import ToolRegistry; from tools.builtin import register_builtin_tools
    from memory.short_term import ShortTermMemory; from memory.long_term import LongTermMemory
    from memory import MemoryManager; from agent.loop import AgentLoop
    llm = DeepSeekAdapter(api_key=key)
    reg = ToolRegistry(safe_mode=False); register_builtin_tools(reg, llm=llm)
    mem = MemoryManager(short=ShortTermMemory(), long=LongTermMemory(llm))
    return AgentLoop(llm=llm, registry=reg, memory=mem, max_steps=20)

# State
chat_history = []
current_status = "Ready"
tool_calls_log = []

def build_layout():
    return Layout(name="root")

def render():
    out = []
    out.append(Text(" One-Code", style="bold cyan") +
               Text(f"  {current_status}", style="green" if current_status == "Ready" else "yellow"))
    out.append(Text("─" * console.width, style="dim"))
    for msg in chat_history[-20:]:
        role, text = msg
        if role == "you":
            out.append(Text(f"\n▸ {text}", style="bold blue"))
        elif role == "agent":
            out.append(Text(f"\n{text[:800]}", style="white"))
        elif role == "tool":
            out.append(Text(f"  ⚙ {text}", style="dim yellow"))
    out.append(Text(""))
    out.append(Text("─" * console.width, style="dim"))
    out.append(Text(">>> ", style="bold green") + Text("Type your task (Enter to send, Ctrl+C to quit)", style="dim"))
    return Panel(Text.assemble(*out), box=box.SIMPLE, border_style="blue")

def main():
    console.clear()
    with console.status("[cyan]Starting One-Code...[/cyan]"):
        agent = make_agent()
    chat_history.append(("agent", f"One-Code ready. {len(agent.registry.tool_names)} tools, {agent.llm.model}."))

    while True:
        console.clear()
        console.print(render())
        try:
            user = console.input("")
        except (EOFError, KeyboardInterrupt):
            console.print("\n[yellow]Goodbye[/yellow]")
            break

        user = user.strip()
        if not user:
            continue
        if user.lower() in ("exit", "quit", "/exit"):
            break

        chat_history.append(("you", user))
        current_status = "Thinking..."

        try:
            result = agent.run(user)
            chat_history.append(("agent", result[:2000]))
        except Exception as e:
            chat_history.append(("agent", f"Error: {e}"))
        current_status = "Ready"

if __name__ == "__main__":
    main()
