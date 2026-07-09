# -*- coding: utf-8 -*-
"""One-Code CLI — 简单的 coding 助手"""
import sys, os, time
from tools.registry import ToolRegistry
from tools.builtin import register_builtin_tools
from memory.short_term import ShortTermMemory
from memory.long_term import LongTermMemory
from memory import MemoryManager

class C:
    R="\033[91m";G="\033[92m";Y="\033[93m";B="\033[94m";C="\033[96m";M="\033[95m";W="\033[0m"
    @staticmethod
    def p(text, color): return f"{color}{text}{C.W}"

def main():
    if os.name=="nt": os.system("")

    key = os.environ.get("DEEPSEEK_API_KEY","")
    if not key: return print(C.p("请设置 DEEPSEEK_API_KEY","R"))

    from llm.deepseek_api import DeepSeekAdapter
    llm = DeepSeekAdapter(api_key=key)
    reg = ToolRegistry(safe_mode=False)
    register_builtin_tools(reg, llm=llm)
    mem = MemoryManager(short=ShortTermMemory(), long=LongTermMemory(llm))
    from agent.loop import AgentLoop
    agent = AgentLoop(llm=llm, registry=reg, memory=mem, max_steps=20)

    print(C.p("One-Code ready. Type a task, Enter to send. Ctrl+C to quit.", "C"))
    print(f"  {C.p(str(len(reg.tool_names)),'G')} tools  {C.p(llm.model,'B')}\n")

    while True:
        try:
            raw = input(C.p("> ","G")).strip()
        except (EOFError,KeyboardInterrupt):
            break
        if not raw: continue

        print(C.p(f"  ...thinking...\n","Y"), end="", flush=True)

        # Run with debug to show steps
        result = agent.run(raw, debug=False)
        result = result.replace("[STOPPED]","").strip()
        print(C.p(result,"W"))
        print(C.p("-"*50,"B"))

if __name__=="__main__": main()
