# -*- coding: utf-8 -*-
"""One-Code CLI — 简单的 coding 助手"""
import sys, os, time, argparse
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

    parser = argparse.ArgumentParser(description="One-Code CLI — coding 助手")
    parser.add_argument("--contract-first", "-c", action="store_true",
                        help="启用契约先行模式：先确认方向再拆解执行")
    parser.add_argument("--max-steps", "-m", type=int, default=20,
                        help="最大执行步数 (默认 20)")
    args = parser.parse_args()

    key = os.environ.get("DEEPSEEK_API_KEY","")
    if not key: return print(C.p("请设置 DEEPSEEK_API_KEY","R"))

    from llm.deepseek_api import DeepSeekAdapter
    llm = DeepSeekAdapter(api_key=key)
    reg = ToolRegistry(safe_mode=False)
    register_builtin_tools(reg, llm=llm)
    mem = MemoryManager(short=ShortTermMemory(), long=LongTermMemory(llm))
    from agent.loop import AgentLoop
    agent = AgentLoop(llm=llm, registry=reg, memory=mem,
                      max_steps=args.max_steps,
                      enable_contract_first=args.contract_first)

    mode_str = "契约先行" if args.contract_first else "直接执行"
    print(C.p(f"One-Code ready [{mode_str}]. Type a task, Enter to send. Ctrl+C to quit.", "C"))
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
