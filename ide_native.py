import os, sys, threading
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import tkinter as tk
from tkinter import scrolledtext

BG, FG = "#1e1e1e", "#d4d4d4"

def make_agent():
    k = os.environ.get("DEEPSEEK_API_KEY","")
    if not k: return None
    from llm.deepseek_api import DeepSeekAdapter as D
    from tools.registry import ToolRegistry as R; from tools.builtin import register_builtin_tools as B
    from memory.short_term import ShortTermMemory as S; from memory.long_term import LongTermMemory as L
    from memory import MemoryManager as M; from agent.loop import AgentLoop as A
    llm = D(api_key=k); reg = R(safe_mode=False); B(reg, llm=llm)
    mem = M(short=S(), long=L(llm))
    return A(llm=llm, registry=reg, memory=mem, max_steps=20)

agent = make_agent()
if not agent:
    import tkinter.messagebox
    root = tk.Tk(); root.withdraw()
    tkinter.messagebox.showerror("Error","DEEPSEEK_API_KEY not set")
    sys.exit(1)

root = tk.Tk()
root.title("One-Code")
root.geometry("900x600")
root.configure(bg=BG)

# Status
st = tk.Label(root, text="Ready", fg="#4ec9b0", bg=BG, font=("Consolas",10))
st.pack(pady=(6,0))

# Chat
chat = scrolledtext.ScrolledText(root, bg=BG, fg=FG, bd=0, font=("Consolas",11), wrap=tk.WORD)
chat.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)
chat.tag_config("u", foreground="#007acc"); chat.tag_config("a", foreground=FG)
chat.insert(tk.END, "One-Code ready.\n\n","a")
chat.configure(state=tk.DISABLED)

# Input
inp = tk.Entry(root, bg="#3c3c3c", fg="white", bd=0, font=("Consolas",13), insertbackground="white")
inp.pack(fill=tk.X, padx=8, pady=(0,8), ipady=6)
inp.focus_set()

def send(e=None):
    t = inp.get().strip()
    if not t: return
    inp.delete(0,tk.END)
    chat.configure(state=tk.NORMAL)
    chat.insert(tk.END, f"\n>>> {t}\n\n","u"); chat.see(tk.END)
    chat.configure(state=tk.DISABLED)
    st.configure(text="Thinking...",fg="#dcdcaa")
    threading.Thread(target=_run,args=(t,),daemon=True).start()

def _run(t):
    try: r=agent.run(t);e=None
    except Exception as ex: r,e=None,str(ex)
    root.after(0,lambda:d(r,e))

def d(r,e):
    chat.configure(state=tk.NORMAL)
    if e: chat.insert(tk.END,f"Error: {e}\n","a")
    else: chat.insert(tk.END,f"{r[:5000]}\n","a")
    chat.insert(tk.END,"─"*60+"\n","a"); chat.see(tk.END)
    chat.configure(state=tk.DISABLED)
    st.configure(text="Ready",fg="#4ec9b0")

inp.bind("<Return>",send)
root.mainloop()
