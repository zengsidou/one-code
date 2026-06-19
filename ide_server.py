# -*- coding: utf-8 -*-
"""Micro-Agent IDE Server — Flask Web 界面"""
import os
import sys
import json
import threading

from flask import Flask, request, jsonify, send_from_directory

# Setup
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
IDE_DIR = os.path.dirname(os.path.abspath(__file__))

from tools.registry import ToolRegistry
from tools.builtin import register_builtin_tools
from memory.short_term import ShortTermMemory
from memory.long_term import LongTermMemory
from memory import MemoryManager
from agent.loop import AgentLoop
from llm.deepseek_api import DeepSeekAdapter

app = Flask(__name__, static_folder=os.path.join(IDE_DIR, "ide"), static_url_path="")

# Global agent instance
_agent: AgentLoop | None = None
_current_task: dict = {"running": False, "task": "", "result": ""}

# Color codes for terminal
class C:
    R = "\033[91m"; G = "\033[92m"; Y = "\033[93m"; B = "\033[94m"
    C = "\033[96m"; M = "\033[95m"; W = "\033[0m"


def get_agent() -> AgentLoop:
    global _agent
    if _agent is None:
        llm = DeepSeekAdapter()
        registry = ToolRegistry(safe_mode=True)
        register_builtin_tools(registry, llm=llm)
        memory = MemoryManager(
            short=ShortTermMemory(),
            long=LongTermMemory(llm, persist_dir=os.path.join(IDE_DIR, "chroma_data")),
        )
        _agent = AgentLoop(
            llm=llm, registry=registry, memory=memory, max_steps=15,
            enable_evolution=True, enable_self_optimize=False,
        )
    return _agent


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_input = data.get("message", "").strip()
    if not user_input:
        return jsonify({"error": "Empty message"}), 400

    agent = get_agent()
    agent.memory.clear()

    _current_task["running"] = True
    _current_task["task"] = user_input

    try:
        result = agent.run(user_input)
    except Exception as e:
        result = f"[ERROR] {e}"

    _current_task["running"] = False
    _current_task["result"] = result

    return jsonify({"response": result})


@app.route("/api/chat/stream", methods=["POST"])
def chat_stream():
    """SSE 流式聊天——逐步推送思考过程"""
    import threading, queue
    from flask import Response

    data = request.get_json()
    user_input = data.get("message", "").strip()
    if not user_input:
        return jsonify({"error": "Empty message"}), 400

    agent = get_agent()
    agent.memory.clear()

    event_queue = queue.Queue()

    def run_agent():
        _current_task["running"] = True
        _current_task["task"] = user_input
        try:
            event_queue.put({"event": "status", "data": "thinking"})
            result = agent.run(user_input)
            event_queue.put({"event": "result", "data": result})
        except Exception as e:
            event_queue.put({"event": "error", "data": str(e)})
        finally:
            _current_task["running"] = False
            event_queue.put({"event": "done", "data": ""})

    threading.Thread(target=run_agent, daemon=True).start()

    def generate():
        while True:
            try:
                evt = event_queue.get(timeout=120)
                yield f"event: {evt['event']}\ndata: {evt['data']}\n\n"
                if evt["event"] in ("done", "error"):
                    break
            except queue.Empty:
                yield "event: error\ndata: timeout\n\n"
                break

    return Response(generate(), mimetype="text/event-stream")


@app.route("/api/status", methods=["GET"])
def status():
    agent = get_agent()
    stats = {"tools": len(agent.registry.tool_names), "running": _current_task["running"]}
    if agent.enable_evolution:
        try:
            rpt = agent.get_evolution_report()
            stats["tasks"] = rpt["growth"]["total_tasks"]
            stats["success_rate"] = rpt["growth"]["recent_success_rate"]
            stats["skills"] = rpt["skill_count"]
            stats["trend"] = rpt["growth"]["trend"]
        except Exception:
            pass
    return jsonify(stats)


@app.route("/api/tools", methods=["GET"])
def tools():
    agent = get_agent()
    return jsonify({"tools": agent.registry.tool_names})


@app.route("/api/grow", methods=["POST"])
def grow():
    agent = get_agent()
    try:
        plan = agent.grow()
        return jsonify(plan)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/report", methods=["GET"])
def report():
    agent = get_agent()
    try:
        return jsonify(agent.get_evolution_report())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/clear", methods=["POST"])
def clear():
    agent = get_agent()
    agent.memory.clear()
    return jsonify({"ok": True})


def main():
    port = int(os.environ.get("MICRO_AGENT_PORT", "8765"))
    print(f"{C.C}╔{'═'*58}╗{C.W}")
    print(f"{C.C}║{C.W}  Micro-Agent IDE Server{C.C} {' ' * 32}║{C.W}")
    print(f"{C.C}║{C.W}  {C.G}http://localhost:{port}{C.W}{' ' * (46 - len(str(port)))}║{C.W}")
    print(f"{C.C}╚{'═'*58}╝{C.W}")
    print(f"  {C.Y}启动中...{C.W}")

    # Pre-init agent
    agent = get_agent()
    print(f"  {C.B}Agent 就绪{C.W} | 工具: {len(agent.registry.tool_names)} | 进化: {'ON' if agent.enable_evolution else 'OFF'}")
    print(f"  {C.G}在浏览器中打开 http://localhost:{port}{C.W}")
    print()

    app.run(host="127.0.0.1", port=port, debug=False)


if __name__ == "__main__":
    main()
