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
from memory.context_boot import ContextBootstrapper
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
            llm=llm, registry=registry, memory=memory, max_steps=20,
            enable_evolution=True, enable_self_optimize=False,
        )
        _bootstrapper = ContextBootstrapper()
        _boot_context = _bootstrapper.build_boot_context()
        if _boot_context:
            for msg in _boot_context:
                _agent.memory.add_message(msg)
    return _agent

_bootstrapper = None
_boot_context = None


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

    # 重新注入项目/用户/会话上下文（可能已更新）
    global _boot_context
    if _boot_context:
        for msg in _boot_context:
            agent.memory.add_message(msg)

    _current_task["running"] = True
    _current_task["task"] = user_input

    try:
        result = agent.run(user_input)
    except Exception as e:
        result = f"[ERROR] {e}"

    _current_task["running"] = False
    _current_task["result"] = result

    # 学习用户偏好
    _learn_from_exchange(user_input, result)

    # 保存会话状态（下次启动恢复）
    try:
        b = ContextBootstrapper()
        files = _extract_files_from_result(result)
        b.save_session(user_input, files)
    except Exception:
        pass

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

    # 重新注入上下文
    global _boot_context
    if _boot_context:
        for msg in _boot_context:
            agent.memory.add_message(msg)

    event_queue = queue.Queue()

    def run_agent():
        _current_task["running"] = True
        _current_task["task"] = user_input
        try:
            event_queue.put({"event": "status", "data": "thinking"})
            result = agent.run(user_input)
            event_queue.put({"event": "result", "data": result})
            # 学习偏好
            _learn_from_exchange(user_input, result)
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


def _learn_from_exchange(user_input: str, agent_response: str):
    """从一次对话中学习用户偏好"""
    global _boot_context
    if not user_input or not agent_response:
        return

    boots = ContextBootstrapper()

    # 显式反馈关键词
    feedback_keywords = {
        "太啰嗦": ("回复风格", "更简洁"),
        "短一点": ("回复风格", "更简洁"),
        "太长了": ("回复风格", "更简洁"),
        "详细": ("回复风格", "更详细"),
        "解释一下": ("回复风格", "多解释"),
        "说中文": ("语言", "中文"),
        "做得好": ("反馈", "正面"),
        "对了": ("反馈", "正面"),
        "错了": ("反馈", "纠正"),
        "不对": ("反馈", "纠正"),
        "不要 commit": ("Git", "不要自动提交"),
        "太慢了": ("速度", "加快"),
        "简洁": ("回复风格", "简洁直接"),
        "不要加注释": ("代码风格", "不添加注释"),
    }

    for keyword, (pref_key, pref_val) in feedback_keywords.items():
        if keyword in user_input:
            boots.update_preference(pref_key, pref_val)
            boots.record_feedback(f"用户说: {user_input[:100]}")

    # 重新构建启动上下文（下次回答生效）
    _boot_context = boots.build_boot_context()
    """从 agent 返回中提取涉及的文件名"""
    import re
    files = set()
    # 匹配 File written: path, 已替换 path, 修复 path
    for pat in [r"File written:\s*(\S+)", r"已替换\s*(\S+)", r"修复\s*(\S+\.py)"]:
        for m in re.finditer(pat, result):
            files.add(m.group(1))
    return list(files)[:5]


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
