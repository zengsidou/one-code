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

    # 学习用户偏好（不阻塞主响应）
    try:
        _learn_from_exchange(user_input, result)
    except Exception:
        pass

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
            # 学习偏好（不阻塞）
            try:
                _learn_from_exchange(user_input, result)
            except Exception:
                pass
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
    """Agent 自反思对话质量"""
    global _boot_context
    if not user_input or not agent_response or len(agent_response) < 10:
        return

    boots = ContextBootstrapper()
    reflection = ""

    # 用独立 LLM 做对话质量自反思（不复用 agent 的 LLM 上下文）
    try:
        from llm.deepseek_api import DeepSeekAdapter
        reflection_llm = DeepSeekAdapter(timeout=10)
        resp = reflection_llm.generate(
            [Message(role="user", content=(
                "你刚和用户完成了一次对话。请反思你的回答质量。\n\n"
                f"用户说: {user_input[:200]}\n\n"
                f"你的回答（摘要）: {agent_response[:400]}\n\n"
                "请用一句话评价: 回答长度(太长/合适/太短)，风格(太啰嗦/合适/太干瘪)，用户满意度。"
                "只输出评价，不要解释。"
            ))],
            tools=None,
        )
        reflection = (resp.content or "")[:200]
    except Exception as e:
        reflection = f"[反思失败: {e}]"[:100]

    if reflection:
        # 提取行为指导
        guidance = _extract_self_guidance(reflection)
        if guidance:
            boots.update_preference("上次反思", guidance)
        boots.record_feedback(f"[自反思] 用户: {user_input[:60]} | 反思: {reflection}")

    # 重建启动上下文
    _boot_context = boots.build_boot_context()
    if reflection:
        _boot_context.append(Message(
            role="system",
            content=f"[自我提示] 上次对话反思: {reflection}"
        ))
    """从 agent 返回中提取涉及的文件名"""
    import re
    files = set()
    # 匹配 File written: path, 已替换 path, 修复 path
    for pat in [r"File written:\s*(\S+)", r"已替换\s*(\S+)", r"修复\s*(\S+\.py)"]:
        for m in re.finditer(pat, result):
            files.add(m.group(1))
    return list(files)[:5]


def _extract_self_guidance(reflection: str) -> str:
    """从 LLM 自反思中提取行为指导"""
    guidance_parts = []
    if "太长" in reflection:
        guidance_parts.append("回答应该更简洁")
    if "太短" in reflection or "太干瘪" in reflection:
        guidance_parts.append("回答应该更详细")
    if "太啰嗦" in reflection:
        guidance_parts.append("去掉冗余废话，直接给结论")
    if "不满意" in reflection:
        guidance_parts.append("用户可能不满意，需要调整回答方式")
    return ", ".join(guidance_parts) if guidance_parts else ""


def _extract_files_from_result(result: str) -> list[str]:
    """从 agent 返回中提取涉及的文件名"""
    import re
    files = set()
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
