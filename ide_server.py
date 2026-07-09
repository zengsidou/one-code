# -*- coding: utf-8 -*-
"""One-Code IDE Server — Flask Web 界面 + WebSocket 流式"""
import os
import sys
import json
import threading

from flask import Flask, request, jsonify, send_from_directory
from flask_sock import Sock

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
IDE_DIR = os.path.dirname(os.path.abspath(__file__))

from tools.registry import ToolRegistry
from tools.builtin import register_builtin_tools
from memory.short_term import ShortTermMemory
from memory.long_term import LongTermMemory
from memory import MemoryManager
from memory.context_boot import ContextBootstrapper
from agent.loop import AgentLoop
from agent.models import Message, ToolCall
from llm.deepseek_api import DeepSeekAdapter

app = Flask(__name__, static_folder=os.path.join(IDE_DIR, "ide"), static_url_path="")
sock = Sock(app)

_agent: AgentLoop | None = None
_current_task: dict = {"running": False, "task": "", "result": ""}

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
        )
    return _agent

_boot_context: list[Message] | None = None

def _build_context() -> list[Message]:
    global _boot_context
    if _boot_context is not None:
        return _boot_context
    boots = ContextBootstrapper()
    _boot_context = boots.build_boot_context() or []
    return _boot_context


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/models", methods=["GET"])
def list_models():
    return jsonify({"models": [
        {"id": "deepseek-v4-pro", "name": "DeepSeek V4 Pro", "provider": "deepseek"},
        {"id": "deepseek-v4-flash", "name": "DeepSeek V4 Flash", "provider": "deepseek"},
        {"id": "deepseek-reasoner", "name": "DeepSeek R1", "provider": "deepseek"},
        {"id": "gemini-2.5-flash", "name": "Gemini 2.5 Flash", "provider": "gemini"},
        {"id": "gemini-2.5-pro", "name": "Gemini 2.5 Pro", "provider": "gemini"},
        {"id": "gpt-4o", "name": "GPT-4o", "provider": "openai"},
        {"id": "gpt-4.1", "name": "GPT-4.1", "provider": "openai"},
    ]})


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_input = data.get("message", "").strip()
    if not user_input:
        return jsonify({"error": "Empty message"}), 400

    agent = get_agent()

    for msg in _build_context():
        agent.memory.add_message(msg)

    _current_task["running"] = True
    _current_task["task"] = user_input

    try:
        result = agent.run(user_input)
    except Exception as e:
        result = f"[ERROR] {e}"

    _current_task["running"] = False
    _current_task["result"] = result

    try:
        _learn_from_exchange(user_input, result)
    except Exception:
        pass
    try:
        b = ContextBootstrapper()
        files = _extract_files_from_result(result)
        b.save_session(user_input, files)
    except Exception:
        pass

    return jsonify({"response": result})


@app.route("/api/chat/stream", methods=["POST"])
def chat_stream():
    from flask import Response
    import threading, queue, time

    data = request.get_json()
    user_input = data.get("message", "").strip()
    if not user_input:
        return jsonify({"error": "Empty message"}), 400

    agent = get_agent()
    for msg in _build_context():
        agent.memory.add_message(msg)

    def generate():
        # Build initial context
        agent.memory.add_message(Message(role="user", content=user_input))
        step = 0
        idle = 0
        max_steps = agent.max_steps

        while step < max_steps:
            step += 1
            context = agent.memory.get_context(query=user_input[:200])
            system_prompt = agent.system_prompt
            context.insert(0, Message(role="system", content=system_prompt))

            # Call LLM with streaming, buffer into larger chunks
            text_buf = ""
            tool_calls = []
            for event in agent.llm.generate_stream(context, tools=agent.get_active_schemas()):
                if event["type"] == "text":
                    text_buf += event["text"]
                    if len(text_buf) >= 5:
                        yield f"data: {json.dumps({'type':'text-delta','text':text_buf})}\n\n"
                        text_buf = ""
                elif event["type"] == "tool":
                    if text_buf:  # Flush remaining text
                        yield f"data: {json.dumps({'type':'text','text':text_buf})}\n\n"
                        text_buf = ""
                    tool_calls.append(event)
                elif event["type"] == "error":
                    yield f"data: {json.dumps(event)}\n\n"
                    return
                elif event["type"] == "done":
                    break

            # Flush remaining buffered text
            if text_buf:
                yield f"data: {json.dumps({'type':'text-delta','text':text_buf})}\n\n"
                text_buf = ""

            # Handle tool calls
            if tool_calls:
                idle = 0
                has_tool_msg = False
                for tc in tool_calls:
                    try:
                        args = tc.get("args", {})
                        if isinstance(args, str):
                            import json as _json
                            args = _json.loads(args) if args.strip() else {}
                    except Exception:
                        args = {}
                    tc_obj = ToolCall(id=tc["name"], name=tc["name"], arguments=args)
                    result = agent.registry.execute(tc["name"], args)
                    agent.memory.add_message(Message(role="assistant", tool_calls=[tc_obj]))
                    agent.memory.add_message(Message(role="tool", content=result, tool_call_id=tc["name"], tool_name=tc["name"]))
                    has_tool_msg = True
                if has_tool_msg:
                    continue

            # No tool calls — text response
            idle += 1
            if text_content.strip():
                agent.memory.add_message(Message(role="assistant", content=text_content))
            if idle >= 2 or step >= max_steps:
                yield f"data: {json.dumps({'type': 'done', 'text': text_content})}\n\n"
                return

        yield f"data: {json.dumps({'type': 'done', 'text': text_content if 'text_content' in dir() else ''})}\n\n"

    return Response(generate(), mimetype="text/event-stream",
                    headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"})


@sock.route("/ws/chat")
def chat_ws(ws):
    import threading, queue
    while True:
        try:
            data = json.loads(ws.receive())
        except Exception:
            break
        text = data.get("message", "").strip()
        if not text:
            continue
        agent = get_agent()
        for m in _build_context():
            agent.memory.add_message(m)

        def process():
            try:
                result = agent.run(text)
                # Chunk and send for visual streaming effect
                chunk = 3
                for i in range(0, len(result), chunk):
                    ws.send(json.dumps({"type": "text", "text": result[i:i+chunk]}))
                    import time; time.sleep(0.01)
                ws.send(json.dumps({"type": "done"}))
            except Exception as e:
                ws.send(json.dumps({"type": "error", "text": str(e)}))

        threading.Thread(target=process, daemon=True).start()


@app.route("/api/files", methods=["GET"])
def file_list():
    cwd = os.getcwd()
    tree = _build_file_tree(cwd, cwd, depth=2)
    return jsonify({"files": tree})


def _build_file_tree(base: str, path: str, depth: int) -> list[dict]:
    if depth <= 0:
        return []
    entries = []
    try:
        names = sorted(os.listdir(path))
    except PermissionError:
        return []
    skip = {".git", "__pycache__", ".pytest_cache", "chroma_data", "memory_db",
            "node_modules", ".trash", "tmp_chroma", "arch_backups"}
    for name in names:
        if name in skip or name.startswith("."):
            continue
        full = os.path.join(path, name)
        is_dir = os.path.isdir(full)
        entry = {"name": name, "path": os.path.relpath(full, base),
                 "type": "dir" if is_dir else "file"}
        if is_dir:
            entry["children"] = _build_file_tree(base, full, depth - 1)
        entries.append(entry)
    return entries


@app.route("/api/status", methods=["GET"])
def status():
    agent = get_agent()
    stats = {"tools": len(agent.registry.tool_names), "running": _current_task["running"]}
    try:
        stats["token_count"] = agent.memory.short_term.get_token_count()
    except Exception:
        pass
    return jsonify(stats)


@app.route("/api/tools", methods=["GET"])
def tools():
    agent = get_agent()
    return jsonify({"tools": agent.registry.tool_names})


@app.route("/api/clear", methods=["POST"])
def clear():
    agent = get_agent()
    agent.memory.clear()
    return jsonify({"ok": True})


@app.route("/api/usage", methods=["GET"])
def usage():
    agent = get_agent()
    try:
        tokens = agent.memory.short_term.get_token_count()
        # DeepSeek V4 pricing: ¥1/M input, ¥2/M output (est 50/50 split)
        cost = round(tokens / 1_000_000 * 1.5, 6)
    except Exception:
        tokens, cost = 0, 0
    return jsonify({"tokens": tokens, "cost": cost, "max_tokens": agent.memory.short_term.max_tokens, "model": getattr(agent.llm, "model", "?"), "pricing": "¥1/M input ¥2/M output"})


@app.route("/api/cwd", methods=["GET"])
def cwd():
    return jsonify({"cwd": os.getcwd()})


@app.route("/api/tasks", methods=["GET"])
def tasks():
    agent = get_agent()
    tasks = getattr(agent.registry, '_tasks', [])
    return jsonify({"tasks": [{"id": t.id, "title": t.title, "status": t.status} for t in tasks]})


@app.route("/api/mode", methods=["POST"])
def set_mode():
    agent = get_agent()
    data = request.get_json()
    mode = data.get("mode", "build")
    if mode in ("build", "plan", "compose"):
        agent.mode = mode
        return jsonify({"mode": mode, "ok": True})
    return jsonify({"error": f"Invalid mode: {mode}"}), 400


@app.route("/api/model", methods=["POST"])
def switch_model():
    agent = get_agent()
    data = request.get_json()
    model = data.get("model", "")
    try:
        if model.startswith("deepseek"):
            from llm.deepseek_api import DeepSeekAdapter
            agent.llm = DeepSeekAdapter(model=model)
        elif model.startswith("gemini"):
            from llm.gemini_api import GeminiAdapter
            agent.llm = GeminiAdapter(model=model)
        elif model.startswith("gpt"):
            from llm.openai_api import OpenAIAdapter
            agent.llm = OpenAIAdapter(model=model)
        else:
            return jsonify({"error": f"Unknown model: {model}"}), 400
        return jsonify({"ok": True, "model": model})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
@app.route("/api/suggest", methods=["GET"])
def suggest():
    q = request.args.get("q", "").lower()
    tasks = [
        "修复 bug", "重构代码", "写单元测试",
        "添加新功能", "优化性能", "分析代码",
        "搜索文档", "配置环境", "写 API",
        "做网页", "写脚本", "解释代码",
        "写 README", "合并代码", "创建组件",
    ]
    matches = [t for t in tasks if q in t.lower()][:5] if q else tasks[:5]
    return jsonify({"suggestions": matches})


def _learn_from_exchange(user_input: str, agent_response: str):
    global _boot_context
    if not user_input or not agent_response or len(agent_response) < 10:
        return

    boots = ContextBootstrapper()
    try:
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
    except Exception:
        return

    guidance = _extract_self_guidance(reflection)
    if guidance:
        boots.update_preference("上次反思", guidance)
    boots.record_feedback(f"[自反思] 用户: {user_input[:60]} | 反思: {reflection}")

    _boot_context = boots.build_boot_context() or []
    if reflection:
        _boot_context.append(Message(
            role="system",
            content=f"[自我提示] 上次对话反思: {reflection}"
        ))


def _extract_self_guidance(reflection: str) -> str:
    parts = []
    if "太长" in reflection:
        parts.append("回答应该更简洁")
    if "太短" in reflection or "太干瘪" in reflection:
        parts.append("回答应该更详细")
    if "太啰嗦" in reflection:
        parts.append("去掉冗余废话，直接给结论")
    if "不满意" in reflection:
        parts.append("用户可能不满意，需要调整回答方式")
    return ", ".join(parts) if parts else ""


def _extract_files_from_result(result: str) -> list[str]:
    import re
    files = set()
    for pat in [r"File written:\s*(\S+)", r"已替换\s*(\S+)", r"修复\s*(\S+\.py)",
                r"已删除\s*(\S+)", r"已重命名\s*(\S+)", r"OK:\s*(\S+)"]:
        for m in re.finditer(pat, result):
            files.add(m.group(1))
    return list(files)[:5]


def main():
    port = int(os.environ.get("ONE_CODE_PORT", "8765"))
    print(f"{C.C}╔{'═'*58}╗{C.W}")
    print(f"{C.C}║{C.W}  One-Code IDE Server{C.C} {' ' * 32}║{C.W}")
    print(f"{C.C}║{C.W}  {C.G}http://localhost:{port}{C.W}{' ' * (46 - len(str(port)))}║{C.W}")
    print(f"{C.C}╚{'═'*58}╝{C.W}")

    agent = get_agent()
    print(f"  Agent:  就绪 | 工具: {len(agent.registry.tool_names)}")
    print()

    import webbrowser
    webbrowser.open(f"http://127.0.0.1:{port}")

    app.run(host="127.0.0.1", port=port, debug=False)


if __name__ == "__main__":
    main()
