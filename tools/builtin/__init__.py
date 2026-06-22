# -*- coding: utf-8 -*-
"""内置工具集"""
import os
from tools.registry import run_shell


def register_builtin_tools(registry, sandbox=None, llm=None) -> None:
    @registry.register("read_file", "读取文件内容，支持 offset(起始行，1开始)和 limit(行数上限)")
    def read_file(path: str, offset: int = 0, limit: int = 2000) -> str:
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
            total = len(lines)
            if offset > 0:
                lines = lines[offset - 1:]
            if limit > 0:
                lines = lines[:limit]
            content = "".join(lines)
            if total > offset + len(lines):
                content += f"\n... (共 {total} 行，已显示第 {max(offset,1)}-{min(offset+len(lines)-1,total)} 行)"
            return content[:8000] if len(content) > 8000 else content
        except FileNotFoundError:
            return f"[ERROR] File not found: {path}"
        except PermissionError:
            return f"[ERROR] Permission denied: {path}"
        except Exception as e:
            return f"[ERROR] Read file failed: {e}"

    @registry.register("grep", "在目录中搜索匹配正则表达式的内容，返回文件路径和行号。支持 include 参数过滤文件名(如 '*.py')")
    def grep(pattern: str, path: str = ".", include: str = "*") -> str:
        import re
        import fnmatch
        from pathlib import Path

        try:
            compiled = re.compile(pattern)
        except re.error as e:
            return f"[ERROR] 正则表达式错误: {e}"

        results = []
        try:
            base = Path(path).resolve()
            for f in base.rglob("*"):
                if not f.is_file():
                    continue
                if include != "*" and not fnmatch.fnmatch(f.name, include):
                    continue
                if f.suffix.lower() in (".exe", ".dll", ".pyd", ".pyc", ".so", ".o", ".obj", ".bin", ".png", ".jpg", ".zip", ".tar", ".gz"):
                    continue
                try:
                    if f.stat().st_size > 1024 * 1024:
                        continue
                    with open(f, encoding="utf-8", errors="replace") as fh:
                        for i, line in enumerate(fh, 1):
                            if compiled.search(line):
                                results.append(f"{f.relative_to(base)}:{i}: {line.rstrip()[:200]}")
                                if len(results) >= 100:
                                    break
                except Exception:
                    continue
                if len(results) >= 100:
                    break
        except Exception as e:
            return f"[ERROR] 搜索失败: {e}"

        if not results:
            return f"未找到匹配 '{pattern}' 的内容 (路径: {path}, 文件过滤: {include})"
        suffix = f"\n... (超过 100 条结果，已截断)" if len(results) >= 100 else ""
        return f"找到 {len(results)} 处匹配:\n" + "\n".join(results) + suffix

    @registry.register("glob", "按 glob 模式匹配文件，如 '**/*.py' 或 '*.json'")
    def glob_files(pattern: str, path: str = ".") -> str:
        from pathlib import Path
        if not isinstance(pattern, str):
            return f"[ERROR] glob 参数错误: pattern 应为字符串，实际为 {type(pattern).__name__}"
        if not isinstance(path, str):
            return f"[ERROR] glob 参数错误: path 应为字符串，实际为 {type(path).__name__}"
        try:
            matches = sorted(Path(path).rglob(pattern))
            if not matches:
                return f"未找到匹配 '{pattern}' 的文件 (路径: {path})"
            items = []
            for m in matches[:100]:
                items.append(str(m))
            suffix = f"\n... (共 {len(matches)} 个结果，已截断至前 100 个)" if len(matches) > 100 else ""
            return f"找到 {len(matches)} 个匹配:\n" + "\n".join(items) + suffix
        except Exception as e:
            return f"[ERROR] glob 失败: {e}"

    @registry.register("edit_file", "精确替换文件中的指定字符串。old_string 必须唯一出现，否则替换失败")
    def edit_file(path: str, old_string: str, new_string: str) -> str:
        try:
            with open(path, encoding="utf-8") as f:
                content = f.read()
        except FileNotFoundError:
            return f"[ERROR] 文件不存在: {path}"
        except Exception as e:
            return f"[ERROR] 读取文件失败: {e}"

        count = content.count(old_string)
        if count == 0:
            return f"[ERROR] 未找到要替换的文本 (在 {path} 中)。请确认 old_string 与文件内容精确匹配（包括缩进、换行）"
        if count > 1:
            return f"[ERROR] 找到 {count} 处匹配，请提供更多上下文以唯一确定要替换的位置"

        try:
            new_content = content.replace(old_string, new_string, 1)
            with open(path, "w", encoding="utf-8") as f:
                f.write(new_content)
            return f"已替换 {path} 中的 1 处匹配 ({len(content)} → {len(new_content)} 字符)"
        except Exception as e:
            return f"[ERROR] 写入文件失败: {e}"

    @registry.register("write_file", "写入内容到指定路径的文件")
    def write_file(path: str, content: str) -> str:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            return f"File written: {path} ({len(content)} chars)"
        except PermissionError:
            return f"[ERROR] Permission denied: {path}"
        except Exception as e:
            return f"[ERROR] Write file failed: {e}"

    @registry.register("list_dir", "列出目录内容，默认为当前目录")
    def list_dir(path: str = ".") -> str:
        try:
            entries = os.listdir(path)
            if not entries:
                return f"Directory '{path}' is empty."
            lines = [f"  {'D' if os.path.isdir(os.path.join(path, e)) else 'F'}  {e}" for e in sorted(entries)]
            return f"Contents of '{path}':\n" + "\n".join(lines[:100])
        except FileNotFoundError:
            return f"[ERROR] Directory not found: {path}"
        except PermissionError:
            return f"[ERROR] Permission denied: {path}"
        except Exception as e:
            return f"[ERROR] List dir failed: {e}"

    @registry.register("run_shell", "执行 Shell 命令并返回输出，超时默认 30 秒")
    def _run_shell(command: str, timeout: int = 30) -> str:
        if sandbox:
            result = sandbox.execute(command)
            if result["ok"]:
                out = result["output"]
                if result["error"]:
                    out += "\n[stderr] " + result["error"]
                return f"{out}\n[done in {result['duration_ms']}ms]"
            else:
                return f"[{result['blocked_by'] or 'ERROR'}] {result['error']}"
        return run_shell(command, timeout)

    @registry.register("search_web", "Bing 网络搜索（免费，国内可用，超时 10 秒）")
    def search_web(query: str) -> str:
        import re, os, threading

        # DuckDuckGo 备用（优先避免被封 IP）
        try:
            from ddgs import DDGS
            results_holder = []
            def _ddg():
                try:
                    results_holder.extend(list(DDGS().text(query, max_results=5)))
                except Exception:
                    pass
            t = threading.Thread(target=_ddg, daemon=True)
            t.start()
            t.join(timeout=8)
            if results_holder:
                lines = [f"搜索 '{query}':"]
                for i, r in enumerate(results_holder[:5], 1):
                    lines.append(f"{i}. {r.get('title','')[:80]}\n   {r.get('href','')}\n   {r.get('body','')[:200]}")
                return "\n".join(lines)
        except ImportError:
            pass

        # Bing 搜索（主方案，国内稳定）
        try:
            import httpx
            r = httpx.get(
                "https://cn.bing.com/search",
                params={"q": query, "setlang": "zh-cn"},
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"},
                follow_redirects=True, timeout=5,
            )
            if r.status_code == 200:
                # 提取搜索结果
                results = []
                blocks = re.split(r'<li class="b_algo"', r.text)
                for block in blocks[1:6]:
                    title_m = re.search(r'<a[^>]*href="(https?://[^"]+)"[^>]*>(.+?)</a>', block)
                    if title_m:
                        url = title_m.group(1)
                        title = re.sub(r'<[^>]+>', '', title_m.group(2)).strip()
                        snippet_m = re.search(r'<p[^>]*>(.+?)</p>', block, re.DOTALL)
                        snippet = re.sub(r'<[^>]+>', '', snippet_m.group(1)).strip()[:200] if snippet_m else ""
                        results.append((title, url, snippet))
                if results:
                    lines = [f"Bing 搜索 '{query}':"]
                    for i, (title, url, snippet) in enumerate(results[:5], 1):
                        lines.append(f"{i}. {title}\n   {url}\n   {snippet}")
                    return "\n".join(lines)
        except ImportError:
            pass
        except Exception:
            pass

        # Tavily 备用
        tavily_key = os.environ.get("TAVILY_API_KEY") or os.environ.get("tavily_api_key")
        if tavily_key:
            try:
                import httpx
                resp = httpx.post("https://api.tavily.com/search", json={"api_key": tavily_key, "query": query, "max_results": 5}, timeout=15)
                if resp.status_code == 200:
                    data = resp.json()
                    results = data.get("results", [])
                    if results:
                        lines = [f"Tavily 搜索结果:"]
                        for i, r in enumerate(results[:5], 1):
                            lines.append(f"{i}. {r.get('title','')}\n   {r.get('url','')}\n   {r.get('content','')[:200]}")
                        return "\n".join(lines)
            except Exception:
                pass

        return f"搜索 '{query}' 无结果。DuckDuckGo/Bing/Tavily 均不可用。"

    @registry.register("fetch_url", "抓取指定 URL 的完整网页内容（Markdown 格式）。配合 search_web 使用：搜出结果→选 URL→fetch 全文。format 可选 markdown/text/html")
    def fetch_url(url: str, format: str = "markdown") -> str:
        import re, httpx

        try:
            r = httpx.get(
                url,
                headers={"User-Agent": "Mozilla/5.0 (compatible; MicroAgent/1.0)"},
                follow_redirects=True,
                timeout=15,
            )
            if r.status_code != 200:
                return f"[ERROR] HTTP {r.status_code}"

            content_type = r.headers.get("content-type", "")
            charset = "utf-8"
            m = re.search(r'charset=([^\s;]+)', content_type)
            if m:
                charset = m.group(1)
            html = r.content.decode(charset, errors="replace")

            if format == "html":
                return html[:8000]

            if format == "text":
                text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
                text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
                text = re.sub(r"<[^>]+>", " ", text)
                text = re.sub(r"\s+", " ", text).strip()
                return text[:6000]

            # Default: markdown
            try:
                import html2text
                h = html2text.HTML2Text()
                h.ignore_links = False
                h.ignore_images = True
                h.body_width = 0
                h.ignore_emphasis = False
                md = h.handle(html)
                return md[:6000]
            except ImportError:
                text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
                text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
                text = re.sub(r"<[^>]+>", " ", text)
                text = re.sub(r"\s+", " ", text).strip()
                return f"(html2text 未安装，返回纯文本)\n{text[:6000]}"

        except httpx.ConnectTimeout:
            return f"[ERROR] 连接 {url} 超时"
        except Exception as e:
            return f"[ERROR] {e}"

    @registry.register("calculate", "执行数学计算，支持加减乘除、幂运算、三角函数等")
    def calculate(expression: str) -> str:
        import math
        allowed = {"__builtins__": {}, **{k: getattr(math, k) for k in dir(math) if not k.startswith("_")}}
        try:
            result = eval(expression, allowed)
            return f"计算结果: {expression} = {result}"
        except Exception as e:
            return f"[ERROR] 计算失败: {e}"

    @registry.register("delegate_task", "委派子 Agent 执行独立子任务，可指定工具白名单。参数: task=任务描述, tools=允许的工具名列表(逗号分隔), 可选 max_steps=最大步骤数")
    def delegate_task(task: str, tools: str = "", max_steps: int = 5) -> str:
        """Spawn a SubAgent to handle a specific subtask"""
        from tools.registry import ToolRegistry
        from agent.subagent import SubAgent

        tool_names = [t.strip() for t in tools.split(",") if t.strip()] if tools else []
        sub_registry = ToolRegistry(safe_mode=False)

        if not tool_names:
            readable_tools = ["read_file", "list_dir", "calculate"]
            tool_names = readable_tools

        for name in tool_names:
            if name in registry._tools:
                # Copy existing tool to sub-registry
                func = registry._tools[name]
                meta = registry._tool_metadata.get(name, {})
                sub_registry._tools[name] = func
                sub_registry._tool_metadata[name] = meta

        if not sub_registry._tools:
            return "[ERROR] No valid tools available for sub-agent"

        sub = SubAgent(
            llm=llm,
            registry=sub_registry,
            prompt="你是子任务执行 Agent。用赋予的工具完成任务，然后给出简洁中文结果。",
            max_steps=max_steps,
        )
        try:
            result = sub.run(task)
            return f"[SubAgent完成] {result}"
        except Exception as e:
            return f"[ERROR] SubAgent 执行失败: {e}"
