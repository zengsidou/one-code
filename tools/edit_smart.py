# -*- coding: utf-8 -*-
"""智能编辑 — 多策略模糊匹配，对标 MiMo Code 的 9-replacer"""
import os
import difflib


def edit_smart(file_path: str, old_string: str, new_string: str) -> str:
    """智能文件编辑，支持多种匹配策略

    Strategy order:
      1. Exact match
      2. Line-trimmed (strip whitespace per line)
      3. Indentation-flexible (ignore leading whitespace)
      4. Whitespace-normalized (collapse all whitespace)
      5. Block-anchor (match first+last lines, fuzzy middle)
      6. Levenshtein similarity fallback

    Returns: (status: "ok"|"error", message: str)
    """
    if not os.path.exists(file_path):
        return f"[ERROR] File not found: {file_path}"

    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    # ── Strategy 1: Exact ──
    if old_string in content:
        count = content.count(old_string)
        if count > 1:
            return f"[ERROR] Found {count} matches. Provide more surrounding context to make it unique."
        new_content = content.replace(old_string, new_string, 1)
        _write_atomic(file_path, new_content)
        return f"OK: 精确替换 1 处"

    # ── Strategy 2: Line-trimmed ──
    result = _try_line_trimmed(content, old_string, new_string)
    if result:
        _write_atomic(file_path, result)
        return "OK: 去行尾空白后替换 1 处"

    # ── Strategy 3: Indentation-flexible ──
    result = _try_indent_flexible(content, old_string, new_string)
    if result:
        _write_atomic(file_path, result)
        return "OK: 忽略缩进差异后替换 1 处"

    # ── Strategy 4: Whitespace-normalized ──
    result = _try_ws_normalized(content, old_string, new_string)
    if result:
        _write_atomic(file_path, result)
        return "OK: 空白标准化后替换 1 处"

    # ── Strategy 5: Block-anchor ──
    result = _try_block_anchor(content, old_string, new_string)
    if result:
        _write_atomic(file_path, result)
        return "OK: 首尾锚点匹配替换 1 处"

    # ── Strategy 6: Levenshtein similarity ──
    suggestions = _find_similar_lines(content, old_string)
    if suggestions:
        return f"[ERROR] 未找到精确匹配。最接近的代码段:\n{suggestions}"

    return f"[ERROR] old_string 在文件中未找到匹配"


# ── matching strategies ──

def _try_line_trimmed(content: str, old: str, new: str) -> str | None:
    old_trimmed = "\n".join(line.rstrip() for line in old.split("\n"))
    content_lines = content.split("\n")
    content_trimmed = "\n".join(line.rstrip() for line in content_lines)
    if old_trimmed in content_trimmed:
        count = content_trimmed.count(old_trimmed)
        if count > 1:
            return None
        return content_trimmed.replace(old_trimmed, new, 1)
    return None


def _try_indent_flexible(content: str, old: str, new: str) -> str | None:
    def _bare(s: str) -> str:
        return "\n".join(line.lstrip() for line in s.split("\n"))
    old_bare = _bare(old)
    lines = content.split("\n")
    for i in range(len(lines)):
        for j in range(i, len(lines)):
            chunk = "\n".join(lines[i:j + 1])
            if _bare(chunk) == old_bare:
                # Build replacement preserving 1st line indentation
                indent = lines[i][:len(lines[i]) - len(lines[i].lstrip())]
                new_indented = "\n".join(
                    indent + line if k == 0 else line
                    for k, line in enumerate(new.split("\n"))
                )
                result_lines = lines[:i] + [new_indented] + lines[j + 1:]
                return "\n".join(result_lines)
    return None


def _try_ws_normalized(content: str, old: str, new: str) -> str | None:
    import re
    def _norm(s: str) -> str:
        return re.sub(r"\s+", " ", s).strip()
    old_norm = _norm(old)
    lines = content.split("\n")
    for i in range(len(lines)):
        for j in range(i, min(len(lines), i + 20)):
            chunk = "\n".join(lines[i:j + 1])
            if _norm(chunk) == old_norm:
                result_lines = lines[:i] + [new] + lines[j + 1:]
                return "\n".join(result_lines)
    return None


def _try_block_anchor(content: str, old: str, new: str) -> str | None:
    old_lines = old.strip().split("\n")
    if len(old_lines) < 3:
        return None
    first = old_lines[0].strip()
    last = old_lines[-1].strip()
    content_lines = content.split("\n")
    for i in range(len(content_lines)):
        if content_lines[i].strip() == first:
            for j in range(i + 1, min(len(content_lines), i + 30)):
                if content_lines[j].strip() == last and j - i >= len(old_lines) - 2:
                    result_lines = content_lines[:i] + [new] + content_lines[j + 1:]
                    return "\n".join(result_lines)
    return None


def _find_similar_lines(content: str, old: str) -> str:
    content_lines = content.split("\n")
    old_head = old.strip().split("\n")[0].strip() if old.strip() else old[:40]
    candidates = []
    for i, line in enumerate(content_lines):
        ratio = difflib.SequenceMatcher(None, old_head, line.strip()).ratio()
        if ratio > 0.4:
            start = max(0, i - 1)
            end = min(len(content_lines), i + 3)
            snippet = "\n".join(
                f"  {k+1}: {content_lines[k][:120]}"
                for k in range(start, end)
            )
            candidates.append(f"  行 {i+1} (相似度 {ratio:.0%}):\n{snippet}")
    if not candidates:
        return ""
    return "\n".join(candidates[:3])


def _write_atomic(path: str, content: str):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(content)
    os.replace(tmp, path)
