# -*- coding: utf-8 -*-
"""多文件结构化补丁应用 — 对标 MiMo Code 的 apply_patch 工具

支持单次调用中跨多个文件进行 add/update/delete/move 操作。
"""
import os
import json
import shutil
from datetime import datetime


def apply_patch(patch_json: str) -> str:
    """应用结构化多文件补丁

    输入 JSON 数组，每个元素:
    {
        "action": "add" | "update" | "delete" | "move",
        "file": "目标文件路径",
        "content": "文件内容 (add/update)",
        "old_string": "要替换的文本 (update)",
        "new_string": "替换为 (update)",
        "new_file": "目标路径 (move)"
    }

    所有操作在内存中累积，全部解析成功后才写入磁盘。
    任一失败 → 全部回滚。
    """
    try:
        patches = json.loads(patch_json)
        if not isinstance(patches, list):
            return "[ERROR] patch 必须是 JSON 数组"
    except json.JSONDecodeError as e:
        return f"[ERROR] JSON 解析失败: {e}"

    # 阶段 1: 预检查 — 所有操作在内存中验证
    snapshots = {}  # file_path → (original_exists, original_content)
    results = []    # (file_path, new_content) for apply

    for i, p in enumerate(patches):
        action = p.get("action", "")
        file_path = p.get("file", "")

        if not file_path:
            return f"[ERROR] 补丁 {i}: 缺少 file 字段"

        exists = os.path.exists(file_path)
        orig_content = None

        if exists:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    orig_content = f.read()
            except Exception as e:
                return f"[ERROR] 补丁 {i}: 无法读取 {file_path}: {e}"

        snapshots[file_path] = (exists, orig_content)

        if action == "add":
            if exists:
                return f"[ERROR] 补丁 {i}: 文件已存在 {file_path}，用 update 代替 add"
            content = p.get("content", "")
            if not content:
                return f"[ERROR] 补丁 {i}: add 操作需要 content"
            results.append((file_path, content))

        elif action == "update":
            if not exists:
                return f"[ERROR] 补丁 {i}: 文件不存在 {file_path}"
            old = p.get("old_string", "")
            new = p.get("new_string", "")
            if not old:
                return f"[ERROR] 补丁 {i}: update 操作需要 old_string"
            count = orig_content.count(old)
            if count == 0:
                return f"[ERROR] 补丁 {i}: old_string 未在 {file_path} 中找到"
            if count > 1:
                return f"[ERROR] 补丁 {i}: old_string 在 {file_path} 中出现 {count} 次，需更多上下文"
            results.append((file_path, orig_content.replace(old, new, 1)))

        elif action == "delete":
            if not exists:
                return f"[ERROR] 补丁 {i}: 文件不存在 {file_path}"
            trash_dir = ".trash"
            os.makedirs(trash_dir, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup = os.path.join(trash_dir, f"{ts}_{os.path.basename(file_path)}")
            results.append((file_path, None, backup))  # None = delete, backup = trash path

        elif action == "move":
            if not exists:
                return f"[ERROR] 补丁 {i}: 源文件不存在 {file_path}"
            new_path = p.get("new_file", "")
            if not new_path:
                return f"[ERROR] 补丁 {i}: move 操作需要 new_file"
            if os.path.exists(new_path):
                return f"[ERROR] 补丁 {i}: 目标已存在 {new_path}"
            results.append((file_path, new_path))  # (old, new) for rename

        else:
            return f"[ERROR] 补丁 {i}: 未知操作 '{action}'，支持 add/update/delete/move"

    # 阶段 2: 全部预检通过 → 原子应用
    applied = []
    try:
        for item in results:
            if len(item) == 2:
                file_path, content = item
                if content is None:
                    # delete — 第三个元素会在下面处理
                    continue
                os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)
                applied.append(file_path)
            elif len(item) == 3 and item[1] is None:
                file_path, _, backup = item
                shutil.move(file_path, backup)
                applied.append(file_path)
            else:
                old, new = item
                os.makedirs(os.path.dirname(new) or ".", exist_ok=True)
                shutil.move(old, new)
                applied.append(old)

        actions = {a: 0 for a in ["add", "update", "delete", "move"]}
        for p in patches:
            actions[p["action"]] = actions.get(p["action"], 0) + 1
        summary = ", ".join(f"{k}: {v}" for k, v in actions.items() if v > 0)
        return f"OK: 应用 {len(patches)} 个补丁 ({summary})"

    except Exception as e:
        # 回滚已应用的更改
        for i, (file_path, _, _) in enumerate(results):
            if i >= len(applied):
                break
            snap = snapshots.get(file_path)
            if snap is None:
                continue
            existed, orig = snap
            try:
                if existed and orig is not None:
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(orig)
                elif not existed:
                    if os.path.exists(file_path):
                        os.remove(file_path)
            except Exception:
                pass
        return f"[ERROR] 补丁应用失败，已回滚: {e}"
