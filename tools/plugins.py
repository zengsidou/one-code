# -*- coding: utf-8 -*-
"""工具插件系统 — 从配置目录自动发现和加载自定义工具"""
import os
import importlib.util
from pathlib import Path

PLUGIN_DIRS = [
    os.path.expanduser("~/.onecode/tools"),
    ".onecode/tools",
]


def load_plugin_tools(registry):
    """从插件目录扫描并加载自定义工具"""
    loaded = 0
    for dir_path in PLUGIN_DIRS:
        d = Path(dir_path)
        if not d.exists():
            d.mkdir(parents=True, exist_ok=True)
            # 创建示例文件
            example = d / "_example.py"
            if not example.exists():
                example.write_text(
                    '# 自定义工具示例 — 复制此文件并改名来创建新工具\n'
                    'def register(registry):\n'
                    '    @registry.register("hello", "示例: 打招呼的工具")\n'
                    '    def hello(name: str = "World") -> str:\n'
                    '        return f"Hello, {name}!"\n',
                    encoding="utf-8",
                )
            continue

        for f in sorted(d.glob("*.py")):
            if f.name.startswith("_"):
                continue
            try:
                spec = importlib.util.spec_from_file_location(f"plugin_{f.stem}", f)
                if spec and spec.loader:
                    mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(mod)
                    if hasattr(mod, "register"):
                        mod.register(registry)
                        loaded += 1
            except Exception as e:
                print(f"  [PLUGIN] 加载 {f.name} 失败: {e}")

    if loaded:
        print(f"  [PLUGIN] 加载了 {loaded} 个自定义工具")
    return loaded
