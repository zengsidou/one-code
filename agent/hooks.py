# -*- coding: utf-8 -*-
"""Hook 系统 — 对标 MiMo Code 的事件钩子

支持注册回调到工具执行前后、Agent 启动/停止等生命周期事件。
用户可在 ~/.onecode/hooks/ 下放置 .py 文件注册钩子。
"""
import os
import importlib.util
from pathlib import Path


HOOK_DIRS = [
    os.path.expanduser("~/.onecode/hooks"),
    ".onecode/hooks",
]


class HookRegistry:
    """事件钩子注册中心"""

    def __init__(self):
        self._hooks: dict[str, list] = {
            "tool.before": [],
            "tool.after": [],
            "agent.start": [],
            "agent.stop": [],
            "agent.error": [],
        }

    def register(self, event: str, callback):
        if event in self._hooks:
            self._hooks[event].append(callback)

    def fire(self, event: str, **kwargs):
        for cb in self._hooks.get(event, []):
            try:
                cb(**kwargs)
            except Exception:
                pass  # Hooks must not crash the agent

    def load_user_hooks(self):
        """从 ~/.onecode/hooks/ 加载用户自定义钩子"""
        for dir_path in HOOK_DIRS:
            d = Path(dir_path)
            if not d.exists():
                continue
            for f in sorted(d.glob("*.py")):
                try:
                    spec = importlib.util.spec_from_file_location(f.stem, f)
                    if spec and spec.loader:
                        mod = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(mod)
                        if hasattr(mod, "register"):
                            mod.register(self)
                except Exception:
                    pass


# 全局单例
_hooks = HookRegistry()


def get_hooks() -> HookRegistry:
    return _hooks
