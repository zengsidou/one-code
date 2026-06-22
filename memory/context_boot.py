# -*- coding: utf-8 -*-
"""上下文引导器 — 启动时自动加载项目/用户/会话三层记忆"""
import os
import json
import time
from agent.models import Message

PROJECT_MEMORY_PATH = os.path.join(os.path.expanduser("~"), ".local", "share", "mimocode", "memory", "projects", "global", "MEMORY.md")
USER_PROFILE_PATH = "./user_profile.json"
SESSION_STATE_PATH = "./session_state.json"


class ContextBootstrapper:
    def __init__(self):
        self._user_profile = self._load_json(USER_PROFILE_PATH)
        self._session_state = self._load_json(SESSION_STATE_PATH)

    def build_boot_context(self) -> list[Message]:
        """构建启动上下文消息，包含项目记忆 + 用户偏好 + 上次会话"""
        messages = []

        # 第一层：项目记忆
        project_ctx = self._load_project_memory()
        if project_ctx:
            messages.append(Message(
                role="system",
                content=(
                    "[项目记忆] 以下是当前项目的重要背景信息，请在回答时参考:\n\n"
                    + project_ctx
                )
            ))

        # 第二层：用户偏好
        user_ctx = self._build_user_context()
        if user_ctx:
            messages.append(Message(
                role="system",
                content=(
                    "[用户偏好] 以下是你应该遵守的用户个人偏好:\n\n"
                    + user_ctx
                )
            ))

        # 第三层：上次会话
        session_ctx = self._build_session_context()
        if session_ctx:
            messages.append(Message(
                role="system",
                content=(
                    "[上次会话] 以下是之前未完成的工作，你可以询问用户是否继续:\n\n"
                    + session_ctx
                )
            ))

        return messages

    def record_feedback(self, feedback: str):
        """记录用户反馈，更新偏好"""
        ts = time.strftime("%Y-%m-%d %H:%M")
        feedbacks = self._user_profile.get("feedbacks", [])
        feedbacks.append({"time": ts, "text": feedback})
        if len(feedbacks) > 20:
            feedbacks = feedbacks[-20:]
        self._user_profile["feedbacks"] = feedbacks
        self._save_json(USER_PROFILE_PATH, self._user_profile)

    def update_preference(self, key: str, value: str):
        """更新单项偏好"""
        prefs = self._user_profile.get("preferences", {})
        prefs[key] = value
        self._user_profile["preferences"] = prefs
        self._save_json(USER_PROFILE_PATH, self._user_profile)

    def save_session(self, task: str, files_touched: list[str]):
        """保存当前会话状态"""
        self._session_state = {
            "last_task": task,
            "last_files": files_touched,
            "last_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        self._save_json(SESSION_STATE_PATH, self._session_state)

    def _load_project_memory(self) -> str:
        """加载项目 MEMORY.md"""
        try:
            if os.path.exists(PROJECT_MEMORY_PATH):
                with open(PROJECT_MEMORY_PATH, encoding="utf-8") as f:
                    return f.read()[-8000:]  # 最近 8000 字
        except Exception:
            pass
        return ""

    def _build_user_context(self) -> str:
        """构建用户偏好上下文"""
        parts = []
        prefs = self._user_profile.get("preferences", {})
        if prefs:
            parts.append("偏好设置:")
            for k, v in prefs.items():
                parts.append(f"  - {k}: {v}")

        feedbacks = self._user_profile.get("feedbacks", [])
        if feedbacks:
            parts.append(f"\n近期反馈 (最近 {min(3, len(feedbacks))} 条):")
            for fb in feedbacks[-3:]:
                parts.append(f"  [{fb['time']}] {fb['text'][:200]}")

        if not parts:
            # 默认偏好
            parts.append("默认偏好:")
            parts.append("  - 语言: 中文为主")
            parts.append("  - 代码风格: Python, 简洁, 不添加多余注释")
            parts.append("  - 回复风格: 直接、简洁")
            parts.append("  - 不要主动提交代码")
        return "\n".join(parts)

    def _build_session_context(self) -> str:
        """构建上次会话上下文"""
        last_task = self._session_state.get("last_task", "")
        last_files = self._session_state.get("last_files", [])
        last_time = self._session_state.get("last_time", "")
        if not last_task:
            return ""
        parts = [f"上次时间: {last_time}", f"上次任务: {last_task}"]
        if last_files:
            parts.append(f"涉及文件: {', '.join(last_files[:5])}")
        return "\n".join(parts)

    @staticmethod
    def _load_json(path: str) -> dict:
        try:
            if os.path.exists(path):
                with open(path, encoding="utf-8") as f:
                    return json.load(f)
        except Exception:
            pass
        return {}

    @staticmethod
    def _save_json(path: str, data: dict):
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
