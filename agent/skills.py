# -*- coding: utf-8 -*-
"""Skills 系统 — 从 ~/.onecode/skills/ 加载 SKILL.md 领域知识

对标 MiMo Code 的 skill 系统。
每个 skill 是一个目录，包含 SKILL.md（YAML frontmatter + markdown）。
Agent 自动根据任务关键词匹配相关 skill，注入 system prompt。
"""
import os
import yaml
from pathlib import Path
from typing import Any

SKILL_DIRS = [
    os.path.expanduser("~/.onecode/skills"),
    ".onecode/skills",
]


class Skill:
    def __init__(self, name: str, description: str, body: str, triggers: list[str], path: Path):
        self.name = name
        self.description = description
        self.body = body
        self.triggers = triggers
        self.path = path


class SkillLibrary:
    """Skills 管理器"""

    def __init__(self):
        self.skills: list[Skill] = []
        self._loaded = False

    def load_all(self):
        if self._loaded:
            return
        for dir_path in SKILL_DIRS:
            d = Path(dir_path)
            if not d.exists():
                d.mkdir(parents=True, exist_ok=True)
                continue
            for skill_dir in sorted(d.iterdir()):
                if not skill_dir.is_dir():
                    continue
                skill_md = skill_dir / "SKILL.md"
                if not skill_md.exists():
                    continue
                try:
                    skill = self._parse_skill(skill_md)
                    if skill:
                        self.skills.append(skill)
                except Exception:
                    pass
        self._loaded = True

    def match(self, task: str, top_k: int = 2) -> list[Skill]:
        """根据任务关键词匹配相关 skills"""
        task_lower = task.lower()
        scored = []
        for s in self.skills:
            hits = sum(1 for t in s.triggers if t.lower() in task_lower or t.lower() in s.name.lower())
            if hits > 0:
                scored.append((hits, s))
        scored.sort(key=lambda x: -x[0])
        return [s for _, s in scored[:top_k]]

    def to_prompt(self, skills: list[Skill]) -> str:
        """将匹配的 skills 转为 system prompt 注入"""
        if not skills:
            return ""
        parts = ["[Skills 参考 — 这些是已加载的领域知识，请遵循其中指导:]"]
        for s in skills:
            parts.append(f"\n## {s.name}\n{s.body[:800]}")
        return "\n".join(parts) + "\n[/Skills]\n"

    @staticmethod
    def _parse_skill(filepath: Path) -> Skill | None:
        text = filepath.read_text(encoding="utf-8")
        # Parse YAML frontmatter
        if text.startswith("---"):
            parts = text.split("---", 2)
            if len(parts) >= 3:
                fm = yaml.safe_load(parts[1]) or {}
                body = parts[2].strip()
                return Skill(
                    name=fm.get("name", filepath.parent.name),
                    description=fm.get("description", ""),
                    body=body,
                    triggers=fm.get("triggers", []) or [fm.get("name", "")],
                    path=filepath,
                )
        # Fallback: use first line as name
        first_line = text.split("\n")[0].lstrip("#").strip()
        return Skill(
            name=filepath.parent.name,
            description=first_line[:100],
            body=text[:2000],
            triggers=[filepath.parent.name],
            path=filepath,
        )


_skill_lib: SkillLibrary | None = None


def get_skills() -> SkillLibrary:
    global _skill_lib
    if _skill_lib is None:
        _skill_lib = SkillLibrary()
        _skill_lib.load_all()
    return _skill_lib
