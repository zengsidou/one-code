# -*- coding: utf-8 -*-
"""可复用能力库 — 从复盘中提取的技能/策略/模式，跨任务复用

SkillLibrary 是 Agent 的"经验笔记本"。每次复盘发现可复用技能时，
存入这里。执行新任务时，按任务特征查询相关技能注入上下文。
"""
import json
import os
from datetime import datetime


class SkillLibrary:
    def __init__(self, filepath: str = "./skill_library.json"):
        self.filepath = filepath
        self.skills: list[dict] = []
        self._llm = None
        self._embeddings: dict[int, list[float]] = {}
        self._load()

    def _load(self):
        if os.path.exists(self.filepath):
            with open(self.filepath, "r", encoding="utf-8") as f:
                self.skills = json.load(f)

    def _save(self):
        os.makedirs(os.path.dirname(self.filepath) or ".", exist_ok=True)
        with open(self.filepath, "w", encoding="utf-8") as f:
            json.dump(self.skills, f, ensure_ascii=False, indent=2)

    def add_from_post_mortem(self, reflection: dict):
        """从复盘报告中提取并存入新技能

        Args:
            reflection: TaskPostMortem.reflect() 的返回结果
        """
        skill_data = reflection.get("new_skill_gained", {})
        name = skill_data.get("name", "").strip()
        if not name or not skill_data.get("reusable"):
            return None

        # 检查是否已存在同名技能 → 强化而非重复添加
        for s in self.skills:
            if s["name"] == name:
                s["reinforce_count"] = s.get("reinforce_count", 0) + 1
                s["last_used"] = datetime.now().isoformat()
                s["strength"] = min(1.0, s.get("strength", 0.5) + 0.1)
                self._save()
                return s

        skill = {
            "name": name,
            "description": skill_data.get("description", ""),
            "trigger": skill_data.get("trigger", ""),
            "steps": skill_data.get("steps", ""),
            "source_task": reflection.get("task_desc", "")[:120],
            "strength": 0.5,  # 初始强度
            "reinforce_count": 1,
            "created_at": datetime.now().isoformat(),
            "last_used": datetime.now().isoformat(),
        }
        self.skills.append(skill)
        self._save()
        return skill

    def reinforce(self, skill_name: str):
        """强化已有技能（使用后调用）"""
        for s in self.skills:
            if s["name"] == skill_name:
                s["reinforce_count"] = s.get("reinforce_count", 0) + 1
                s["strength"] = min(1.0, s.get("strength", 0.5) + 0.1)
                s["last_used"] = datetime.now().isoformat()
                self._save()
                return s
        return None

    def decay(self, min_strength: float = 0.1, decay_rate: float = 0.1):
        """技能衰减 — 长时间未用的技能逐渐弱化"""
        for s in self.skills:
            s["strength"] = max(min_strength, s.get("strength", 0.5) - decay_rate)
        self._save()

    def set_llm(self, llm):
        """设置LLM适配器，启用embedding相似度匹配"""
        self._llm = llm

    def query(self, task_desc: str, top_k: int = 3, min_strength: float = 0.2) -> list[dict]:
        """根据任务描述查询相关技能 — 优先用embedding相似度，fallback到关键词"""
        scored = []
        use_embedding = self._llm is not None

        if use_embedding:
            query_emb = self._get_embedding(task_desc[:200])

        for i, s in enumerate(self.skills):
            if s.get("strength", 0.5) < min_strength:
                continue

            if use_embedding and query_emb:
                # Compute or reuse embedding for this skill
                if i not in self._embeddings:
                    skill_text = f"{s['name']} {s.get('description','')} {s.get('trigger','')}"[:200]
                    emb = self._get_embedding(skill_text)
                    if emb:
                        self._embeddings[i] = emb
                skill_emb = self._embeddings.get(i)
                if skill_emb:
                    sim = self._cosine_similarity(query_emb, skill_emb)
                    if sim > 0.3:
                        scored.append((sim + s.get("strength", 0.5) * 0.3, s))
                else:
                    # Fallback to keyword for this skill
                    self._keyword_score(task_desc, s, scored)
            else:
                self._keyword_score(task_desc, s, scored)

        scored.sort(key=lambda x: -x[0])
        return [s[1] for s in scored[:top_k]]

    @staticmethod
    def _keyword_score(task_desc: str, skill: dict, scored: list):
        keywords = set(task_desc.lower().split())
        text = f"{skill['name']} {skill.get('description','')} {skill.get('trigger','')}".lower()
        kw_hits = sum(1 for kw in keywords if kw in text)
        # Also check bigrams for better phrase matching
        words = task_desc.lower().split()
        bigrams = set(f"{words[i]} {words[i+1]}" for i in range(len(words)-1))
        bg_hits = sum(1 for bg in bigrams if bg in text)
        score = kw_hits + bg_hits * 2
        if score > 0:
            scored.append((score + skill.get("strength", 0.5), skill))

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        if not a or not b:
            return 0.0
        dot = sum(x*y for x,y in zip(a,b))
        norm_a = sum(x*x for x in a) ** 0.5
        norm_b = sum(x*x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def _get_embedding(self, text: str) -> list[float] | None:
        try:
            return self._llm.embed(text)
        except Exception:
            return None

    def to_prompt_hint(self, skills: list[dict]) -> str:
        """技能提示（精简版）"""
        if not skills:
            return ""
        lines = ["\n\n## 已学技能\n"]
        for s in skills:
            lines.append(f"- **{s['name']}**(强度{s.get('strength',0.5):.1f}): {s.get('trigger','')[:60]}")
        return "\n".join(lines)

    def get_top_skills(self, n: int = 3) -> list[dict]:
        """获取强度最高的 N 个技能（不依赖查询匹配）"""
        sorted_skills = sorted(self.skills, key=lambda s: s.get("strength", 0.5), reverse=True)
        return sorted_skills[:n]

    def get_stats(self) -> dict:
        """获取能力库统计"""
        return {
            "total_skills": len(self.skills),
            "avg_strength": sum(s.get("strength", 0.5) for s in self.skills) / max(len(self.skills), 1),
            "strongest": sorted(self.skills, key=lambda s: s.get("strength", 0), reverse=True)[:3],
            "weakest": sorted(self.skills, key=lambda s: s.get("strength", 1))[:3],
        }
