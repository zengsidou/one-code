# -*- coding: utf-8 -*-
"""架构自进化模块 — Agent 读自己的代码，识别架构瓶颈，生成改动方案

当所有 Config 层优化（prompt/参数/阈值）都已穷尽但任务仍然失败时，
瓶颈在 Architecture 层。此模块让 Agent：
1. 读自己的源代码
2. 分析为什么当前架构无法处理某类任务
3. 生成具体的代码改动方案
4. 安全应用 + 验证 + 回滚
"""
import os
import json
import shutil
from datetime import datetime
from agent.models import Message

BOTTLENECK_DIAGNOSIS_PROMPT = """你是一个 Agent 架构师。你的 Agent 系统在以下任务上多次失败，
且所有参数级优化（改 prompt、调阈值、修工具描述）都已穷尽。
瓶颈在架构层面：当前代码结构无法提供足够的能力。

## 失败症状
{ symptoms }

## 当前架构摘要（各模块职责）
{ current_arch }

## 任务需求 vs 架构能力缺口
{ capability_gap }

## 要求
判断最关键的架构瓶颈是什么。只从以下选择：
- context_too_small: 上下文窗口太小，装不下大项目的文件内容
- no_task_decomposition: 无法将复杂任务拆成子任务
- no_execution_loop: 写代码后不验证，缺少 run→error→fix 循环
- single_agent_limit: 单 Agent 无法并行处理多文件
- no_shared_memory: Worker 之间无法共享上下文
- tool_too_simple: 现有工具太粗糙，需要更高层工具

同时：需要修改哪个文件来突破瓶颈？从以下选择：
- agent/loop.py (主循环)
- memory/short_term.py (短期记忆)
- agent/orchestrator.py (任务编排)
- agent/subagent.py (子Agent)
- tools/builtin/__init__.py (内置工具)

输出严格 JSON。"""

ARCHITECTURE_PROPOSAL_PROMPT = """你是一个 Agent 架构师。你需要修改 Agent 的源代码来突破架构瓶颈。

## 瓶颈
{ bottleneck }

## 当前代码 ({ file_path })
```python
{ file_content }
```

## 能力缺口
{ capability_gap }

## 要求
生成这个文件的具体修改方案。以 JSON 格式输出：
{{
  "file_path": "要修改的文件路径",
  "rationale": "为什么做这个修改",
  "change_type": "add_class / add_method / modify_method / add_import",
  "target_location": "在哪个位置插入或替换（描述行号或函数名）",
  "new_code": "完整的新代码片段（注意缩进）",
  "old_code_hint": "需要被替换的旧代码片段（如为新增则为空）",
  "expected_effect": "预期这个改动如何突破瓶颈"
}}

只输出 JSON，不要任何解释文字。"""


class ArchitectureBottleneckDetector:
    """架构瓶颈识别器

    当连续多轮自优化都无法产生有效修复时，诊断是否到了架构瓶颈。
    """

    def __init__(self, llm_adapter):
        self.llm = llm_adapter
        self.failure_log: list[dict] = []

    def record_failure(self, task_desc: str, optimize_reports: list[dict]):
        """记录一次优化失败的尝试"""
        self.failure_log.append({
            "task_desc": task_desc[:200],
            "reports": [{
                "analyzed": r.get("analyzed", 0),
                "kept": r.get("fixes_kept", 0),
                "rolled": r.get("fixes_rolled_back", 0),
            } for r in optimize_reports],
            "timestamp": datetime.now().isoformat(),
        })

    def is_bottleneck(self, min_consecutive_failures: int = 3) -> bool:
        """判断是否已到架构瓶颈"""
        if len(self.failure_log) < min_consecutive_failures:
            return False
        recent = self.failure_log[-min_consecutive_failures:]
        for log in recent:
            total_kept = sum(r.get("kept", 0) for r in log.get("reports", []))
            if total_kept > 0:
                return False
        return True

    def diagnose(self, agent_instance, capability_gap: str = "") -> dict:
        """诊断架构瓶颈：用 LLM 分析为什么当前架构不够

        Args:
            agent_instance: AgentLoop 实例
            capability_gap: 手动描述的能力缺口

        Returns:
            { bottleneck_type, target_file, rationale, capability_gap }
        """
        symptoms_lines = []
        for log in self.failure_log[-5:]:
            symptoms_lines.append(f"- 任务: {log.get('task_desc', '')[:100]}")
            for i, r in enumerate(log.get("reports", [])):
                symptoms_lines.append(f"  优化{i+1}: analyzed={r['analyzed']} kept={r['kept']}")

        symptoms_text = "\n".join(symptoms_lines[-20:]) or "(无失败记录)"

        arch_text = (
            "agent/loop.py: ReAct 循环 + 熔断 + 回路检测\n"
            "memory/short_term.py: Token 感知滑动窗口裁剪\n"
            "agent/orchestrator.py: 多 Agent 扇出/流水线调度\n"
            "agent/subagent.py: 轻量子 Agent 委派\n"
            "tools/builtin/: read_file, write_file, run_shell, search_web 等\n"
            "agent/evolve/: 复盘反思 + 技能库 + 能力画像 + 挑战生成"
        )

        gap_text = capability_gap or "Agent 无法完成需要多文件协调、复杂依赖或长上下文的任务"

        prompt = (BOTTLENECK_DIAGNOSIS_PROMPT
            .replace("{ symptoms }", symptoms_text)
            .replace("{ current_arch }", arch_text)
            .replace("{ capability_gap }", gap_text))

        try:
            resp = self.llm.generate(
                [Message(role="system", content="你是一个 Agent 架构师。只输出 JSON。"),
                 Message(role="user", content=prompt)],
                tools=None,
            )
            return self._parse(json.loads(self._clean_json(resp.content or "")))
        except Exception as e:
            return {
                "bottleneck_type": "context_too_small",
                "target_file": "memory/short_term.py",
                "rationale": f"自动诊断失败: {e}",
                "capability_gap": gap_text,
                "confidence": 0.3,
            }

    @staticmethod
    def _parse(raw: dict) -> dict:
        valid_types = {
            "context_too_small", "no_task_decomposition", "no_execution_loop",
            "single_agent_limit", "no_shared_memory", "tool_too_simple",
        }
        valid_files = {
            "agent/loop.py", "memory/short_term.py", "agent/orchestrator.py",
            "agent/subagent.py", "tools/builtin/__init__.py",
        }
        bt = raw.get("bottleneck_type", "context_too_small")
        tf = raw.get("target_file", "memory/short_term.py")
        conf = float(raw.get("confidence", 0.85))
        return {
            "bottleneck_type": bt if bt in valid_types else "context_too_small",
            "target_file": tf if tf in valid_files else "memory/short_term.py",
            "rationale": raw.get("rationale", ""),
            "capability_gap": raw.get("capability_gap", ""),
            "confidence": conf,
        }

    @staticmethod
    def _clean_json(text: str) -> str:
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:]) if len(lines) > 1 else text[3:]
        if text.endswith("```"):
            text = text[:-3]
        return text


class ArchitectureProposalGenerator:
    """架构改动方案生成器

    读取 Agent 自己的源代码，用 LLM 生成具体的代码改动方案。
    """

    def __init__(self, llm_adapter, source_root: str | None = None):
        self.llm = llm_adapter
        self.source_root = source_root or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def generate_proposal(self, bottleneck: dict) -> dict | None:
        """生成架构改动方案"""
        file_path = bottleneck.get("target_file", "memory/short_term.py")
        
        # Try multiple paths: absolute, relative to source_root, relative to project root
        candidates = [
            file_path,  # absolute or relative to cwd
            os.path.join(self.source_root, file_path),  # relative to agent/evolve
            os.path.join(os.path.dirname(self.source_root), file_path),  # relative to agent/
            os.path.join(os.path.dirname(os.path.dirname(self.source_root)), file_path),  # relative to project/
        ]
        full_path = None
        for p in candidates:
            if os.path.exists(p):
                full_path = p
                break

        if full_path is None:
            return None

        try:
            with open(full_path, "r", encoding="utf-8") as f:
                file_content = f.read()
        except Exception:
            return None

        prompt = (ARCHITECTURE_PROPOSAL_PROMPT
            .replace("{ bottleneck }", json.dumps(bottleneck, ensure_ascii=False, indent=2))
            .replace("{ file_path }", file_path)
            .replace("{ file_content }", file_content[-8000:])
            .replace("{ capability_gap }", bottleneck.get("capability_gap", "")))

        try:
            resp = self.llm.generate(
                [Message(role="system", content="你是一个 Agent 架构师。只输出 JSON，不要任何解释。"),
                 Message(role="user", content=prompt)],
                tools=None,
            )
            text = resp.content or ""
            text = ArchitectureBottleneckDetector._clean_json(text)
            proposal = json.loads(text)
            proposal["full_path"] = full_path
            return proposal
        except Exception:
            return None


class ArchitectureApplier:
    """架构改动安全应用器

    支持备份 → 应用 → 重载 → 验证 → 回滚的完整流程。
    """

    # 文件路径 → Python 模块名映射
    MODULE_MAP = {
        "memory/short_term.py": "memory.short_term",
        "agent/loop.py": "agent.loop",
        "agent/orchestrator.py": "agent.orchestrator",
        "agent/subagent.py": "agent.subagent",
        "tools/builtin/__init__.py": "tools.builtin",
        "tools/registry.py": "tools.registry",
    }

    # 文件 → agent 组件属性映射
    COMPONENT_MAP = {
        "memory/short_term.py": ("memory", "short_term"),
        "agent/loop.py": (None, None),  # loop 自身，需特殊处理
    }

    def __init__(self, backup_dir: str = "./arch_backups"):
        self.backup_dir = backup_dir
        os.makedirs(backup_dir, exist_ok=True)
        self.applied: list[dict] = []

    def backup(self, file_path: str) -> str:
        """备份文件"""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{os.path.basename(file_path)}.{ts}.bak"
        backup_path = os.path.join(self.backup_dir, backup_name)
        shutil.copy2(file_path, backup_path)
        return backup_path

    def apply(self, proposal: dict) -> bool:
        """应用架构改动"""
        file_path = proposal.get("full_path", "")
        old_code = proposal.get("old_code_hint", "")
        new_code = proposal.get("new_code", "")

        if not file_path or not os.path.exists(file_path) or not new_code:
            return False

        try:
            backup_path = self.backup(file_path)
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            if old_code and old_code in content:
                content = content.replace(old_code, new_code, 1)
            else:
                # Try smart insertion: find target function/class and insert new code
                target = proposal.get("target_location", "")
                if target and target in content:
                    # Insert after the target line
                    idx = content.index(target) + len(target)
                    content = content[:idx] + "\n" + new_code + content[idx:]
                else:
                    content = content.rstrip() + "\n\n" + new_code + "\n"

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)

            self.applied.append({
                "file_path": file_path,
                "backup": backup_path,
                "proposal": proposal,
                "timestamp": datetime.now().isoformat(),
            })
            return True
        except Exception:
            return False

    def apply_and_reload(self, proposal: dict, agent_instance) -> bool:
        """应用架构改动并动态重载受影响的模块

        Args:
            proposal: 代码改动方案
            agent_instance: AgentLoop 实例

        Returns:
            是否成功
        """
        if not self.apply(proposal):
            return False

        file_path = proposal.get("full_path", "")
        rel_path = self._get_relative_path(file_path)

        # 导到模块名并重载
        module_name = self.MODULE_MAP.get(rel_path)
        if module_name:
            try:
                import importlib
                mod = importlib.import_module(module_name)
                importlib.reload(mod)
            except Exception:
                pass

        # 更新 agent 中的组件引用
        comp_info = self.COMPONENT_MAP.get(rel_path)
        if comp_info:
            parent_attr, child_attr = comp_info
            if parent_attr and child_attr:
                parent = getattr(agent_instance, parent_attr, None)
                if parent and hasattr(parent, child_attr):
                    # Recreate the component with updated code
                    old = getattr(parent, child_attr)
                    try:
                        new_instance = type(old)(max_tokens=old.max_tokens)
                        setattr(parent, child_attr, new_instance)
                    except Exception:
                        pass

        return True

    @staticmethod
    def _get_relative_path(file_path: str) -> str:
        """从绝对路径提取相对路径"""
        parts = file_path.replace("\\", "/").split("/")
        for i, p in enumerate(parts):
            if p in ("agent", "memory", "tools", "llm", "sandbox", "mcp"):
                return "/".join(parts[i:])
        return file_path

    def rollback_last(self) -> bool:
        """回滚最近一次改动"""
        if not self.applied:
            return False
        last = self.applied.pop()
        try:
            # Find and restore the latest backup
            base = os.path.basename(last["file_path"])
            backups = sorted(
                [f for f in os.listdir(self.backup_dir) if f.startswith(base)],
                reverse=True,
            )
            if backups:
                shutil.copy2(
                    os.path.join(self.backup_dir, backups[0]),
                    last["file_path"],
                )
                return True
        except Exception:
            pass
        return False


class ArchitectureValidator:
    """架构改动验证器

    用一组基准任务验证架构改动是否有效。
    """

    def __init__(self):
        self.benchmarks: list[str] = []

    def add_benchmark(self, task: str):
        """添加基准任务"""
        self.benchmarks.append(task)

    def validate(self, agent_instance) -> dict:
        """运行基准任务验证改动效果

        Returns:
            { total, success_count, success_rate, improved (bool) }
        """
        if not self.benchmarks:
            return {"total": 0, "success_count": 0, "success_rate": 0, "improved": False}

        success = 0
        for task in self.benchmarks:
            try:
                result = agent_instance.run(task)
                if "[STOPPED]" not in result and "[LLM error]" not in result:
                    success += 1
            except Exception:
                pass

        rate = success / len(self.benchmarks)
        return {
            "total": len(self.benchmarks),
            "success_count": success,
            "success_rate": round(rate, 2),
            "improved": rate >= 0.5,
        }
