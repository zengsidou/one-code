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
- prompt_inadequate: 系统提示不够好，导致 agent 对话质量差、缺乏协作能力、回答方式不恰当

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
  "change_type": "add_class / add_method / modify_method / add_import / add_tool",
  "target_location": "在哪个位置插入或替换（描述行号或函数名）",
  "new_code": "完整的新代码片段（注意缩进）。如果是新工具函数，必须包含 @registry.register 装饰器，格式见现有工具",
  "old_code_hint": "需要被替换的旧代码片段（如为新增则为空）",
  "expected_effect": "预期这个改动如何突破瓶颈"
}}

只输出 JSON，不要任何解释文字。"""

TOOL_GENERATION_HINT = """
注意：你在为 Agent 添加新工具函数。请遵循以下规范：
1. 函数需要 return str 类型
2. 使用 @registry.register("tool_name", "工具描述") 装饰器
3. 函数参数需要类型标注 (str, int, bool 等)
4. 工具描述要清晰说明用途和参数
5. 参考文件中现有工具的格式
"""

PROMPT_OPTIMIZATION_PROMPT = """你是 Agent 系统提示优化专家。当前的系统提示存在不足，需要改进。

## 当前系统提示
```
{ current_prompt }
```

## 问题反馈
{ feedback }

## 修改要求
生成改进后的完整系统提示。要求：
1. 保留原版所有有效的规则和能力（核心工作流、行为准则、代码规范、Git 安全、工具策略）
2. 根据问题反馈，在对话与协作能力区补充或调整指令
3. 输出格式就是最终的 Python 字符串，不要加 def、类、或任何代码包装
4. 保持中文为主，关键术语用英文

只输出改进后的完整提示文本，不要任何解释。"""


class ArchitectureBottleneckDetector:
    """架构瓶颈识别器

    当连续多轮自优化都无法产生有效修复时，诊断是否到了架构瓶颈。
    """

    def __init__(self, llm_adapter):
        self.llm = llm_adapter
        self.failure_log: list[dict] = []

    def record_failure(self, task_desc: str, optimize_reports: list[dict], failure_cases: list[dict] | None = None):
        """记录一次优化失败的尝试"""
        self.failure_log.append({
            "task_desc": task_desc[:200],
            "reports": [{
                "analyzed": r.get("analyzed", 0),
                "kept": r.get("fixes_kept", 0),
                "rolled": r.get("fixes_rolled_back", 0),
            } for r in optimize_reports],
            "cases": failure_cases or [],
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
        """诊断架构瓶颈"""
        symptoms_lines = []
        unknown_tool_count = 0
        tool_call_patterns: dict[str, int] = {}
        error_patterns: dict[str, int] = {}
        
        for log in self.failure_log[-5:]:
            task = log.get("task_desc", "")[:100]
            symptoms_lines.append(f"- 任务: {task}")
            for i, r in enumerate(log.get("reports", [])):
                symptoms_lines.append(f"  优化{i+1}: analyzed={r['analyzed']} kept={r['kept']}")
            # 统计工具调用模式 — 从所有上下文消息中提取
            for case in log.get("cases", []):
                err = case.get("error_msg", "")
                if "Unknown tool" in err:
                    unknown_tool_count += 1
                for msg in case.get("context_snapshot", []):
                    content = str(msg.get("content", ""))
                    # 统计 ERROR 出现次数
                    if "ERROR" in content:
                        error_patterns["ERROR"] = error_patterns.get("ERROR", 0) + 1
                    if "调用工具" in content:
                        error_patterns["调用工具"] = error_patterns.get("调用工具", 0) + 1
                    # 提取工具名
                    import re
                    matches = re.findall(r"调用工具:\s*(\w+)", content)
                    for m in matches:
                        tool_call_patterns[m] = tool_call_patterns.get(m, 0) + 1

        symptoms_text = "\n".join(symptoms_lines[-20:]) or "(无失败记录)"
        
        # 工具调用分析 — 辅助区分瓶颈类型
        if tool_call_patterns:
            top_tools = sorted(tool_call_patterns.items(), key=lambda x: -x[1])[:5]
            symptoms_text += f"\n\n工具调用统计:\n"
            for name, count in top_tools:
                symptoms_text += f"  - {name}: {count}次\n"
            # 诊断提示
            if len(tool_call_patterns) == 1 and list(tool_call_patterns.values())[0] >= 4:
                symptoms_text += "\n[推断] 仅依赖1个工具且重复≥4次 → 可能是 no_execution_loop 或 tool_too_simple\n"
            elif len(tool_call_patterns) >= 3:
                symptoms_text += "\n[推断] 尝试了多种工具但均失败 → 可能缺少调试闭环 (no_execution_loop)\n"
        
        if error_patterns:
            symptoms_text += f"\n错误模式统计: {error_patterns}"
            if error_patterns.get("ERROR", 0) >= 5:
                symptoms_text += "\n[推断] 大量ERROR但仍不读错误输出 → 大概率 no_execution_loop (无 read→error→fix 循环)\n"

        gap_text = capability_gap or "Agent 无法完成需要多文件协调、复杂依赖或长上下文的任务"

        # 快速判定：Unknown tool → tool_too_simple
        if unknown_tool_count >= 1:
            return {
                "bottleneck_type": "tool_too_simple",
                "target_file": "tools/builtin/__init__.py",
                "rationale": f"Agent 多次尝试调用不存在的工具({unknown_tool_count}次)，当前工具集不足",
                "capability_gap": gap_text,
                "confidence": 0.95,
            }

        # 快速判定：大量 delegate_task 调用 → single_agent_limit
        delegate_count = tool_call_patterns.get("delegate_task", 0)
        if delegate_count >= 3:
            return {
                "bottleneck_type": "single_agent_limit",
                "target_file": "agent/loop.py",
                "rationale": f"Agent 多次尝试委派子任务({delegate_count}次)但始终无法完成，单Agent无法处理大规模并行工作",
                "capability_gap": gap_text,
                "confidence": 0.90,
            }
        
        # 快速判定：仅1个工具失败大量+大量ERROR → no_execution_loop
        if (len(tool_call_patterns) <= 2 
            and error_patterns.get("ERROR", 0) >= 5
            and error_patterns.get("调用工具", 0) >= 5):
            return {
                "bottleneck_type": "no_execution_loop",
                "target_file": "agent/loop.py",
                "rationale": f"Agent 在大量工具ERROR后({error_patterns.get('ERROR',0)}次)从未尝试读错误输出或修正策略，缺乏 write→test→read_error→fix 的执行闭环",
                "capability_gap": gap_text,
                "confidence": 0.90,
            }

        arch_text = (
            "agent/loop.py: ReAct 循环 + 熔断 + 回路检测\n"
            "memory/short_term.py: Token 感知滑动窗口裁剪\n"
            "agent/orchestrator.py: 多 Agent 扇出/流水线调度\n"
            "agent/subagent.py: 轻量子 Agent 委派\n"
            "tools/builtin/: read_file, write_file, run_shell, search_web 等\n"
            "agent/evolve/: 复盘反思 + 技能库 + 能力画像 + 挑战生成"
        )

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
            "prompt_inadequate",
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

        # ━━━ 系统提示自优化 ━━━
        if bottleneck.get("bottleneck_type") == "prompt_inadequate":
            return self._generate_prompt_optimization(bottleneck, full_path)

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

        # 追加重试提示（如果之前尝试失败了）
        retry_hint = bottleneck.get("retry_hint", "")
        if retry_hint:
            prompt += f"\n\n## ⚠️ 重试提示\n上次方案导致测试失败，请换思路。\n{retry_hint}"

        # 追加工具生成规范提示
        if "builtin" in file_path or bottleneck.get("bottleneck_type") == "tool_too_simple":
            prompt += TOOL_GENERATION_HINT

        system_msg = "你是一个 Agent 架构师。只输出 JSON，不要任何解释。"
        if "tool_too_simple" == bottleneck.get("bottleneck_type"):
            system_msg = "你是一个 Agent 工具开发专家。为 Agent 生成新的工具函数。只输出 JSON。"

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

    def _generate_prompt_optimization(self, bottleneck: dict, full_path: str) -> dict | None:
        """生成系统提示优化方案"""
        import re

        try:
            with open(full_path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception:
            return None

        m = re.search(r'DEFAULT_SYSTEM_PROMPT = \((.*?)\)', content, re.DOTALL)
        if not m:
            return None
        current_prompt = m.group(1)

        feedback = bottleneck.get("capability_gap", "Prompt needs improvement for better dialogue and collaboration.")

        prompt = PROMPT_OPTIMIZATION_PROMPT.replace("{ current_prompt }", current_prompt[:4000])
        prompt = prompt.replace("{ feedback }", feedback)

        try:
            resp = self.llm.generate(
                [Message(role="system", content="你是 Agent 系统提示优化专家。只输出改进后的提示文本。"),
                 Message(role="user", content=prompt)],
                tools=None,
            )
            new_prompt = (resp.content or "").strip()
            if not new_prompt or len(new_prompt) < 200:
                return None
        except Exception:
            return None

        old_block = f"DEFAULT_SYSTEM_PROMPT = ({current_prompt})"
        new_block = f"DEFAULT_SYSTEM_PROMPT = ({new_prompt})"

        return {
            "full_path": full_path,
            "old_code_hint": old_block,
            "new_code": new_block,
            "change_type": "modify_prompt",
            "rationale": f"系统提示优化: {bottleneck.get('rationale', '')[:100]}",
            "target_location": "DEFAULT_SYSTEM_PROMPT",
        }


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
        "tools/builtin/__init__.py": ("registry", None),  # 需重新注册所有工具
    }

    def __init__(self, backup_dir: str = "./arch_backups"):
        self.backup_dir = backup_dir
        os.makedirs(backup_dir, exist_ok=True)
        self.applied: list[dict] = []
        self._pre_reload_functions: dict[str, str] = {}
        self._last_test_output = ""
        self._test_passed = False

    def backup(self, file_path: str) -> str:
        """备份文件"""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{os.path.basename(file_path)}.{ts}.bak"
        backup_path = os.path.join(self.backup_dir, backup_name)
        shutil.copy2(file_path, backup_path)
        return backup_path

    def apply(self, proposal: dict) -> bool:
        """应用架构改动（带语法安全门）"""
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
                new_content = content.replace(old_code, new_code, 1)
            else:
                target = proposal.get("target_location", "")
                if target and target in content:
                    idx = content.index(target) + len(target)
                    new_content = content[:idx] + "\n" + new_code + content[idx:]
                else:
                    new_content = content.rstrip() + "\n\n" + new_code + "\n"

            # ━━━ 安全门：语法检查 ━━━
            if not self._validate_syntax(file_path, new_content):
                return False

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(new_content)

            self.applied.append({
                "file_path": file_path,
                "backup": backup_path,
                "proposal": proposal,
                "timestamp": datetime.now().isoformat(),
            })
            return True
        except Exception:
            return False

    @staticmethod
    def _validate_syntax(file_path: str, content: str) -> bool:
        """语法安全门：Python 文件用 py_compile 检查，JSON 文件用 json.loads 检查"""
        ext = os.path.splitext(file_path)[1].lower()
        try:
            if ext == ".py":
                import py_compile
                import tempfile
                with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as tmp:
                    tmp.write(content)
                    tmp_path = tmp.name
                try:
                    py_compile.compile(tmp_path, doraise=True)
                    return True
                except py_compile.PyCompileError as e:
                    print(f"  [L4-GATE] 语法检查失败，拒绝应用: {e}")
                    return False
                finally:
                    os.unlink(tmp_path)
            elif ext == ".json":
                json.loads(content)
                return True
            else:
                return True
        except Exception as e:
            print(f"  [L4-GATE] 验证失败: {e}")
            return False

    @staticmethod
    def _recreate_instance(new_class: type, old_instance) -> object | None:
        """用新加载的类重建实例，保留旧实例的状态。

        不使用硬编码构造参数；而是遍历旧实例的公有属性，
        尝试以对应参数调用新类的 __init__，如果失败则退回到浅拷贝 __dict__。
        """
        import inspect
        try:
            sig = inspect.signature(new_class.__init__)
            params = sig.parameters
            kwargs = {}
            for name in params:
                if name == "self":
                    continue
                if hasattr(old_instance, name):
                    kwargs[name] = getattr(old_instance, name)
                elif params[name].default is not inspect.Parameter.empty:
                    kwargs[name] = params[name].default
            return new_class(**kwargs)
        except Exception:
            pass

        try:
            inst = new_class.__new__(new_class)
            inst.__dict__.update({
                k: v for k, v in old_instance.__dict__.items()
                if not k.startswith("_")
            })
            return inst
        except Exception:
            return None

    def apply_and_reload(self, proposal: dict, agent_instance, run_tests: bool = False) -> dict:
        """应用架构改动并动态重载受影响的模块

        Args:
            proposal: 代码改动方案
            agent_instance: AgentLoop 实例
            run_tests: 是否在应用后跑自测试验证

        Returns:
            {"success": True/False, "test_result": "passed"/"failed"/"skipped", "test_output": "..."}
        """
        result = {"success": False, "test_result": "skipped", "test_output": ""}

        if not self.apply(proposal):
            result["error"] = "apply_failed"
            return result

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
                    old = getattr(parent, child_attr)
                    new_instance = self._recreate_instance(type(old), old)
                    if new_instance is not None:
                        setattr(parent, child_attr, new_instance)
            elif parent_attr == "registry" and child_attr is None:
                try:
                    from tools.builtin import register_builtin_tools
                    saved_tools = dict(agent_instance.registry._tools)
                    saved_metadata = dict(agent_instance.registry._tool_metadata)
                    agent_instance.registry._tools.clear()
                    agent_instance.registry._tool_metadata.clear()
                    register_builtin_tools(agent_instance.registry)
                except Exception:
                    agent_instance.registry._tools.update(saved_tools)
                    agent_instance.registry._tool_metadata.update(saved_metadata)

        # ━━━ 自测试闭环：跑 pytest 验证改动没破坏任何东西 ━━━
        if run_tests:
            test_passed, test_output = self._run_tests()
            self._test_passed = test_passed
            self._last_test_output = test_output
            if not test_passed:
                self.rollback_last()
                result["test_result"] = "failed"
                result["test_output"] = test_output[:500]
                result["error"] = "test_failed"
                return result
            result["test_result"] = "passed"
            result["test_output"] = test_output[:200]

        result["success"] = True
        return result

    @staticmethod
    def _run_tests() -> tuple:
        """运行 pytest 验证代码改动是否破坏现有功能

        Returns:
            (passed: bool, output: str)
        """
        import subprocess
        try:
            proc = subprocess.run(
                ["python", "-m", "pytest", "tests/", "-q", "--tb=line"],
                capture_output=True, timeout=60,
                cwd=os.getcwd(),
            )
            raw = proc.stdout + proc.stderr
            output = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else str(raw)
            return proc.returncode == 0, output
        except FileNotFoundError:
            return False, "pytest not found"
        except subprocess.TimeoutExpired:
            return False, "pytest 超时 (60s)"
        except Exception as e:
            return False, f"pytest 执行异常: {e}"

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
    验证标准：功能等价性 — 改动前后的输出结构一致、关键行为保持。
    """

    # 用于判断输出是否实质为空/退化的哨兵模式
    _DEGRADED_PATTERNS = [
        r"^[重复\s]*$",
        r"^error[:\s]",
        r"^traceback",
        r"^exception",
        r"^\s*$",
    ]

    def __init__(self):
        self.benchmarks: list[str] = []
        self._baseline: dict[str, str] = {}

    def add_benchmark(self, task: str):
        """添加基准任务"""
        self.benchmarks.append(task)

    def capture_baseline(self, agent_instance) -> None:
        """记录改动前的基准输出"""
        self._baseline = {}
        for task in self.benchmarks:
            try:
                result = agent_instance.run(task)
                self._baseline[task] = result
            except Exception:
                self._baseline[task] = ""

    def validate(self, agent_instance) -> dict:
        """运行基准任务验证改动效果

        Returns:
            { total, success_count, success_rate, improved (bool),
              failures: [{task, reason}] }
        """
        if not self.benchmarks:
            return {"total": 0, "success_count": 0, "success_rate": 0, "improved": False}

        failures = []
        for task in self.benchmarks:
            try:
                result = agent_instance.run(task)
                reason = self._check_degraded(task, result)
                if reason:
                    failures.append({"task": task, "reason": reason})
            except Exception as e:
                failures.append({"task": task, "reason": f"Exception: {e}"})

        success = len(self.benchmarks) - len(failures)
        rate = success / len(self.benchmarks)
        return {
            "total": len(self.benchmarks),
            "success_count": success,
            "success_rate": round(rate, 2),
            "improved": rate >= 0.5,
            "failures": failures,
        }

    def _check_degraded(self, task: str, result: str) -> str | None:
        """检查输出是否退化。返回 None 表示通过，返回字符串表示失败原因。"""
        # 1. 无输出
        if not result or not result.strip():
            return "空输出"

        # 2. 硬崩溃信号
        if "[STOPPED]" in result:
            return "触发熔断 [STOPPED]"
        if "[LLM error]" in result:
            return "LLM 调用错误"

        # 3. 退化模式检查
        import re
        for pattern in self._DEGRADED_PATTERNS:
            if re.search(pattern, result, re.IGNORECASE):
                return f"输出匹配退化模式: {pattern}"

        # 4. 与基线对比（如有）
        if task in self._baseline and self._baseline[task]:
            baseline = self._baseline[task]
            # 长度大幅退化 (减少 80% 以上)
            if len(result) < len(baseline) * 0.2:
                return f"输出严重退化 (基线 {len(baseline)} → {len(result)} 字符)"
            # 基线有实质内容但新结果无实质信号
            if self._has_substance(baseline) and not self._has_substance(result):
                return "输出失去实质内容"

        return None

    @staticmethod
    def _has_substance(text: str) -> bool:
        """判断文本是否有实质内容（非纯状态/错误消息）"""
        stripped = text.strip()
        if len(stripped) < 20:
            return False
        if stripped.startswith("[STOPPED]") or stripped.startswith("[LLM error]"):
            return False
        return True
