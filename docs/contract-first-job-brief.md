# 契约先行（Contract-First）求职精简稿

> 定位：DeepSeek Harness / Coding Agent 求职材料  
> 完整论述见：https://zengsidou.github.io/papers/contract-first-paper.pdf  
> 工程实现：`agent/contract_first.py`（One-Code）  
> 小实验：`eval/contract_first/`

---

## 一句话

Plan-Then-Execute 让人确认「怎么做」；契约先行让人确认「做成什么样」——在执行前用低成本产物预览做硬门控，再按预览逆向拆解执行。

这对应终极目标「指哪打哪」里的 **指得准**：先把目标钉死，再动手。

---

## 问题

自然语言有 gap。现有验证对象多为：

| 机制 | 验证对象 | 问题 |
|------|----------|------|
| 事后看结果 | 完整产物 | 反馈周期 = 全量执行 |
| 逐步确认 | 单步动作 | 看不到最终形态 |
| Plan-Then-Execute | 文本步骤 | 猜产物，猜不准 |
| 代码断言（Karpathy） | 可测属性 | 测不出「丑 / 不对味」 |
| Inline diff | 代码变更 | 执行后才见，且偏局部 |

共同缺口：**执行前缺少「人一眼能否决」的产物方向预览。**

---

## 定义（工程可落地）

**契约先行** = 执行前，用最小 token 生成人可直观判断方向的产物预览 → 人确认 → 从预览逆向拆解执行 → 每步对照预览。

三原则：

1. **人对齐优先**：未确认不执行（硬门控）
2. **最小 token**：预览 << 完整产物
3. **逆向拆解**：步骤以契约为验收锚点，而非纯正向推进

---

## 与相关工作的差异（面试口述版）

| | PTE | Karpathy | Magentic-UI | 本文 / One-Code |
|--|-----|----------|-------------|-----------------|
| 契约 | 步骤 | 测试断言 | 可编辑步骤 | **多模态产物预览** |
| 判断者 | 人 | 机器 | 人 | 人 |
| 执行 | 正向 | 正向 | 正向 | **逆向拆解** |

不是新范式，是 PTE 内把契约模态从「how」升级为「what」。

---

## 六种契约类型（实现已有）

见 `agent/contract_types.py`：visual / dialog / code_api / config / data / narrative。  
当前用关键词自适应选择（启发式，非完整类型学——面试主动承认）。

---

## 逆向拆解（算法摘要）

```
输入: task, contract_preview
输出: steps[]，每步含 goal + contract_checkpoint

1. 以 contract_preview 为终点状态 S*
2. LLM 从 S* 反推 3–8 步，每步写出「对照契约哪一段验收」
3. 执行时：完成 step_i 后检查 checkpoint；偏离则修正后再继续
4. 结束时做契约一致性总结（preview vs 最终产物）
```

实现：`agent/reverse_decomposer.py`。  
**未验证假设**：逆向比正向更能降低方向偏离——需用 `eval/contract_first` 实验检验。

---

## 适用边界（必须会讲）

适用：「人看一眼就知道方向对不对」。  
不适用或仅部分适用：要「用一用才知道」的体验、纯后端无直观形态、对话运行时长尾质量。

---

## 与 One-Code / 求职叙事

- 论文 = 产品原则  
- One-Code `--contract-first` = 最小实现  
- 运维背景 → 门控、验收、返工成本（可靠性迁移）  
- DeepSeek Harness：模型之外，目标确认与执行锚定属于 harness 核心

---

## 未验证假设（诚实列表）

1. 预览能显著减少返工轮数  
2. 预览 token 显著低于全量执行  
3. 逆向拆解优于同任务正向 plan  
4. 关键词选类型在真实任务上足够准  
5. 执行结果与预览的一致性可被自动/半自动度量

用 `eval/contract_first` 的 5 任务协议逐项收集证据。
