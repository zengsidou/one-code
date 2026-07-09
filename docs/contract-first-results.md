# 契约先行小实验结果（2026-07-10）

单次自动跑批：5 任务 × 3 条件 = **15 runs**。模型 DeepSeek；`max_steps=12`；contract 模式 **auto-confirm**（未测人工多轮改预览）。

原始数据：`eval/contract_first/results/rows.jsonl`  
产物：`eval/contract_first/workspaces/`  
打分脚本：`python -m eval.contract_first.score_results`

## 总表（success / consistency）

| Task | direct | plan | contract |
|------|--------|------|----------|
| T1 visual 落地页 | 0 / 1（无 index.html） | 0 / 2（无 index.html） | **1 / 5** |
| T2 Todo API | 1 / 3 | 1 / 3 | **1 / 4** |
| T3 客服对话 | 1 / 4 | 1 / 4 | 0 / 1（未写出 md） |
| T4 流失框架 | 1 / 4 | 1 / 4 | 1 / 4 |
| T5 README 章节 | 1 / 4 | 1 / 4 | **1 / 5** |
| **合计** | **4/5** | **4/5** | **4/5** |
| 平均 consistency | 3.2 | 3.4 | **3.8** |
| 平均耗时 (s) | 62 | 96 | 109 |

## 观察（诚实）

1. **成功率打平（都是 4/5）**：本轮不能声称「契约先行显著提高成功率」。  
2. **一致性均值 contract 最高（3.8）**：尤其 T1/T5，预览对最终形态约束更明显。  
3. **T1 是最强 case study**：direct/plan 未产出页面；contract 产出完整四段落地页。  
4. **T3 contract 失败**：预览后执行未落盘 `customer_support_bot.md`——说明门控≠交付保证，仍需执行可靠性。  
5. **耗时**：contract/plan 更慢（多一轮规划/预览），符合预期。  
6. **未测量**：人工返工轮数、token usage（API 未接入精确计量）、交互改预览。

## 面试可讲的一句话

> 在 15 次自动对照中，三种模式成功率同为 4/5；契约先行未抬成功率，但平均一致性更高，并在视觉落地页任务上单独打穿 direct/plan 的失败。样本小，结论是方向性观察，不是证明。

## 工程修复（本轮为跑通实验）

- Windows GBK 下 reasoning 输出崩溃 → `_safe_stdout`
- 实验禁用共享 ChromaDB（compaction 竞态）
- `write_file` 缺参保护；拒绝 agent 内长驻 Flask/uvicorn
- T2 任务改为「不要启动常驻服务」
