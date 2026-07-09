# 契约先行：5 任务小实验协议

目标：给「返工更少 / token 更省 / 方向更准」收集**最小可展示证据**，不是发论文级实验。

## 对照条件

| 条件 | 含义 | 命令示意 |
|------|------|----------|
| A. Direct | 直接执行，无计划门控 | `python -m eval.contract_first.run --mode direct --task-id T1` |
| B. Plan | 文本步骤计划后执行（模拟 PTE） | `--mode plan` |
| C. Contract | 产物预览门控 + 逆向拆解后执行 | `--mode contract` |

每人/每任务固定同一模型（建议 DeepSeek），同一工作目录沙箱，最多 N 步（默认 15）。

## 五个任务（覆盖多种契约类型）

| ID | 类型 | 任务简述 | 成功标准（人工一眼可判） |
|----|------|----------|--------------------------|
| T1 | visual | 做一个产品介绍落地页：顶部导航、中间英雄区、三列特性、底部 CTA | 结构符合四段布局；不是空白页 |
| T2 | code_api | 给本地 `todo` 模块加 REST：list/create/complete，并画模块关系 | 三个端点可调用或有清晰路由文件；有架构说明 |
| T3 | dialog | 设计售后客服 bot：先安抚再排查再升级人工 | 示例对话含安抚→排查→升级；语气克制 |
| T4 | data | 根据假销售 CSV 给出流失分析框架（维度+结论方向） | 输出含维度表+一句话方向；不含无关竞品维度 |
| T5 | narrative | 写 One-Code README 的「契约先行」小节大纲+语气样本 | 大纲 3–5 节；语气偏工程说明而非营销 |

任务全文与验收清单：`tasks.json`。

## 指标（每任务每条件记一行）

| 指标 | 定义 |
|------|------|
| `preview_rounds` | 执行前预览/计划修改轮数（A=0） |
| `direction_ok_before_exec` | 执行前人工是否认为方向正确（0/1；A 记 N/A） |
| `rework_rounds` | 看完最终结果后，因「方向不对」要求重做的轮数 |
| `tokens_preview` | 预览/计划阶段 token（无则 0） |
| `tokens_exec` | 执行阶段 token |
| `tokens_total` | 两者之和 |
| `steps` | agent 步数 |
| `success` | 是否满足成功标准（0/1） |
| `consistency_1_5` | 最终产物与预览/计划一致性（1–5；A 对「用户心里目标」打分） |
| `notes` | 失败模式一句话 |

## 流程（单任务）

1. 清空沙箱工作区 `eval/contract_first/workspaces/<task_id>/<mode>/`
2. 跑对应 mode，记录日志与产物路径
3. 人工按成功标准打 `success` / `consistency`
4. 写入 `results/rows.jsonl` 一行
5. 五个任务 × 三条件 = 最多 15 行；时间紧可先跑 T1/T2/T5

## 如何报告（简历/面试）

不要说「实验证明契约先行更好」。说：

> 在 5 个任务、三条件对照下收集了返工轮数与 token；当前样本为 N，观察是……；局限是……。

有数字就上表；没有就只展示协议与 1–2 个 case study。

## 自动化范围

Runner 自动采集：mode、步数、预览文本、执行日志、粗略 token（若 API 返回 usage）。  
`success` / `consistency` / `rework_rounds` **必须人工**（这正是契约先行的哲学：人判断方向）。
