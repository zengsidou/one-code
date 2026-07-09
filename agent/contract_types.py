# -*- coding: utf-8 -*-
"""契约类型定义 — 6种多模态契约 + 任务类型自动检测"""
from enum import Enum


class ContractType(Enum):
    VISUAL = "visual"          # 视觉任务: 线框图/SVG
    DIALOG = "dialog"          # 对话Agent: 示例对话轨迹
    CODE_API = "code_api"      # 代码库: API签名 + 架构图
    CONFIG = "config"          # 配置系统: 预期行为表
    DATA = "data"              # 数据分析: 输出格式+结论方向
    NARRATIVE = "narrative"    # 写作/内容: 大纲+风格样本


TYPE_META = {
    ContractType.VISUAL: {
        "format": "svg+ascii",
        "keywords": ["网页", "UI", "页面", "PPT", "海报", "界面", "前端", "布局",
                     "设计", "样式", "组件", "html", "css", "landing", "dashboard",
                     "网站", "web", "page", "banner", "画布"],
        "prompt_hint": "输出一个简洁的线框图，标注布局结构和信息层级",
        "human_judge": "一眼看结构——导航在哪、内容怎么排列",
    },
    ContractType.DIALOG: {
        "format": "json",
        "keywords": ["对话", "客服", "助手", "Agent", "聊天", "bot", "回复",
                     "问答", "交互", "对话流", "语音", "多轮", "对话策略",
                     "conversation", "chat", "assistant"],
        "prompt_hint": "输出3-5轮关键对话，展示语气和逻辑走向",
        "human_judge": "读几轮感受到风格——热情随和还是专业克制",
    },
    ContractType.CODE_API: {
        "format": "mermaid+text",
        "keywords": ["代码", "API", "架构", "模块", "接口", "重构", "refactor",
                     "拆分", "服务", "微服务", "后端", "类", "函数", "SDK",
                     "库", "library", "package", "server", "service"],
        "prompt_hint": "输出API签名列表 + Mermaid架构图",
        "human_judge": "看接口和模块关系——拆分是否合理、数据流对不对",
    },
    ContractType.CONFIG: {
        "format": "table",
        "keywords": ["配置", "开关", "参数", "规则", "config", "yaml", "json",
                     "环境变量", "env", "settings", "选项", "feature flag",
                     "权限", "阈值", "常量", "模板"],
        "prompt_hint": "输出'条件→结果'行为映射表",
        "human_judge": "看条件到结果的映射——逻辑是否完整、有无遗漏",
    },
    ContractType.DATA: {
        "format": "table+text",
        "keywords": ["数据", "分析", "报表", "统计", "图表", "SQL", "查询",
                     "可视化", "BI", "指标", "metrics", "analytics", "report",
                     "dashboard", "ETL", "清洗", "导入", "导出", "日志分析"],
        "prompt_hint": "输出分析维度和一句话结论方向",
        "human_judge": "看分析框架和维度——是否覆盖了关心的角度",
    },
    ContractType.NARRATIVE: {
        "format": "text",
        "keywords": ["文章", "报告", "文案", "写作", "博客", "blog", "文档",
                     "README", "说明", "教程", "总结", "会议纪要", "周报",
                     "PRD", "spec", "方案", "提案", "邮件", "公告", "推文"],
        "prompt_hint": "输出大纲 + 语气样本",
        "human_judge": "看大纲和语气——结构是否合理、文风是否合适",
    },
}


def detect_contract_type(task: str) -> ContractType:
    """根据任务描述中的关键词自动匹配契约类型

    遍历所有类型的关键词，按匹配数量排序，返回最佳匹配。
    无匹配时默认返回 CODE_API（代码任务最常见）。
    """
    task_lower = task.lower()
    scores: dict[ContractType, int] = {}

    for ct, meta in TYPE_META.items():
        for kw in meta["keywords"]:
            if kw.lower() in task_lower:
                scores[ct] = scores.get(ct, 0) + 1

    if scores:
        return max(scores, key=scores.get)

    return ContractType.CODE_API


def get_contract_meta(ct: ContractType) -> dict:
    """获取契约类型的元信息"""
    return TYPE_META.get(ct, TYPE_META[ContractType.CODE_API])
