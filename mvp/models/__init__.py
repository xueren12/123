"""models 包导出
- AIAdvisor：大模型辅助解读
- build_payload：便捷构造输入
- 数据结构：AIInputPayload/MarketSnapshot/WhaleTransfer/SentimentSummary/AIAdvice
"""
from .ai_advisor import (
    AIAdvisor,
    build_payload,
    AIInputPayload,
    MarketSnapshot,
    WhaleTransfer,
    SentimentSummary,
    AIAdvice,
)
__all__ = [
    "AIAdvisor",
    "build_payload",
    "AIInputPayload",
    "MarketSnapshot",
    "WhaleTransfer",
    "SentimentSummary",
    "AIAdvice",
]