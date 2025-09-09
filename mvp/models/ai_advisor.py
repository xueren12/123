"""
AI 辅助解读模块（轻量）。
- 输入：结构化行情 + 链上大额转账 + 社交舆情摘要
- 输出：交易信号建议与市场风险说明
- 支持：本地轻量 LLM（transformers）或 OpenAI 兼容 API（如 GPT-4/mini）
- 降本：将输入压缩为 JSON/特征向量，仅在异常事件触发时调用

注意：本地推理需要安装 PyTorch/CPU 或 GPU 才能正常运行；若未安装，请将 AI_PROVIDER 设为 openai 并配置 OPENAI_API_KEY。
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from enum import Enum
from typing import List, Dict, Any, Tuple, Optional
import json

from loguru import logger
import httpx

from utils.config import AppConfig


# ====== 基础数据结构（精简） ======
@dataclass
class MarketSnapshot:
    """行情快照（尽量精简字段以便压缩）"""
    inst: str
    ts: int  # 毫秒时间戳
    last: float
    prev_last: Optional[float] = None  # 若提供则自动计算涨跌幅
    buy_vol_60s: float = 0.0  # 近 60s 买量
    sell_vol_60s: float = 0.0  # 近 60s 卖量
    spread_bps: Optional[float] = None  # 买一卖一价差（基点）


@dataclass
class WhaleTransfer:
    """链上巨鲸转账（已换算 USD）"""
    chain: str
    symbol: str
    usd_value: float
    direction: str  # inflow/outflow
    ts: int


@dataclass
class SentimentSummary:
    """社交舆情摘要"""
    score: float  # [-1,1]
    delta: float  # 相对上期变化 [-1,1]
    summary: str  # 简要摘要，建议 <= 280 字符


@dataclass
class AIInputPayload:
    market: MarketSnapshot
    whales: List[WhaleTransfer]
    sentiment: SentimentSummary


class AdviceSignal(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


@dataclass
class AIAdvice:
    """AI 输出建议（统一结构）"""
    signal: AdviceSignal
    confidence: float
    risk_level: RiskLevel
    rationale: str
    risks: str
    triggered: bool  # 是否触发了异常而调用了大模型
    triggered_by: List[str]
    raw_model_output: Optional[str] = None


# ====== 异常检测器（阈值来源于配置） ======
class AnomalyDetector:
    @staticmethod
    def check(payload: AIInputPayload, cfg: AppConfig) -> Tuple[bool, List[str]]:
        ai = cfg.ai
        triggers: List[str] = []
        # 价格跳变
        if payload.market.prev_last:
            pct = abs((payload.market.last - payload.market.prev_last) / payload.market.prev_last)
            if pct >= ai.price_jump_pct:
                triggers.append(f"price_jump>={ai.price_jump_pct:.2%} 实际 {pct:.2%}")
        # 巨鲸转账
        big_whales = [w for w in payload.whales if w.usd_value >= ai.whale_usd_threshold]
        if big_whales:
            total = sum(w.usd_value for w in big_whales)
            triggers.append(f"whale_total_usd>={ai.whale_usd_threshold:.0f} 共 {len(big_whales)} 笔, 累计 ${total:,.0f}")
        # 情绪骤降
        if payload.sentiment.delta <= -ai.sentiment_drop:
            triggers.append(f"sentiment_drop>={ai.sentiment_drop:.2f} 实际 {payload.sentiment.delta:.2f}")
        return (len(triggers) > 0) or (not ai.call_on_anomaly_only), triggers


# ====== LLM 提供方抽象 ======
class LLMProvider:
    def generate(self, prompt: str, cfg: AppConfig) -> str:
        raise NotImplementedError


class OpenAIProvider(LLMProvider):
    """OpenAI 兼容 API（/v1/chat/completions）"""
    def generate(self, prompt: str, cfg: AppConfig) -> str:
        url = cfg.ai.openai_base_url.rstrip("/") + "/v1/chat/completions"
        headers = {"Authorization": f"Bearer {cfg.ai.openai_api_key}", "Content-Type": "application/json"}
        body = {
            "model": cfg.ai.openai_model,
            "temperature": cfg.ai.temperature,
            "max_tokens": cfg.ai.max_tokens,
            "messages": [
                {"role": "system", "content": "你是一名专业的加密市场分析师，请用简洁中文输出严格的 JSON。"},
                {"role": "user", "content": prompt},
            ],
        }
        try:
            r = httpx.post(url, headers=headers, json=body, timeout=30)
            r.raise_for_status()
            data = r.json()
            return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            logger.exception("OpenAIProvider 调用失败: {}", e)
            raise


class LocalLLMProvider(LLMProvider):
    """本地 transformers 轻量模型。需要安装 torch。
    默认模型：Qwen/Qwen2.5-0.5B-Instruct（可通过环境变量覆盖）。
    """
    _pipe = None

    def _ensure_pipe(self, cfg: AppConfig):
        if self._pipe is not None:
            return
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
            model_id = cfg.ai.local_model_id
            tok = AutoTokenizer.from_pretrained(model_id)
            mdl = AutoModelForCausalLM.from_pretrained(model_id)
            self._pipe = pipeline("text-generation", model=mdl, tokenizer=tok)
        except Exception as e:
            logger.error("本地模型初始化失败，请安装 PyTorch 或切换 AI_PROVIDER=openai: {}", e)
            raise

    def generate(self, prompt: str, cfg: AppConfig) -> str:
        self._ensure_pipe(cfg)
        out = self._pipe(prompt, max_new_tokens=cfg.ai.max_tokens, temperature=cfg.ai.temperature, do_sample=True)
        text = out[0]["generated_text"]
        return text[len(prompt):].strip() if text.startswith(prompt) else text.strip()


# ====== 主类：AIAdvisor ======
class AIAdvisor:
    """大模型辅助解读器
    用法：
    advisor = AIAdvisor(AppConfig())
    result = advisor.run_if_needed(payload)  # 返回 AIAdvice
    """

    def __init__(self, cfg: Optional[AppConfig] = None):
        self.cfg = cfg or AppConfig()
        if self.cfg.ai.provider == "openai":
            self.provider: LLMProvider = OpenAIProvider()
        else:
            self.provider = LocalLLMProvider()

    # 压缩/特征化输入，降低 token 消耗
    def _compress(self, p: AIInputPayload) -> Dict[str, Any]:
        last, prev = p.market.last, p.market.prev_last
        pct = (last - prev) / prev if (prev and prev != 0) else 0.0
        imb = p.market.buy_vol_60s - p.market.sell_vol_60s
        whale_total = sum(w.usd_value for w in p.whales)
        whales_top3 = sorted((w.usd_value for w in p.whales), reverse=True)[:3]
        return {
            "inst": p.market.inst,
            "ts": p.market.ts,
            "last": round(last, 6),
            "pct_change": round(pct, 6),
            "vol_imbalance_60s": round(imb, 3),
            "spread_bps": p.market.spread_bps,
            "whale_cnt": len(p.whales),
            "whale_total_usd": round(whale_total, 2),
            "whales_top3_usd": whales_top3,
            "sent_score": round(p.sentiment.score, 3),
            "sent_delta": round(p.sentiment.delta, 3),
            "sent_summary": (p.sentiment.summary or "")[:280],
        }

    def _build_prompt(self, feats: Dict[str, Any]) -> str:
        """构造严格 JSON 指令，要求输出：signal/confidence/risk_level/rationale/risks"""
        schema = {
            "signal": "BUY/SELL/HOLD 三选一",
            "confidence": "0-1 浮点，表示置信度",
            "risk_level": "LOW/MEDIUM/HIGH 三选一",
            "rationale": "50字内中文理由",
            "risks": "50字内中文风险提示"
        }
        return (
            "以下为加密市场压缩特征(JSON)。请基于其中的价格变动、买卖盘不平衡、巨鲸转账与情绪变化，给出短线交易建议。"
            "只允许输出严格 JSON，键为: signal, confidence, risk_level, rationale, risks。\n"
            f"Schema: {json.dumps(schema, ensure_ascii=False)}\n"
            f"Features: {json.dumps(feats, ensure_ascii=False)}\n"
            "禁止输出多余文本。"
        )

    def _parse(self, text: str) -> Dict[str, Any]:
        """尽力解析为 JSON（容错括号外多余文本）"""
        try:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                return json.loads(text[start:end+1])
            return json.loads(text)
        except Exception:
            logger.warning("模型输出非 JSON，将回退为 HOLD: {}", text[:200])
            return {"signal": "HOLD", "confidence": 0.3, "risk_level": "MEDIUM", "rationale": "输出异常", "risks": "格式不规范"}

    def run_if_needed(self, payload: AIInputPayload) -> AIAdvice:
        need_call, triggers = AnomalyDetector.check(payload, self.cfg)
        feats = self._compress(payload)
        if not need_call:
            # 不触发时返回保守 HOLD，且不调用大模型
            return AIAdvice(
                signal=AdviceSignal.HOLD,
                confidence=0.5,
                risk_level=RiskLevel.MEDIUM,
                rationale="未发现显著异常，保持观望",
                risks="常规波动风险",
                triggered=False,
                triggered_by=triggers,
                raw_model_output=None,
            )
        # 触发 -> 调用 LLM
        prompt = self._build_prompt(feats)
        raw = self.provider.generate(prompt, self.cfg)
        obj = self._parse(raw)
        sig = obj.get("signal", "HOLD").upper()
        if sig not in ("BUY", "SELL", "HOLD"):
            sig = "HOLD"
        risk = obj.get("risk_level", "MEDIUM").upper()
        if risk not in ("LOW", "MEDIUM", "HIGH"):
            risk = "MEDIUM"
        return AIAdvice(
            signal=AdviceSignal(sig),
            confidence=float(obj.get("confidence", 0.5)),
            risk_level=RiskLevel(risk),
            rationale=str(obj.get("rationale", ""))[:200],
            risks=str(obj.get("risks", ""))[:200],
            triggered=True,
            triggered_by=triggers,
            raw_model_output=raw,
        )


# ====== 便捷构造函数 ======
def build_payload(market: Dict[str, Any], whales: List[Dict[str, Any]], sentiment: Dict[str, Any]) -> AIInputPayload:
    """从原始 dict 构造输入载荷，便于外部快速接入。"""
    ms = MarketSnapshot(**market)
    ws = [WhaleTransfer(**w) for w in whales]
    ss = SentimentSummary(**sentiment)
    return AIInputPayload(market=ms, whales=ws, sentiment=ss)