"""
集中式配置加载器。
优先级：环境变量 > .env 文件默认值。
"""
from __future__ import annotations

import os
import json  # 新增：支持从 JSON 载入交易纪律配置
from dataclasses import dataclass, field
from typing import List, Optional, Any
from dotenv import load_dotenv

load_dotenv()


@dataclass
class WSConfig:
    okx_ws_public: str = os.getenv("OKX_WS_PUBLIC", "wss://ws.okx.com:8443/ws/v5/public")
    instruments: List[str] = field(default_factory=lambda: os.getenv("INSTRUMENTS", "BTC-USDT,ETH-USDT").split(","))
    # 订阅的频道列表
    channels: List[str] = field(default_factory=lambda: ["trades", "books5"])


@dataclass
class DBEnvConfig:
    host: str = os.getenv("DB_HOST", "localhost")
    port: int = int(os.getenv("DB_PORT", "15432"))
    dbname: str = os.getenv("DB_NAME", "crypto")
    user: str = os.getenv("DB_USER", "postgres")
    password: str = os.getenv("DB_PASSWORD", "123456")
    sslmode: str = os.getenv("DB_SSLMODE", "prefer")


# 新增：OKX REST API 配置
@dataclass
class OKXRESTConfig:
    """OKX REST API 配置（从环境变量读取）。
    - base_url：REST 接口基础域名
    - api_key/secret_key/passphrase：OKX API 凭证
    - simulated_trading：是否使用 OKX 模拟盘（Demo Trading），会自动在请求头加入 x-simulated-trading: 1
    - td_mode：交易模式（现货：cash；全仓/逐仓杠杆：cross/isolated；合约：cross/isolated）
    - tgt_ccy：现货下单计量方式（base_ccy 或 quote_ccy）
    """
    base_url: str = os.getenv("OKX_BASE_URL", "https://www.okx.com")
    api_key: str = os.getenv("OKX_API_KEY", "")
    secret_key: str = os.getenv("OKX_API_SECRET", "")
    passphrase: str = os.getenv("OKX_API_PASSPHRASE", "")
    simulated_trading: bool = os.getenv("OKX_SIMULATED", "0") == "1"
    td_mode: str = os.getenv("OKX_TD_MODE", "cash")
    tgt_ccy: str = os.getenv("OKX_TGT_CCY", "base_ccy")


# 新增：AI 模块配置
@dataclass
class AIConfig:
    """AI 模块配置
    - provider：local（本地 transformers）或 openai（OpenAI 兼容 API）
    - local_model_id：本地小模型名称，如 Qwen/Qwen2.5-0.5B-Instruct
    - openai_base_url：OpenAI 兼容 API 的基础地址
    - openai_api_key：用于鉴权的 API Key
    - openai_model：服务端模型名，如 gpt-4o-mini / qwen-plus 等
    - temperature/max_tokens：采样温度与最大生成长度
    - call_on_anomaly_only：仅在触发异常事件时才调用大模型
    - 异常阈值：价格跳变阈值、巨鲸转账阈值、情绪骤降阈值
    """
    provider: str = os.getenv("AI_PROVIDER", "local")  # local / openai
    local_model_id: str = os.getenv("AI_LOCAL_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
    openai_base_url: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    temperature: float = float(os.getenv("AI_TEMPERATURE", "0.2"))
    max_tokens: int = int(os.getenv("AI_MAX_TOKENS", "512"))
    call_on_anomaly_only: bool = os.getenv("AI_ONLY_ANOMALY", "1") == "1"
    # 异常触发阈值
    price_jump_pct: float = float(os.getenv("AI_TH_PRICE_JUMP", "0.01"))  # 1%
    whale_usd_threshold: float = float(os.getenv("AI_TH_WHALE_USD", "1000000"))  # 100 万美金
    sentiment_drop: float = float(os.getenv("AI_TH_SENT_DROP", "0.3"))  # 情绪分跌幅 0.3


# 新增：Telegram 告警配置
@dataclass
class TelegramConfig:
    """Telegram 机器人告警配置"""
    enabled: bool = os.getenv("TG_ENABLED", "0") == "1"
    bot_token: str = os.getenv("TG_BOT_TOKEN", "")
    chat_id: str = os.getenv("TG_CHAT_ID", "")  # 可为用户/群组 ID


# 新增：Slack 告警配置
@dataclass
class SlackConfig:
    """Slack 机器人告警配置"""
    enabled: bool = os.getenv("SLACK_ENABLED", "0") == "1"
    bot_token: str = os.getenv("SLACK_BOT_TOKEN", "")
    channel: str = os.getenv("SLACK_CHANNEL", "")


# 新增：监控告警阈值配置
@dataclass
class AlertConfig:
    """监控模块的异常触发阈值"""
    enabled: bool = os.getenv("ALERTS_ENABLED", "1") == "1"
    price_jump_pct: float = float(os.getenv("ALERT_PRICE_JUMP_PCT", "0.02"))  # 价格瞬时跳变阈值（2%）
    stale_book_sec: int = int(os.getenv("ALERT_STALE_BOOK_SEC", "10"))        # 盘口延迟判定（秒）
    trade_fail_window_sec: int = int(os.getenv("ALERT_TRADE_FAIL_WIN", "60")) # 交易失败窗口（秒）
    rate_limit_sec: int = int(os.getenv("ALERT_RATE_LIMIT_SEC", "60"))        # 同类告警限频（秒）


# 新增：监控模块参数
@dataclass
class MonitorConfig:
    """实时监控模块配置"""
    poll_interval_sec: float = float(os.getenv("MONITOR_POLL_SEC", "1"))
    # 若未显式配置，则复用 WS 订阅的交易对
    instruments: List[str] = field(default_factory=lambda: os.getenv("MONITOR_INSTRUMENTS", "").split(",") if os.getenv("MONITOR_INSTRUMENTS") else [])
    use_web_ui: bool = os.getenv("MONITOR_WEB_UI", "0") == "1"  # 预留：是否启用 Web UI（默认仅终端）


 # 新增：执行与熔断配置
@dataclass
class ExecConfig:
    """执行与熔断配置
    - mode: 运行模式三态：mock / real / paused
    - control_file: 外部控制文件路径（JSON），可被HTTP或人工修改
    - http_admin_enabled: 是否开启HTTP管理接口
    - http_host/http_port: HTTP管理服务监听地址与端口
    - base_equity_usd: 基准权益（用于按百分比计算亏损阈值）
    - daily_max_loss_pct: 单日最大亏损百分比（超过则暂停）
    - single_trade_max_loss_pct: 单笔最大亏损百分比（超过则暂停）
    - risk_notional_usd: 用于将百分比变化近似折算为单笔USD损益名义规模
    """
    mode: str = os.getenv("EXEC_MODE", "mock")  # mock / real / paused
    control_file: str = os.getenv("EXEC_CONTROL_FILE", os.path.join("data", "exec_switch.json"))
    http_admin_enabled: bool = os.getenv("EXEC_HTTP_ENABLED", "1") == "1"
    http_host: str = os.getenv("EXEC_HTTP_HOST", "127.0.0.1")
    http_port: int = int(os.getenv("EXEC_HTTP_PORT", "8088"))
    base_equity_usd: float = float(os.getenv("BASE_EQUITY_USD", "10000"))
    daily_max_loss_pct: float = float(os.getenv("DAILY_MAX_LOSS_PCT", "0.03"))  # 3%
    single_trade_max_loss_pct: float = float(os.getenv("SINGLE_TRADE_MAX_LOSS_PCT", "0.01"))  # 1%
    risk_notional_usd: float = float(os.getenv("RISK_NOTIONAL_USD", "100"))


# 新增：风控规则配置
@dataclass
class RiskConfig:
    """风控规则配置（供下单前校验使用）
    - max_position_usd: 每个交易对的最大持仓名义金额（USD）
    - max_open_orders: 每个交易对允许的最大未完成订单数量
    - single_order_max_pct_equity: 单次下单占用权益的最大比例（0-1）
    - max_slippage_pct: 允许的最大滑点比例（相对于最优价/中间价）
    - max_single_trade_loss_usd: 单笔最大允许亏损（按预估极端移动计算）
    - max_daily_loss_usd: 单日最大允许亏损（与熔断阈值解耦，属于事前校验）
    """
    max_position_usd: float = float(os.getenv("RISK_MAX_POSITION_USD", "2000"))
    max_open_orders: int = int(os.getenv("RISK_MAX_OPEN_ORDERS", "5"))
    single_order_max_pct_equity: float = float(os.getenv("RISK_SINGLE_ORDER_MAX_PCT_EQUITY", "0.2"))
    max_slippage_pct: float = float(os.getenv("RISK_MAX_SLIPPAGE_PCT", "0.002"))  # 0.2%
    max_single_trade_loss_usd: float = float(os.getenv("RISK_MAX_SINGLE_TRADE_LOSS_USD", "100"))
    max_daily_loss_usd: float = float(os.getenv("RISK_MAX_DAILY_LOSS_USD", "300"))


# 新增：审计日志配置
@dataclass
class AuditConfig:
    """审计与结构化日志配置
    - enabled: 是否启用审计
    - to_file: 是否写入本地 JSONL 文件
    - to_db: 是否写入 PostgreSQL (audit_logs 表)
    - dir: 本地日志目录（按天滚动文件）
    - file_prefix: 文件名前缀
    """
    enabled: bool = os.getenv("AUDIT_ENABLED", "1") == "1"
    to_file: bool = os.getenv("AUDIT_TO_FILE", "1") == "1"
    to_db: bool = os.getenv("AUDIT_TO_DB", "1") == "1"
    dir: str = os.getenv("AUDIT_DIR", os.path.join("data", "audit"))
    file_prefix: str = os.getenv("AUDIT_FILE_PREFIX", "audit")


# 新增：交易纪律/仓位与止盈止损集中配置（JSON）
@dataclass
class DisciplineConfig:
    """交易纪律与风险参数（优先从 JSON 文件加载）。
    默认文件路径：RULES_CONFIG_FILE 环境变量；未设置则使用 mvp/rules.json。
    可配置项示例：
    {
      "risk_percent": 0.01,
      "max_add_positions": 3,
      "atr_stop_loss": 2.0,
      "stop_loss_pct_btc": 0.03,
      "stop_loss_pct_eth": 0.04,
      "take_profit_split": [0.3, 0.3],
      "trailing_atr": 1.5,
      "cooldown_bars": 2,
      "max_consecutive_losses": 3
    }
    """
    file_path: str = field(default_factory=lambda: os.getenv("RULES_CONFIG_FILE", os.path.join("mvp", "rules.json")))
    risk_percent: float = 0.01
    max_add_positions: int = 3
    atr_stop_loss: float = 2.0
    stop_loss_pct_btc: float = 0.03
    stop_loss_pct_eth: float = 0.04
    take_profit_split: List[float] = field(default_factory=lambda: [0.3, 0.3])
    trailing_atr: float = 1.5
    cooldown_bars: int = 0  # 0 表示按秒级冷却配置（MA_COOLDOWN_SEC/COOLDOWN_SEC）
    max_consecutive_losses: int = 3


def load_discipline_config() -> DisciplineConfig:
    """从 JSON 文件加载交易纪律配置；文件不存在或解析失败则使用默认值。
    - 仅覆盖 JSON 中出现的字段，其余保留默认值。
    """
    cfg = DisciplineConfig()
    path = cfg.file_path
    try:
        if path and os.path.exists(path) and os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as f:
                data: Any = json.load(f)
            if isinstance(data, dict):
                # 逐字段覆盖
                for k, v in data.items():
                    if hasattr(cfg, k):
                        try:
                            setattr(cfg, k, v)
                        except Exception:
                            pass
    except Exception:
        # 静默失败，使用默认
        pass
    return cfg


@dataclass
class AppConfig:
    ws: WSConfig = field(default_factory=WSConfig)
    db: DBEnvConfig = field(default_factory=DBEnvConfig)
    # 新增：OKX REST API 配置
    okx: OKXRESTConfig = field(default_factory=OKXRESTConfig)
    # 新增：AI 配置
    ai: AIConfig = field(default_factory=AIConfig)
    # 新增：告警通道与阈值配置
    telegram: TelegramConfig = field(default_factory=TelegramConfig)
    slack: SlackConfig = field(default_factory=SlackConfig)
    alerts: AlertConfig = field(default_factory=AlertConfig)
    monitor: MonitorConfig = field(default_factory=MonitorConfig)
    # 新增：执行与熔断配置
    exec: ExecConfig = field(default_factory=ExecConfig)
    # 新增：风控规则配置
    risk: RiskConfig = field(default_factory=RiskConfig)
    # 新增：审计日志配置
    audit: AuditConfig = field(default_factory=AuditConfig)
    # 新增：交易纪律与集中风险参数（JSON）
    discipline: DisciplineConfig = field(default_factory=load_discipline_config)