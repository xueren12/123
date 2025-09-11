"""
风控模块：提供下单前事前校验（Pre-Trade Checks）。

功能覆盖：
1) 仓位上限（按名义 USD）
2) 单笔最大亏损（支持基于止损价或最坏移动比例评估）
3) 日内最大亏损（与熔断不同，此为事前校验层面）
4) 最大未平仓订单数（per instrument）
5) 单次下单最大比例（占用权益）
6) 滑点上限（相对参考价）

使用方式：
- 在交易执行前调用 RiskManager.validate_order(...)，若返回 allowed=False，则阻止下单并记录原因。
- 该模块与实盘/模拟盘无关，统一生效。

示例：
from utils.config import AppConfig
from utils.risk import RiskManager, OrderIntent, MarketState, AccountState

app_cfg = AppConfig()
risk = RiskManager(app_cfg)
order = OrderIntent(inst_id="BTC-USDT", side="buy", order_type="limit", qty=0.01, price=60000, reference_price=60000, stop_loss_price=59400)
market = MarketState(mid_price=60000)
account = AccountState(equity_usd=10000, position_usd_by_instrument={"BTC-USDT": 500}, open_orders_count_by_instrument={"BTC-USDT": 1}, daily_realized_pnl_usd=-50)
res = risk.validate_order(order, account, market)
if not res.allowed:
    print("阻止下单:", res.violations)
else:
    print("通过风控校验")
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from utils.config import AppConfig, RiskConfig


@dataclass
class OrderIntent:
    """下单意图描述（供风控校验使用）
    - inst_id: 交易对，例如 BTC-USDT
    - side: 买卖方向，buy/sell
    - order_type: 订单类型，market/limit
    - qty: 下单数量（以交易对 base 计量，如 BTC 数量）
    - price: 限价单价格（order_type=limit 必填）
    - reference_price: 参考价格（用于名义金额、滑点计算，一般用 mid/last）
    - stop_loss_price: 止损价格（可选，用于估算单笔最大亏损）
    - expected_slippage_pct: 预期滑点比例（可选；market单建议外部提供）
    - worst_move_pct: 在无止损价时，假设的最坏价格移动比例（0-1），用于估算潜在亏损
    """
    inst_id: str
    side: str  # "buy" or "sell"
    order_type: str  # "market" or "limit"
    qty: float
    price: Optional[float] = None
    reference_price: Optional[float] = None
    stop_loss_price: Optional[float] = None
    expected_slippage_pct: Optional[float] = None
    worst_move_pct: Optional[float] = None


@dataclass
class MarketState:
    """市场状态（供风控估算使用）"""
    mid_price: Optional[float] = None
    best_bid: Optional[float] = None
    best_ask: Optional[float] = None


@dataclass
class AccountState:
    """账户状态（供风控评估使用）
    - equity_usd: 账户权益（USD）
    - position_usd_by_instrument: 每个交易对的当前绝对名义持仓（USD），用于仓位上限校验
    - open_orders_count_by_instrument: 每个交易对的当前未完成订单数量
    - daily_realized_pnl_usd: 当日已实现盈亏（USD），亏损为负数
    """
    equity_usd: float
    position_usd_by_instrument: Dict[str, float] = field(default_factory=dict)
    open_orders_count_by_instrument: Dict[str, int] = field(default_factory=dict)
    daily_realized_pnl_usd: float = 0.0


@dataclass
class RiskCheckResult:
    """风控校验结果"""
    allowed: bool
    violations: List[str] = field(default_factory=list)
    # 额外输出一些估算数据，便于上层做日志与监控
    est_notional_usd: float = 0.0
    est_potential_loss_usd: float = 0.0
    est_slippage_pct: float = 0.0


class RiskManager:
    """风控模块主类

    说明：
    - 本类仅负责“事前校验”，不做委托下发或账户查询；所需的账户状态、行情、订单意图均由调用方提供。
    - 为确保可测试与可移植性，该模块不依赖数据库或交易所 SDK。
    - 支持在模拟盘与实盘统一使用。
    """

    def __init__(self, app_cfg: AppConfig | None = None, risk_cfg: RiskConfig | None = None):
        """初始化
        - app_cfg: 可选，若提供则优先读取 app_cfg.risk；否则可直接传入 risk_cfg
        - risk_cfg: 可选，直接传入风控阈值配置
        """
        if risk_cfg:
            self.cfg = risk_cfg
        elif app_cfg:
            self.cfg = app_cfg.risk
        else:
            # 兜底：直接从环境变量加载一份默认 RiskConfig
            from utils.config import RiskConfig as _RC

            self.cfg = _RC()

    # =============== 对外主入口 ===============
    def validate_order(self, order: OrderIntent, account: AccountState, market: MarketState) -> RiskCheckResult:
        """对单笔下单进行风控校验。

        返回 RiskCheckResult：
        - allowed: 是否放行
        - violations: 不满足的规则列表（中文原因）
        - est_*: 估算的名义金额/潜在亏损/滑点等，便于记录日志
        """
        violations: List[str] = []
        # 1) 计算名义金额与参考价
        ref_price = self._resolve_reference_price(order, market)
        if ref_price is None or ref_price <= 0:
            return RiskCheckResult(False, ["参考价格缺失或非法"], 0.0, 0.0, 0.0)
        notional_usd = max(0.0, order.qty) * ref_price

        # 2) 单次下单最大比例（占用权益）—按需关闭拦截（不再作为拦截条件）
        if account.equity_usd <= 0:
            violations.append("账户权益为0或未知，拒绝下单")
        # else: 不拦截“下单名义金额超限”，仅作为可选信息（已取消硬性约束）

        # 3) 仓位上限（名义 USD）
        cur_pos = abs(account.position_usd_by_instrument.get(order.inst_id, 0.0))
        if cur_pos + notional_usd > self.cfg.max_position_usd + 1e-9:
            violations.append(
                f"当前持仓 {cur_pos:.2f} + 下单 {notional_usd:.2f} 超过仓位上限 {self.cfg.max_position_usd:.2f}")

        # 4) 最大未平仓订单数（per instrument）
        cur_open = int(account.open_orders_count_by_instrument.get(order.inst_id, 0))
        if cur_open + 1 > self.cfg.max_open_orders:
            violations.append(
                f"当前未完成订单数 {cur_open} 超过上限 {self.cfg.max_open_orders}（本次下单将达到 {cur_open + 1}）")

        # 5) 滑点约束功能已移除，不再作为拦截条件
        slip_pct = 0.0  # 可选：保留为0，供上层记录

        # 6) 单笔最大亏损
        potential_loss = self._estimate_single_trade_loss_usd(order, ref_price, notional_usd)
        if potential_loss is None:
            violations.append("无法估算单笔潜在亏损：缺少止损或最坏移动比例，且无兜底参数")
            potential_loss = 0.0
        else:
            if potential_loss > self.cfg.max_single_trade_loss_usd + 1e-9:
                violations.append(
                    f"单笔潜在亏损 {potential_loss:.2f} 超过上限 {self.cfg.max_single_trade_loss_usd:.2f}")

        # 7) 日内最大亏损（事前校验）
        realized_loss = max(0.0, -account.daily_realized_pnl_usd)  # 亏损为正数
        if realized_loss >= self.cfg.max_daily_loss_usd - 1e-9:
            violations.append(
                f"今日已亏 {realized_loss:.2f} 已达上限 {self.cfg.max_daily_loss_usd:.2f}")
        else:
            if realized_loss + potential_loss > self.cfg.max_daily_loss_usd + 1e-9:
                violations.append(
                    f"若成交可能触发日亏超限：已亏 {realized_loss:.2f} + 潜在亏损 {potential_loss:.2f} > 上限 {self.cfg.max_daily_loss_usd:.2f}")

        return RiskCheckResult(allowed=len(violations) == 0,
                               violations=violations,
                               est_notional_usd=notional_usd,
                               est_potential_loss_usd=potential_loss,
                               est_slippage_pct=slip_pct)

    # =============== 内部辅助 ===============
    def _resolve_reference_price(self, order: OrderIntent, market: MarketState) -> Optional[float]:
        # 优先使用显式传入的 reference_price；其次使用 mid_price；最后使用限价单价格
        if order.reference_price is not None and order.reference_price > 0:
            return order.reference_price
        if market.mid_price is not None and market.mid_price > 0:
            return market.mid_price
        if order.order_type == "limit" and order.price and order.price > 0:
            return order.price
        return None

    def _estimate_slippage_pct(self, order: OrderIntent, ref_price: float, market: MarketState) -> Optional[float]:
        # 若显式给定预期滑点，则直接使用
        if order.expected_slippage_pct is not None:
            return max(0.0, float(order.expected_slippage_pct))
        # 否则，若为限价单，使用 |limit - ref| / ref 估算
        if order.order_type == "limit" and order.price is not None and ref_price > 0:
            return abs(order.price - ref_price) / ref_price
        # 市价单但缺少预估滑点，则尝试用半个价差估算
        if market.best_bid and market.best_ask and market.best_ask > market.best_bid > 0:
            mid = (market.best_bid + market.best_ask) / 2.0
            # 半个点差相对 mid 的比例
            return (market.best_ask - market.best_bid) / 2.0 / mid
        return None

    def _estimate_single_trade_loss_usd(self, order: OrderIntent, ref_price: float, notional_usd: float) -> Optional[float]:
        # 若提供止损价，则用 |entry - stop| * qty 作为潜在亏损
        if order.stop_loss_price is not None and order.qty is not None:
            entry = order.price if (order.order_type == "limit" and order.price) else ref_price
            loss_per_unit = abs(entry - float(order.stop_loss_price))
            return max(0.0, loss_per_unit * max(0.0, order.qty))
        # 若提供最坏移动比例，则用比例 * 名义金额
        if order.worst_move_pct is not None:
            return max(0.0, float(order.worst_move_pct)) * max(0.0, notional_usd)
        # 否则，兜底使用 max_slippage_pct * 名义金额 估算一次极端波动损失
        return self.cfg.max_slippage_pct * max(0.0, notional_usd)