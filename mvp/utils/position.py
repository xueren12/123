# -*- coding: utf-8 -*-
"""
仓位管理模块（PositionManager）：负责根据账户权益与风控参数，结合当前价格，计算单次下单数量。

设计目标：
- 解耦策略/执行器的下单手数逻辑，统一在此集中配置与计算
- 支持从配置中读取基础权益、单次下单最大权益占比、最大持仓名义金额等
- 支持外部注入价格/权益/持仓等提供器，便于后续接入交易所账户接口
- 新增：支持基于风险百分比（risk_percent）与止损距离的“风险等权”开仓

用法示例：
    from utils.position import PositionManager
    pm = PositionManager(cfg, price_provider=executor.guard.get_last_price)
    size = pm.compute_size("BTC-USDT", side="buy", price=None)  # 若不传 price 则会通过 price_provider 获取

注意：
- 目前默认场景为 USDT 计价标的（如 BTC-USDT），size 返回为“币数量”；
- 若行情价格不可用或计算出现异常，将返回 None，调用方需自行回退到默认手数。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Dict, Any

from utils.config import AppConfig


@dataclass
class PositionManagerConfig:
    """仓位管理计算所需的关键参数（从 AppConfig.exec / AppConfig.risk / AppConfig.discipline 中提取）。"""
    base_equity_usd: float
    single_order_max_pct_equity: float
    max_position_usd: float
    risk_percent: float  # 基于权益的单笔风险比例（来自 DisciplineConfig）


class PositionManager:
    """仓位管理器：根据价格与风险参数计算建议下单数量。

    可注入以下提供器：
    - price_provider(inst_id) -> Optional[float]：返回标的最新中间价/合理价
    - equity_provider() -> Optional[float]：返回账户当前权益（USD），不提供则使用配置中的 base_equity_usd
    - position_usd_provider(inst_id) -> Optional[float]：返回该标的当前持仓名义金额（USD），可用于风控限制（暂未使用）
    """

    def __init__(
        self,
        cfg: AppConfig,
        price_provider: Optional[Callable[[str], Optional[float]]] = None,
        equity_provider: Optional[Callable[[], Optional[float]]] = None,
        position_usd_provider: Optional[Callable[[str], Optional[float]]] = None,
    ) -> None:
        # 保存配置与提供器
        self.cfg = cfg
        self._price_provider = price_provider
        self._equity_provider = equity_provider
        self._position_usd_provider = position_usd_provider
        # 从配置抽取关键参数
        self.pm_cfg = PositionManagerConfig(
            base_equity_usd=float(self.cfg.exec.base_equity_usd),
            single_order_max_pct_equity=float(self.cfg.risk.single_order_max_pct_equity),
            max_position_usd=float(self.cfg.risk.max_position_usd),
            risk_percent=float(getattr(self.cfg.discipline, "risk_percent", 0.0) or 0.0),
        )

    def compute_size(self, inst_id: str, side: str, price: Optional[float] = None, stop_px: Optional[float] = None) -> Optional[float]:
        """计算建议下单手数（币数量）。

        计算逻辑：
        1) 获取价格 price（若未传入则尝试从 price_provider 获取），若仍无则返回 None
        2) 获取账户权益 equity_usd（若无提供器则使用 base_equity_usd）
        3) 计算单次下单名义金额上限：min(equity_usd * 单次占比, max_position_usd)
        4) 计算默认手数：notional_cap / price
        5) 若为 BUY 且配置了 risk_percent>0，则尝试基于“风险等权”计算：
           - 优先使用入参 stop_px；否则按标的固定百分比止损推导 stop_px = price*(1 - pct)
           - 单笔风险资金 risk_usd = equity_usd * risk_percent
           - 单币风险（USD）= price - stop_px（若 <=0 则回退默认手数）
           - 风险等权手数 risk_size = risk_usd / 单币风险
           - 最终 size = min(默认手数, 风险等权手数)
        6) 对结果进行安全裁剪与精度处理；失败则返回 None
        """
        try:
            # 价格获取
            px = price
            if px is None and self._price_provider is not None:
                try:
                    px = self._price_provider(inst_id)
                except Exception:
                    px = None
            if px is None or (isinstance(px, (int, float)) and float(px) <= 0):
                return None  # 无价格时无法按名义金额换算手数

            # 权益获取
            equity = None
            if self._equity_provider is not None:
                try:
                    equity = self._equity_provider()
                except Exception:
                    equity = None
            if equity is None:
                equity = float(self.pm_cfg.base_equity_usd)

            # 计算名义金额上限
            max_by_pct = float(equity) * float(self.pm_cfg.single_order_max_pct_equity)
            max_by_pos = float(self.pm_cfg.max_position_usd)
            notional_cap = max(0.0, min(max_by_pct, max_by_pos))
            if notional_cap <= 0:
                return None

            # 默认名义金额 -> 数量（币）
            default_size = notional_cap / float(px)
            # 基线校验
            if default_size <= 0:
                return None

            final_size = default_size

            # —— 风险等权 sizing（仅 BUY 场景）——
            try:
                if str(side).lower() == "buy" and float(self.pm_cfg.risk_percent) > 0:
                    spx = stop_px
                    if spx is None:
                        # 按标的固定止损百分比推导（优先使用 DisciplineConfig 中 BTC/ETH 百分比）
                        sym = str(inst_id).upper()
                        pct = None
                        if sym.startswith("BTC"):
                            pct = float(getattr(self.cfg.discipline, "stop_loss_pct_btc", 0.0) or 0.0)
                        elif sym.startswith("ETH"):
                            pct = float(getattr(self.cfg.discipline, "stop_loss_pct_eth", 0.0) or 0.0)
                        if pct is not None and pct > 0:
                            spx = float(px) * (1.0 - float(pct))
                    # 计算“单币风险”（USD）
                    if spx is not None and float(spx) > 0 and float(spx) < float(px):
                        per_unit_risk = float(px) - float(spx)
                        risk_usd = float(equity) * float(self.pm_cfg.risk_percent)
                        if per_unit_risk > 0 and risk_usd > 0:
                            risk_size = risk_usd / per_unit_risk
                            if risk_size > 0:
                                final_size = min(final_size, risk_size)
            except Exception:
                # 风险等权 sizing 失败则回退默认手数
                pass

            # 保留 6 位小数（大部分交易所常见精度；若后续接入交易规则可再精细化）
            size = round(final_size, 6)
            # 避免四舍五入到 0
            if size == 0.0:
                size = max(0.0, final_size)
            return size if size > 0 else None
        except Exception:
            # 任何异常均回退为 None，由调用方决定 fallback 策略
            return None

    # 允许在运行时更新提供器（例如切换为真实账户接口）
    def set_price_provider(self, fn: Callable[[str], Optional[float]]) -> None:
        """设置价格提供器。"""
        self._price_provider = fn

    def set_equity_provider(self, fn: Callable[[], Optional[float]]) -> None:
        """设置权益提供器。"""
        self._equity_provider = fn

    def set_position_usd_provider(self, fn: Callable[[str], Optional[float]]) -> None:
        """设置标的持仓名义金额提供器（预留）。"""
        self._position_usd_provider = fn