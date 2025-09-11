# -*- coding: utf-8 -*-
"""
风控模块与执行器集成的单元测试
运行：
    pytest -q
或：
    python -m pytest -q
"""
from __future__ import annotations

import os
from datetime import datetime, timezone

import pytest

from utils.config import AppConfig, RiskConfig
from utils.risk import RiskManager, OrderIntent, MarketState, AccountState
from executor.trade_executor import TradeExecutor, TradeSignal


@pytest.fixture
def app_cfg() -> AppConfig:
    # 使用默认环境变量配置
    return AppConfig()


def test_risk_basic_rules(app_cfg: AppConfig):
    risk_cfg = RiskConfig(
        max_position_usd=1000,
        max_open_orders=1,
        single_order_max_pct_equity=0.2,
        max_slippage_pct=0.001,  # 0.1%
        max_single_trade_loss_usd=50,
        max_daily_loss_usd=100,
    )
    rm = RiskManager(app_cfg, risk_cfg)

    # 账户与市场
    account = AccountState(
        equity_usd=1000,
        position_usd_by_instrument={"BTC-USDT": 900},
        open_orders_count_by_instrument={"BTC-USDT": 1},
        daily_realized_pnl_usd=-99,  # 调整为-99，使潜在亏损与已亏之和超过上限，触发“日亏超限”提示
    )
    market = MarketState(mid_price=60000, best_bid=59990, best_ask=60010)

    # 订单：名义 0.01 * 60000 = 600 USD；会超过仓位上限（900 + 600 > 1000）
    order = OrderIntent(
        inst_id="BTC-USDT",
        side="buy",
        order_type="limit",
        qty=0.01,
        price=60000,
        reference_price=60000,
        stop_loss_price=59500,  # 潜在亏损 0.01*500 = 5 < 50（通过该条）
    )

    res = rm.validate_order(order, account, market)
    assert not res.allowed
    # 应至少包含仓位上限与未平委托限制、单次下单比例（1000*0.2=200 < 600）与日亏校验
    joined = ";".join(res.violations)
    assert "仓位上限" in joined
    assert "当前未完成订单数" in joined
    assert "超过单次上限" in joined
    assert "日亏" in joined


def test_risk_slippage_and_single_loss(app_cfg: AppConfig):
    risk_cfg = RiskConfig(
        max_position_usd=1000000,
        max_open_orders=10,
        single_order_max_pct_equity=1.0,
        max_slippage_pct=0.0005,  # 0.05%
        max_single_trade_loss_usd=10,  # 比较严格
        max_daily_loss_usd=1000,
    )
    rm = RiskManager(app_cfg, risk_cfg)
    account = AccountState(equity_usd=10000)
    market = MarketState(mid_price=2000)

    # 限价单：与 mid 偏差 1% >> 0.05% -> 违规
    order1 = OrderIntent(inst_id="ETH-USDT", side="buy", order_type="limit", qty=1, price=2020, reference_price=2000)
    r1 = rm.validate_order(order1, account, market)
    assert not r1.allowed and "滑点" in ";".join(r1.violations)

    # 市价单：无止损，worst_move_pct 估算亏损，若超过 10 USD 则违规
    order2 = OrderIntent(inst_id="ETH-USDT", side="buy", order_type="market", qty=0.5, reference_price=2000, worst_move_pct=0.02)
    r2 = rm.validate_order(order2, account, market)
    assert not r2.allowed and "单笔潜在亏损" in ";".join(r2.violations)


def test_executor_integrated_risk_block(app_cfg: AppConfig, tmp_path):
    # 强化风控使其容易阻断
    os.environ["RISK_MAX_POSITION_USD"] = "100"
    os.environ["RISK_MAX_OPEN_ORDERS"] = "0"
    os.environ["RISK_SINGLE_ORDER_MAX_PCT_EQUITY"] = "0.01"
    os.environ["RISK_MAX_SLIPPAGE_PCT"] = "0.0001"
    os.environ["RISK_MAX_SINGLE_TRADE_LOSS_USD"] = "1"
    os.environ["RISK_MAX_DAILY_LOSS_USD"] = "1"
    # 禁用 HTTP 管理端，避免测试时占用默认端口
    os.environ["EXEC_HTTP_ENABLED"] = "0"

    cfg = AppConfig()
    log_file = tmp_path / "trade_log.csv"
    exe = TradeExecutor(cfg, mode="mock", log_path=str(log_file))

    # 注入账户与行情状态，制造失败：
    def account_provider(inst_id: str) -> AccountState:
        return AccountState(
            equity_usd=1000,
            position_usd_by_instrument={inst_id: 200},
            open_orders_count_by_instrument={inst_id: 0},
            daily_realized_pnl_usd=-0.5,
        )

    def market_provider(inst_id: str, ref_price):
        return MarketState(mid_price=60000, best_bid=59990, best_ask=60010)

    exe.set_account_state_provider(account_provider)
    exe.set_market_state_provider(market_provider)

    sig = TradeSignal(
        ts=datetime.now(timezone.utc),
        symbol="BTC-USDT",
        side="buy",
        price=60600,  # 相对 mid 偏差 1% -> 超过 0.01%
        size=0.05,    # 名义 3000 -> 超过单次比例 1000*1% = 10
        reason="risk-test",
        meta={"ordType": "limit", "expectedSlippagePct": 0.02},
    )
    r = exe.execute(sig)
    assert not r.ok and r.err and r.err.startswith("risk_reject")