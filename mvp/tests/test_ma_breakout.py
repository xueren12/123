# -*- coding: utf-8 -*-
# 注释一律用中文
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

from strategies.ma_breakout import MABreakoutStrategy, MABreakoutConfig


def _mk_series(vals):
    # 以 1 分钟频率生成 UTC 时间索引
    idx = pd.date_range(end=pd.Timestamp.now(tz="UTC"), periods=len(vals), freq="min", tz="UTC")
    s = pd.Series(vals, index=idx, dtype=float)
    return s


def test_data_insufficient_returns_none():
    # slow_ma 或 brk 需要的长度 + 2 之前，都应返回 None
    cfg = MABreakoutConfig(inst_id="TEST", timeframe="1min", fast_ma=2, slow_ma=5, breakout_lookback=4)
    strat = MABreakoutStrategy(cfg)
    s = _mk_series([100, 101, 102, 103, 104])  # 长度小于 max(5,4)+2=7
    strat._load_price_series = lambda: (s, None)  # monkey patch
    assert strat.compute_signal() is None


def test_buy_signal_with_breakout_and_sl():
    # 构造序列：在最后一根产生金叉上破前N高点 -> BUY
    # 参数：fast=2, slow=3, brk=3
    cfg = MABreakoutConfig(inst_id="TEST", timeframe="1min", fast_ma=2, slow_ma=3, breakout_lookback=3, breakout_buffer_pct=0.0, stop_loss_pct=0.01)
    strat = MABreakoutStrategy(cfg)
    # 序列解释：
    # prev bar: fast(100,99)=99.5 <= slow(100,100,99)=99.67 -> 满足 f0<=s0
    # last bar: fast(105,100)=102.5 > slow(105,100,99)=101.33 -> 金叉
    # 前3根高点（不含当前）= max(100,99,100)=100；收盘105>100 -> 突破
    s = _mk_series([100, 100, 100, 99, 100, 105])
    strat._load_price_series = lambda: (s, None)
    sig = strat.compute_signal()
    assert sig is not None
    assert sig["signal"] == "BUY"
    # 止损价应为 c1*(1-0.01)
    assert abs(sig["sl"] - (105 * 0.99)) < 1e-9
    # timeframe 透出
    assert sig["timeframe"] == cfg.timeframe
    # brk_high 为“前N根”的高点
    assert abs(sig["brk_high"] - 100.0) < 1e-9


def test_sell_signal_with_breakdown_and_sl():
    # 构造序列：在最后一根产生死叉下破前N低点 -> SELL
    cfg = MABreakoutConfig(inst_id="TEST", timeframe="1min", fast_ma=2, slow_ma=3, breakout_lookback=3, breakout_buffer_pct=0.0, stop_loss_pct=0.01)
    strat = MABreakoutStrategy(cfg)
    # prev bar: fast(101,100)=100.5 >= slow(100,100,101)=100.33 -> 满足 f0>=s0
    # last bar: fast(95,100)=97.5 < slow(95,100,101)=98.67 -> 死叉
    # 前3根低点（不含当前）= min(100,101,100)=100；收盘95<100 -> 跌破
    s = _mk_series([100, 100, 100, 101, 100, 95])
    strat._load_price_series = lambda: (s, None)
    sig = strat.compute_signal()
    assert sig is not None
    assert sig["signal"] == "SELL"
    # 止损价应为 c1*(1+0.01)
    assert abs(sig["sl"] - (95 * 1.01)) < 1e-9
    # brk_low 为“前N根”的低点
    assert abs(sig["brk_low"] - 100.0) < 1e-9