# -*- coding: utf-8 -*-
"""
K线驱动的均线+突破策略回测（独立模块）

说明：
- 从 TimescaleDB 读取 trades（优先）或 orderbook(mid) 数据，重采样为指定 timeframe 的收盘价序列；
- 计算快/慢均线与近N高低点，生成 BUY/SELL/HOLD 信号；
- 采用简单持仓规则：
  - BUY -> 持仓 = +1，SELL -> 持仓 = -1，HOLD -> 持仓沿用上一根；
- 收益计算：ret = pos.shift(1) * close.pct_change()，可选扣除双边手续费；
- 输出：回测摘要、CSV（data/backtest_multi_indicator.csv）、SVG 图（data/backtest_multi_indicator.svg）。

用法示例（PowerShell）：
  # 从数据库读取（默认）
  py backtest/ma_backtest.py --inst BTC-USDT --start "2025-01-01" --end "2025-01-07" \
      --timeframe 5min --fast 10 --slow 30 --brk 20 --buffer 0.001 --fee_bps 2 --plot 1

  # 从本地CSV读取（方案一：离线拉取 → 保存 → 回测）
  py backtest/ma_backtest.py --source csv --csv "data/ohlcv_BTC-USDT_1m_2025-01-01_2025-01-07.csv" \
      --inst BTC-USDT --start "2025-01-01" --end "2025-01-07" --timeframe 1min

注意：
- 需要数据库中存在该交易对在时间区间内的 trades 或 orderbook 数据，或本地 CSV 含有 ts/open/high/low/close/volume 列。
- 本模块仅为示例级回测，未考虑滑点/成交细节/资金曲线极端值稳定性等问题。
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
import json  # 新增：用于写出回测汇总 JSON

# 兼容脚本直接运行的导入路径（将 mvp 目录加入 sys.path）
# 说明：当直接执行 mvp/backtest/ma_backtest.py 时，默认工作目录可能无法解析 from utils.xxx
# 因此将上级目录（mvp）加入模块搜索路径，保证内部包可正常导入。
import os as _os
import sys as _sys
_CUR_DIR = _os.path.dirname(_os.path.abspath(__file__))
_MVP_DIR = _os.path.abspath(_os.path.join(_CUR_DIR, ".."))
if _MVP_DIR not in _sys.path:
    _sys.path.insert(0, _MVP_DIR)

from utils.db import TimescaleDB


@dataclass
class MABacktestConfig:
    # 基础参数
    inst: str
    start: datetime
    end: datetime
    timeframe: str = "5min"
    # 旧参数（兼容用，不再直接用于信号计算）
    fast: int = 10
    slow: int = 30
    brk: int = 20
    buffer: float = 0.0
    # 成本与输出
    fee_bps: float = 0.0  # 双边手续费，单位: bp（基点）。例如 2 表示 0.02%
    slip_bps: float = 0.0  # 新增：滑点/点差成本，单位: bp（基点），按换手额收取
    # ===== 合约交易相关参数 =====
    leverage: float = 50              # 杠杆倍数（仅做多方向），>=1
    funding_bps_8h: float = 0.0        # 资金费率：每8小时的bp，可为负表示收取资金费
    mmr_bps: float = 50.0              # 维持保证金率（bp），默认0.50%
    liq_penalty_bps: float = 10.0      # 强平附加惩罚（bp）
    # ==========================
    stop_loss_pct: Optional[float] = None  # 可选：对称止损百分比（仅记录，不强制执行）
    plot: bool = True
    # 数据来源与CSV路径
    source: str = "db"  # db/csv
    csv_path: Optional[str] = None  # 当 source=csv 时必填
    # 新策略参数（多指标确认）
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_period: int = 20
    bb_k: float = 2.0
    rsi_period: int = 14
    rsi_buy: float = 55.0
    rsi_sell: float = 45.0
    aroon_period: int = 25
    aroon_buy: float = 70.0
    aroon_sell: float = 30.0
    confirm_min: int = 3


class MABreakoutBacktester:
    def __init__(self, cfg: MABacktestConfig) -> None:
        self.cfg = cfg
        # 仅当需要从数据库读取时再初始化连接，避免纯CSV模式下的无谓依赖与失败
        self.db = None
        # 新增：基于起止日期的独立输出目录，例如 data/start_YYYYMMDD_end_YYYYMMDD
        try:
            start_str = cfg.start.strftime("%Y%m%d")
            end_str = cfg.end.strftime("%Y%m%d")
        except Exception:
            start_str = "start"
            end_str = "end"
        tf_str = str(cfg.timeframe).strip().replace("/", "-")
        # 在目录名中加入 inst（仅取基础币种，如 ETH-USDT/ETH/USDT → ETH）
        inst_base = str(cfg.inst).upper().replace("-SWAP", "")
        inst_base = inst_base.split("-")[0].split("/")[0]
        self.output_dir = _os.path.join("data", f"start_{start_str}_end_{end_str}_tf_{tf_str}_inst_{inst_base}")
        _os.makedirs(self.output_dir, exist_ok=True)

    @staticmethod
    def _parse_timeframe(tf: str) -> Tuple[str, int]:
        """timeframe 转为 pandas 频率与每bar秒数"""
        s = str(tf).lower().strip().replace("min", "m")
        if s.endswith("m"):
            mins = int(s[:-1])
            return f"{mins}min", mins * 60
        if s.endswith("s"):
            secs = int(s[:-1])
            return f"{secs}s", secs
        return "1min", 60

    def _load_close_series(self) -> pd.Series:
        """加载收盘价序列：支持 DB 与 本地CSV 两种来源。"""
        freq, step_secs = self._parse_timeframe(self.cfg.timeframe)
        end_db = self.cfg.end + timedelta(seconds=step_secs)
        # =========== 分支一：从CSV加载（方案一的离线文件） ==========
        if str(self.cfg.source).lower() == "csv":
            path = self.cfg.csv_path
            if not path:
                raise RuntimeError("当 source=csv 时必须提供 --csv 路径")
            try:
                # 兼容两种列名风格：ts/open/high/low/close/volume 或 timestamp/o/h/l/c/v
                df = pd.read_csv(path)
            except Exception as e:
                raise RuntimeError(f"读取CSV失败：{e}")
            # 解析时间列
            ts_col = None
            for c in ["ts", "timestamp", "time", "date"]:
                if c in df.columns:
                    ts_col = c
                    break
            if ts_col is None:
                raise RuntimeError("CSV 缺少时间列（期望: ts/timestamp/time/date 之一）")
            try:
                s_ts = pd.to_datetime(df[ts_col], utc=True)
            except Exception:
                # 若为毫秒时间戳
                s_ts = pd.to_datetime(df[ts_col].astype("int64"), unit="ms", utc=True)
            df = df.assign(ts=s_ts).set_index("ts").sort_index()
            # 解析 close 列
            close_col = None
            for c in ["close", "c", "Close", "last"]:
                if c in df.columns:
                    close_col = c
                    break
            if close_col is None:
                raise RuntimeError("CSV 缺少收盘价列（期望: close/c/last 之一）")
            s = df[close_col].astype(float).resample(freq).last().ffill()
            # 按回测窗口过滤，保持与 DB 分支一致
            s = s[(s.index >= self.cfg.start) & (s.index <= self.cfg.end)].dropna()
            if s.empty:
                raise RuntimeError("CSV 转换后序列为空，请检查时间区间与列名是否正确")
            return s

        # =========== 分支二：从DB加载（原有逻辑） ==========
        # 读取 trades -> resample last -> ffill
        tr = None
        try:
            # 懒初始化数据库连接
            if self.db is None:
                self.db = TimescaleDB()
            tr = self.db.fetch_trades_window(start=self.cfg.start, end=end_db, inst_id=self.cfg.inst, ascending=True)
        except Exception as e:
            logger.debug("读取 trades 失败，将回退 orderbook：{}", e)
        if tr is not None and not tr.empty:
            tr["ts"] = pd.to_datetime(tr["ts"], utc=True)
            s = tr.set_index("ts").sort_index()["price"].astype(float).resample(freq).last()
            s = s.ffill()
            s = s[(s.index >= self.cfg.start) & (s.index <= self.cfg.end)].dropna()
            if len(s) > 0:
                return s
        # 回退到 orderbook mid
        # 若此前未成功初始化，则此处再尝试一次（以便直接回退到 orderbook）
        if self.db is None:
            self.db = TimescaleDB()
        ob = self.db.fetch_orderbook_window(start=self.cfg.start, end=end_db, inst_id=self.cfg.inst, ascending=True)
        if ob is None or ob.empty:
            raise RuntimeError("数据库中缺少可用的 trades/orderbook 数据以构造收盘价序列")
        ob["ts"] = pd.to_datetime(ob["ts"], utc=True)
        ob = ob.set_index("ts").sort_index()
        def _mid(row):
            try:
                bb = float(row["bids"][0][0]) if row["bids"] else np.nan
                ba = float(row["asks"][0][0]) if row["asks"] else np.nan
                return np.nanmean([bb, ba])
            except Exception:
                return np.nan
        ob["mid"] = ob.apply(_mid, axis=1)
        s = ob["mid"].resample(freq).last().ffill()
        s = s[(s.index >= self.cfg.start) & (s.index <= self.cfg.end)].dropna()
        if s.empty:
            raise RuntimeError("构造收盘价序列失败：orderbook 数据不足")
        return s

    def run(self) -> dict:
        """多指标确认 + 风控/止盈止损 的回测主循环（仅做多，支持分批止盈与移动止损）"""
        # 1) 加载收盘价（OHLC 近似：用前一根与当前收盘生成高低，避免 ATR=0）
        close = self._load_close_series().astype(float)
        prev_close = close.shift(1)
        high = pd.concat([close, prev_close], axis=1).max(axis=1).astype(float)
        low = pd.concat([close, prev_close], axis=1).min(axis=1).astype(float)

        # 2) 指标计算
        # MACD
        ema_fast = close.ewm(span=int(self.cfg.macd_fast), adjust=False).mean()
        ema_slow = close.ewm(span=int(self.cfg.macd_slow), adjust=False).mean()
        macd_line = ema_fast - ema_slow
        macd_signal = macd_line.ewm(span=int(self.cfg.macd_signal), adjust=False).mean()
        # 布林
        bb_mid = close.rolling(int(self.cfg.bb_period)).mean()
        bb_std = close.rolling(int(self.cfg.bb_period)).std(ddof=0)
        bb_upper = bb_mid + float(self.cfg.bb_k) * bb_std
        bb_lower = bb_mid - float(self.cfg.bb_k) * bb_std
        # RSI（Wilder）
        delta = close.diff()
        up = np.where(delta > 0, delta, 0.0)
        down = np.where(delta < 0, -delta, 0.0)
        roll_up = pd.Series(up, index=close.index).ewm(alpha=1 / float(self.cfg.rsi_period), adjust=False).mean()
        roll_down = pd.Series(down, index=close.index).ewm(alpha=1 / float(self.cfg.rsi_period), adjust=False).mean()
        rs = roll_up / roll_down.replace(0, np.nan)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        # Aroon（基于窗口内 argmax/argmin 位置近似）
        n_aroon = int(self.cfg.aroon_period)
        argmax = close.rolling(n_aroon).apply(lambda x: float(np.argmax(x)), raw=True)
        argmin = close.rolling(n_aroon).apply(lambda x: float(np.argmin(x)), raw=True)
        aroon_up = (argmax + 1.0) / n_aroon * 100.0
        aroon_down = (argmin + 1.0) / n_aroon * 100.0
        # ATR（True Range，周期14）
        atr_period = 14
        tr1 = (high - low).abs()
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(int(atr_period)).mean()

        # 3) 回测主循环（逐bar状态机）
        idx = list(close.index)
        signals, pos_list, ret_list, eq_list = [], [], [], []
        fast_list, slow_list, hi_list, lo_list = [], [], [], []

        _pos = 0               # 0=空，1=多
        _entry_price = None
        _tp1_done = False
        _tp2_done = False
        _remain_frac = 0.0
        _trail_stop = None
        _entry_high = None
        _init_risk = None

        fee_rate = abs(self.cfg.fee_bps) / 10000.0
        slip_rate = abs(self.cfg.slip_bps) / 10000.0  # 新增：滑点按换手额收取
        cost_rate = fee_rate + slip_rate  # 总成本率 = 手续费 + 滑点
        equity = 1.0
        last_pos_frac = 0.0
        # ===== 合约交易：杠杆/资金费/强平参数（每bar计提） =====
        L = max(1.0, float(self.cfg.leverage))  # 杠杆倍数，最小为1
        _, step_secs = self._parse_timeframe(self.cfg.timeframe)  # 每bar秒数
        # 资金费换算：传入为每8小时bp -> 每秒费率 -> 乘以bar秒数得到每bar费率
        funding_rate_bar = (float(self.cfg.funding_bps_8h) / 10000.0) / (8.0 * 3600.0) * float(step_secs)
        mmr_rate = float(self.cfg.mmr_bps) / 10000.0  # 维持保证金率（比例）
        liq_pen = float(self.cfg.liq_penalty_bps) / 10000.0  # 强平惩罚（按权益比例扣减）
        # ====================================================
        min_need = max(
            int(self.cfg.macd_slow + self.cfg.macd_signal + 5),
            int(self.cfg.bb_period + 5),
            int(self.cfg.rsi_period + 5),
            int(self.cfg.aroon_period + 1),
            int(atr_period + 5),
        )

        for i, t in enumerate(idx):
            c1 = float(close.iloc[i]) if pd.notna(close.iloc[i]) else None
            c0 = float(close.iloc[i-1]) if i > 0 and pd.notna(close.iloc[i-1]) else None
            macd1 = float(macd_line.iloc[i]) if pd.notna(macd_line.iloc[i]) else None
            sig1 = float(macd_signal.iloc[i]) if pd.notna(macd_signal.iloc[i]) else None
            macd0 = float(macd_line.iloc[i-1]) if i > 0 and pd.notna(macd_line.iloc[i-1]) else None
            sig0 = float(macd_signal.iloc[i-1]) if i > 0 and pd.notna(macd_signal.iloc[i-1]) else None
            bb_up1 = float(bb_upper.iloc[i]) if pd.notna(bb_upper.iloc[i]) else np.nan
            bb_low1 = float(bb_lower.iloc[i]) if pd.notna(bb_lower.iloc[i]) else np.nan
            rsi1 = float(rsi.iloc[i]) if pd.notna(rsi.iloc[i]) else None
            a_up1 = float(aroon_up.iloc[i]) if pd.notna(aroon_up.iloc[i]) else None
            a_dn1 = float(aroon_down.iloc[i]) if pd.notna(aroon_down.iloc[i]) else None
            a_up0 = float(aroon_up.iloc[i-1]) if i > 0 and pd.notna(aroon_up.iloc[i-1]) else None
            a_dn0 = float(aroon_down.iloc[i-1]) if i > 0 and pd.notna(aroon_down.iloc[i-1]) else None
            high1 = float(high.iloc[i]) if pd.notna(high.iloc[i]) else None
            atr1 = float(atr.iloc[i]) if pd.notna(atr.iloc[i]) else None

            out_sig = "HOLD"
            size_frac_suggest = None

            if i >= (min_need - 1) and c1 is not None:
                # 多指标确认
                macd_cross_up = (macd0 is not None and sig0 is not None and macd1 is not None and sig1 is not None and (macd0 <= sig0) and (macd1 > sig1))
                macd_cross_dn = (macd0 is not None and sig0 is not None and macd1 is not None and sig1 is not None and (macd0 >= sig0) and (macd1 < sig1))
                macd_bull_ok = (macd_cross_up or (macd1 is not None and sig1 is not None and macd1 > sig1 and macd1 > 0))
                macd_bear_ok = (macd_cross_dn or (macd1 is not None and sig1 is not None and macd1 < sig1 and macd1 < 0))
                bb_bull_ok = (not np.isnan(bb_up1)) and (c1 > bb_up1)
                bb_bear_ok = (not np.isnan(bb_low1)) and (c1 < bb_low1)
                rsi_bull_ok = (rsi1 is not None) and (rsi1 >= float(self.cfg.rsi_buy))
                rsi_bear_ok = (rsi1 is not None) and (rsi1 <= float(self.cfg.rsi_sell))
                bear_thresh = max(float(self.cfg.aroon_buy), 100.0 - float(self.cfg.aroon_sell))
                aroon_bull_ok = (a_up1 is not None and a_dn1 is not None) and (a_up1 >= float(self.cfg.aroon_buy)) and (a_up1 > a_dn1)
                aroon_bear_ok = (a_up1 is not None and a_dn1 is not None) and (a_dn1 >= bear_thresh) and (a_dn1 > a_up1)
                buy_confirms = int(macd_bull_ok) + int(bb_bull_ok) + int(rsi_bull_ok) + int(aroon_bull_ok)
                sell_confirms = int(macd_bear_ok) + int(bb_bear_ok) + int(rsi_bear_ok) + int(aroon_bear_ok)
                need = int(max(1, min(4, int(self.cfg.confirm_min))))
                if buy_confirms >= need and buy_confirms > sell_confirms:
                    out_sig = "BUY"
                elif sell_confirms >= need and sell_confirms > buy_confirms:
                    out_sig = "SELL"

                # 止盈（仅多头）
                tp_triggered = False
                if _pos > 0 and _entry_price is not None:
                    if high1 is not None:
                        _entry_high = high1 if _entry_high is None else max(_entry_high, high1)
                    if atr1 is not None and atr1 > 0 and _entry_high is not None:
                        trail_cand = float(_entry_high) - 1.5 * float(atr1)
                        _trail_stop = trail_cand if _trail_stop is None else max(_trail_stop, trail_cand)
                    # Aroon 反转：全平
                    aroon_rev = False
                    if a_up0 is not None and a_dn0 is not None and a_up1 is not None and a_dn1 is not None:
                        aroon_rev = (a_up0 >= a_dn0) and (a_dn1 > a_up1)
                    if aroon_rev and (_remain_frac is None or _remain_frac > 0):
                        out_sig = "SELL"; size_frac_suggest = float(_remain_frac or 1.0); tp_triggered = True
                    # 移动止损
                    if (not tp_triggered) and _trail_stop is not None and c1 <= float(_trail_stop):
                        out_sig = "SELL"; size_frac_suggest = float(_remain_frac or 1.0); tp_triggered = True
                    # R 倍数分批止盈
                    if (not tp_triggered) and _init_risk is not None and _init_risk > 0:
                        r_mult = (c1 - float(_entry_price)) / float(_init_risk)
                        if (not _tp2_done) and r_mult >= 2.0 and (_remain_frac or 0.0) > 0:
                            out_sig = "SELL"; size_frac_suggest = float(min(0.3, float(_remain_frac or 1.0))); tp_triggered = True; _tp2_done = True
                        elif (not _tp1_done) and r_mult >= 1.0 and (_remain_frac or 0.0) > 0:
                            out_sig = "SELL"; size_frac_suggest = float(min(0.3, float(_remain_frac or 1.0))); tp_triggered = True; _tp1_done = True
                    # RSI 技术止盈（部分）
                    if (not tp_triggered) and rsi1 is not None and (rsi1 >= 80.0 or rsi1 <= 20.0):
                        out_sig = "SELL"; size_frac_suggest = float(min(0.2, float(_remain_frac or 1.0)))

                # 止损（若未触发止盈）
                if _pos > 0 and _entry_price is not None and (size_frac_suggest is None or size_frac_suggest >= 0.999):
                    stop_hit = None
                    # ATR 止损
                    if atr1 is not None and atr1 > 0:
                        atr_stop = float(_entry_price) - 2.0 * float(atr1)
                        if c1 <= atr_stop:
                            stop_hit = True
                    # 百分比止损（按BTC/ETH不同）
                    if stop_hit is None:
                        sym = str(self.cfg.inst).upper()
                        pct_default = 0.03 if sym.startswith("BTC") else (0.04 if sym.startswith("ETH") else None)
                        stop_pct = float(self.cfg.stop_loss_pct) if (self.cfg.stop_loss_pct is not None) else pct_default
                        if stop_pct is not None and stop_pct > 0:
                            pct_stop = float(_entry_price) * (1.0 - float(stop_pct))
                            if c1 <= pct_stop:
                                stop_hit = True
                    # 技术指标止损
                    if stop_hit is None:
                        macd_cross_dn2 = (macd0 is not None and sig0 is not None and macd1 is not None and sig1 is not None and (macd0 >= sig0) and (macd1 < sig1))
                        if (not np.isnan(bb_low1) and c1 < bb_low1) or macd_cross_dn2:
                            stop_hit = True
                    if stop_hit:
                        out_sig = "SELL"; size_frac_suggest = float(_remain_frac or 1.0)

                # 状态更新（支持部分止盈）
                if out_sig == "BUY" and _pos == 0:
                    _pos = 1
                    _entry_price = c1
                    _entry_high = high1
                    _tp1_done = False
                    _tp2_done = False
                    _remain_frac = 1.0
                    _trail_stop = None
                    # 入场时估算初始风险 R
                    eff_stop = None
                    if atr1 is not None and atr1 > 0:
                        eff_stop = max(eff_stop or -1e18, float(_entry_price) - 2.0 * float(atr1))
                    sym = str(self.cfg.inst).upper()
                    pct_default = 0.03 if sym.startswith("BTC") else (0.04 if sym.startswith("ETH") else None)
                    stop_pct = float(self.cfg.stop_loss_pct) if (self.cfg.stop_loss_pct is not None) else pct_default
                    if stop_pct is not None and stop_pct > 0:
                        eff_stop = max(eff_stop or -1e18, float(_entry_price) * (1.0 - float(stop_pct)))
                    _init_risk = (float(_entry_price) - eff_stop) if (eff_stop is not None and eff_stop < float(_entry_price)) else float(_entry_price) * 0.01
                elif out_sig == "SELL" and _pos == 1:
                    if size_frac_suggest is not None and size_frac_suggest < 0.999:
                        _remain_frac = max(0.0, float(_remain_frac or 0.0) - float(size_frac_suggest))
                        if _remain_frac <= 1e-6:
                            _pos = 0; _entry_price = None; _entry_high = None; _trail_stop = None; _init_risk = None; _tp1_done = False; _tp2_done = False
                    else:
                        _pos = 0; _entry_price = None; _entry_high = None; _trail_stop = None; _init_risk = None; _tp1_done = False; _tp2_done = False

            # 当根结束：计算净值变化（上一根仓位 * 本根涨跌 - 换手手续费）
            cur_pos_frac = float(_remain_frac if _pos == 1 else 0.0)
            bar_ret = 0.0 if c0 is None or c1 is None else (c1 / c0 - 1.0)
            # ===== 基于合约的收益、成本与资金费 =====
            IM = float(last_pos_frac)                                            # 初始保证金占比（上一根）
            gross = float(last_pos_frac) * float(L) * float(bar_ret)             # 杠杆放大的毛收益（占比）
            # 先按策略意图计算换手，再根据是否强平做调整
            delta_pos_pre = abs(cur_pos_frac - last_pos_frac)
            cost_turn_pre = delta_pos_pre * cost_rate * float(L)                 # 成交成本（按名义额，即乘以L）
            fund_cost = float(last_pos_frac) * float(L) * float(funding_rate_bar)  # 资金费（正为支付，负为收取）
            MB_pre = IM + gross - fund_cost - cost_turn_pre                       # 预估保证金余额（占比）
            MM = IM * float(L) * float(mmr_rate)                                  # 维持保证金占比
            forced_liq = (IM > 0.0) and (MB_pre <= MM)
            if forced_liq:
                # 若触发强平：按强制平仓到0仓位计算最终成本，并施加额外惩罚
                cur_pos_frac = 0.0
                delta_pos = abs(cur_pos_frac - last_pos_frac)
                cost_turn = delta_pos * cost_rate * float(L)
                MB = IM + gross - fund_cost - cost_turn
                net = (MB - IM) - float(liq_pen)
                _pos = 0; _remain_frac = 0.0; _entry_price = None; _entry_high = None; _trail_stop = None; _init_risk = None; _tp1_done = False; _tp2_done = False
            else:
                delta_pos = delta_pos_pre
                cost_turn = cost_turn_pre
                net = gross - cost_turn - fund_cost
            equity *= (1.0 + net)

            signals.append(out_sig)
            pos_list.append(cur_pos_frac)
            ret_list.append(net)
            eq_list.append(equity)
            fast_list.append(float(macd1) if macd1 is not None and not np.isnan(macd1) else np.nan)
            slow_list.append(float(sig1) if sig1 is not None and not np.isnan(sig1) else np.nan)
            hi_list.append(bb_up1)
            lo_list.append(bb_low1)

            last_pos_frac = cur_pos_frac

        df = pd.DataFrame({
            "close": close,
            "fast": pd.Series(fast_list, index=close.index),
            "slow": pd.Series(slow_list, index=close.index),
            "highN": pd.Series(hi_list, index=close.index),
            "lowN": pd.Series(lo_list, index=close.index),
            "highN_prev": pd.Series(hi_list, index=close.index),
            "lowN_prev": pd.Series(lo_list, index=close.index),
            "signal": pd.Series(signals, index=close.index),
            "pos": pd.Series(pos_list, index=close.index),
            "ret": pd.Series(ret_list, index=close.index),
            "equity": pd.Series(eq_list, index=close.index),
        })

        stats = self._summary_stats(df)
        self._export(df)
        
        # 新增：计算胜率（winrate）并写出回测汇总 JSON，供 gradual_deploy.py 阈值检查使用
        # 计算规则：将连续持仓区间视为一笔交易（pos!=0 的连续片段），以该区间净收益(ret)之和判断盈亏
        try:
            pos_series = df["pos"].fillna(0.0)
            ret_series = df["ret"].fillna(0.0)
            in_pos = pos_series.ne(0.0)
            prev = in_pos.shift(1)
            next_ = in_pos.shift(-1)
            starts = in_pos & (~prev.fillna(False).astype(bool))
            ends = in_pos & (~next_.fillna(False).astype(bool))
            start_idx = list(df.index[starts])
            end_idx = list(df.index[ends])
            wins = 0
            trades = 0

            for s, e in zip(start_idx, end_idx):
                seg_pnl = float(ret_series.loc[s:e].sum())
                trades += 1
                if seg_pnl > 0:
                    wins += 1
            winrate = float(wins / trades) if trades > 0 else 0.0
        except Exception:
            # 兜底：异常时将胜率置 0.0（避免阻塞后续流程）
            winrate = 0.0

        # 约定输出路径：data/backtest_summary.json（与 scripts/gradual_deploy.py 默认读取一致）
        try:
            summary_path = _os.path.join(self.output_dir, "backtest_summary.json")
            _os.makedirs(_os.path.dirname(summary_path), exist_ok=True)
            summary_payload = {
                "metrics": {
                    # 夏普与回撤直接来自统计结果；回撤输出为正值幅度，便于与 0.05 之类的阈值比较
                    "sharpe": float(stats.get("sharpe", 0.0)),
                    "winrate": float(winrate),
                    "max_drawdown": float(abs(stats.get("max_dd", 0.0))),
                },
                "context": {
                    "inst": self.cfg.inst,
                    "start": self.cfg.start.isoformat(),
                    "end": self.cfg.end.isoformat(),
                    "timeframe": self.cfg.timeframe,
                    "bars": int(len(df)),
                },
            }
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary_payload, f, ensure_ascii=False, indent=2)
            logger.info("已生成回测汇总 JSON -> {}", summary_path)
        except Exception as e:
            logger.warning("写出回测汇总 JSON 失败：{}", e)

        return {"df": df, "stats": stats}

    def _summary_stats(self, df: pd.DataFrame) -> dict:
        """统计汇总：
        - 夏普改为按真实bar间隔年化：以索引时间差的中位数估算每bar秒数
        - 年化收益=bar平均收益*年化bar数；年化波动=bar标准差*sqrt(年化bar数)
        - 风险自由利率默认0
        """
        eq = df["equity"].dropna()
        if eq.empty:
            return {"final_equity": 1.0, "return_pct": 0.0, "sharpe": 0.0, "max_dd": 0.0}
        final_eq = float(eq.iloc[-1])
        ret = df["ret"].fillna(0.0)
        # 推断每bar秒数（若失败则回退到timeframe解析结果）
        step_secs = None
        try:
            idx = df.index
            if hasattr(idx, "to_series"):
                dts = idx.to_series().diff().dt.total_seconds().dropna()
                if len(dts) > 0:
                    step_secs = float(dts.median())
        except Exception:
            step_secs = None
        if step_secs is None or not np.isfinite(step_secs) or step_secs <= 0:
            # 回退：使用配置的timeframe估算
            _, step_secs = self._parse_timeframe(self.cfg.timeframe)
        # 年化bar数（按加密市场7x24）：365天 * 24小时 * 3600秒 / 每bar秒数
        bars_per_year = (365.0 * 24.0 * 3600.0) / float(step_secs)
        mu = ret.mean() * bars_per_year
        sigma = ret.std(ddof=0) * np.sqrt(bars_per_year)
        sharpe = (mu / sigma) if sigma > 1e-12 else 0.0
        cummax = eq.cummax()
        dd = (eq / cummax - 1.0).min() if len(eq) > 0 else 0.0
        return {"final_equity": final_eq, "return_pct": (final_eq - 1.0) * 100.0, "sharpe": float(sharpe), "max_dd": float(dd)}

    def _export(self, df: pd.DataFrame) -> None:
        csv_path = _os.path.join(self.output_dir, "backtest_multi_indicator.csv")
        svg_path = _os.path.join(self.output_dir, "backtest_multi_indicator.svg")
        try:
            df.to_csv(csv_path, index=True)
            logger.info("已导出回测明细CSV：{}", csv_path)
        except Exception as e:
            logger.warning("导出CSV失败：{}", e)
        # 绘图（价格与权益）
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax1 = plt.subplots(figsize=(10, 5))
            df[["close"]].plot(ax=ax1, color="tab:blue", label="Close")
            ax2 = ax1.twinx()
            df[["equity"]].plot(ax=ax2, color="tab:green", label="Equity")
            ax1.set_title("Multi-Indicator Backtest")
            ax1.set_xlabel("Time (UTC)")
            ax1.set_ylabel("Price")
            ax2.set_ylabel("Equity")
            plt.tight_layout()
            fig.savefig(svg_path, format="svg")
            plt.close(fig)
            logger.info("已导出回测图表SVG：{}", svg_path)
        except Exception as e:
            logger.warning("导出SVG失败：{}", e)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="多指标确认策略回测（K线驱动）")
    p.add_argument("--inst", required=True, help="交易对，如 BTC-USDT")
    p.add_argument("--start", required=True, help="起始时间，ISO或日期，如 2025-01-01")
    p.add_argument("--end", required=True, help="结束时间，ISO或日期，如 2025-01-07")
    p.add_argument("--timeframe", default="5min", help="时间粒度，如 1min/5min/15min")
    # 旧参数（兼容保留）
    p.add_argument("--fast", type=int, default=10, help="兼容旧参数：快线窗口（映射为 MACD 快线窗口）")
    p.add_argument("--slow", type=int, default=30, help="兼容旧参数：慢线窗口（映射为 MACD 慢线窗口）")
    p.add_argument("--brk", type=int, default=20, help="兼容旧参数：突破回看窗口（映射为布林带窗口）")
    p.add_argument("--buffer", type=float, default=0.0, help="兼容旧参数：突破缓冲比例（不再使用）")
    p.add_argument("--fee_bps", type=float, default=0.0, help="双边手续费，基点，例如 2=0.02%")
    p.add_argument("--slip_bps", type=float, default=0.0, help="滑点/点差成本，基点，例如 2=0.02%")
    # ===== 合约交易相关参数 =====
    p.add_argument("--leverage", type=float, default=1.0, help="杠杆倍数（仅做多方向），>=1")
    p.add_argument("--funding_bps_8h", type=float, default=0.0, help="资金费率：每8小时的bp，可为负表示收取资金费")
    p.add_argument("--mmr_bps", type=float, default=50.0, help="维持保证金率（bp），默认0.50%")
    p.add_argument("--liq_penalty_bps", type=float, default=10.0, help="强平附加惩罚（bp）")
    # ==========================
    p.add_argument("--sl_pct", type=float, default=None, help="可选：止损百分比（不在回测中强制执行，仅记录）")
    p.add_argument("--plot", type=int, default=1, help="是否导出SVG图：1是 0否")
    # 新策略参数（若提供则覆盖旧参数所映射的默认值）
    p.add_argument("--macd_fast", type=int, default=None, help="MACD EMA 快线窗口")
    p.add_argument("--macd_slow", type=int, default=None, help="MACD EMA 慢线窗口")
    p.add_argument("--macd_signal", type=int, default=None, help="MACD 信号线窗口")
    p.add_argument("--bb_period", type=int, default=None, help="布林带窗口")
    p.add_argument("--bb_k", type=float, default=None, help="布林带倍数（标准差倍数）")
    p.add_argument("--rsi_period", type=int, default=None, help="RSI 窗口")
    p.add_argument("--rsi_buy", type=float, default=None, help="RSI 多头阈值")
    p.add_argument("--rsi_sell", type=float, default=None, help="RSI 空头阈值")
    p.add_argument("--aroon_period", type=int, default=None, help="Aroon 窗口")
    p.add_argument("--aroon_buy", type=float, default=None, help="Aroon 多头阈值（Up）")
    p.add_argument("--aroon_sell", type=float, default=None, help="Aroon 空头阈值（Down）")
    p.add_argument("--confirm_min", type=int, default=None, help="最少确认数（1~4）")
    # 数据来源与CSV路径
    p.add_argument("--source", choices=["db", "csv"], default="db", help="数据来源：db 从数据库读取；csv 从本地CSV读取")
    p.add_argument("--csv", dest="csv_path", default=None, help="当 --source=csv 时，指定本地CSV文件路径")
    return p.parse_args()


def _parse_dt(s: str) -> datetime:
    s = s.strip()
    txt = s.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(txt)
        # 若无时区信息，则按 UTC 解释（不做本地->UTC 的转换）
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        # 仅日期格式
        from datetime import datetime as _dt
        return _dt.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)


if __name__ == "__main__":
    args = _parse_args()
    # 组装新策略参数：若未提供新参数则使用旧参数映射/默认值
    macd_fast = int(args.macd_fast) if args.macd_fast is not None else int(args.fast)
    macd_slow = int(args.macd_slow) if args.macd_slow is not None else int(args.slow)
    macd_signal = int(args.macd_signal) if args.macd_signal is not None else 9
    bb_period = int(args.bb_period) if args.bb_period is not None else int(args.brk)
    bb_k = float(args.bb_k) if args.bb_k is not None else 2.0
    rsi_period = int(args.rsi_period) if args.rsi_period is not None else 14
    rsi_buy = float(args.rsi_buy) if args.rsi_buy is not None else 55.0
    rsi_sell = float(args.rsi_sell) if args.rsi_sell is not None else 45.0
    aroon_period = int(args.aroon_period) if args.aroon_period is not None else 25
    aroon_buy = float(args.aroon_buy) if args.aroon_buy is not None else 70.0
    aroon_sell = float(args.aroon_sell) if args.aroon_sell is not None else 30.0
    confirm_min = int(args.confirm_min) if args.confirm_min is not None else 3

    cfg = MABacktestConfig(
        inst=args.inst,
        start=_parse_dt(args.start),
        end=_parse_dt(args.end),
        timeframe=args.timeframe,
        # 旧参数原样保留（仅用于向后兼容日志/导出）
        fast=max(2, int(args.fast)),
        slow=max(3, int(args.slow)),
        brk=max(2, int(args.brk)),
        buffer=float(args.buffer),
        fee_bps=float(args.fee_bps),
        slip_bps=float(args.slip_bps),
        # ===== 合约参数传入 =====
        leverage=float(args.leverage),
        funding_bps_8h=float(args.funding_bps_8h),
        mmr_bps=float(args.mmr_bps),
        liq_penalty_bps=float(args.liq_penalty_bps),
        # ======================
        stop_loss_pct=(float(args.sl_pct) if args.sl_pct is not None else None),
        plot=bool(int(args.plot)),
        source=str(args.source),
        csv_path=args.csv_path,
        # 新策略参数
        macd_fast=macd_fast,
        macd_slow=macd_slow,
        macd_signal=macd_signal,
        bb_period=bb_period,
        bb_k=bb_k,
        rsi_period=rsi_period,
        rsi_buy=rsi_buy,
        rsi_sell=rsi_sell,
        aroon_period=aroon_period,
        aroon_buy=aroon_buy,
        aroon_sell=aroon_sell,
        confirm_min=confirm_min,
    )
    bt = MABreakoutBacktester(cfg)
    res = bt.run()
    stats = res["stats"]
    logger.info("回测完成：final_eq={:.4f} return={:.2f}% sharpe={:.3f} maxDD={:.2%}",
                stats["final_equity"], stats["return_pct"], stats["sharpe"], stats["max_dd"])