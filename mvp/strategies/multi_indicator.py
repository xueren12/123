"""
中低频（日内/几小时）趋势/波段策略：多指标确认（MACD + 布林带 + RSI + Aroon）

思路（升级版）：
- 使用成交(trades)重采样为指定周期的收盘价，若无成交则回退使用盘口中间价(mid)；
- 计算 MACD、布林带、RSI、Aroon 四类指标；
- 多重确认：四项指标中至少满足 cfg.confirm_min 个看多/看空条件时触发 BUY/SELL，否则 HOLD；
- 输出字段保持兼容：
  - fast/slow 映射为 MACD 线与信号线；
  - brk_high/brk_low 映射为布林带上轨/下轨；
  - 对外 reason 字段默认为 "multi_indicator"（保留对历史数据中旧值 "ma_breakout" 的兼容读取，但不再作为默认产出）。

说明：
- 本策略为“计算信号”组件，不直接下单。实盘由 main.SystemOrchestrator 调用并交由 TradeExecutor 统一执行与风控。
- 所需数据来自 utils.db.TimescaleDB，使用已有 trades/orderbook 查询接口；
- 运行频率建议与 timeframe 对齐（例如 1-5 分钟一次）。
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from utils.db import TimescaleDB


@dataclass
class MABreakoutConfig:
    """策略参数配置（升级为多指标确认，保留原字段以兼容外部调用）"""
    inst_id: str = "BTC-USDT"            # 标的
    timeframe: str = "15min"               # 时间粒度：如 1min/5min/15min

    # —— 兼容旧参数（用于默认回退）——
    fast_ma: int = 10                      # 兼容：作为 MACD 快线窗口默认回退
    slow_ma: int = 30                      # 兼容：作为 MACD 慢线窗口默认回退
    breakout_lookback: int = 20            # 兼容：作为布林带窗口默认回退
    breakout_buffer_pct: float = 0.0       # 保留字段（不再使用）

    lookback_bars: int = 300               # 每次计算读取的最小历史 bar 数
    prefer_trades_price: bool = True       # 优先使用成交价构造收盘价

    # 运行节奏
    poll_sec: float = 60.0                 # 轮询周期（秒）

    # 风控/执行辅助（仅作为信号建议透传到 meta）
    order_type: str = "market"             # 市价/限价（ordType）
    stop_loss_pct: Optional[float] = 0.005  # 止损百分比；None 表示不建议 SL（通用兜底）

    # —— 新增：多指标参数 ——
    macd_fast: int = 12                    # MACD EMA 快线窗口
    macd_slow: int = 26                    # MACD EMA 慢线窗口
    macd_signal: int = 9                   # MACD 信号线窗口

    bb_period: int = 20                    # 布林带窗口
    bb_k: float = 2.0                      # 布林带倍数（标准差倍数）

    rsi_period: int = 14                   # RSI 窗口
    rsi_buy: float = 55.0                  # RSI 多头阈值
    rsi_sell: float = 45.0                 # RSI 空头阈值

    aroon_period: int = 25                 # Aroon 窗口
    aroon_buy: float = 70.0                # Aroon 多头阈值（Up）
    aroon_sell: float = 30.0               # Aroon 空头阈值（Down），与 Up 配合用于判断领先关系

    confirm_min: int = 3                   # 至少满足的确认数（1~4）

    # —— 新增：止损参数 ——
    atr_n: float = 2.0                     # ATR 止损倍数 N（止损阈值 = 入场价 - N * ATR），ATR 周期固定为 14
    stop_loss_pct_btc: float = 0.03        # BTC 固定百分比止损（默认 3%）
    stop_loss_pct_eth: float = 0.04        # ETH 固定百分比止损（默认 4%）

    # —— 新增：止盈参数 ——
    tp_r1: float = 1.0                     # 第一级 R 倍数（达到即部分止盈）
    tp_r2: float = 2.0                     # 第二级 R 倍数（达到即再次部分止盈）
    tp_frac1: float = 0.3                  # 第一级止盈平掉比例（默认 30%）
    tp_frac2: float = 0.3                  # 第二级止盈再平掉比例（默认 30%）
    tp_trail_atr_mult: float = 1.5         # 移动止损 ATR 倍数（默认 1.5）
    rsi_tp_high: float = 80.0              # RSI 技术止盈上阈值（部分止盈）
    rsi_tp_low: float = 20.0               # RSI 技术止盈下阈值（部分止盈）
    rsi_tp_frac: float = 0.2               # RSI 技术止盈建议平掉比例（默认 20%）

class MABreakoutStrategy:
    """多指标确认策略（MACD + 布林带 + RSI + Aroon）"""

    def __init__(self, cfg: MABreakoutConfig) -> None:
        self.cfg = cfg
        self.db = TimescaleDB()
        # —— 策略内轻量持仓状态（用于止损/止盈判定） ——
        self._pos: int = 0                  # 0=空仓；1=持有多头（当前实现不做做空）
        self._entry_price: Optional[float] = None
        self._entry_ts: Optional[datetime] = None
        # —— 止盈管理相关状态 ——
        self._tp1_done: bool = False        # 是否已完成 1R 部分止盈
        self._tp2_done: bool = False        # 是否已完成 2R 部分止盈
        self._remain_frac: float = 0.0      # 估计剩余仓位比例（仅用于策略内止盈序列控制）
        self._trail_stop: Optional[float] = None  # 移动止损价格（不可下降）
        self._entry_high: Optional[float] = None  # 入场后最高价（用于移动止损）
        self._init_risk: Optional[float] = None   # 以“最近一次入场的有效止损价”计算的每单位风险 R（entry - stop）

    # --------------------
    # 内部工具
    # --------------------
    @staticmethod
    def _parse_timeframe(tf: str) -> Tuple[str, int]:
        """将 timeframe 转换为 pandas 频率字符串与秒数。
        返回 (pandas_freq, seconds_per_bar)
        """
        s = str(tf).lower().strip()
        # 标准化
        s = s.replace("min", "m")
        if s.endswith("m"):
            try:
                mins = int(s[:-1])
                mins = max(1, mins)
                return f"{mins}min", mins * 60
            except Exception:
                pass
        if s.endswith("s"):
            try:
                secs = int(s[:-1])
                secs = max(1, secs)
                return f"{secs}s", secs
            except Exception:
                pass
        # 兜底 1 分钟
        logger.warning("无法解析 timeframe={}，回退为 1min", tf)
        return "1min", 60

    def _load_price_series(self) -> pd.DataFrame:
        """加载近一段时间的 OHLC 序列，索引为 UTC 时间。
        优先使用 trades，若无则回退 orderbook mid；均以目标 timeframe 重采样。
        返回 DataFrame，列为 ['open','high','low','close']。
        """
        freq, sec_per_bar = self._parse_timeframe(self.cfg.timeframe)
        # 依据多指标所需窗口，动态扩大最小读取 bars 数量，避免指标前期不稳定
        needed = max(
            int(self.cfg.lookback_bars),
            int(self.cfg.slow_ma),
            int(self.cfg.breakout_lookback),
            int(self.cfg.macd_slow + self.cfg.macd_signal + 5),  # MACD 慢线+signal 需要的最小长度
            int(self.cfg.bb_period + 5),
            int(self.cfg.rsi_period + 5),
            int(self.cfg.aroon_period + 1),
            int(self.cfg.atr_n + 5),  # ATR 需要足够窗口
        )
        bars = int(needed * 1.5)
        # 读取窗口：多取一些，避免边界缺数据
        window_secs = int((bars + 5) * sec_per_bar)
        end = datetime.now(timezone.utc)
        start = end - timedelta(seconds=window_secs)

        # 优先 trades：直接重采样为 OHLC
        ohlc: Optional[pd.DataFrame] = None
        try:
            tr = self.db.fetch_trades_window(start=start, end=end, inst_id=self.cfg.inst_id, ascending=True)
            if tr is not None and not tr.empty:
                tr["ts"] = pd.to_datetime(tr["ts"], utc=True)
                tr = tr.set_index("ts").sort_index()
                s = tr["price"].astype(float)
                ohlc_tr = s.resample(freq).ohlc()
                ohlc = ohlc_tr.dropna(how="all").ffill()
        except Exception as e:
            logger.debug("读取 trades 失败，准备回退：{}", e)

        # 回退：使用 orderbook mid 构造 OHLC
        if ohlc is None or ohlc.empty:
            try:
                ob = self.db.fetch_orderbook_window(start=start, end=end, inst_id=self.cfg.inst_id, ascending=True)
                if ob is not None and not ob.empty:
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
                    s = ob["mid"].astype(float)
                    df = s.resample(freq).agg(["first", "max", "min", "last"]).rename(columns={
                        "first": "open", "max": "high", "min": "low", "last": "close"
                    })
                    ohlc = df.dropna(how="all").ffill()
            except Exception as e:
                logger.error("读取 orderbook 失败：{}", e)

        if ohlc is None or ohlc.empty:
            raise RuntimeError("无法构造 OHLC 序列：请确认数据库中存在 trades 或 orderbook 数据。")

        # 确保列顺序
        for col in ("open", "high", "low", "close"):
            if col not in ohlc.columns:
                ohlc[col] = np.nan
        ohlc = ohlc[["open", "high", "low", "close"]]
        return ohlc

    # --------------------
    # 信号计算
    # --------------------
    def compute_signal(self) -> Optional[dict]:
        """计算一次信号，返回统一结构：
        {
          'ts': <bar收盘时间UTC>, 'inst_id': <交易对>, 'timeframe': str,
          'signal': 'BUY'|'SELL'|'HOLD',
          'close': float, 'fast': float, 'slow': float,
          'brk_high': float, 'brk_low': float,
          # 产出信号结构示例（供调用方参考）：{'ts': datetime, 'side': 'BUY'|'SELL'|'HOLD', 'price': float,
          #  'reason': 'multi_indicator'|'stoploss_*', 'sl': Optional[float], 'ord_type': 'market'|'limit'
        }
        说明：本策略中 fast/slow 分别为 MACD 线与信号线，brk_high/brk_low 为布林带上/下轨；
        若数据不足则返回 None。
        """
        try:
            ohlc = self._load_price_series()
            close = ohlc["close"]
        except Exception as e:
            logger.warning("获取数据失败：{}", e)
            return None

        # 计算指标所需的最小数据长度
        atr_period = 14  # ATR 周期固定 14（若需可后续扩展为可配置）
        min_need = max(
            int(self.cfg.macd_slow + self.cfg.macd_signal + 5),
            int(self.cfg.bb_period + 5),
            int(self.cfg.rsi_period + 5),
            int(self.cfg.aroon_period + 1),
            int(atr_period + 5),
        )
        if len(close) < min_need:
            return None

        # ============ 指标计算：全部基于 pandas ============
        # 1) MACD（EMA）
        ema_fast = close.ewm(span=int(self.cfg.macd_fast), adjust=False).mean()
        ema_slow = close.ewm(span=int(self.cfg.macd_slow), adjust=False).mean()
        macd_line = ema_fast - ema_slow
        macd_signal = macd_line.ewm(span=int(self.cfg.macd_signal), adjust=False).mean()
        macd_hist = macd_line - macd_signal

        # 2) 布林带（移动均值 + 标准差）
        bb_mid = close.rolling(int(self.cfg.bb_period)).mean()
        bb_std = close.rolling(int(self.cfg.bb_period)).std(ddof=0)
        bb_upper = bb_mid + float(self.cfg.bb_k) * bb_std
        bb_lower = bb_mid - float(self.cfg.bb_k) * bb_std

        # 3) RSI（Wilder 平滑）
        delta = close.diff()
        up = np.where(delta > 0, delta, 0.0)
        down = np.where(delta < 0, -delta, 0.0)
        roll_up = pd.Series(up, index=close.index).ewm(alpha=1 / float(self.cfg.rsi_period), adjust=False).mean()
        roll_down = pd.Series(down, index=close.index).ewm(alpha=1 / float(self.cfg.rsi_period), adjust=False).mean()
        rs = roll_up / roll_down.replace(0, np.nan)
        rsi = 100.0 - (100.0 / (1.0 + rs))

        # 4) Aroon（Up/Down）
        n_aroon = int(self.cfg.aroon_period)
        argmax = close.rolling(n_aroon).apply(lambda x: float(np.argmax(x)), raw=True)
        argmin = close.rolling(n_aroon).apply(lambda x: float(np.argmin(x)), raw=True)
        aroon_up = (argmax + 1.0) / n_aroon * 100.0
        aroon_down = (argmin + 1.0) / n_aroon * 100.0

        # 5) ATR（波动性）：使用标准 True Range 定义
        high = ohlc["high"].astype(float)
        low = ohlc["low"].astype(float)
        prev_close = close.shift(1)
        tr1 = (high - low).abs()
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(int(atr_period)).mean()
        atr1 = float(atr.iloc[-1]) if not np.isnan(float(atr.iloc[-1])) else None

        # —— 从序列提取当前/前一根数值，便于后续条件判断 ——
        # 收盘价与最高价（最高价用于移动止损）
        c1 = float(close.iloc[-1]) if not np.isnan(float(close.iloc[-1])) else float('nan')
        high1 = (float(high.iloc[-1]) if not np.isnan(float(high.iloc[-1])) else None)

        # MACD 当前与前一根（保持为浮点或 NaN，避免 None 参与比较导致异常）
        macd1 = float(macd_line.iloc[-1]) if not np.isnan(float(macd_line.iloc[-1])) else float('nan')
        macd0 = float(macd_line.iloc[-2]) if not np.isnan(float(macd_line.iloc[-2])) else float('nan')
        sig1 = float(macd_signal.iloc[-1]) if not np.isnan(float(macd_signal.iloc[-1])) else float('nan')
        sig0 = float(macd_signal.iloc[-2]) if not np.isnan(float(macd_signal.iloc[-2])) else float('nan')

        # 布林带当前上/下轨（保留 NaN，后续使用 np.isnan 判断）
        bb_up1 = float(bb_upper.iloc[-1])
        bb_low1 = float(bb_lower.iloc[-1])

        # RSI 当前值（若为 NaN 则置为 None，便于后续 None 判断）
        _rsi_tmp = float(rsi.iloc[-1]) if not np.isnan(float(rsi.iloc[-1])) else float('nan')
        rsi1 = None if np.isnan(_rsi_tmp) else _rsi_tmp

        # Aroon 当前/前一根（若为 NaN 则置为 None，避免无效比较）
        a_up1 = (float(aroon_up.iloc[-1]) if not np.isnan(float(aroon_up.iloc[-1])) else None)
        a_up0 = (float(aroon_up.iloc[-2]) if not np.isnan(float(aroon_up.iloc[-2])) else None)
        a_dn1 = (float(aroon_down.iloc[-1]) if not np.isnan(float(aroon_down.iloc[-1])) else None)
        a_dn0 = (float(aroon_down.iloc[-2]) if not np.isnan(float(aroon_down.iloc[-2])) else None)

        # ============ 多重确认逻辑 ==========  # 补回缺失的确认布尔量定义
        # MACD 确认：金叉/死叉 或者 位于信号线上下且方向一致
        macd_cross_up = (macd0 <= sig0) and (macd1 > sig1)
        macd_cross_dn = (macd0 >= sig0) and (macd1 < sig1)
        macd_bull_ok = (macd_cross_up or (macd1 > sig1 and macd1 > 0))
        macd_bear_ok = (macd_cross_dn or (macd1 < sig1 and macd1 < 0))

        # 布林带确认：突破上轨/下轨
        bb_bull_ok = (False if np.isnan(bb_up1) else (c1 > bb_up1))
        bb_bear_ok = (False if np.isnan(bb_low1) else (c1 < bb_low1))

        # RSI 确认：多头 >= rsi_buy；空头 <= rsi_sell
        rsi_bull_ok = (rsi1 is not None) and (rsi1 >= float(self.cfg.rsi_buy))
        rsi_bear_ok = (rsi1 is not None) and (rsi1 <= float(self.cfg.rsi_sell))

        # Aroon 确认：Up 高且领先；Down 高且领先
        bear_thresh = max(float(self.cfg.aroon_buy), 100.0 - float(self.cfg.aroon_sell))
        aroon_bull_ok = (a_up1 is not None and a_dn1 is not None) and (a_up1 >= float(self.cfg.aroon_buy)) and (a_up1 > a_dn1)
        aroon_bear_ok = (a_up1 is not None and a_dn1 is not None) and (a_dn1 >= bear_thresh) and (a_dn1 > a_up1)

        # 统计确认数
        buy_confirms = int(macd_bull_ok) + int(bb_bull_ok) + int(rsi_bull_ok) + int(aroon_bull_ok)
        sell_confirms = int(macd_bear_ok) + int(bb_bear_ok) + int(rsi_bear_ok) + int(aroon_bear_ok)
        need = int(max(1, min(4, int(self.cfg.confirm_min))))

        # 生成“基础信号”：至少满足确认数，且方向上不“打架”
        if buy_confirms >= need and buy_confirms > sell_confirms:
            base_sig = "BUY"
        elif sell_confirms >= need and sell_confirms > buy_confirms:
            base_sig = "SELL"
        else:
            base_sig = "HOLD"

        reason = "multi_indicator"
        out_sig = base_sig
        size_frac_suggest: Optional[float] = None  # 建议的部分平仓比例（仅在 SELL 且部分平仓时给出）

        # ============ 止盈模块（TakeProfit）============
        # 1) 分批止盈：达到 1R/2R 分别止盈 30%/30%；
        # 2) 移动止损：多单 = 入场后最高价 - ATR*1.5（trail 不回撤）；
        # 3) 技术止盈：RSI 极值 -> 部分止盈；Aroon 反转 -> 全平。
        tp_triggered = False
        if self._pos > 0 and self._entry_price is not None:
            # —— 入场后最高价 & 移动止损 ——
            if high1 is not None:
                if self._entry_high is None:
                    self._entry_high = high1
                else:
                    self._entry_high = max(self._entry_high, high1)
            if atr1 is not None and atr1 > 0 and self._entry_high is not None:
                trail_cand = float(self._entry_high) - float(self.cfg.tp_trail_atr_mult) * float(atr1)
                if self._trail_stop is None:
                    self._trail_stop = trail_cand
                else:
                    # 移动止损只抬不降
                    self._trail_stop = max(self._trail_stop, trail_cand)

            # —— Aroon 反转：全平 ——
            aroon_rev = False
            if a_up0 is not None and a_dn0 is not None and a_up1 is not None and a_dn1 is not None:
                aroon_rev = (a_up0 >= a_dn0) and (a_dn1 > a_up1)
            if aroon_rev and (self._remain_frac is None or self._remain_frac > 0):
                out_sig = "SELL"
                reason = "takeprofit_aroon_rev"
                size_frac_suggest = float(self._remain_frac or 1.0)
                tp_triggered = True

            # —— 移动止损：剩余仓位交给移动止损 ——
            if not tp_triggered and self._trail_stop is not None and c1 <= float(self._trail_stop):
                out_sig = "SELL"
                reason = "takeprofit_trail"
                size_frac_suggest = float(self._remain_frac or 1.0)
                tp_triggered = True

            # —— R 倍数分批止盈 ——
            if not tp_triggered and self._init_risk is not None and self._init_risk > 0:
                r_mult = (c1 - float(self._entry_price)) / float(self._init_risk)
                # 先判断 2R，再判断 1R（避免 2R 到达却先触发 1R）
                if (not self._tp2_done) and r_mult >= float(self.cfg.tp_r2) and (self._remain_frac or 0.0) > 0:
                    out_sig = "SELL"
                    reason = "takeprofit_r2"
                    size_frac_suggest = float(min(float(self.cfg.tp_frac2), float(self._remain_frac or 1.0)))
                    tp_triggered = True
                elif (not self._tp1_done) and r_mult >= float(self.cfg.tp_r1) and (self._remain_frac or 0.0) > 0:
                    out_sig = "SELL"
                    reason = "takeprofit_r1"
                    size_frac_suggest = float(min(float(self.cfg.tp_frac1), float(self._remain_frac or 1.0)))
                    tp_triggered = True

            # —— 技术止盈：RSI 极值（部分止盈）——
            if not tp_triggered and rsi1 is not None and (rsi1 >= float(self.cfg.rsi_tp_high) or rsi1 <= float(self.cfg.rsi_tp_low)):
                out_sig = "SELL"
                reason = "takeprofit_rsi"
                size_frac_suggest = float(min(float(self.cfg.rsi_tp_frac), float(self._remain_frac or 1.0) if (self._remain_frac is not None) else 1.0))
                tp_triggered = True

        # ============ 止损模块（StopLoss）============
        # 若已触发止盈，则本轮不再评估止损（均会触发 SELL）。
        if (not tp_triggered) and self._pos > 0 and self._entry_price is not None:
            stop_hit = None
            stop_pct = None
            # 1) ATR 止损
            if atr1 is not None and atr1 > 0:
                atr_stop = float(self._entry_price) - float(self.cfg.atr_n) * float(atr1)
                if c1 <= atr_stop:
                    stop_hit = "atr"
            # 2) 固定百分比止损（按标的默认或全局覆盖）
            sym = str(self.cfg.inst_id).upper()
            pct_default = None
            if sym.startswith("BTC"):
                pct_default = float(self.cfg.stop_loss_pct_btc)
            elif sym.startswith("ETH"):
                pct_default = float(self.cfg.stop_loss_pct_eth)
            stop_pct = float(self.cfg.stop_loss_pct) if (self.cfg.stop_loss_pct is not None) else pct_default
            if stop_pct is not None and stop_pct > 0:
                pct_stop = float(self._entry_price) * (1.0 - float(stop_pct))
                if c1 <= pct_stop and stop_hit is None:
                    stop_hit = "pct"
            # 3) 技术指标止损（多头反向）
            macd_cross_dn = (macd0 >= sig0) and (macd1 < sig1)
            tech_stop = (c1 < bb_low1) or macd_cross_dn
            if tech_stop and stop_hit is None:
                stop_hit = "tech"

            if stop_hit is not None:
                out_sig = "SELL"
                reason = f"stoploss_{stop_hit}"
                size_frac_suggest = float(self._remain_frac or 1.0)

        # ============ 策略内简单持仓状态机更新（支持部分止盈）============
        try:
            if out_sig == "BUY" and self._pos == 0:
                # 开仓：记录入场与初始风险 R
                self._pos = 1
                self._entry_price = c1
                self._entry_ts = close.index[-1].to_pydatetime()
                self._entry_high = high1
                self._tp1_done = False
                self._tp2_done = False
                self._remain_frac = 1.0
                self._trail_stop = None
                # 计算入场时的“有效止损价”（更近的那一个）：max(ATR止损价, 百分比止损价)
                eff_stop = None
                if atr1 is not None and atr1 > 0:
                    eff_stop = (eff_stop if eff_stop is not None else -1e18)
                    eff_stop = max(eff_stop, float(self._entry_price) - float(self.cfg.atr_n) * float(atr1))
                sym = str(self.cfg.inst_id).upper()
                pct_default = float(self.cfg.stop_loss_pct_btc) if sym.startswith("BTC") else (float(self.cfg.stop_loss_pct_eth) if sym.startswith("ETH") else None)
                stop_pct = float(self.cfg.stop_loss_pct) if (self.cfg.stop_loss_pct is not None) else pct_default
                if stop_pct is not None and stop_pct > 0:
                    eff_stop = max(eff_stop or -1e18, float(self._entry_price) * (1.0 - float(stop_pct)))
                if eff_stop is None or eff_stop <= 0 or eff_stop >= float(self._entry_price):
                    # 兜底：若无法得到有效止损，按 1% 计算 R
                    self._init_risk = float(self._entry_price) * 0.01
                else:
                    self._init_risk = float(self._entry_price) - float(eff_stop)

            elif out_sig == "SELL" and self._pos == 1:
                # 判断是否为“部分平仓”
                if size_frac_suggest is not None and size_frac_suggest < 0.999:
                    # 部分减仓：更新剩余比例，但不将仓位清零
                    self._remain_frac = max(0.0, float(self._remain_frac or 0.0) - float(size_frac_suggest))
                    # 标记已完成的分批止盈级别
                    if reason == "takeprofit_r1":
                        self._tp1_done = True
                    elif reason == "takeprofit_r2":
                        self._tp2_done = True
                    # 若已全部减完，则视为平仓
                    if self._remain_frac <= 1e-6:
                        self._pos = 0
                        self._entry_price = None
                        self._entry_ts = None
                        self._entry_high = None
                        self._trail_stop = None
                        self._init_risk = None
                        self._tp1_done = False
                        self._tp2_done = False
                else:
                    # 全量卖出/止损/反转：清空状态
                    self._pos = 0
                    self._entry_price = None
                    self._entry_ts = None
                    self._entry_high = None
                    self._trail_stop = None
                    self._init_risk = None
                    self._tp1_done = False
                    self._tp2_done = False
        except Exception:
            # 状态更新失败不影响信号输出
            pass

        return {
            "ts": close.index[-1].to_pydatetime(),
            "inst_id": self.cfg.inst_id,
            "timeframe": self.cfg.timeframe,
            "signal": out_sig,
            "close": float(c1),
            "fast": float(macd1),
            "slow": float(sig1),
            "brk_high": float(bb_up1),
            "brk_low": float(bb_low1),
            "reason": reason,
            # 建议部分平仓比例（仅 SELL 且部分平仓时非空）；执行层可选使用
            "size_frac": (float(size_frac_suggest) if size_frac_suggest is not None else None),
            "sl": None,
            "ord_type": self.cfg.order_type,
        }

    # 可选：独立循环（主要用于快速本地验证）
    def run_loop(self) -> None:
        logger.info(
            "启动多指标确认策略：inst={} tf={} MACD(f,s,sign)=({},{},{}) BB(p,k)=({},{}) RSI(p,buy,sell)=({},{},{}) Aroon(p,buy,sell)=({},{},{}) confirm_min={} poll={}s",
            self.cfg.inst_id, self.cfg.timeframe,
            self.cfg.macd_fast, self.cfg.macd_slow, self.cfg.macd_signal,
            self.cfg.bb_period, self.cfg.bb_k,
            self.cfg.rsi_period, self.cfg.rsi_buy, self.cfg.rsi_sell,
            self.cfg.aroon_period, self.cfg.aroon_buy, self.cfg.aroon_sell,
            self.cfg.confirm_min, self.cfg.poll_sec,
        )
        try:
            while True:
                sig = self.compute_signal()
                if sig is None:
                    logger.info("[{}] 数据不足或暂未产生信号", self.cfg.inst_id)
                else:
                    logger.info(
                        "[{}] 信号={} close={:.4f} MACD={:.4f}/{:.4f} BB(U/L)={:.4f}/{:.4f}",
                        self.cfg.inst_id, sig["signal"], sig["close"], sig["fast"], sig["slow"], sig["brk_high"], sig["brk_low"],
                    )
                # 控制节奏
                from time import sleep
                sleep(max(1.0, float(self.cfg.poll_sec)))
        except KeyboardInterrupt:
            logger.info("收到中断信号，退出多指标确认策略循环。")
        finally:
            try:
                self.db.close()
            except Exception:
                pass

# ===== 同文件顶部 _parse_timeframe 的返回值频率字符串修正：'T'->'min'、'S'->'s' =====