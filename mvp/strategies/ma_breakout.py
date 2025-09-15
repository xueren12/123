"""
中低频（日内/几小时）趋势/波段策略：多指标确认（MACD + 布林带 + RSI + Aroon）

思路（升级版）：
- 使用成交(trades)重采样为指定周期的收盘价，若无成交则回退使用盘口中间价(mid)；
- 计算 MACD、布林带、RSI、Aroon 四类指标；
- 多重确认：四项指标中至少满足 cfg.confirm_min 个看多/看空条件时触发 BUY/SELL，否则 HOLD；
- 输出字段保持兼容：
  - fast/slow 映射为 MACD 线与信号线；
  - brk_high/brk_low 映射为布林带上轨/下轨；
  - reason 字段保留为 "ma_breakout" 以兼容外部透传逻辑（尽管策略已更换为多指标确认）。

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
    timeframe: str = "5min"               # 时间粒度：如 1min/5min/15min

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
    stop_loss_pct: Optional[float] = 0.005  # 止损百分比；None 表示不建议 SL

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


class MABreakoutStrategy:
    """多指标确认策略（MACD + 布林带 + RSI + Aroon）"""

    def __init__(self, cfg: MABreakoutConfig) -> None:
        self.cfg = cfg
        self.db = TimescaleDB()

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
                return f"{mins}T", mins * 60
            except Exception:
                pass
        if s.endswith("s"):
            try:
                secs = int(s[:-1])
                secs = max(1, secs)
                return f"{secs}S", secs
            except Exception:
                pass
        # 兜底 1 分钟
        logger.warning("无法解析 timeframe={}，回退为 1min", tf)
        return "1T", 60

    def _load_price_series(self) -> Tuple[pd.Series, Optional[pd.Series]]:
        """加载近一段时间的收盘价序列。
        返回 (close_series, ts_index_series)，索引为 UTC 时间。
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
        )
        bars = int(needed * 1.5)
        # 读取窗口：多取一些，避免边界缺数据
        window_secs = int((bars + 5) * sec_per_bar)
        end = datetime.now(timezone.utc)
        start = end - timedelta(seconds=window_secs)

        # 优先 trades
        close_series: Optional[pd.Series] = None
        try:
            tr = self.db.fetch_trades_window(start=start, end=end, inst_id=self.cfg.inst_id, ascending=True)
            if tr is not None and not tr.empty:
                tr["ts"] = pd.to_datetime(tr["ts"], utc=True)
                tr = tr.set_index("ts").sort_index()
                s = tr["price"].astype(float).resample(freq).last()
                close_series = s.ffill().dropna()
        except Exception as e:
            logger.debug("读取 trades 失败，准备回退：{}", e)

        # 回退：使用 orderbook mid
        if close_series is None or close_series.empty:
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
                    s = ob["mid"].resample(freq).last()
                    close_series = s.ffill().dropna()
            except Exception as e:
                logger.error("读取 orderbook 失败：{}", e)

        if close_series is None:
            raise RuntimeError("无法构造收盘价序列：请确认数据库中存在 trades 或 orderbook 数据。")

        return close_series, close_series.index.to_series()

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
          'reason': 'ma_breakout', 'sl': Optional[float], 'ord_type': 'market'|'limit'
        }
        说明：本策略中 fast/slow 分别为 MACD 线与信号线，brk_high/brk_low 为布林带上/下轨；
        若数据不足则返回 None。
        """
        try:
            close, _ = self._load_price_series()
        except Exception as e:
            logger.warning("获取数据失败：{}", e)
            return None

        # 计算指标所需的最小数据长度
        min_need = max(
            int(self.cfg.macd_slow + self.cfg.macd_signal + 5),
            int(self.cfg.bb_period + 5),
            int(self.cfg.rsi_period + 5),
            int(self.cfg.aroon_period + 1),
        )
        if len(close) < min_need:
            # 数据不足以稳定计算多指标
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

        # 取最后两根用于交叉与阈值判定
        c0, c1 = float(close.iloc[-2]), float(close.iloc[-1])
        macd0, macd1 = float(macd_line.iloc[-2]), float(macd_line.iloc[-1])
        sig0, sig1 = float(macd_signal.iloc[-2]), float(macd_signal.iloc[-1])
        bb_up1, bb_low1 = float(bb_upper.iloc[-1]), float(bb_lower.iloc[-1])
        rsi1 = float(rsi.iloc[-1]) if not np.isnan(float(rsi.iloc[-1])) else None
        a_up1 = float(aroon_up.iloc[-1]) if not np.isnan(float(aroon_up.iloc[-1])) else None
        a_dn1 = float(aroon_down.iloc[-1]) if not np.isnan(float(aroon_down.iloc[-1])) else None

        # ============ 多重确认逻辑 ============
        # MACD 确认：金叉/死叉 或者 位于信号线上下且方向一致
        macd_cross_up = (macd0 <= sig0) and (macd1 > sig1)
        macd_cross_dn = (macd0 >= sig0) and (macd1 < sig1)
        macd_bull_ok = macd_cross_up or (macd1 > sig1 and macd1 > 0)
        macd_bear_ok = macd_cross_dn or (macd1 < sig1 and macd1 < 0)

        # 布林带确认：突破上轨/下轨
        bb_bull_ok = c1 > bb_up1 if not np.isnan(bb_up1) else False
        bb_bear_ok = c1 < bb_low1 if not np.isnan(bb_low1) else False

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

        # 生成信号：至少满足确认数，且方向上不“打架”
        if buy_confirms >= need and buy_confirms > sell_confirms:
            sig = "BUY"
        elif sell_confirms >= need and sell_confirms > buy_confirms:
            sig = "SELL"
        else:
            sig = "HOLD"

        # 止损建议（对称百分比）
        sl: Optional[float] = None
        if self.cfg.stop_loss_pct and self.cfg.stop_loss_pct > 0:
            if sig == "BUY":
                sl = c1 * (1.0 - float(self.cfg.stop_loss_pct))
            elif sig == "SELL":
                sl = c1 * (1.0 + float(self.cfg.stop_loss_pct))

        return {
            "ts": close.index[-1].to_pydatetime(),
            "inst_id": self.cfg.inst_id,
            "timeframe": self.cfg.timeframe,
            "signal": sig,
            "close": float(c1),
            # 兼容输出：fast/slow -> MACD 线/信号线
            "fast": float(macd1),
            "slow": float(sig1),
            # 兼容输出：brk_high/brk_low -> 布林带上/下轨
            "brk_high": float(bb_up1),
            "brk_low": float(bb_low1),
            # 保留 reason 字段名与旧值，便于外层兼容
            "reason": "ma_breakout",
            "sl": float(sl) if sl is not None else None,
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