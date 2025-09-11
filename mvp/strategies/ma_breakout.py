"""
中低频（日内/几小时）趋势/波段策略：均线 + 突破

思路（简化版）：
- 使用成交(trades)重采样为指定周期的收盘价，若无成交则回退使用盘口中间价(mid)；
- 计算快/慢均线（SMA），并检测金叉/死叉；
- 同时检测向上/向下突破（近 N 根最高/最低价）并叠加一个缓冲百分比；
- 触发条件：
  - BUY：快线上穿慢线，且收盘价 > 近N高点*(1+buffer)
  - SELL：快线下穿慢线，且收盘价 < 近N低点*(1-buffer)
- 输出信号：BUY/SELL/HOLD，并给出止损价（可选）以便风控/执行层使用。

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
    """策略参数配置"""
    inst_id: str = "BTC-USDT"           # 标的
    timeframe: str = "5min"              # 时间粒度：如 1min/5min/15min（支持 '1m','5m','15m','1min','5min' 等写法）
    fast_ma: int = 10                    # 快线窗口
    slow_ma: int = 30                    # 慢线窗口
    breakout_lookback: int = 20          # 突破回看窗口（bar 数）
    breakout_buffer_pct: float = 0.0     # 突破缓冲比例（如 0.001=0.1%）
    lookback_bars: int = 300             # 每次计算读取的最小历史 bar 数
    prefer_trades_price: bool = True     # 优先使用成交价构造收盘价

    # 运行节奏
    poll_sec: float = 60.0               # 轮询周期（秒）

    # 风控/执行辅助（仅作为信号建议透传到 meta）
    order_type: str = "market"           # 市价/限价（ordType）
    stop_loss_pct: Optional[float] = 0.005  # 止损百分比（0.5%）；None 表示不建议 SL


class MABreakoutStrategy:
    """均线 + 突破 中低频策略"""

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
        bars = int(max(self.cfg.lookback_bars, self.cfg.slow_ma, self.cfg.breakout_lookback) * 1.5)
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
        若数据不足则返回 None。
        """
        try:
            close, _ = self._load_price_series()
        except Exception as e:
            logger.warning("获取数据失败：{}", e)
            return None

        if len(close) < max(self.cfg.slow_ma, self.cfg.breakout_lookback) + 2:
            # 数据不足以判断交叉/突破
            return None

        fast = close.rolling(self.cfg.fast_ma).mean()
        slow = close.rolling(self.cfg.slow_ma).mean()
        brk_high = close.rolling(self.cfg.breakout_lookback).max()
        brk_low = close.rolling(self.cfg.breakout_lookback).min()
        # 使用“前N根”的高低点（不包含当前bar）
        brk_high_prev = brk_high.shift(1)
        brk_low_prev = brk_low.shift(1)

        # 取最后两根用于交叉判定
        c0, c1 = close.iloc[-2], close.iloc[-1]
        f0, f1 = float(fast.iloc[-2]), float(fast.iloc[-1])
        s0, s1 = float(slow.iloc[-2]), float(slow.iloc[-1])
        h1 = float(brk_high_prev.iloc[-1])
        l1 = float(brk_low_prev.iloc[-1])
        buf = float(self.cfg.breakout_buffer_pct)

        # 金叉/死叉 + 突破（基于前N根高低点）
        buy_cond = (f0 <= s0 and f1 > s1) and (c1 > h1 * (1.0 + buf))
        sell_cond = (f0 >= s0 and f1 < s1) and (c1 < l1 * (1.0 - buf))

        if buy_cond:
            sig = "BUY"
        elif sell_cond:
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
            "fast": float(f1),
            "slow": float(s1),
            "brk_high": float(h1),
            "brk_low": float(l1),
            "reason": "ma_breakout",
            "sl": float(sl) if sl is not None else None,
            "ord_type": self.cfg.order_type,
        }

    # 可选：独立循环（主要用于快速本地验证）
    def run_loop(self) -> None:
        logger.info(
            "启动均线+突破策略：inst={} tf={} fast={} slow={} brkN={} buffer={} poll={}s",
            self.cfg.inst_id, self.cfg.timeframe, self.cfg.fast_ma, self.cfg.slow_ma,
            self.cfg.breakout_lookback, self.cfg.breakout_buffer_pct, self.cfg.poll_sec,
        )
        try:
            while True:
                sig = self.compute_signal()
                if sig is None:
                    logger.info("[{}] 数据不足或暂未产生信号", self.cfg.inst_id)
                else:
                    logger.info(
                        "[{}] 信号={} close={:.4f} fast={:.4f} slow={:.4f} highN={:.4f} lowN={:.4f}",
                        self.cfg.inst_id, sig["signal"], sig["close"], sig["fast"], sig["slow"], sig["brk_high"], sig["brk_low"],
                    )
                # 控制节奏
                from time import sleep
                sleep(max(1.0, float(self.cfg.poll_sec)))
        except KeyboardInterrupt:
            logger.info("收到中断信号，退出均线+突破策略循环。")
        finally:
            try:
                self.db.close()
            except Exception:
                pass