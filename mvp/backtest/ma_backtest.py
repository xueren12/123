# -*- coding: utf-8 -*-
"""
K线驱动的均线+突破策略回测（独立模块）

说明：
- 从 TimescaleDB 读取 trades（优先）或 orderbook(mid) 数据，重采样为指定 timeframe 的收盘价序列；
- 计算快/慢均线与近N高低点，生成 BUY/SELL/HOLD 信号；
- 采用简单持仓规则：
  - BUY -> 持仓 = +1，SELL -> 持仓 = -1，HOLD -> 持仓沿用上一根；
- 收益计算：ret = pos.shift(1) * close.pct_change()，可选扣除双边手续费；
- 输出：回测摘要、CSV（data/backtest_ma_breakout.csv）、SVG 图（data/backtest_ma_breakout.svg）。

用法示例（PowerShell）：
  py backtest/ma_backtest.py --inst BTC-USDT --start "2025-01-01" --end "2025-01-07" \
      --timeframe 5min --fast 10 --slow 30 --brk 20 --buffer 0.001 --fee_bps 2 --plot 1

注意：
- 需要数据库中存在该交易对在时间区间内的 trades 或 orderbook 数据。
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

from utils.db import TimescaleDB


@dataclass
class MABacktestConfig:
    inst: str
    start: datetime
    end: datetime
    timeframe: str = "5min"
    fast: int = 10
    slow: int = 30
    brk: int = 20
    buffer: float = 0.0
    fee_bps: float = 0.0  # 双边手续费，单位: bp（基点）。例如 2 表示 0.02%
    stop_loss_pct: Optional[float] = None  # 可选：对称止损百分比
    plot: bool = True


class MABreakoutBacktester:
    def __init__(self, cfg: MABacktestConfig) -> None:
        self.cfg = cfg
        self.db = TimescaleDB()

    @staticmethod
    def _parse_timeframe(tf: str) -> Tuple[str, int]:
        """timeframe 转为 pandas 频率与每bar秒数"""
        s = str(tf).lower().strip().replace("min", "m")
        if s.endswith("m"):
            mins = int(s[:-1])
            return f"{mins}T", mins * 60
        if s.endswith("s"):
            secs = int(s[:-1])
            return f"{secs}S", secs
        return "1T", 60

    def _load_close_series(self) -> pd.Series:
        freq, _ = self._parse_timeframe(self.cfg.timeframe)
        # 读取 trades -> resample last -> ffill
        tr = None
        try:
            tr = self.db.fetch_trades_window(start=self.cfg.start, end=self.cfg.end, inst_id=self.cfg.inst, ascending=True)
        except Exception as e:
            logger.debug("读取 trades 失败，将回退 orderbook：{}", e)
        if tr is not None and not tr.empty:
            tr["ts"] = pd.to_datetime(tr["ts"], utc=True)
            s = tr.set_index("ts").sort_index()["price"].astype(float).resample(freq).last()
            s = s.ffill().dropna()
            if len(s) > 0:
                return s
        # 回退到 orderbook mid
        ob = self.db.fetch_orderbook_window(start=self.cfg.start, end=self.cfg.end, inst_id=self.cfg.inst, ascending=True)
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
        s = ob["mid"].resample(freq).last().ffill().dropna()
        if s.empty:
            raise RuntimeError("构造收盘价序列失败：orderbook 数据不足")
        return s

    def run(self) -> dict:
        close = self._load_close_series()
        fast = close.rolling(self.cfg.fast).mean()
        slow = close.rolling(self.cfg.slow).mean()
        highn = close.rolling(self.cfg.brk).max()
        lown = close.rolling(self.cfg.brk).min()
        # 使用“前N根”的高低点进行突破判定，避免包含当前bar造成自证突破
        highn_prev = highn.shift(1)
        lown_prev = lown.shift(1)
        buf = float(self.cfg.buffer)

        # 生成信号（向量化）
        f0 = fast.shift(1)
        s0 = slow.shift(1)
        buy = (f0 <= s0) & (fast > slow) & (close > highn_prev * (1.0 + buf))
        sell = (f0 >= s0) & (fast < slow) & (close < lown_prev * (1.0 - buf))
        signal = pd.Series("HOLD", index=close.index)
        signal = signal.mask(buy, "BUY").mask(sell, "SELL")

        # 仓位：BUY -> +1，SELL -> -1，HOLD -> 延续
        pos = pd.Series(0.0, index=close.index)
        pos = pos.mask(buy, 1.0).mask(sell, -1.0)
        pos = pos.replace(0.0, np.nan).ffill().fillna(0.0)

        # 收益计算
        ret = close.pct_change().fillna(0.0)
        gross = pos.shift(1).fillna(0.0) * ret
        # 手续费（双边）：当仓位变化时收取一次换手成本近似；
        turn = (pos != pos.shift(1)).astype(float)
        fee_rate = abs(self.cfg.fee_bps) / 10000.0
        net = gross - turn * fee_rate

        equity = (1.0 + net).cumprod()

        df = pd.DataFrame({
            "close": close,
            "fast": fast,
            "slow": slow,
            "highN": highn,
            "lowN": lown,
            "highN_prev": highn_prev,
            "lowN_prev": lown_prev,
            "signal": signal,
            "pos": pos,
            "ret": net,
            "equity": equity,
        })

        stats = self._summary_stats(df)
        self._export(df)
        return {"df": df, "stats": stats}

    @staticmethod
    def _summary_stats(df: pd.DataFrame) -> dict:
        eq = df["equity"].dropna()
        if eq.empty:
            return {"final_equity": 1.0, "return_pct": 0.0, "sharpe": 0.0, "max_dd": 0.0}
        final_eq = float(eq.iloc[-1])
        ret = df["ret"].fillna(0.0)
        ann_fac = 365 * 24 * 60  # 以分钟为单位近似年化（保守）
        mu = ret.mean() * ann_fac
        sigma = ret.std(ddof=0) * np.sqrt(ann_fac)
        sharpe = (mu / sigma) if sigma > 1e-12 else 0.0
        cummax = eq.cummax()
        dd = (eq / cummax - 1.0).min() if len(eq) > 0 else 0.0
        return {"final_equity": final_eq, "return_pct": (final_eq - 1.0) * 100.0, "sharpe": float(sharpe), "max_dd": float(dd)}

    @staticmethod
    def _export(df: pd.DataFrame) -> None:
        import os
        os.makedirs("data", exist_ok=True)
        csv_path = "data/backtest_ma_breakout.csv"
        svg_path = "data/backtest_ma_breakout.svg"
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
            ax1.set_title("MA+Breakout Backtest")
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
    p = argparse.ArgumentParser(description="MA+Breakout 策略回测（K线驱动）")
    p.add_argument("--inst", required=True, help="交易对，如 BTC-USDT")
    p.add_argument("--start", required=True, help="起始时间，ISO或日期，如 2025-01-01")
    p.add_argument("--end", required=True, help="结束时间，ISO或日期，如 2025-01-07")
    p.add_argument("--timeframe", default="5min", help="时间粒度，如 1min/5min/15min")
    p.add_argument("--fast", type=int, default=10, help="快线窗口")
    p.add_argument("--slow", type=int, default=30, help="慢线窗口")
    p.add_argument("--brk", type=int, default=20, help="突破回看窗口")
    p.add_argument("--buffer", type=float, default=0.0, help="突破缓冲比例，如 0.001")
    p.add_argument("--fee_bps", type=float, default=0.0, help="双边手续费，基点，例如 2=0.02%")
    p.add_argument("--sl_pct", type=float, default=None, help="可选：止损百分比（不在回测中强制执行，仅记录）")
    p.add_argument("--plot", type=int, default=1, help="是否导出SVG图：1是 0否")
    return p.parse_args()


def _parse_dt(s: str) -> datetime:
    s = s.strip()
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        # 仅日期
        from datetime import datetime as _dt
        return _dt.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)


if __name__ == "__main__":
    args = _parse_args()
    cfg = MABacktestConfig(
        inst=args.inst,
        start=_parse_dt(args.start),
        end=_parse_dt(args.end),
        timeframe=args.timeframe,
        fast=max(2, int(args.fast)),
        slow=max(3, int(args.slow)),
        brk=max(2, int(args.brk)),
        buffer=float(args.buffer),
        fee_bps=float(args.fee_bps),
        stop_loss_pct=(float(args.sl_pct) if args.sl_pct is not None else None),
        plot=bool(int(args.plot)),
    )
    bt = MABreakoutBacktester(cfg)
    res = bt.run()
    stats = res["stats"]
    logger.info("回测完成：final_eq={:.4f} return={:.2f}% sharpe={:.3f} maxDD={:.2%}",
                stats["final_equity"], stats["return_pct"], stats["sharpe"], stats["max_dd"])