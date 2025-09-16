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
            summary_path = _os.path.join("data", "backtest_summary.json")
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
    p = argparse.ArgumentParser(description="MA+Breakout 策略回测（K线驱动，已升级为多指标确认）")
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