"""
回测引擎（基于历史 trades 与 orderbook 数据）

功能概述：
- 从 PostgreSQL/TimescaleDB 读取指定时间窗口内的历史盘口与成交数据；
- 支持按秒级或分钟级的时间粒度回测（例如 1s、1min）；
- 复用“盘口深度不平衡”思路生成信号（BUY/SELL/HOLD），并进行简单执行假设；
- 计算并输出：收益曲线、夏普比率、最大回撤；可选择导出 CSV 与生成 SVG 图表。

使用示例（命令行）：
    python -m backtest.engine \
        --inst BTC-USDT \
        --start "2024-09-01T00:00:00Z" \
        --end   "2024-09-01T12:00:00Z" \
        --timeframe 1s \
        --levels 5 --mode notional --threshold 0.02 \
        --fee-bps 1.0 --slippage-bps 1.0 \
        --initial-capital 10000 \
        --csv-out data/backtest_BTC-USDT_1s.csv \
        --plot-svg data/backtest_BTC-USDT_1s.svg

注意：
- 本模块以“演示级别”的简化执行模型为主，假设以中间价（mid）建仓/平仓；
- 交易费用（费率）与滑点通过基点（bps）近似计提，在持仓切换的时刻扣减；
- 若时间粒度为 1s，则年化换算采用 365*24*3600；若 1min，则使用 365*24*60。
"""
from __future__ import annotations

import argparse
import math
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

# 非交互式后端，便于服务器/CI 上生成图表文件
import matplotlib
matplotlib.use("Agg")  # 设置为无窗口后端
import matplotlib.pyplot as plt

from utils.db import TimescaleDB


@dataclass
class BacktestConfig:
    """回测参数配置（可通过命令行覆盖）。"""
    inst_id: str = "BTC-USDT"                 # 回测标的（交易对）
    start: str = "2024-09-01T00:00:00Z"       # 起始时间（ISO8601 或 "YYYY-mm-dd HH:MM:SS"），默认 UTC
    end: str = "2024-09-01T06:00:00Z"         # 结束时间（包含端点的右侧可能根据采样方式而不同）
    timeframe: str = "1s"                     # 时间粒度："1s" 或 "1min" 等

    # 策略相关（与 strategies.depth_strategy 的思想一致）
    levels: int = 5
    mode: str = "notional"                    # "size" 或 "notional"
    threshold: float = 0.02                    # 相对不平衡阈值（例如 0.02 表示 2%）

    # 执行与成本（以基点计，1bp=0.01%）
    fee_bps: float = 1.0                       # 手续费（往返或单边由自身策略决定，这里按持仓切换计提）
    slippage_bps: float = 1.0                  # 滑点（按持仓切换时计提）

    # 资金与输出
    initial_capital: float = 10000.0
    csv_out: Optional[str] = None              # 导出回测结果 CSV 的路径（可选）
    plot_svg: Optional[str] = None             # 生成收益曲线 SVG 图路径（可选）
    prefer_trades_price: bool = False          # 是否优先使用成交价格构造价格序列（否则使用盘口 mid）


class Backtester:
    """简化回测引擎：基于 orderbook 快照生成信号，并在给定时间粒度上回放。"""

    def __init__(self, cfg: BacktestConfig) -> None:
        self.cfg = cfg
        self.db = TimescaleDB()

    # =====================
    # 工具函数
    # =====================
    @staticmethod
    def _to_utc(ts: str) -> datetime:
        """将字符串解析为 UTC 时间。"""
        dt = pd.to_datetime(ts, utc=True)
        if dt.tzinfo is None:
            return dt.tz_localize(timezone.utc)
        return dt.tz_convert(timezone.utc)

    @staticmethod
    def _parse_timeframe_to_freq(tf: str) -> Tuple[str, float]:
        """将 timeframe 文本转换为 pandas 频率标识与“每年 bar 数”。
        返回 (pandas_freq_str, bars_per_year)。
        支持：1s、5s、15s、1min、5min、15min。
        """
        s_match = re.fullmatch(r"(\d+)s", tf)
        m_match = re.fullmatch(r"(\d+)(min|m)", tf)
        if s_match:
            secs = int(s_match.group(1))
            secs = max(1, secs)
            bars_per_year = (365 * 24 * 3600) / secs
            return f"{secs}S", bars_per_year
        if m_match:
            mins = int(m_match.group(1))
            mins = max(1, mins)
            bars_per_year = (365 * 24 * 60) / mins
            return f"{mins}T", bars_per_year
        # 兜底：默认按秒
        logger.warning("无法解析 timeframe=%s，回退为 1s", tf)
        return "1S", 365 * 24 * 3600

    @staticmethod
    def _sum_depth(levels: int, side, mode: str) -> float:
        """聚合某一侧（bids 或 asks）的前 N 档深度。side 为 [[price, size, ...], ...]。"""
        total = 0.0
        if not isinstance(side, (list, tuple)):
            return 0.0
        for lvl in list(side)[: max(levels, 0)]:
            if not isinstance(lvl, (list, tuple)) or len(lvl) < 2:
                continue
            try:
                px = float(lvl[0])
                sz = float(lvl[1])
            except Exception:
                continue
            if mode == "size":
                total += sz
            else:
                total += px * sz
        return total

    def _compute_signal_row(self, bids, asks) -> Tuple[str, float, float, float]:
        """基于一条盘口快照计算信号与相关指标。
        返回 (signal, buy_depth, sell_depth, mid)。
        """
        buy_depth = self._sum_depth(self.cfg.levels, bids, self.cfg.mode)
        sell_depth = self._sum_depth(self.cfg.levels, asks, self.cfg.mode)
        # 信号判定
        up_bound = sell_depth * (1.0 + self.cfg.threshold)
        down_bound = buy_depth * (1.0 + self.cfg.threshold)
        if buy_depth > up_bound:
            sig = "BUY"
        elif sell_depth > down_bound:
            sig = "SELL"
        else:
            sig = "HOLD"
        # 估算中间价（取最优 bid/ask）
        try:
            best_bid = float(bids[0][0]) if bids else np.nan
            best_ask = float(asks[0][0]) if asks else np.nan
            mid = np.nanmean([best_bid, best_ask])
        except Exception:
            mid = np.nan
        return sig, buy_depth, sell_depth, float(mid) if not math.isnan(mid) else np.nan

    # =====================
    # 数据加载与预处理
    # =====================
    def _load_data(self) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """加载 orderbook 与（可选）trades 数据。
        返回 (ob_df, tr_df)。
        - ob_df: 索引为 ts（UTC），包含列 [signal, buy_depth, sell_depth, mid]
        - tr_df: 若 prefer_trades_price=True 则返回按 timeframe 重采样后的成交价序列，否则为 None
        """
        start = self._to_utc(self.cfg.start)
        end = self._to_utc(self.cfg.end)
        freq, _ = self._parse_timeframe_to_freq(self.cfg.timeframe)

        logger.info("从数据库读取 orderbook: inst={} start={} end={}", self.cfg.inst_id, start, end)
        ob_raw = self.db.fetch_orderbook_window(start, end, inst_id=self.cfg.inst_id, ascending=True)
        if ob_raw.empty:
            raise RuntimeError("未查询到任何 orderbook 数据，请确认采集器是否已写入历史数据。")
        # 逐行计算 signal/buy_depth/sell_depth/mid
        ob_raw["signal"], ob_raw["buy_depth"], ob_raw["sell_depth"], ob_raw["mid"] = zip(
            *ob_raw.apply(lambda r: self._compute_signal_row(r["bids"], r["asks"]), axis=1)
        )
        # 设置时间索引，并按 timeframe 取每个区间内的“最后一条”作为当期信号与 mid
        ob_raw["ts"] = pd.to_datetime(ob_raw["ts"], utc=True)
        ob_df = ob_raw.set_index("ts").sort_index()
        ob_res = ob_df.resample(freq).last().dropna(subset=["signal", "mid"], how="any")

        tr_res: Optional[pd.DataFrame] = None
        if self.cfg.prefer_trades_price:
            logger.info("从数据库读取 trades: inst={} start={} end={}", self.cfg.inst_id, start, end)
            tr_raw = self.db.fetch_trades_window(start, end, inst_id=self.cfg.inst_id, ascending=True)
            if not tr_raw.empty:
                tr_raw["ts"] = pd.to_datetime(tr_raw["ts"], utc=True)
                tr_df = tr_raw.set_index("ts").sort_index()
                # 以最后一笔成交价作为该周期收盘价
                tr_res = tr_df[["price"]].resample(freq).last().rename(columns={"price": "close"})
                # 缺失用 forward fill，避免断档
                tr_res["close"] = tr_res["close"].ffill()
            else:
                logger.warning("trades 数据为空，回退使用盘口 mid 作为价格序列。")
        return ob_res, tr_res

    # =====================
    # 回测与绩效计算
    # =====================
    def run(self) -> dict:
        """运行回测主流程，返回汇总指标字典。"""
        ob_res, tr_res = self._load_data()

        # 构造价格序列：优先 trades 的收盘价，否则使用 mid
        price_series = None
        if self.cfg.prefer_trades_price and tr_res is not None and not tr_res.empty:
            price_series = tr_res["close"].reindex(ob_res.index).ffill()
        if price_series is None:
            price_series = ob_res["mid"].copy()
        price_series = price_series.ffill()

        # 将信号映射为目标持仓：BUY=+1, SELL=-1, HOLD=前值
        signal_map = {"BUY": 1, "SELL": -1, "HOLD": np.nan}
        target_pos = ob_res["signal"].map(signal_map)
        # HOLD 使用前一个目标持仓填充，初值为 0（空仓）
        target_pos = target_pos.ffill().fillna(0.0)
        # 实际持仓假设在下一根 bar 生效（避免未来函数）
        pos = target_pos.shift(1).fillna(0.0)

        # 价格收益（对数或算术收益均可，这里使用简单收益）
        ret = price_series.pct_change().fillna(0.0)
        gross = pos * ret

        # 交易成本：当持仓变化时计提（abs(delta_pos) * (fee+slip)/1e4）
        delta_pos = pos.diff().abs().fillna(0.0)
        cost_bps = (self.cfg.fee_bps + self.cfg.slippage_bps) / 10000.0
        cost = delta_pos * cost_bps
        net_ret = gross - cost

        # 权益曲线
        equity = (1.0 + net_ret).cumprod() * float(self.cfg.initial_capital)

        # 指标计算
        # 夏普比率：sqrt(bars_per_year) * mean(ret) / std(ret)
        _, bars_per_year = self._parse_timeframe_to_freq(self.cfg.timeframe)
        ret_std = net_ret.std(ddof=0)
        sharpe = float(np.sqrt(bars_per_year) * net_ret.mean() / ret_std) if ret_std > 1e-12 else float("nan")

        # 最大回撤
        roll_max = equity.cummax()
        drawdown = (equity - roll_max) / roll_max
        max_dd = float(drawdown.min()) if len(drawdown) else float("nan")

        # 组织结果表
        result = pd.DataFrame({
            "price": price_series,
            "signal": ob_res["signal"],
            "target_pos": target_pos,
            "pos": pos,
            "ret": ret,
            "gross": gross,
            "cost": cost,
            "net_ret": net_ret,
            "equity": equity,
        })

        # 导出 CSV（可选）
        if self.cfg.csv_out:
            out_path = self.cfg.csv_out
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            result.to_csv(out_path, index_label="ts")
            logger.info("已导出回测结果 CSV -> {}", out_path)

        # 生成 SVG 图（可选）
        if self.cfg.plot_svg:
            out_svg = self.cfg.plot_svg
            os.makedirs(os.path.dirname(out_svg), exist_ok=True)
            fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

            axes[0].plot(price_series.index, price_series.values, color="tab:blue", label="Price")
            axes[0].set_title(f"{self.cfg.inst_id} 价格 ({self.cfg.timeframe})")
            axes[0].legend(loc="upper left")

            axes[1].plot(equity.index, equity.values, color="tab:green", label="Equity")
            axes[1].set_title("策略权益曲线")
            axes[1].legend(loc="upper left")

            plt.tight_layout()
            fig.savefig(out_svg, format="svg")
            plt.close(fig)
            logger.info("已生成回测图表 SVG -> {}", out_svg)

        summary = {
            "inst_id": self.cfg.inst_id,
            "timeframe": self.cfg.timeframe,
            "start": self.cfg.start,
            "end": self.cfg.end,
            "initial_capital": float(self.cfg.initial_capital),
            "final_equity": float(equity.iloc[-1]) if len(equity) else float("nan"),
            "return": float(equity.iloc[-1] / self.cfg.initial_capital - 1.0) if len(equity) else float("nan"),
            "sharpe": sharpe,
            "max_drawdown": max_dd,
            "bars": int(len(result)),
        }

        logger.info(
            "回测完成 | inst={} tf={} 样本={} 最终权益={:.2f} 收益={:.2%} 夏普={:.3f} 最大回撤={:.2%}",
            summary["inst_id"], summary["timeframe"], summary["bars"],
            summary["final_equity"], summary["return"], summary["sharpe"], summary["max_drawdown"],
        )
        return summary


# =====================
# 命令行入口
# =====================

def parse_args() -> BacktestConfig:
    parser = argparse.ArgumentParser(description="基于历史盘口（可选成交）数据的简化回测引擎")
    parser.add_argument("--inst", dest="inst_id", default="BTC-USDT", help="交易对，例如 BTC-USDT")
    parser.add_argument("--start", dest="start", required=True, help="起始时间（ISO8601，例如 2024-09-01T00:00:00Z）")
    parser.add_argument("--end", dest="end", required=True, help="结束时间（ISO8601，例如 2024-09-01T06:00:00Z）")
    parser.add_argument("--timeframe", dest="timeframe", default="1s", help="时间粒度：1s/5s/15s/1min/5min 等")
    parser.add_argument("--levels", dest="levels", type=int, default=5, help="用于计算不平衡的深度档位数")
    parser.add_argument("--mode", dest="mode", default="notional", choices=["size", "notional"], help="聚合模式")
    parser.add_argument("--threshold", dest="threshold", type=float, default=0.02, help="相对不平衡阈值，例 0.02=2%")
    parser.add_argument("--fee-bps", dest="fee_bps", type=float, default=1.0, help="手续费基点，1bp=0.01%")
    parser.add_argument("--slippage-bps", dest="slippage_bps", type=float, default=1.0, help="滑点基点，1bp=0.01%")
    parser.add_argument("--initial-capital", dest="initial_capital", type=float, default=10000.0, help="初始资金")
    parser.add_argument("--csv-out", dest="csv_out", default=None, help="导出结果 CSV 的路径（可选）")
    parser.add_argument("--plot-svg", dest="plot_svg", default=None, help="导出收益曲线 SVG 的路径（可选）")
    parser.add_argument("--prefer-trades-price", dest="prefer_trades_price", action="store_true", help="优先使用成交价构造价格序列")
    args = parser.parse_args()

    return BacktestConfig(
        inst_id=args.inst_id,
        start=args.start,
        end=args.end,
        timeframe=args.timeframe,
        levels=args.levels,
        mode=args.mode,
        threshold=args.threshold,
        fee_bps=args.fee_bps,
        slippage_bps=args.slippage_bps,
        initial_capital=args.initial_capital,
        csv_out=args.csv_out,
        plot_svg=args.plot_svg,
        prefer_trades_price=args.prefer_trades_price,
    )


def main() -> None:
    cfg = parse_args()
    bt = Backtester(cfg)
    bt.db.connect()
    try:
        bt.run()
    finally:
        bt.db.close()


if __name__ == "__main__":
    main()