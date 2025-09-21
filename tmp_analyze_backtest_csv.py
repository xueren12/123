# -*- coding: utf-8 -*-
"""
临时工具：分析回测明细CSV（backtest_multi_indicator.csv）并输出关键统计（JSON）。
使用方式：
  py tmp_analyze_backtest_csv.py --path "c:\\path\\to\\backtest_multi_indicator.csv" [--extended]

注：仅作为分析辅助脚本，后续可删除。
"""
import argparse
import json
import math
import sys
from typing import Optional, List, Dict, Any

import pandas as pd


def safe_float(x: Optional[float]) -> Optional[float]:
    """将值安全转换为float，无法转换返回None。"""
    try:
        if x is None:
            return None
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except Exception:
        return None


def segment_trades(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """根据 pos!=0 的连续片段切分交易，返回每笔交易的起止索引、方向、区间PnL与bar数。"""
    bars = len(df)
    pos = df["pos"].fillna(0.0) if "pos" in df.columns else pd.Series([0.0] * bars)
    ret = df["ret"].fillna(0.0) if "ret" in df.columns else pd.Series([0.0] * bars)

    in_pos = pos.ne(0.0)
    prev = in_pos.shift(1, fill_value=False)
    next_ = in_pos.shift(-1, fill_value=False)
    starts = in_pos & (~prev)
    ends = in_pos & (~next_)

    trades: List[Dict[str, Any]] = []
    for s, e in zip(df.index[starts], df.index[ends]):
        seg = ret.iloc[s : e + 1]
        seg_pnl = safe_float(seg.sum()) or 0.0
        direction = "long" if (pos.iloc[s] > 0) else ("short" if (pos.iloc[s] < 0) else "flat")
        trades.append({
            "start": int(s),
            "end": int(e),
            "bars": int(e - s + 1),
            "side": direction,
            "pnl": seg_pnl,
        })
    return trades


def analyze_csv(path: str, extended: bool = False) -> dict:
    """读取CSV并计算：交易次数、胜率、平均持仓bar、收益均值/波动、累计收益、最终净值、最大回撤等。
    若 extended=True，额外输出：列名、净值首/峰/谷/末、分多空统计、交易分布分位数等。
    """
    # 读取CSV
    df = pd.read_csv(path)

    # 时间列（可选）：若存在则尝试解析为UTC
    ts = None
    if "ts" in df.columns:
        try:
            ts = pd.to_datetime(df["ts"], errors="coerce", utc=True)
        except Exception:
            ts = None

    bars = len(df)
    cols = list(df.columns)

    # 信号分布（若有）
    signal_counts = {}
    if "signal" in df.columns:
        try:
            signal_counts = df["signal"].value_counts(dropna=False).to_dict()
        except Exception:
            signal_counts = {}

    # 仓位与收益序列（缺失则用0填充）
    pos = df["pos"].fillna(0.0) if "pos" in df.columns else pd.Series([0.0] * bars)
    ret = df["ret"].fillna(0.0) if "ret" in df.columns else pd.Series([0.0] * bars)

    # 识别交易分段：pos!=0 的连续片段视为一笔交易
    in_pos = pos.ne(0.0)
    prev = in_pos.shift(1, fill_value=False)
    next_ = in_pos.shift(-1, fill_value=False)
    starts = in_pos & (~prev)
    ends = in_pos & (~next_)
    start_idx = list(df.index[starts])
    end_idx = list(df.index[ends])

    wins = 0
    trades_n = 0
    durations = []
    pnl_list = []

    for s, e in zip(start_idx, end_idx):
        seg = ret.iloc[s : e + 1]
        seg_pnl = safe_float(seg.sum()) or 0.0
        pnl_list.append(seg_pnl)
        durations.append(int(e - s + 1))
        trades_n += 1
        if seg_pnl > 0:
            wins += 1

    winrate = (wins / trades_n) if trades_n > 0 else None

    # 净值与回撤
    final_equity = None
    cum_return = None
    max_dd = None
    equity_profile = None

    if "equity" in df.columns:
        try:
            eq = pd.to_numeric(df["equity"], errors="coerce").dropna()
            if len(eq) > 1:
                first = safe_float(eq.iloc[0])
                peak = safe_float(eq.cummax().max())
                trough = safe_float(eq.min())
                last = safe_float(eq.iloc[-1])
                final_equity = last
                base = first
                cum_return = (last / base - 1.0) if (last is not None and base and base != 0) else None
                roll_max = eq.cummax()
                dd = (roll_max - eq) / roll_max
                max_dd = safe_float(dd.max())
                equity_profile = {"first": first, "peak": peak, "trough": trough, "last": last}
        except Exception:
            pass

    # 时间覆盖信息
    start_ts = None
    end_ts = None
    if ts is not None and not ts.isna().all():
        start_ts = ts.iloc[0].isoformat()
        end_ts = ts.iloc[-1].isoformat()

    # 其他统计：bar级收益均值与波动、持仓占比
    try:
        ret_mean = safe_float(ret.mean())
        ret_std = safe_float(ret.std(ddof=0))
        pos_time_frac = safe_float(in_pos.mean())
    except Exception:
        ret_mean = None
        ret_std = None
        pos_time_frac = None

    res = {
        "file": path,
        "shape": {"rows": bars, "columns": len(cols)},
        "time": {"start": start_ts, "end": end_ts},
        "signals": signal_counts,
        "pos": {
            "time_in_pos_frac": pos_time_frac,
            "trades": trades_n,
            "winrate": winrate,
            "avg_trade_len_bars": (sum(durations) / len(durations) if durations else None),
        },
        "returns": {
            "bar_mean": ret_mean,
            "bar_std": ret_std,
            "cum_return": cum_return,
            "final_equity": final_equity,
            "max_drawdown": max_dd,
        },
    }

    # 扩展统计
    if extended:
        res["columns"] = cols
        # 交易列表与分布
        trades = segment_trades(df)
        res["trades_detail"] = {
            "count": len(trades),
            "pnl_sum": safe_float(sum(t["pnl"] for t in trades)) if trades else None,
            "pnl_mean": safe_float(sum(t["pnl"] for t in trades) / len(trades)) if trades else None,
            "pnl_min": safe_float(min((t["pnl"] for t in trades), default=None)),
            "pnl_max": safe_float(max((t["pnl"] for t in trades), default=None)),
            "bars_mean": safe_float(sum(t["bars"] for t in trades) / len(trades)) if trades else None,
        }
        # 分位数（若样本量足够）
        pnl_series = pd.Series([t["pnl"] for t in trades]) if trades else pd.Series(dtype=float)
        if len(pnl_series) > 0:
            quant = pnl_series.quantile([0.1, 0.25, 0.5, 0.75, 0.9]).to_dict()
            res["trades_detail"]["pnl_quantiles"] = {str(k): safe_float(v) for k, v in quant.items()}
            res["trades_detail"]["win_avg"] = safe_float(pnl_series[pnl_series > 0].mean())
            res["trades_detail"]["loss_avg"] = safe_float(pnl_series[pnl_series < 0].mean())
            # 期望与盈亏比
            winrate = res["pos"]["winrate"] or 0.0
            win_avg = res["trades_detail"]["win_avg"] or 0.0
            loss_avg = res["trades_detail"]["loss_avg"] or 0.0
            payoff = (abs(win_avg / loss_avg) if (loss_avg is not None and loss_avg != 0) else None)
            expectancy = winrate * win_avg + (1 - winrate) * loss_avg
            res["trades_detail"]["payoff_ratio"] = safe_float(payoff)
            res["trades_detail"]["expectancy_per_trade"] = safe_float(expectancy)
        # 多空分组
        side_df = pd.DataFrame(trades)
        if not side_df.empty:
            grouped = side_df.groupby("side")["pnl"].agg(["count", "sum", "mean"]).to_dict()
            # 转换为普通dict，带中文key
            res["by_side"] = {}
            for side in ("long", "short", "flat"):
                if side in grouped["count"]:
                    res["by_side"][side] = {
                        "trades": int(grouped["count"][side]),
                        "pnl_sum": safe_float(grouped["sum"][side]),
                        "pnl_mean": safe_float(grouped["mean"][side]),
                    }
        # 净值概况
        if equity_profile is not None:
            res["equity_profile"] = equity_profile

    return res


def main():
    parser = argparse.ArgumentParser(description="分析回测明细CSV并输出关键统计（JSON）")
    parser.add_argument("--path", required=True, help="CSV 文件路径")
    parser.add_argument("--extended", action="store_true", help="输出扩展统计信息")
    args = parser.parse_args()

    res = analyze_csv(args.path, extended=args.extended)
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()