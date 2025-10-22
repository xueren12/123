# -*- coding: utf-8 -*-
"""
回测CSV可视化工具：显示仓位与余额
用法示例：
  py visualize_backtest.py \
    --path "c:\\Users\\64789\\Desktop\\123\\data\\start_20250601_end_20251021_tf_15min_inst_ETH-USDT-SWAP\\backtest_multi_indicator.csv" \
    --initial 10000 \
    --out "c:\\Users\\64789\\Desktop\\123\\data\\visual_eth_15m.png"

参数说明：
- --path    CSV 文件路径（默认指向你给出的 ETH 15m 产物）
- --initial 初始金额（单位同报价货币，如 USDT），用于换算余额
- --out     输出图片路径（PNG/SVG）。不传则默认保存在 CSV 同目录下
- --start   可选：过滤起始时间（ISO 或 YYYY-MM-DD）
- --end     可选：过滤结束时间（ISO 或 YYYY-MM-DD）
"""

import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter


def parse_args():
    p = argparse.ArgumentParser(description="可视化回测CSV，显示仓位与余额")
    p.add_argument(
        "--path",
        default=r"c:\\Users\\64789\\Desktop\\123\\data\\start_20250601_end_20251021_tf_15min_inst_ETH-USDT-SWAP\\backtest_multi_indicator.csv",
        help="CSV 文件路径",
    )
    p.add_argument("--initial", type=float, default=10000.0, help="初始金额（如 USDT）")
    p.add_argument("--out", default=None, help="输出图片路径（PNG/SVG），默认保存在CSV同目录")
    p.add_argument("--start", default=None, help="可选：起始时间（ISO或YYYY-MM-DD）")
    p.add_argument("--end", default=None, help="可选：结束时间（ISO或YYYY-MM-DD）")
    return p.parse_args()


def load_df(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"CSV 文件不存在：{path}")
    df = pd.read_csv(path)
    if "ts" not in df.columns:
        raise RuntimeError("CSV 缺少 ts 列")
    # 解析时间与基本列
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts"]).sort_values("ts").set_index("ts")
    for c in ["equity", "pos", "ret", "close"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # 若 equity 缺失，则按 ret 复合计算（一般不缺失）
    if "equity" not in df.columns and "ret" in df.columns:
        eq = (1.0 + df["ret"].fillna(0.0)).cumprod()
        df["equity"] = eq
    # 缺列保护
    if "equity" not in df.columns:
        raise RuntimeError("CSV 缺少 equity 列，无法计算余额")
    if "pos" not in df.columns:
        df["pos"] = 0.0
    return df


def clip_window(df: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
    if start:
        try:
            s = pd.to_datetime(start, utc=True)
            df = df[df.index >= s]
        except Exception:
            pass
    if end:
        try:
            e = pd.to_datetime(end, utc=True)
            df = df[df.index <= e]
        except Exception:
            pass
    return df


def fmt_currency():
    return FuncFormatter(lambda x, pos: f"{x:,.0f}")


def visualize(df: pd.DataFrame, initial: float, out_path: Path) -> dict:
    df = df.copy()
    df["balance"] = df["equity"].astype(float) * float(initial)
    df["pos"] = df["pos"].astype(float).fillna(0.0)
    df["pos_value"] = df["pos"] * df["balance"]

    final_equity = float(df["equity"].iloc[-1])
    final_balance = float(df["balance"].iloc[-1])
    trades = int(((df.get("signal") == "BUY").sum() if "signal" in df.columns else 0) + ((df.get("signal") == "SELL").sum() if "signal" in df.columns else 0))

    # 画图：上余额，下仓位
    fig = plt.figure(figsize=(12, 6))
    gs = plt.GridSpec(2, 1, height_ratios=[2, 1], hspace=0.15)
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1], sharex=ax1)

    ax1.plot(df.index, df["balance"], color="#1f77b4", linewidth=1.6, label=f"余额 (初始={initial:g})")
    ax1.set_ylabel("余额")
    ax1.yaxis.set_major_formatter(fmt_currency())
    ax1.grid(True, linestyle="--", alpha=0.3)

    # 标注买卖信号（若存在）
    if "signal" in df.columns:
        buy_ts = df.index[df["signal"] == "BUY"]
        sell_ts = df.index[df["signal"] == "SELL"]
        ax1.scatter(buy_ts, df.loc[buy_ts, "balance"], marker="^", color="#2ca02c", s=18, label="BUY")
        ax1.scatter(sell_ts, df.loc[sell_ts, "balance"], marker="v", color="#d62728", s=18, label="SELL")
    ax1.legend(loc="upper left")

    # 仓位比例与持仓区间高亮
    ax2.plot(df.index, df["pos"], color="#ff7f0e", linewidth=1.3, label="仓位比例")
    pos_mask = df["pos"] > 0
    ax2.fill_between(df.index, 0, df["pos"], where=pos_mask, color="#ff7f0e", alpha=0.15)
    ax2.set_ylabel("仓位(0~1)")
    ax2.set_xlabel("时间")
    ax2.grid(True, linestyle="--", alpha=0.3)

    # x轴日期格式
    locator = mdates.AutoDateLocator(minticks=6, maxticks=12)
    formatter = mdates.ConciseDateFormatter(locator)
    ax2.xaxis.set_major_locator(locator)
    ax2.xaxis.set_major_formatter(formatter)

    # 标题与说明
    plt.suptitle(
        f"回测可视化 | 最终余额={final_balance:,.2f}  (equity={final_equity:.4f})  交易次数={trades}",
        fontsize=12,
    )

    # 保存图片
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "final_equity": final_equity,
        "final_balance": final_balance,
        "trades": trades,
        "out": str(out_path),
    }


def main():
    args = parse_args()
    csv_path = Path(args.path)
    df = load_df(csv_path)
    df = clip_window(df, args.start, args.end)

    # 默认输出路径：CSV 同目录，文件名含初始金额
    if args.out:
        out_path = Path(args.out)
    else:
        out_name = f"visual_balance_pos_init_{int(args.initial)}.png"
        out_path = csv_path.parent / out_name

    info = visualize(df, args.initial, out_path)

    print("===== 可视化完成 =====")
    print(f"输出文件: {info['out']}")
    print(f"最终 equity: {info['final_equity']:.6f}")
    print(f"最终余额: {info['final_balance']:,.2f}")
    print(f"交易次数(买+卖): {info['trades']}")


if __name__ == "__main__":
    raise SystemExit(main())