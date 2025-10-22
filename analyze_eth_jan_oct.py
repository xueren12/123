import json
from pathlib import Path
import pandas as pd


BASE_DIR = Path(r"c:\Users\64789\Desktop\123\data\start_20250101_end_20251015_tf_15min_inst_ETH")
CSV_PATH = BASE_DIR / "backtest_multi_indicator.csv"
SUMMARY_PATH = BASE_DIR / "backtest_summary.json"


def print_summary_metrics(summary: dict):
    print("=== 回测汇总指标 ===")
    # 尝试打印常见指标，存在则显示
    fields = [
        ("annualized_sharpe", "年化夏普"),
        ("annualized_return", "年化收益"),
        ("annualized_volatility", "年化波动"),
        ("max_drawdown", "最大回撤"),
        ("win_rate", "胜率"),
        ("total_trades", "交易笔数"),
        ("start_equity", "起始权益"),
        ("end_equity", "结束权益"),
    ]
    for key, label in fields:
        if key in summary:
            val = summary[key]
            # 百分比类统一用百分比显示
            if key in {"annualized_return", "annualized_volatility", "max_drawdown", "win_rate"}:
                try:
                    print(f"- {label}: {float(val) * 100:.2f}%")
                except Exception:
                    print(f"- {label}: {val}")
            else:
                print(f"- {label}: {val}")
    # 打印未知但可能有用的其他字段
    extra = {k: v for k, v in summary.items() if k not in {f[0] for f in fields}}
    if extra:
        print("- 其他字段:")
        for k, v in extra.items():
            print(f"  * {k}: {v}")


def monthly_stats(df: pd.DataFrame) -> pd.DataFrame:
    # 按月分组
    df = df.copy()
    df["ts"] = pd.to_datetime(df["ts"])
    df["month"] = df["ts"].dt.to_period("M").astype(str)
    # 月度收益用当月末/当月初的权益之比
    def month_ret(group: pd.DataFrame) -> float:
        start_eq = group.iloc[0]["equity"]
        end_eq = group.iloc[-1]["equity"]
        return (end_eq / start_eq) - 1.0

    # 月度最大回撤（当月内以当月最高点为基准）
    def month_max_dd(group: pd.DataFrame) -> float:
        eq = group["equity"].astype(float)
        running_max = eq.cummax()
        dd = (eq / running_max) - 1.0
        return dd.min()

    # 买卖次数
    def count_buy(group: pd.DataFrame) -> int:
        return (group["signal"] == "BUY").sum()

    def count_sell(group: pd.DataFrame) -> int:
        return (group["signal"] == "SELL").sum()

    # 平均仓位（反映当月风险敞口）
    def avg_pos(group: pd.DataFrame) -> float:
        return group["pos"].astype(float).mean()

    grouped = df.groupby("month")
    out = pd.DataFrame({
        "month_return": grouped.apply(month_ret),
        "max_drawdown": grouped.apply(month_max_dd),
        "buy_count": grouped.apply(count_buy),
        "sell_count": grouped.apply(count_sell),
        "avg_pos": grouped.apply(avg_pos),
        "bars": grouped.size(),
    })
    return out


def overall_metrics(df: pd.DataFrame) -> dict:
    start_eq = float(df.iloc[0]["equity"]) if not df.empty else 1.0
    end_eq = float(df.iloc[-1]["equity"]) if not df.empty else start_eq
    total_ret = (end_eq / start_eq) - 1.0
    eq = df["equity"].astype(float)
    running_max = eq.cummax()
    dd = (eq / running_max) - 1.0
    overall_max_dd = float(dd.min()) if len(dd) else 0.0
    min_eq = float(eq.min()) if len(eq) else start_eq
    max_eq = float(eq.max()) if len(eq) else start_eq
    buy_total = int((df["signal"] == "BUY").sum())
    sell_total = int((df["signal"] == "SELL").sum())
    return {
        "start_equity": start_eq,
        "end_equity": end_eq,
        "total_return": total_ret,
        "overall_max_drawdown": overall_max_dd,
        "min_equity": min_eq,
        "max_equity": max_eq,
        "buy_total": buy_total,
        "sell_total": sell_total,
    }


def main():
    # 读取summary
    if SUMMARY_PATH.exists():
        try:
            summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
            print_summary_metrics(summary)
        except Exception as e:
            print(f"无法读取summary JSON: {e}")
    else:
        print("未找到backtest_summary.json")

    # 读取CSV
    if not CSV_PATH.exists():
        print("未找到backtest_multi_indicator.csv")
        return

    df = pd.read_csv(CSV_PATH)
    # 基本字段校验
    required_cols = {"ts", "equity", "signal", "pos"}
    if not required_cols.issubset(df.columns):
        print(f"CSV缺少必要字段: {required_cols - set(df.columns)}")
        return

    df["ts"] = pd.to_datetime(df["ts"])  # 确保时间戳格式

    print("\n=== 整体指标 ===")
    overall = overall_metrics(df)
    print(f"- 起始权益: {overall['start_equity']:.4f}")
    print(f"- 结束权益: {overall['end_equity']:.4f}")
    print(f"- 总收益: {overall['total_return']*100:.2f}%")
    print(f"- 整体最大回撤: {overall['overall_max_drawdown']*100:.2f}%")
    print(f"- 买入总数: {overall['buy_total']} / 卖出总数: {overall['sell_total']}")
    print(f"- 最低权益: {overall['min_equity']:.4f} / 最高权益: {overall['max_equity']:.4f}")

    # 爆仓风险估计（简化：权益低于0.1视为爆仓风险）
    print("\n=== 爆仓风险评估 ===")
    if overall["min_equity"] < 0.1:
        print("⚠️ 可能存在爆仓风险（最低权益 < 0.1）")
    else:
        print("✅ 未检测到爆仓风险（权益维持在安全区间）")

    # 月度统计
    print("\n=== 月度统计（2025-01 至 2025-10）===")
    mstats = monthly_stats(df)
    # 只显示2025-01到2025-10的月份
    mstats = mstats.loc[[m for m in mstats.index if m.startswith("2025-")]]
    for m, row in mstats.iterrows():
        print(
            f"- {m}: 收益 {row['month_return']*100:.2f}%, 最大回撤 {row['max_drawdown']*100:.2f}%, "
            f"买 {int(row['buy_count'])} / 卖 {int(row['sell_count'])}, 平均仓位 {row['avg_pos']:.2f}, Bars {int(row['bars'])}"
        )

    # 将月度统计保存到同目录，便于复用
    out_path = BASE_DIR / "analysis_monthly.csv"
    try:
        mstats.to_csv(out_path, index=True)
        print(f"\n月度统计已保存到: {out_path}")
    except Exception as e:
        print(f"月度统计保存失败: {e}")


if __name__ == "__main__":
    main()