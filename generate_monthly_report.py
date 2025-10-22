import json
from pathlib import Path
import pandas as pd

BASE_DIR = Path(r"c:\Users\64789\Desktop\123\data\start_20250101_end_20251015_tf_15min_inst_ETH")
CSV_PATH = BASE_DIR / "backtest_multi_indicator.csv"
SUMMARY_PATH = BASE_DIR / "backtest_summary.json"
MONTHLY_CSV = BASE_DIR / "analysis_monthly.csv"
REPORT_MD = BASE_DIR / "monthly_report.md"


def load_monthly_stats() -> pd.DataFrame:
    if MONTHLY_CSV.exists():
        df = pd.read_csv(MONTHLY_CSV, index_col=0)
        return df
    # 回退：从CSV重新计算
    src = pd.read_csv(CSV_PATH)
    src["ts"] = pd.to_datetime(src["ts"]) 
    src["month"] = src["ts"].dt.to_period("M").astype(str)
    grouped = src.groupby("month")
    def month_ret(g):
        start_eq = float(g.iloc[0]["equity"]) 
        end_eq = float(g.iloc[-1]["equity"]) 
        return (end_eq / start_eq) - 1.0
    def month_max_dd(g):
        eq = g["equity"].astype(float)
        running_max = eq.cummax()
        dd = (eq / running_max) - 1.0
        return float(dd.min())
    out = pd.DataFrame({
        "month_return": grouped.apply(month_ret),
        "max_drawdown": grouped.apply(month_max_dd),
        "buy_count": grouped.apply(lambda g: (g["signal"] == "BUY").sum()),
        "sell_count": grouped.apply(lambda g: (g["signal"] == "SELL").sum()),
        "avg_pos": grouped["pos"].mean(),
        "bars": grouped.size(),
    })
    out.to_csv(MONTHLY_CSV)
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
    return {
        "start_equity": start_eq,
        "end_equity": end_eq,
        "total_return": total_ret,
        "overall_max_drawdown": overall_max_dd,
        "min_equity": min_eq,
        "max_equity": max_eq,
    }


def try_make_charts(mstats: pd.DataFrame, full_df: pd.DataFrame):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("未安装matplotlib，跳过图表生成")
        return []

    outputs = []
    # 排序月份
    months = sorted(mstats.index.tolist())

    # 月度收益柱状图
    ret_vals = mstats.loc[months, "month_return"].values
    colors = ["tab:green" if v >= 0 else "tab:red" for v in ret_vals]
    plt.figure(figsize=(12, 5))
    plt.bar(months, ret_vals, color=colors)
    plt.axhline(0, color="gray", linewidth=0.8)
    plt.title("Monthly Returns")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Return")
    out1 = BASE_DIR / "monthly_return_bar.png"
    plt.tight_layout()
    plt.savefig(out1)
    outputs.append(out1)
    plt.close()

    # 月度最大回撤柱状图
    dd_vals = mstats.loc[months, "max_drawdown"].values
    plt.figure(figsize=(12, 5))
    plt.bar(months, dd_vals, color="tab:orange")
    plt.title("Monthly Max Drawdown")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Drawdown")
    out2 = BASE_DIR / "monthly_drawdown_bar.png"
    plt.tight_layout()
    plt.savefig(out2)
    outputs.append(out2)
    plt.close()

    # 权益曲线（按日采样）
    full_df = full_df.copy()
    full_df["ts"] = pd.to_datetime(full_df["ts"]) 
    daily = full_df.set_index("ts")["equity"].astype(float).resample("D").last().dropna()
    plt.figure(figsize=(12, 4))
    plt.plot(daily.index, daily.values, color="tab:blue")
    plt.title("Equity Curve (Daily)")
    plt.ylabel("Equity")
    plt.tight_layout()
    out3 = BASE_DIR / "equity_curve_daily.png"
    plt.savefig(out3)
    outputs.append(out3)
    plt.close()

    return outputs


def main():
    # 读取全量CSV
    if not CSV_PATH.exists():
        print("未找到backtest_multi_indicator.csv")
        return
    df = pd.read_csv(CSV_PATH)
    df["ts"] = pd.to_datetime(df["ts"]) 

    # 月度统计
    mstats = load_monthly_stats()

    # 总体指标
    overall = overall_metrics(df)

    # 汇总JSON（可选）
    summary = None
    if SUMMARY_PATH.exists():
        try:
            summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
        except Exception:
            summary = None

    # 排名与亮点
    sorted_by_ret = mstats.sort_values("month_return", ascending=False)
    top3 = sorted_by_ret.head(3)
    bottom3 = sorted_by_ret.tail(3).sort_values("month_return")
    worst_dd = mstats.sort_values("max_drawdown").head(3)  # 更负更靠前

    # 累计收益（按月复利）
    cum = (1.0 + mstats["month_return"]).prod() - 1.0

    # 生成Markdown报告
    lines = []
    lines.append("# ETH 15m 回测月度报告（2025-01 至 2025-10）")
    lines.append("")
    lines.append(f"- 数据目录: `{BASE_DIR}`")
    lines.append(f"- 起始权益: {overall['start_equity']:.4f}")
    lines.append(f"- 结束权益: {overall['end_equity']:.4f}")
    lines.append(f"- 总收益: {overall['total_return']*100:.2f}%")
    lines.append(f"- 整体最大回撤: {overall['overall_max_drawdown']*100:.2f}%")
    lines.append(f"- 月度累计收益(复利): {cum*100:.2f}%")
    if summary:
        # 若有年化指标则显示
        if "annualized_return" in summary:
            try:
                lines.append(f"- 年化收益: {float(summary['annualized_return'])*100:.2f}%")
            except Exception:
                lines.append(f"- 年化收益: {summary['annualized_return']}")
        if "annualized_sharpe" in summary:
            lines.append(f"- 年化夏普: {summary['annualized_sharpe']}")
    lines.append("")

    # 月度表格
    lines.append("## 月度统计")
    lines.append("- 列: 月份、收益%、最大回撤%、买入数、卖出数、平均仓位、样本条数")
    for m, row in mstats.iterrows():
        lines.append(
            f"- {m}: 收益 {row['month_return']*100:.2f}%, 最大回撤 {row['max_drawdown']*100:.2f}%, "
            f"买 {int(row['buy_count'])} / 卖 {int(row['sell_count'])}, 平均仓位 {row['avg_pos']:.2f}, Bars {int(row['bars'])}"
        )
    lines.append("")

    # 亮点与风险
    lines.append("## 亮点")
    for m, r in top3.iterrows():
        lines.append(f"- 高收益月份: {m}，收益 {r['month_return']*100:.2f}%，最大回撤 {r['max_drawdown']*100:.2f}%")
    lines.append("\n## 风险点")
    for m, r in worst_dd.iterrows():
        lines.append(f"- 深回撤月份: {m}，最大回撤 {r['max_drawdown']*100:.2f}%，当月收益 {r['month_return']*100:.2f}%")
    lines.append("")

    # 建议
    lines.append("## 建议")
    lines.append("- 在深回撤月份启用波动率/趋势过滤（如ADX/ATR门槛）以减少进场频次")
    lines.append("- 设定月度回撤阈值（如-30%）触发降杠杆/提高止损灵敏度")
    lines.append("- 连续亏损时应用再入场冷却时间，避免高频反复进场导致连败")

    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")

    # 图表
    chart_paths = try_make_charts(mstats, df)
    for p in chart_paths:
        print(f"图表已生成: {p}")

    print(f"月度报告已保存: {REPORT_MD}")


if __name__ == "__main__":
    main()