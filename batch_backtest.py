import os
import json
import time
import subprocess
from datetime import datetime
import pandas as pd

# Fixed dataset and period for ETH 15m
ROOT = os.path.dirname(__file__)
CSV_PATH = os.path.join(ROOT, "data", "ohlcv_ETH-USDT-SWAP_15m_2025-06-01_2025-10-21.csv")
INST = "ETH-USDT-SWAP"
START = "2025-06-01"
END = "2025-10-21"
TF = "15m"
BASE_EQUITY = 10000.0

# Build output dir consistent with ma_backtest.py

def build_output_dir(start: str, end: str, tf: str, inst: str):
    start_str = datetime.strptime(start, "%Y-%m-%d").strftime("%Y%m%d")
    end_str = datetime.strptime(end, "%Y-%m-%d").strftime("%Y%m%d")
    # 与 ma_backtest 输出目录一致：将 15m 规范化为 15min
    tf_str = str(tf).strip().replace("m", "min").replace("/", "-")
    inst_label = str(inst).upper().replace("/", "-")
    return os.path.join(ROOT, "data", f"start_{start_str}_end_{end_str}_tf_{tf_str}_inst_{inst_label}")

OUTDIR = build_output_dir(START, END, TF, INST)
SUMMARY_JSON = os.path.join(OUTDIR, "backtest_summary.json")
CSV_OUT = os.path.join(OUTDIR, "backtest_multi_indicator.csv")

# Stage A grid（粗扫，加入冷却与金字塔选项，控制组合数量）
# 注：选择更贴近当前默认建议的范围，以减少无效组合
LEVERAGE_BASE = 50
GRID_SINGLE_PCT = [0.4, 0.6]
GRID_MAX_POS_USD = [12000.0]
GRID_RISK_PCT = [0.015, 0.02]
GRID_CONFIRM_MIN = [2, 3]
GRID_COOLDOWN_SEC = [600, 900]
GRID_PYRAMID_NEED_CONSEC = [0, 1]  # 保持允许加仓，测试是否需要连续BUY

# Stage B refinement around best (filled after Stage A)
REFINE_LEVERAGES = [20, 30]
REFINE_RISKS = [0.015, 0.025]


def run_backtest(leverage, single_pct, max_pos_usd, risk_pct, confirm_min, cooldown_sec, pyramid_need_consecutive):
    cmd = [
        "py", os.path.join(ROOT, "mvp", "backtest", "ma_backtest.py"),
        "--source", "csv",
        "--csv", CSV_PATH,
        "--inst", INST,
        "--start", START,
        "--end", END,
        "--timeframe", TF,
        "--leverage", str(leverage),
        "--base_equity_usd", str(BASE_EQUITY),
        "--single_order_max_pct_equity", str(single_pct),
        "--max_position_usd", str(max_pos_usd),
        "--risk_percent", str(risk_pct),
        "--confirm_min", str(confirm_min),
        "--cooldown_sec", str(cooldown_sec),
        "--sell_cooldown_sec", "0",
        "--pyramid_on_buy", "1",
        "--pyramid_need_consecutive", str(pyramid_need_consecutive),
        "--fee_bps", "0",
        "--slip_bps", "0",
        "--funding_bps_8h", "0",
        "--plot", "0",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    # Allow filesystem to flush
    time.sleep(0.2)
    # Read summary JSON
    summary = {}
    try:
        with open(SUMMARY_JSON, "r", encoding="utf-8") as f:
            summary = json.load(f)
    except Exception:
        summary = {"metrics": {"sharpe": 0.0, "winrate": 0.0, "max_drawdown": 0.0}}
    # Read final equity from CSV
    final_eq = None
    try:
        df = pd.read_csv(CSV_OUT)
        # 兼容列名：有的版本为 equity，有的是 eq
        if "equity" in df.columns:
            final_eq = float(df["equity"].iloc[-1])
        elif "eq" in df.columns:
            final_eq = float(df["eq"].iloc[-1])
    except Exception:
        final_eq = None
    return {
        "params": {
            "leverage": leverage,
            "single_order_max_pct_equity": single_pct,
            "max_position_usd": max_pos_usd,
            "risk_percent": risk_pct,
            "confirm_min": confirm_min,
            "cooldown_sec": cooldown_sec,
            "pyramid_need_consecutive": pyramid_need_consecutive,
        },
        "metrics": {
            "final_eq": final_eq,
            "return_pct": (final_eq - 1.0) * 100.0 if final_eq is not None else None,
            "sharpe": float(summary.get("metrics", {}).get("sharpe", 0.0)),
            "winrate": float(summary.get("metrics", {}).get("winrate", 0.0)),
            "max_drawdown": float(summary.get("metrics", {}).get("max_drawdown", 0.0)),
        },
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "rc": proc.returncode,
    }


def sort_key(item):
    m = item["metrics"]
    # None-safe values
    rp = m.get("return_pct") if m.get("return_pct") is not None else -1e9
    sh = m.get("sharpe") if m.get("sharpe") is not None else -1e9
    dd = m.get("max_drawdown") if m.get("max_drawdown") is not None else 1e9
    # Sort: Sharpe desc, Return desc, Drawdown asc
    return (-sh, -rp, dd)


def print_top(results, topn=10, header="Stage A Results"):
    print(f"\n=== {header} ===")
    print("leverage | single_pct | max_pos_usd | risk_pct | confirm_min | cooldown | pyramid_consec || sharpe | return% | maxDD | winrate")
    for r in results[:topn]:
        p = r["params"]; m = r["metrics"]
        print(f"{p['leverage']:>8} | {p['single_order_max_pct_equity']:.2f}      | {p['max_position_usd']:>12.0f} | {p['risk_percent']:.3f}  | {p['confirm_min']:>11} | {p['cooldown_sec']:>8} | {p['pyramid_need_consecutive']:>14} || "
              f"{m['sharpe']:.3f} | {m['return_pct'] if m['return_pct'] is not None else float('nan'):.2f} | {m['max_drawdown']:.2%} | {m['winrate']:.2%}")


def main():
    results_a = []
    for single_pct in GRID_SINGLE_PCT:
        for max_pos in GRID_MAX_POS_USD:
            for risk_pct in GRID_RISK_PCT:
                for cmin in GRID_CONFIRM_MIN:
                    for cd in GRID_COOLDOWN_SEC:
                        for pyc in GRID_PYRAMID_NEED_CONSEC:
                            r = run_backtest(LEVERAGE_BASE, single_pct, max_pos, risk_pct, cmin, cd, pyc)
                            results_a.append(r)
    results_a_sorted = sorted(results_a, key=sort_key)
    print_top(results_a_sorted, header="Stage A Results (ETH 15m 2025-06~10)")

    # Pick top 2 for refinement
    top2 = results_a_sorted[:2]
    results_b = []
    for t in top2:
        p = t["params"]
        for L in REFINE_LEVERAGES:
            for rk in REFINE_RISKS:
                # 以冠军组合为中心，仅微调杠杆与风险，保留最佳 cooldown 与金字塔设置
                r = run_backtest(L, p["single_order_max_pct_equity"], p["max_position_usd"], rk, p["confirm_min"], p["cooldown_sec"], p["pyramid_need_consecutive"])
                results_b.append(r)
    results_b_sorted = sorted(results_b, key=sort_key)
    print_top(results_b_sorted, header="Stage B Refinement")

    best = results_b_sorted[0] if results_b_sorted else results_a_sorted[0]
    print("\n=== Recommended Parameters ===")
    print(json.dumps(best["params"], indent=2))
    print("Metrics:")
    print(json.dumps(best["metrics"], indent=2))


if __name__ == "__main__":
    main()