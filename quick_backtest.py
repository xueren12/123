# -*- coding: utf-8 -*-
"""
一键回测脚本（Python 版）
目标：点一下就跑（双击或 py quick_backtest.py 即可）

功能：
1) 支持两种模式：
   - download：用 CCXT 下载 OHLCV 到 CSV，然后立刻回测
   - offline：直接读取 data/ 下最新的 ohlcv_*.csv 回测
2) 回测产物：
   - data/backtest_multi_indicator.csv
   - data/backtest_multi_indicator.svg
   - data/backtest_summary.json

说明：
- 按需修改下面 CONFIG 中的默认参数即可
- 脚本内部使用 py 命令调用已有脚本：
  - mvp/scripts/download_ohlcv_okx.py
  - mvp/backtest/ma_backtest.py
- 全程中文日志
"""

import os
import sys
import json
import glob
import subprocess
from pathlib import Path
from datetime import datetime
import re
import argparse

# ====================== 可配置参数（按需修改） ======================
CONFIG = {
    # 运行模式：download / offline
    "mode": "download",

    # 交易所（ccxt 名称）。示例：okx、binanceusdm
    "exchange": "okx",

    # 标的。例如 OKX 合约：ETH-USDT-SWAP、BTC-USDT-SWAP
    "inst": "ETH-USDT-SWAP",

    # K线周期：1m/5m/15m/1h/4h/1d 等
    "timeframe": "15m",

    # 新增：回测杠杆倍数（>=1，影响名义杠杆、资金费与换手成本的放大系数）
    "leverage": 20,
    # 是否允许做空（0=不允许，仅做多；1=允许双向）
    "allow_short": 0,

    # ===== 仓位与风险配置（影响实盘一致性） =====
    # 基准权益（USD）：用于计算开仓名义上限与风险资金
    "base_equity_usd": 10000,
    # 单次下单最大权益占比（0-1），默认更积极一些
    "single_order_max_pct_equity": 0.6,
    # 单标的最大持仓名义金额（USD），适度放宽上限以避免过小暴露
    "max_position_usd": 12000,
    # 风险等权开仓占权益百分比（仅 BUY 场景启用）
    "risk_percent": 0.025,

    # ===== 信号确认与节流 =====
    # 最少确认数（1~4），与多指标确认一致，默认 2 较为宽松
    "confirm_min": 2,
    # 通用冷却秒数（BUY/SELL均节流），避免过于频繁的同向/反向触发
    "cooldown_sec": 600,
    # 仅针对 SELL 的额外冷却（如无需额外节流设为 0）
    "sell_cooldown_sec": 0,
    # BUY 时允许加仓、是否要求连续 BUY 才加仓
    "pyramid_on_buy": 1,
    "pyramid_need_consecutive": 0,

    # 起止日期（UTC）
    "start": "2025-06-02",
    "end": "2025-10-21",


    # 代理（如无需代理留空）。形如 "http://127.0.0.1:7890"
    "proxy": "http://127.0.0.1:7890",

    # 数据目录
    "data_dir": "data",

    # （可选）直接指定已有 CSV 文件路径；若填写将优先使用该文件并自动切换为离线模式
    "csv_path": "",

    # 下载器相关参数
    "timeout": 30000,
    "max_retries": 5,
}
# ===============================================================

# 为了兼容双击运行，切换到脚本所在目录
PROJECT_DIR = Path(__file__).resolve().parent
os.chdir(PROJECT_DIR)


def run_cmd(cmd: list, env: dict | None = None) -> int:
    """运行外部命令并实时输出，返回退出码。"""
    print("执行命令:", " ".join(cmd))
    try:
        proc = subprocess.run(cmd, env=env, cwd=PROJECT_DIR, text=True)
        return proc.returncode
    except FileNotFoundError:
        print("错误：命令未找到，请确认已安装 Python 且可通过 'py' 命令调用。")
        return 1


def ensure_dir(p: Path):
    """确保目录存在。"""
    p.mkdir(parents=True, exist_ok=True)


def pick_latest_csv(data_dir: Path) -> Path | None:
    """从 data 目录挑选最新的 ohlcv_*.csv。"""
    files = sorted(data_dir.glob("*ohlcv*.csv"), key=lambda x: x.stat().st_mtime, reverse=True)
    return files[0] if files else None


def parse_args():
    """解析命令行参数（支持将 CSV 文件路径作为位置参数或 --csv 传入）。"""
    p = argparse.ArgumentParser(description="一键回测入口（支持以 CSV 文件路径作为参数）")
    # 位置参数：直接给 CSV 文件路径即可
    p.add_argument("csv", nargs="?", help="CSV 文件路径（可选，提供则直接离线回测该文件）")
    # 可选参数：与位置参数等价
    p.add_argument("--csv", dest="csv_opt", help="CSV 文件路径（等同于位置参数）")
    # 允许做空开关（覆盖 CONFIG.allow_short）
    p.add_argument("--allow-short", dest="allow_short", type=int, choices=[0, 1], help="允许做空：1是 0否（默认0，长仅）")
    return p.parse_args()


def main() -> int:
    # 解析命令行参数，支持以 CSV 路径一键离线回测
    args = parse_args()
    user_csv = (getattr(args, "csv_opt", None) or getattr(args, "csv", None))
    user_csv_path = None
    if user_csv:
        p = Path(user_csv).expanduser()
        if not p.exists():
            print(f"错误：指定的 CSV 文件不存在：{p}")
            input("按回车键退出...")
            return 1
        # 若传入了 CSV 文件，则强制进入离线模式
        CONFIG["mode"] = "offline"
        # 尝试从文件名解析 inst/timeframe/start/end（若不匹配则保持原配置）
        m = re.search(r"ohlcv_(?P<inst>[^_]+)_(?P<tf>[^_]+)_(?P<start>\d{4}-\d{2}-\d{2})_(?P<end>\d{4}-\d{2}-\d{2})\\.csv$", p.name)
        if m:
            CONFIG["inst"] = m.group("inst")
            CONFIG["timeframe"] = m.group("tf")
            CONFIG["start"] = m.group("start")
            CONFIG["end"] = m.group("end")
        user_csv_path = p.resolve()
    else:
        # 未通过命令行提供 CSV，则尝试读取配置中的 csv_path
        cfg_csv = (CONFIG.get("csv_path") or "").strip()
        if cfg_csv:
            p = Path(cfg_csv).expanduser()
            if not p.is_absolute():
                p = (PROJECT_DIR / p).resolve()  # 相对路径按项目根目录解析
            if p.exists():
                CONFIG["mode"] = "offline"  # 指定了文件则强制离线
                m = re.search(r"ohlcv_(?P<inst>[^_]+)_(?P<tf>[^_]+)_(?P<start>\d{4}-\d{2}-\d{2})_(?P<end>\d{4}-\d{2}-\d{2})\\.csv$", p.name)
                if m:
                    CONFIG["inst"] = m.group("inst")
                    CONFIG["timeframe"] = m.group("tf")
                    CONFIG["start"] = m.group("start")
                    CONFIG["end"] = m.group("end")
                user_csv_path = p
                print(f"检测到 CONFIG['csv_path']，将使用指定文件：{p}")
            else:
                print(f"警告：CONFIG['csv_path'] 指定的文件不存在：{p}，将回退到默认逻辑。")

    # 打印配置
    print("===== Python 一键回测 =====")
    for k in ["mode", "exchange", "inst", "timeframe", "start", "end", "proxy", "data_dir", "csv_path"]:
        print(f"{k}: {CONFIG.get(k)}")

    # 补充打印关键策略参数，便于确认
    print("\n===== 策略参数 =====")
    for k in [
        "leverage",
        "allow_short",
        "base_equity_usd",
        "single_order_max_pct_equity",
        "max_position_usd",
        "risk_percent",
        "confirm_min",
        "cooldown_sec",
        "sell_cooldown_sec",
        "pyramid_on_buy",
        "pyramid_need_consecutive",
    ]:
        print(f"{k}: {CONFIG.get(k)}")

    data_dir = PROJECT_DIR / CONFIG["data_dir"]
    ensure_dir(data_dir)

    # 构造 CSV 路径（下载模式会用到）
    csv_name = f"ohlcv_{CONFIG['inst']}_{CONFIG['timeframe']}_{CONFIG['start']}_{CONFIG['end']}.csv"
    csv_path = data_dir / csv_name

    # 准备环境变量（用于代理）
    env = os.environ.copy()
    if CONFIG["proxy"]:
        print(f"设置代理: {CONFIG['proxy']}")
        env["HTTP_PROXY"] = CONFIG["proxy"]
        env["HTTPS_PROXY"] = CONFIG["proxy"]

    # 1) 下载阶段（可选）
    if CONFIG["mode"].lower() == "download":
        print("\n===== 下载阶段 =====")
        download_cmd = [
            "py", "-u", str(PROJECT_DIR / "mvp" / "scripts" / "download_ohlcv_okx.py"),
            "--engine", "ccxt",
            "--ccxt-exchange", CONFIG["exchange"],
            "--inst", CONFIG["inst"],
            "--timeframe", CONFIG["timeframe"],
            "--start", CONFIG["start"],
            "--end", CONFIG["end"],
            "--out", str(csv_path),
            "--timeout", str(CONFIG["timeout"]),
            "--max-retries", str(CONFIG["max_retries"]),
        ]
        if CONFIG["proxy"]:
            download_cmd += ["--proxy", CONFIG["proxy"]]

        code = run_cmd(download_cmd, env=env)
        if code != 0:
            print("错误：数据下载失败，终止。")
            input("按回车键退出...")
            return code

        if not csv_path.exists():
            print("错误：CSV 文件未生成，终止。")
            input("按回车键退出...")
            return 1

        size_kb = round(csv_path.stat().st_size / 1024, 2)
        print(f"下载完成: {csv_path.name} ({size_kb} KB)")

    else:
        print("\n===== 离线模式 =====")
        if user_csv_path:
            # 优先使用用户通过参数传入的 CSV 文件
            csv_path = user_csv_path
            print(f"使用指定文件: {csv_path.name}")
        else:
            latest = pick_latest_csv(data_dir)
            if not latest:
                print("错误：未在 data/ 目录中找到任何 *ohlcv*.csv 文件。请先切换到 download 模式或手动放置 CSV。")
                input("按回车键退出...")
                return 1
            csv_path = latest
            print(f"使用最新文件: {csv_path.name}")
        # 二次从最终 csv_path 解析并覆盖配置，确保目录命名与传参一致
        try:
            m2 = re.search(r"ohlcv_(?P<inst>[^_]+)_(?P<tf>[^_]+)_(?P<start>\d{4}-\d{2}-\d{2})_(?P<end>\d{4}-\d{2}-\d{2})\.csv$", csv_path.name)
            if m2:
                CONFIG["inst"] = m2.group("inst")
                CONFIG["timeframe"] = m2.group("tf")
                CONFIG["start"] = m2.group("start")
                CONFIG["end"] = m2.group("end")
        except Exception:
            pass

    # 覆盖 allow_short（若通过命令行传入）
    try:
        # argparse 返回对象在 main 函数内，若不存在则忽略
        if 'args' in locals() and getattr(args, 'allow_short', None) is not None:
            CONFIG["allow_short"] = int(getattr(args, 'allow_short'))
    except Exception:
        pass

    # 2) 回测阶段
    print("\n===== 回测阶段 =====")
    backtest_inst = CONFIG["inst"]
    backtest_timeframe = CONFIG["timeframe"].replace("m", "min")

    backtest_cmd = [
        "py", "-u", str(PROJECT_DIR / "mvp" / "backtest" / "ma_backtest.py"),
        "--source", "csv",
        "--csv", str(csv_path),
        "--inst", backtest_inst,
        "--start", CONFIG["start"],
        "--end", CONFIG["end"],
        "--timeframe", backtest_timeframe,
        "--leverage", str(CONFIG["leverage"]),
        "--allow_short", str(CONFIG["allow_short"]),
        "--plot", "1",
        # 传递仓位与风险参数（与 ma_backtest CLI 对齐）
        "--base_equity_usd", str(CONFIG["base_equity_usd"]),
        "--single_order_max_pct_equity", str(CONFIG["single_order_max_pct_equity"]),
        "--max_position_usd", str(CONFIG["max_position_usd"]),
        "--risk_percent", str(CONFIG["risk_percent"]),
        # 传递确认与冷却参数
        "--confirm_min", str(CONFIG["confirm_min"]),
        "--cooldown_sec", str(CONFIG["cooldown_sec"]),
        "--sell_cooldown_sec", str(CONFIG["sell_cooldown_sec"]),
        # 金字塔加仓策略开关
        "--pyramid_on_buy", str(CONFIG["pyramid_on_buy"]),
        "--pyramid_need_consecutive", str(CONFIG["pyramid_need_consecutive"]),
    ]

    code = run_cmd(backtest_cmd, env=env)
    if code != 0:
        print("警告：回测执行返回非零退出码，继续检查产物...")

    # 3) 产物检查与摘要
    print("\n===== 回测产物 =====")
    # 产物目录：与回测模块保持一致：data/start_YYYYMMDD_end_YYYYMMDD
    try:
        start_str = datetime.fromisoformat(CONFIG["start"].replace("Z", "")).strftime("%Y%m%d")
    except Exception:
        m = re.search(r"(\d{4})[^0-9]?(\d{2})[^0-9]?(\d{2})", CONFIG["start"]) ; start_str = "".join(m.groups()) if m else "start"
    try:
        end_str = datetime.fromisoformat(CONFIG["end"].replace("Z", "")).strftime("%Y%m%d")
    except Exception:
        m = re.search(r"(\d{4})[^0-9]?(\d{2})[^0-9]?(\d{2})", CONFIG["end"]) ; end_str = "".join(m.groups()) if m else "end"
    tf_str = CONFIG["timeframe"].replace("m", "min")
    # 在目录名中加入 inst（使用完整合约名，保留 -SWAP，去除斜杠）
    inst_label = CONFIG["inst"].upper().replace("/", "-")
    out_dir = data_dir / f"start_{start_str}_end_{end_str}_tf_{tf_str}_inst_{inst_label}"

    products = [
        out_dir / "backtest_multi_indicator.csv",
        out_dir / "backtest_multi_indicator.svg",
        out_dir / "backtest_summary.json",
    ]

    found = []
    for p in products:
        if p.exists():
            found.append(p)
            print(f"[✓] {p}")
        else:
            print(f"[✗] {p}")

    if (out_dir / "backtest_summary.json").exists():
        try:
            summary = json.loads((out_dir / "backtest_summary.json").read_text(encoding="utf-8"))
            m = summary.get("metrics", {})
            ctx = summary.get("context", {})
            print("\n===== 回测结果摘要 =====")
            print(f"夏普比率: {m.get('sharpe')}")
            print(f"胜率: {round((m.get('winrate') or 0) * 100, 2)}%")
            print(f"最大回撤: {round((m.get('max_drawdown') or 0) * 100, 2)}%")
            print(f"数据条数: {ctx.get('bars')}")
        except Exception as e:
            print("读取回测汇总失败：", e)

    if not found:
        print("\n回测失败：未生成任何产物。")
        input("按回车键退出...")
        return 1

    print(f"\n回测完成！产物已保存到 {out_dir} 目录。")
    input("按回车键退出...")
    return 0


if __name__ == "__main__":
    sys.exit(main())