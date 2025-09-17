#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
灰度上实盘部署与运行检查脚本（Windows 可用，使用 py 命令执行）

功能概述：
- 上线前自检（OKX API 健康、数据延迟、回测阈值）
- 执行模式切换（mock/real/paused），优先使用 HTTP 管理接口，其次写控制文件
- 可作为“灰度实盘”操作模板，保障用 0.5%-1% 资金先小流量上线

使用示例：
1) 仅做上线前自检（建议在启动主程序前执行）
   py scripts/gradual_deploy.py preflight --http-admin-url http://127.0.0.1:8001 \
      --backtest-summary data/backtest_summary.json --min-sharpe 0.2 --min-winrate 0.5 --max-drawdown 0.05

2) 切换到 mock（灰度起步阶段）
   py scripts/gradual_deploy.py switch --to mock --http-admin-url http://127.0.0.1:8001

3) 切换到 real（仅在完成自检、灰度演练后）
   py scripts/gradual_deploy.py switch --to real --http-admin-url http://127.0.0.1:8001

4) 紧急回退（暂停）
   py scripts/gradual_deploy.py pause --http-admin-url http://127.0.0.1:8001

环境变量（可选）：
- EXEC_CONTROL_FILE：控制文件路径（默认：data/control/mode.json）
- AUDIT_TO_FILE / AUDIT_TO_DB：审计写入开关（脚本不直接修改，但会在自检中提示）
- DATABASE_URL：数据库连接串（用于数据延迟检测）

注意：
- 若 HTTP 管理接口未开启（EXEC_HTTP_ADMIN_ENABLED=false），脚本将回退为写控制文件。
- 脚本仅执行“检查与切换模式”，不负责拉起系统主进程。请在另一个窗口启动主程序（main.py）。
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Optional

try:
    import requests  # 依赖 requests 库
except Exception:  # 兜底提示
    print("[WARN] 未安装 requests，请先执行: py -m pip install requests", file=sys.stderr)
    raise

# 尝试导入项目内模块（可选，不存在时降级）
try:
    from utils.db import TimescaleDB
except Exception:
    TimescaleDB = None  # type: ignore

OKX_TIME_URL = "https://www.okx.com/api/v5/public/time"


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def okx_api_health(timeout=3.0) -> dict:
    """OKX API 健康检查：获取服务器时间
    返回：{"ok": bool, "latency_ms": float, "error": Optional[str]}
    """
    t0 = time.perf_counter()
    try:
        resp = requests.get(OKX_TIME_URL, timeout=timeout)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        if resp.status_code == 200:
            return {"ok": True, "latency_ms": latency_ms, "error": None}
        return {"ok": False, "latency_ms": latency_ms, "error": f"http {resp.status_code}"}
    except Exception as e:
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return {"ok": False, "latency_ms": latency_ms, "error": str(e)}


def data_latency_check(threshold_seconds: float = 5.0) -> dict:
    """数据延迟检查：读取 trades 表最近成交时间，计算与当前时间差
    需配置 DATABASE_URL 且 TimescaleDB 可用。

    返回：{"ok": bool, "latency_sec": Optional[float], "error": Optional[str]}
    """
    if TimescaleDB is None:
        return {"ok": False, "latency_sec": None, "error": "TimescaleDB 模块不可用"}
    try:
        db = TimescaleDB()
        db.connect()
        # 取最近 30 秒数据，找最大时间戳
        rows = db.fetch_recent_trades(window_seconds=30)
        if not rows:
            return {"ok": False, "latency_sec": None, "error": "最近 30 秒无成交数据，请确认采集已启动"}
        last_ts = max(r["ts"] for r in rows if "ts" in r)
        if not isinstance(last_ts, datetime):
            return {"ok": False, "latency_sec": None, "error": "数据库返回的 ts 类型异常"}
        latency = (_now_utc() - last_ts).total_seconds()
        return {"ok": latency <= threshold_seconds, "latency_sec": latency, "error": None}
    except Exception as e:
        return {"ok": False, "latency_sec": None, "error": str(e)}


def backtest_threshold_check(summary_path: str,
                             min_sharpe: float,
                             min_winrate: float,
                             max_drawdown: float) -> dict:
    """回测阈值检查：读取 JSON 汇总文件，校验关键指标
    约定 JSON 结构之一：
      {
        "metrics": {"sharpe": 0.3, "winrate": 0.55, "max_drawdown": 0.04},
        ...
      }
    返回：{"ok": bool, "metrics": dict, "error": Optional[str]}
    """
    if not os.path.exists(summary_path):
        return {"ok": False, "metrics": {}, "error": f"未找到文件: {summary_path}"}
    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            js = json.load(f)
        metrics = js.get("metrics", js)  # 兼容直接在根上的键
        sharpe = float(metrics.get("sharpe", 0.0))
        winrate = float(metrics.get("winrate", 0.0))
        mdd = float(metrics.get("max_drawdown", 1.0))
        ok = (sharpe >= min_sharpe) and (winrate >= min_winrate) and (mdd <= max_drawdown)
        return {"ok": ok, "metrics": {"sharpe": sharpe, "winrate": winrate, "max_drawdown": mdd}, "error": None}
    except Exception as e:
        return {"ok": False, "metrics": {}, "error": str(e)}


def _write_control_file(mode: str, control_file: str) -> None:
    os.makedirs(os.path.dirname(control_file), exist_ok=True)
    with open(control_file, "w", encoding="utf-8") as f:
        json.dump({"mode": mode, "message": f"switch by gradual_deploy at {_now_utc().isoformat()}"}, f, ensure_ascii=False)


def _switch_by_http(http_admin_url: str, mode: str, timeout=3.0) -> bool:
    try:
        status = requests.get(f"{http_admin_url.rstrip('/')}/status", timeout=timeout)
        if status.status_code != 200:
            print(f"[WARN] HTTP 管理接口 /status 非 200：{status.status_code}")
        resp = requests.post(f"{http_admin_url.rstrip('/')}/mode",
                             json={"mode": mode}, timeout=timeout)
        if resp.status_code == 200:
            print(f"[OK] 已通过 HTTP 管理接口切换到 {mode}")
            return True
        print(f"[WARN] HTTP 管理接口 /mode 切换失败：{resp.status_code} {resp.text}")
        return False
    except Exception as e:
        print(f"[WARN] HTTP 管理接口不可达：{e}")
        return False


def preflight(args) -> int:
    print("==== [1/3] 检查 OKX API 健康 ====")
    h = okx_api_health()
    print(f"OKX API: ok={h['ok']} latency_ms={h['latency_ms']:.1f} err={h['error']}")
    if not h["ok"]:
        print("[FAIL] OKX API 健康检查未通过")
        return 2

    if not args.skip_db_check:
        print("\n==== [2/3] 检查数据延迟（trades） ====")
        d = data_latency_check(args.data_latency_threshold)
        print(f"Data Latency: ok={d['ok']} latency_sec={d['latency_sec']} err={d['error']}")
        if not d["ok"]:
            print("[FAIL] 数据延迟检查未通过，请确认采集已启动且数据库连通")
            return 3
    else:
        print("\n[SKIP] 已跳过数据延迟检查 (--skip-db-check)")

    if not args.skip_backtest_check:
        print("\n==== [3/3] 检查回测阈值 ====")
        b = backtest_threshold_check(args.backtest_summary,
                                     args.min_sharpe, args.min_winrate, args.max_drawdown)
        print(f"Backtest: ok={b['ok']} metrics={b.get('metrics')} err={b.get('error')}")
        if not b["ok"]:
            print("[FAIL] 回测阈值检查未通过，请先完成回测并保存汇总 JSON")
            return 4
    else:
        print("\n[SKIP] 已跳过回测阈值检查 (--skip-backtest-check)")

    print("\n[OK] 上线前自检全部通过")
    return 0


def switch_mode(args) -> int:
    mode = args.to
    assert mode in ("mock", "real", "paused"), "to 必须是 mock|real|paused"
    http_admin_url = args.http_admin_url
    control_file = args.control_file or os.environ.get("EXEC_CONTROL_FILE", os.path.join("data", "control", "mode.json"))

    if http_admin_url:
        ok = _switch_by_http(http_admin_url, mode)
        if ok:
            return 0
        print("[INFO] 回退到控制文件方式切换模式...")
    _write_control_file(mode, control_file)
    print(f"[OK] 已写控制文件切换到 {mode} -> {control_file}")
    return 0


def pause_mode(args) -> int:
    args.to = "paused"
    return switch_mode(args)


def main():
    parser = argparse.ArgumentParser(description="灰度上实盘部署与运行检查脚本")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_pre = sub.add_parser("preflight", help="上线前自检：OKX API、数据延迟、回测阈值")
    p_pre.add_argument("--http-admin-url", type=str, default=os.environ.get("EXEC_HTTP_ADMIN_URL"), help="HTTP 管理接口根地址，例如 http://127.0.0.1:8001")
    p_pre.add_argument("--backtest-summary", type=str, default=os.path.join("data", "backtest_summary.json"))
    # 注意：回测模块已将产物输出到 data/start_YYYYMMDD_end_YYYYMMDD 子目录
    # 建议在调用时通过 --backtest-summary 显式传入该子目录下的 backtest_summary.json 完整路径
    p_pre.add_argument("--min-sharpe", type=float, default=0.2)
    p_pre.add_argument("--min-winrate", type=float, default=0.5)
    p_pre.add_argument("--max-drawdown", type=float, default=0.05)
    p_pre.add_argument("--data-latency-threshold", type=float, default=5.0)
    p_pre.add_argument("--skip-backtest-check", action="store_true")
    p_pre.add_argument("--skip-db-check", action="store_true")
    p_pre.add_argument("--control-file", type=str, default=os.environ.get("EXEC_CONTROL_FILE", os.path.join("data", "control", "mode.json")))

    p_sw = sub.add_parser("switch", help="切换模式：mock/real/paused")
    p_sw.add_argument("--to", type=str, required=True, choices=["mock", "real", "paused"])
    p_sw.add_argument("--http-admin-url", type=str, default=os.environ.get("EXEC_HTTP_ADMIN_URL"))
    p_sw.add_argument("--control-file", type=str, default=os.environ.get("EXEC_CONTROL_FILE", os.path.join("data", "control", "mode.json")))

    p_pause = sub.add_parser("pause", help="紧急暂停（等价切到 paused）")
    p_pause.add_argument("--http-admin-url", type=str, default=os.environ.get("EXEC_HTTP_ADMIN_URL"))
    p_pause.add_argument("--control-file", type=str, default=os.environ.get("EXEC_CONTROL_FILE", os.path.join("data", "control", "mode.json")))

    args = parser.parse_args()

    if args.cmd == "preflight":
        rc = preflight(args)
        sys.exit(rc)
    elif args.cmd == "switch":
        rc = switch_mode(args)
        sys.exit(rc)
    elif args.cmd == "pause":
        rc = pause_mode(args)
        sys.exit(rc)


if __name__ == "__main__":
    main()