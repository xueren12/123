# -*- coding: utf-8 -*-
"""
OKX 历史K线离线拉取脚本（方案一的“离线拉取 → 保存 → 回测” 中的第一步）

功能：
- 通过 OKX 公共 REST 接口，按时间区间分页拉取指定交易对/粒度的 K 线；
- 保存为本地 CSV 文件，列为：ts, open, high, low, close, volume（UTC 时区）；
- 后续可由 backtest/ma_backtest.py 以 --source csv 方式读取并回测。

用法示例（PowerShell）：
  # 拉取 2025-01-01 至 2025-01-07 的 1m K线并保存
  py scripts/download_ohlcv_okx.py --inst BTC-USDT --timeframe 1min \
      --start "2025-01-01" --end "2025-01-07" \
      --out "data/ohlcv_BTC-USDT_1m_2025-01-01_2025-01-07.csv"

随后回测：
  py backtest/ma_backtest.py --source csv --csv "data/ohlcv_BTC-USDT_1m_2025-01-01_2025-01-07.csv" \
      --inst BTC-USDT --start "2025-01-01" --end "2025-01-07" --timeframe 1min --fast 10 --slow 30 --brk 20 --buffer 0.001 --fee_bps 2 --plot 1
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from typing import List

import os
# ===== 关键修复：确保作为脚本直接运行时也能找到 mvp 包 =====
# 说明：当用 “py mvp/scripts/download_ohlcv_okx.py” 直接运行时，sys.path[0] 是脚本所在的 mvp/scripts，
#       这会导致无法以 “from utils.xxx” 或 “from executor.xxx” 导入。
#       这里将 mvp 目录加入 sys.path，保证直接运行也能正常导入内部模块。
import sys
_CUR_DIR = os.path.dirname(os.path.abspath(__file__))
_MVP_DIR = os.path.abspath(os.path.join(_CUR_DIR, ".."))
if _MVP_DIR not in sys.path:
    sys.path.insert(0, _MVP_DIR)

import pandas as pd
from loguru import logger

from utils.config import AppConfig
from executor.okx_rest import OKXRESTClient


def _parse_dt(s: str) -> datetime:
    """解析日期/时间字符串为 UTC aware datetime。
    - 支持 ISO 格式（含Z或+00:00）；
    - 支持仅日期（按 UTC 当日 00:00:00）。
    """
    s = s.strip()
    try:
        # 兼容末尾 Z
        return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        # 仅日期
        from datetime import datetime as _dt
        return _dt.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def _normalize_bar(tf: str) -> str:
    """将通用 timeframe 文本（如 1min/5min/15min/1m/5m）转换为 OKX bar 参数（如 1m/5m/15m）。"""
    s = str(tf).strip().lower().replace("min", "m")
    # 兼容小时/天等（若外部传 1h/4h/1d 亦可直接透传）
    return s


def _build_default_out(inst: str, bar: str, start: datetime, end: datetime) -> str:
    """构造默认输出路径，位于 data/ 目录下。"""
    os.makedirs("data", exist_ok=True)
    s = start.strftime("%Y-%m-%d")
    e = end.strftime("%Y-%m-%d")
    safe_bar = bar.replace(":", "_")
    return os.path.join("data", f"ohlcv_{inst}_{safe_bar}_{s}_{e}.csv")


def main() -> None:
    # ========== 解析参数 ==========
    p = argparse.ArgumentParser(description="OKX 历史K线离线拉取脚本")
    p.add_argument("--inst", required=True, help="交易对，例如 BTC-USDT")
    p.add_argument("--timeframe", required=True, help="时间粒度，例如 1min/5min/15min/1h/1d")
    p.add_argument("--start", required=True, help="起始时间/日期，例如 2025-01-01 或 2025-01-01T00:00:00Z")
    p.add_argument("--end", required=True, help="结束时间/日期，例如 2025-01-07 或 2025-01-07T00:00:00Z")
    p.add_argument("--out", default=None, help="输出CSV文件路径（可选，不填则自动生成到 data/）")
    p.add_argument("--limit", type=int, default=100, help="每次请求的最大条数（默认100，OKX上限通常为100）")
    p.add_argument("--no-history", action="store_true", help="使用 /market/candles 而非 /market/history-candles")
    args = p.parse_args()

    inst = args.inst.strip()
    bar = _normalize_bar(args.timeframe)
    start = _parse_dt(args.start)
    end = _parse_dt(args.end)
    if end <= start:
        raise SystemExit("结束时间必须大于起始时间")

    out_path = args.out or _build_default_out(inst, bar, start, end)
    logger.info("输出路径：{}", out_path)

    # ========== 初始化 REST 客户端 ==========
    app_cfg = AppConfig()
    client = OKXRESTClient(app_cfg)

    # ========== 拉取数据 ==========
    ok, rows = client.fetch_ohlcv_range(
        inst_id=inst,
        bar=bar,
        start=start,
        end=end,
        limit_per_call=int(args.limit),
        use_history=(not args.no_history),
    )
    if not ok or not rows:
        raise SystemExit("拉取失败或无数据，请检查交易对/时间/粒度是否正确")

    # rows 形如：[ts, o, h, l, c, vol, ...]，大多为字符串；
    # 转换为 DataFrame 并按时间升序，去重；
    # 为便于回测，保存 ts 为 ISO 字符串（UTC），列名统一为 ts/open/high/low/close/volume。
    def _ms_to_iso(ms: int) -> str:
        return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).isoformat()

    recs: List[dict] = []
    for r in rows:
        try:
            ts_ms = int(r[0])
            o = float(r[1])
            h = float(r[2])
            l = float(r[3])
            c = float(r[4])
            vol = float(r[5]) if len(r) >= 6 and r[5] is not None else 0.0
        except Exception:
            # 若解析失败则跳过此条
            continue
        recs.append({
            "ts": _ms_to_iso(ts_ms),
            "open": o,
            "high": h,
            "low": l,
            "close": c,
            "volume": vol,
        })

    if not recs:
        raise SystemExit("经解析后无有效数据记录")

    df = pd.DataFrame.from_records(recs)
    df = df.sort_values("ts").drop_duplicates(subset=["ts"], keep="last").reset_index(drop=True)

    # 额外裁剪一次，确保严格在区间内（防止接口边界返回超出数据）
    s_iso = start.isoformat()
    e_iso = end.isoformat()
    df = df[(df["ts"] >= s_iso) & (df["ts"] <= e_iso)]

    # ========== 保存 ==========
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info("已保存 {} 行到 {}", len(df), out_path)


if __name__ == "__main__":
    main()