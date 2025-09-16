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
    # 新增：选择拉取引擎（okx 原生 REST 或 ccxt 封装）
    p.add_argument("--engine", choices=["okx", "ccxt"], default="okx", help="数据拉取引擎：okx 使用官方 REST；ccxt 使用 ccxt.okx")
    # 新增：网络相关参数（针对 CCXT）
    p.add_argument("--proxy", default=None, help="HTTP(S) 代理，例如 http://127.0.0.1:7890")
    p.add_argument("--timeout", type=int, default=20000, help="请求超时（毫秒），默认 20000ms")
    p.add_argument("--max-retries", type=int, default=3, help="网络失败时的最大重试次数（仅 CCXT 分支）")
    # 新增：CCXT 交易所与符号映射控制
    p.add_argument("--ccxt-exchange", default="okx", help="CCXT 交易所 ID，例如 okx / binanceusdm / binance")
    p.add_argument("--ccxt-symbol", default=None, help="覆盖映射后的 CCXT 符号（可选），例如 ETH/USDT 或 ETH/USDT:USDT")
    args = p.parse_args()

    inst = args.inst.strip()
    bar = _normalize_bar(args.timeframe)
    start = _parse_dt(args.start)
    end = _parse_dt(args.end)
    if end <= start:
        raise SystemExit("结束时间必须大于起始时间")

    out_path = args.out or _build_default_out(inst, bar, start, end)
    logger.info("输出路径：{}", out_path)

    # 将下载过程写入文件日志，便于事后排查（旋转 500KB，保留 3 份）
    try:
        os.makedirs(os.path.join("data", "audit"), exist_ok=True)
        logger.add(
            os.path.join("data", "audit", "download.log"),
            rotation="500 KB",
            retention=3,
            enqueue=True,
            encoding="utf-8",
            level="INFO",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
        )
        logger.info("已启用文件日志：data/audit/download.log")
    except Exception as _e:
        logger.warning("启用文件日志失败：{}", _e)

    # ========== 分支：使用 CCXT 拉取 ==========
    if str(getattr(args, "engine", "okx")).lower() == "ccxt":
        # 仅在使用 ccxt 时再导入，避免未安装 ccxt 导致其他功能失败
        try:
            import ccxt  # type: ignore
        except Exception as e:
            raise SystemExit(f"未安装 ccxt，请先执行: python -m pip install ccxt；错误: {e}")

        # 构造 CCXT 交易所实例（公共数据，不需要 API Key）
        # 支持代理/超时配置；若为合约（-SWAP 结尾），根据交易所特性设置默认类型
        is_swap = inst.endswith("-SWAP")
        exch_cfg = {
            "enableRateLimit": True,
            "timeout": int(getattr(args, "timeout", 20000) or 20000),
        }
        proxy = getattr(args, "proxy", None)
        if proxy:
            # CCXT 支持 requests 的 proxies 透传
            exch_cfg["proxies"] = {"http": proxy, "https": proxy}
            logger.info("已设置 HTTP(S) 代理：{}", proxy)

        ex_id = str(getattr(args, "ccxt_exchange", "okx")).lower()
        if not hasattr(ccxt, ex_id):
            raise SystemExit(f"无效的 CCXT 交易所 ID：{ex_id}")
        exchange = getattr(ccxt, ex_id)(exch_cfg)
        try:
            # OKX 需要指定默认类型为 swap 以避免符号歧义
            if exchange.id == "okx" and is_swap:
                exchange.options = dict(getattr(exchange, "options", {}))
                exchange.options["defaultType"] = "swap"
            # Binance 家族：binanceusdm / binancecoinm / binance
            if exchange.id.startswith("binance"):
                # binanceusdm 默认就是 U 本位合约，无需额外设置；此处保留以便将来扩展
                exchange.options = dict(getattr(exchange, "options", {}))
        except Exception:
            pass

        # 载入市场，带重试与退避
        max_retries = max(0, int(getattr(args, "max_retries", 3) or 3))
        lm_ok = False
        last_err = None
        for i in range(max_retries):
            try:
                exchange.load_markets()
                lm_ok = True
                break
            except Exception as e:
                last_err = e
                logger.warning("load_markets 失败（第 {} 次）：{}", i + 1, e)
                try:
                    import time as _t
                    _t.sleep(1.0 + i)
                except Exception:
                    pass
        if not lm_ok:
            raise SystemExit(f"load_markets 失败，请检查网络或代理（可用 --proxy 设置）：{last_err}")

        # ===== 符号映射 =====
        # 1) 若用户提供 --ccxt-symbol，直接使用
        symbol = getattr(args, "ccxt_symbol", None)
        # 2) 尝试通过 markets 反查 id -> symbol
        if not symbol:
            try:
                for m in exchange.markets.values():
                    mid = str(m.get("id", ""))
                    msym = str(m.get("symbol", ""))
                    # 直接匹配 instId 或者去掉连字符后的对比
                    if mid == inst or mid.replace("-", "") == inst.replace("-", ""):
                        symbol = msym
                        break
                    # OKX 市场对象中常带 info.instId，可直接比对
                    info = m.get("info") or {}
                    if isinstance(info, dict) and str(info.get("instId", "")) == inst:
                        symbol = msym
                        break
            except Exception:
                symbol = None
        # 3) 按交易所特性进行退化映射
        if not symbol:
            if exchange.id == "okx":
                if is_swap:
                    parts = inst.split("-")
                    if len(parts) >= 2:
                        base, quote = parts[0], parts[1]
                        symbol = f"{base}/{quote}:USDT"
                    else:
                        symbol = inst.replace("-", "/")
            elif exchange.id.startswith("binance"):
                # Binance 家族：
                parts = inst.split("-")
                if len(parts) >= 2:
                    base, quote = parts[0], parts[1]
                    # USDT 本位合约（binanceusdm）需要带 :USDT 后缀；现货/其他沿用常规符号
                    if exchange.id == "binanceusdm" and is_swap:
                        symbol = f"{base}/{quote}:USDT"
                    else:
                        symbol = f"{base}/{quote}"
                else:
                    symbol = inst.replace("-", "/")
            else:
                # 其它交易所按通用规则退化
                symbol = inst.replace("-", "/")
        logger.info("CCXT 符号映射：inst={} -> exchange={} symbol={}", inst, exchange.id, symbol)

        # ccxt 的时间粒度使用如 "1m/5m/15m/1h/1d"
        bar_ccxt = bar
        limit = int(getattr(args, "limit", 100) or 100)
        start_ms = int(start.timestamp() * 1000)
        end_ms = int(end.timestamp() * 1000)

        _step_map = {
            "1m": 60_000, "3m": 180_000, "5m": 300_000, "15m": 900_000, "30m": 1_800_000,
            "1h": 3_600_000, "2h": 7_200_000, "4h": 14_400_000, "6h": 21_600_000, "12h": 43_200_000,
            "1d": 86_400_000,
        }
        step_ms = _step_map.get(bar_ccxt, 0)

        rows_raw: List[List] = []
        since_ms = start_ms
        page = 0
        while since_ms <= end_ms:
            page += 1
            got = None
            err = None
            for i in range(max_retries):
                try:
                    got = exchange.fetch_ohlcv(symbol, timeframe=bar_ccxt, since=since_ms, limit=limit)
                    err = None
                    break
                except Exception as e:
                    err = e
                    logger.warning("CCXT fetch_ohlcv 失败（第 {} 次）：since={} page={} err={}", i + 1, since_ms, page, e)
                    try:
                        import time as _t
                        _t.sleep(0.5 + 0.5 * i)
                    except Exception:
                        pass
            if err is not None:
                logger.warning("放弃本页：since={} page={}，错误={}（可调大 --max-retries 或设置 --proxy）", since_ms, page, err)
                break
            batch = got or []
            if not batch:
                logger.info("CCXT 返回空 batch，结束分页：page={} since_ms={}", page, since_ms)
                break
            for r in batch:
                ts = int(r[0])
                if ts < start_ms or ts > end_ms:
                    continue
                rows_raw.append([int(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5])])
            last_ts = int(batch[-1][0])
            if step_ms > 0:
                since_ms = last_ts + step_ms
            else:
                since_ms = last_ts + 1
            logger.info("[CCXT] 分页：page={} 累计={} next_since_ms={}", page, len(rows_raw), since_ms)
            if len(batch) < limit:
                break

        if not rows_raw:
            raise SystemExit("拉取失败或无数据，请检查交易对/时间/粒度是否正确 (ccxt)")

        def _ms_to_iso(ms: int) -> str:
            return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).isoformat()

        recs: List[dict] = []
        for r in rows_raw:
            ts_ms, o, h, l, c, vol = r
            recs.append({
                "ts": _ms_to_iso(int(ts_ms)),
                "open": float(o),
                "high": float(h),
                "low": float(l),
                "close": float(c),
                "volume": float(vol),
            })
        df = pd.DataFrame.from_records(recs)
        df = df.sort_values("ts").drop_duplicates(subset=["ts"], keep="last").reset_index(drop=True)
        s_iso = start.isoformat(); e_iso = end.isoformat()
        df = df[(df["ts"] >= s_iso) & (df["ts"] <= e_iso)]
        if df.empty:
            raise SystemExit("经裁剪后为空，请检查时间窗口是否正确 (ccxt)")
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        df.to_csv(out_path, index=False)
        logger.info("[CCXT] 已保存 {} 行到 {}", len(df), out_path)
        return

    # ========== 拉取数据（默认：官方 OKX REST） ==========
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