import os, sys
from datetime import datetime, timezone
from typing import List

try:
    import ccxt  # type: ignore
    import pandas as pd
except Exception as e:
    print(f"[FATAL] 依赖导入失败: {e}")
    sys.exit(2)

INST = os.environ.get("INST_ID", "ETH-USDT-SWAP")
TIMEFRAME = os.environ.get("TF", "5m")
START = os.environ.get("START_ISO", "2025-09-13T00:00:00+00:00")
END = os.environ.get("END_ISO", "2025-09-16T00:00:00+00:00")
OUT = os.environ.get("OUT_CSV") or "data/ohlcv_tmp.csv"

os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)

# instId -> ccxt symbol
if INST.endswith("-SWAP"):
    parts = INST.split("-")
    if len(parts) >= 2:
        base, quote = parts[0], parts[1]
        symbol = f"{base}/{quote}:USDT"
    else:
        symbol = INST.replace("-", "/")
else:
    symbol = INST.replace("-", "/")

# ===== 修复拼写错误并增强：允许指定交易所/代理/超时 =====
EXCHANGE_ID = os.environ.get("EXCHANGE_ID", "okx").lower()
TIMEOUT_MS = int(os.environ.get("TIMEOUT_MS", "30000"))
PROXY = os.environ.get("PROXY")  # 例如 http://127.0.0.1:7890 或 socks5h://127.0.0.1:1080

cfg = {"enableRateLimit": True, "timeout": TIMEOUT_MS}
if PROXY:
    cfg["proxies"] = {"http": PROXY, "https": PROXY}

if not hasattr(ccxt, EXCHANGE_ID):
    print(f"[FATAL] 无效的 EXCHANGE_ID: {EXCHANGE_ID}")
    sys.exit(2)
exchange = getattr(ccxt, EXCHANGE_ID)(cfg)

# OKX 合约场景默认设置 defaultType=swap，避免符号歧义
if exchange.id == "okx" and INST.endswith("-SWAP"):
    exchange.options = dict(getattr(exchange, "options", {}))
    exchange.options["defaultType"] = "swap"

# 带重试的 load_markets，以提升稳定性
_last_err = None
for i in range(5):
    try:
        exchange.load_markets()
        break
    except Exception as e:
        _last_err = e
        print(f"[WARN] load_markets 失败 第{ i+1 }次: {e}")
        import time as _t; _t.sleep(1 + 0.5 * i)
else:
    print(f"[FATAL] load_markets 连续失败: {_last_err}")
    sys.exit(5)

start = datetime.fromisoformat(START)
end = datetime.fromisoformat(END)
start_ms = int(start.timestamp() * 1000)
end_ms = int(end.timestamp() * 1000)

STEP_MS_MAP = {"1m":60000,"3m":180000,"5m":300000,"15m":900000,"30m":1800000,"1h":3600000,"4h":14400000,"1d":86400000}
step_ms = STEP_MS_MAP.get(TIMEFRAME, 300000)
limit = 300

rows: List[List] = []
since = start_ms
page = 0
while since <= end_ms:
    page += 1
    try:
        batch = exchange.fetch_ohlcv(symbol, timeframe=TIMEFRAME, since=since, limit=limit)
    except Exception as e:
        print(f"[WARN] fetch_ohlcv 失败 page={page} since={since}: {e}")
        break
    if not batch:
        print(f"[INFO] 空页，结束. page={page} since={since}")
        break
    for r in batch:
        ts = int(r[0])
        if ts < start_ms or ts > end_ms:
            continue
        rows.append([int(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5])])
    last_ts = int(batch[-1][0])
    since = last_ts + step_ms
    print(f"[PAGE] {page} batch={len(batch)} acc={len(rows)} next={since}")
    if len(batch) < limit:
        break

if not rows:
    print("[FATAL] 无数据，请检查时间/合约/粒度")
    sys.exit(3)

from datetime import datetime as _dt

def _ms_to_iso(ms:int)->str:
    return _dt.fromtimestamp(ms/1000.0, tz=timezone.utc).isoformat()

recs = [{
    "ts": _ms_to_iso(ts),
    "open": o,
    "high": h,
    "low": l,
    "close": c,
    "volume": v,
} for ts,o,h,l,c,v in rows]

import pandas as pd

df = pd.DataFrame.from_records(recs)
df.sort_values("ts", inplace=True)
df.drop_duplicates(subset=["ts"], keep="last", inplace=True)
# 严格裁剪窗口
s_iso = start.isoformat(); e_iso = end.isoformat()
df = df[(df["ts"] >= s_iso) & (df["ts"] <= e_iso)]
if df.empty:
    print("[FATAL] 裁剪后为空")
    sys.exit(4)

df.to_csv(OUT, index=False)
print(f"[OK] 保存 {len(df)} 行 -> {OUT}")
