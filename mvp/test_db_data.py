#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单测试脚本：检查数据库中的实际数据时间范围
用法：
  py mvp/test_db_data.py [inst_id] [start(YYYY-MM-DD[THH:MM:SSZ])] [end]
示例：
  py mvp/test_db_data.py ETH-USDT-SWAP 2025-09-14 2025-09-15
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.db import TimescaleDB
from datetime import datetime, timezone
from typing import Optional

def _parse_dt(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    s = s.strip()
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        try:
            return datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except Exception:
            raise

def main():
    inst = sys.argv[1] if len(sys.argv) > 1 else "ETH-USDT-SWAP"
    start = _parse_dt(sys.argv[2]) if len(sys.argv) > 2 else datetime(2025, 9, 15, 0, 0, 0, tzinfo=timezone.utc)
    end = _parse_dt(sys.argv[3]) if len(sys.argv) > 3 else datetime(2025, 9, 15, 23, 59, 59, tzinfo=timezone.utc)

    db = TimescaleDB()

    # 查询 trades 表的时间范围
    print("=== 检查 trades 表时间范围 ===")
    try:
        result = db._query_df(
            "SELECT MIN(ts) as min_ts, MAX(ts) as max_ts, COUNT(*) as total FROM trades WHERE inst_id = %(inst_id)s",
            {"inst_id": inst}
        )
        print(f"{inst} trades: {result.iloc[0].to_dict()}")
    except Exception as e:
        print(f"查询 trades 失败: {e}")

    # 查询 orderbook 表的时间范围
    print("\n=== 检查 orderbook 表时间范围 ===")
    try:
        result = db._query_df(
            "SELECT MIN(ts) as min_ts, MAX(ts) as max_ts, COUNT(*) as total FROM orderbook WHERE inst_id = %(inst_id)s",
            {"inst_id": inst}
        )
        print(f"{inst} orderbook: {result.iloc[0].to_dict()}")
    except Exception as e:
        print(f"查询 orderbook 失败: {e}")

    # 测试具体时间窗口查询
    print("\n=== 测试时间窗口查询 ===")
    print(f"inst={inst} start={start} end={end}")
    try:
        trades = db.fetch_trades_window(start=start, end=end, inst_id=inst, limit=10)
        print(f"trades 查询结果: {len(trades)} 条记录")
        if not trades.empty:
            print(f"前5条: {trades.head().to_dict('records')}")
    except Exception as e:
        print(f"fetch_trades_window 失败: {e}")

    try:
        orderbook = db.fetch_orderbook_window(start=start, end=end, inst_id=inst, limit=10)
        print(f"orderbook 查询结果: {len(orderbook)} 条记录")
        if not orderbook.empty:
            print(f"前5条: {orderbook.head().to_dict('records')}")
    except Exception as e:
        print(f"fetch_orderbook_window 失败: {e}")

    db.close()

if __name__ == "__main__":
    main()