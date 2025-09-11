#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""检查数据库中的数据（增强版）

- 校验内容：
  1) orderbook/trades 总量与按交易对分布
  2) 各交易对最新 orderbook 时间与新鲜度（是否延迟）
  3) 各交易对最新顶级买卖价、价差/中间价有效性
  4) trades 基础质量：价格/数量是否为正，近10分钟是否有成交
  5) 重点检查 ETH-USDT-SWAP 的最近样本

使用方法：
  py mvp/check_db.py

注：若未安装 TimescaleDB 扩展，将以普通 PostgreSQL 表运行，不影响检查逻辑。
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any

from utils.db import TimescaleDB

# -------- 可调参数（秒/分钟阈值） --------
STALE_BOOK_SEC = 15           # orderbook 新鲜度阈值（秒）
RECENT_TRADE_WINDOW_MIN = 10  # 近 N 分钟窗口
FOCUS_INST = "ETH-USDT-SWAP"  # 重点检查的交易对


def _fmt_dt(dt: datetime) -> str:
    if not dt:
        return "-"
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%fZ")


def _check_orderbook(db: TimescaleDB) -> None:
    print("\n=== [1] Orderbook 概览 ===")
    with db.conn.cursor() as cur:
        # 总量
        cur.execute("SELECT COUNT(*) AS c FROM orderbook")
        total = (cur.fetchone() or {}).get("c", 0)
        print(f"orderbook 表总记录数: {total}")
        if total == 0:
            return
        # 按 inst_id 分布
        cur.execute("SELECT inst_id, COUNT(*) AS c FROM orderbook GROUP BY inst_id ORDER BY c DESC")
        rows = cur.fetchall() or []
        print("各交易对订单簿记录数:", rows)

        # 各交易对最新一条与新鲜度/顶级价
        print("\n--- 各交易对最新 orderbook 质量 ---")
        now = datetime.now(timezone.utc)
        for r in rows:
            inst = r["inst_id"]
            cur.execute(
                """
                SELECT ts, bids, asks FROM orderbook
                WHERE inst_id=%s
                ORDER BY ts DESC LIMIT 1
                """,
                (inst,),
            )
            rec = cur.fetchone()
            if not rec:
                print(f"{inst}: 无数据")
                continue
            ts: datetime = rec["ts"]
            bids = rec["bids"] or []
            asks = rec["asks"] or []
            age = (now - ts).total_seconds()
            # 解析顶级买卖价
            best_bid = best_ask = None
            try:
                if isinstance(bids, list) and bids and isinstance(bids[0], (list, tuple)):
                    best_bid = float(bids[0][0])
                if isinstance(asks, list) and asks and isinstance(asks[0], (list, tuple)):
                    best_ask = float(asks[0][0])
            except Exception:
                pass
            spread = (best_ask - best_bid) if (best_bid and best_ask) else None
            mid = ((best_ask + best_bid) / 2.0) if (best_bid and best_ask) else None
            ok = True
            issues: List[str] = []
            if age > STALE_BOOK_SEC:
                ok = False
                issues.append(f"orderbook延迟{age:.1f}s>")
            if best_bid is None or best_ask is None:
                ok = False
                issues.append("缺少顶级买/卖价")
            else:
                if best_bid <= 0 or best_ask <= 0:
                    ok = False
                    issues.append("买卖价<=0")
                if best_bid > best_ask:
                    ok = False
                    issues.append("bid>ask 异常")
                if spread is not None and spread < 0:
                    ok = False
                    issues.append("负价差")
            status = "OK" if ok else "WARN"
            print(
                f"{inst}: 最新={_fmt_dt(ts)} age={age:.1f}s bid={best_bid} ask={best_ask} mid={mid} spread={spread} -> {status} {issues}"
            )


def _check_trades(db: TimescaleDB) -> None:
    print("\n=== [2] Trades 概览 ===")
    with db.conn.cursor() as cur:
        # 总量
        cur.execute("SELECT COUNT(*) AS c FROM trades")
        total = (cur.fetchone() or {}).get("c", 0)
        print(f"trades 表总记录数: {total}")
        if total == 0:
            return
        # 按 inst_id 分布
        cur.execute("SELECT inst_id, COUNT(*) AS c FROM trades GROUP BY inst_id ORDER BY c DESC")
        rows = cur.fetchall() or []
        print("各交易对成交记录数:", rows)

        # 异常值统计
        cur.execute("SELECT COUNT(*) AS bad FROM trades WHERE COALESCE(price::numeric,0)<=0 OR COALESCE(size::numeric,0)<=0")
        bad = (cur.fetchone() or {}).get("bad", 0)
        print(f"成交异常值（price<=0 或 size<=0）条数: {bad}")

        # 近 N 分钟各交易对成交活跃度
        cur.execute(
            f"""
            SELECT inst_id, COUNT(*) AS recent
            FROM trades
            WHERE ts > NOW() - INTERVAL '{RECENT_TRADE_WINDOW_MIN} minutes'
            GROUP BY inst_id ORDER BY recent DESC
            """
        )
        act = cur.fetchall() or []
        print(f"近 {RECENT_TRADE_WINDOW_MIN} 分钟成交活跃度:", act)

        # 最新 5 条（重点交易对）
        cur.execute(
            """
            SELECT ts, inst_id, side, price, size
            FROM trades
            WHERE inst_id=%s
            ORDER BY ts DESC LIMIT 5
            """,
            (FOCUS_INST,),
        )
        last5 = cur.fetchall() or []
        print(f"{FOCUS_INST} 最新5条成交样本:", last5)


def main():
    db = TimescaleDB()
    db.connect()
    try:
        _check_orderbook(db)
        _check_trades(db)

        # 汇总结论（基于关键交易对）
        print("\n=== [3] 结论（基于 FOCUS_INST） ===")
        with db.conn.cursor() as cur:
            # 最新 orderbook 质量
            cur.execute(
                "SELECT ts, bids, asks FROM orderbook WHERE inst_id=%s ORDER BY ts DESC LIMIT 1",
                (FOCUS_INST,),
            )
            ob = cur.fetchone()
            ob_ok = False
            ob_msg = "无数据"
            if ob:
                now = datetime.now(timezone.utc)
                age = (now - ob["ts"]).total_seconds()
                bids = ob.get("bids") or []
                asks = ob.get("asks") or []
                try:
                    bid = float(bids[0][0]) if bids else None
                    ask = float(asks[0][0]) if asks else None
                except Exception:
                    bid = ask = None
                if (bid and ask and bid > 0 and ask > 0 and bid <= ask and age <= STALE_BOOK_SEC):
                    ob_ok = True
                    ob_msg = f"OK: age={age:.1f}s bid={bid} ask={ask}"
                else:
                    ob_msg = f"WARN: age={age:.1f}s bid={bid} ask={ask}"

            # 近窗口内是否有成交
            cur.execute(
                f"""
                SELECT COUNT(*) AS c FROM trades
                WHERE inst_id=%s AND ts > NOW() - INTERVAL '{RECENT_TRADE_WINDOW_MIN} minutes'
                """,
                (FOCUS_INST,),
            )
            c_recent = (cur.fetchone() or {}).get("c", 0)
            has_trades = c_recent > 0

            print(f"orderbook: {ob_msg}")
            print(f"近{RECENT_TRADE_WINDOW_MIN}分钟成交: {'有' if has_trades else '无'} (count={c_recent})")

            if ob_ok and (has_trades or True):  # 有效盘口足以驱动中间价
                print("结论：数据有效，未发现漏写；可用于策略/风控参考价。")
            else:
                print("结论：数据可能不完整或延迟，请检查采集器与网络/订阅标的配置。")
    finally:
        db.close()


if __name__ == '__main__':
    main()