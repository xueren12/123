# -*- coding: utf-8 -*-
"""
检查数据库中新增表是否存在：strategy_positions 与 pnl_events。
用法：
  py mvp/scripts/check_tables.py
"""
import sys
import os
# 将 mvp 目录加入 sys.path，便于相对导入
_CUR = os.path.dirname(os.path.abspath(__file__))
_MVP_DIR = os.path.abspath(os.path.join(_CUR, ".."))
if _MVP_DIR not in sys.path:
    sys.path.insert(0, _MVP_DIR)

from utils.db import TimescaleDB


def main():
    db = TimescaleDB()
    db.connect()
    with db.conn.cursor() as cur:
        cur.execute("SELECT to_regclass('strategy_positions') AS t")
        sp = cur.fetchone()
        print("strategy_positions:", sp)
        cur.execute("SELECT to_regclass('pnl_events') AS t")
        pe = cur.fetchone()
        print("pnl_events:", pe)
    db.close()


if __name__ == "__main__":
    main()