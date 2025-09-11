# -*- coding: utf-8 -*-
"""
初始化数据库：确保目标数据库存在，并创建所需表（trades、orderbook、risk_events、audit_logs）。
读取环境变量：DB_HOST/DB_PORT/DB_NAME/DB_USER/DB_PASSWORD/DB_SSLMODE
用法：
  python scripts/init_db.py
"""
from __future__ import annotations

import os
import sys
import pathlib

# 将项目根目录加入 sys.path，确保可以导入 utils 等顶层包
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from psycopg import connect
from psycopg import sql
from psycopg.errors import OperationalError, InsufficientPrivilege

from utils.db import TimescaleDB, DBConfig


def ensure_database(cfg: DBConfig) -> None:
    try:
        # 连接到 postgres 系统库，检查/创建目标库
        with connect(host=cfg.host, port=cfg.port, dbname="postgres", user=cfg.user, password=cfg.password, sslmode=cfg.sslmode, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1 FROM pg_database WHERE datname=%s", (cfg.dbname,))
                row = cur.fetchone()
                if not row:
                    cur.execute(sql.SQL("CREATE DATABASE {}" ).format(sql.Identifier(cfg.dbname)))
                    print(f"已创建数据库: {cfg.dbname}")
                else:
                    print(f"数据库已存在: {cfg.dbname}")
    except InsufficientPrivilege:
        print("ERROR: 当前用户缺少创建数据库权限。请手动创建数据库或使用具备 CREATEDB 权限的账户。", file=sys.stderr)
        raise
    except OperationalError as e:
        print(f"ERROR: 无法连接到 PostgreSQL（用于创建数据库）：{e}", file=sys.stderr)
        raise


def main() -> None:
    cfg = DBConfig()
    print(f"准备在 {cfg.host}:{cfg.port} 初始化数据库与表（DB={cfg.dbname} 用户={cfg.user}）...")
    ensure_database(cfg)
    # 使用项目内置初始化逻辑创建表/扩展/索引
    db = TimescaleDB(cfg)
    db.connect()
    db.close()
    print("表结构初始化完成：trades、orderbook、risk_events、audit_logs。")


if __name__ == "__main__":
    main()