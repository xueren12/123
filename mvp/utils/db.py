"""
PostgreSQL（可选 TimescaleDB）数据库工具。
- 提供连接、trades 与 orderbook 表结构初始化
- 提供高吞吐的批量写入方法
- 支持通过环境变量进行配置（支持 .env）

环境变量：
  DB_HOST（默认：localhost）
  DB_PORT（默认：5432）
  DB_NAME（默认：crypto）
  DB_USER（默认：postgres）
  DB_PASSWORD（默认：postgres）
  DB_SSLMODE（默认：prefer）
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Iterable, List, Dict, Any, Optional, TYPE_CHECKING

from dotenv import load_dotenv
from loguru import logger
import psycopg
from psycopg.rows import dict_row
from psycopg.errors import InsufficientPrivilege, OperationalError
from psycopg.types.json import Json

if TYPE_CHECKING:
    import pandas as pd


load_dotenv()


@dataclass
class DBConfig:
    host: str = os.getenv("DB_HOST", "localhost")
    port: int = int(os.getenv("DB_PORT", "15432"))
    dbname: str = os.getenv("DB_NAME", "crypto")
    user: str = os.getenv("DB_USER", "postgres")
    password: str = os.getenv("DB_PASSWORD", "123456")
    sslmode: str = os.getenv("DB_SSLMODE", "prefer")

    def dsn(self) -> str:
        return (
            f"host={self.host} port={self.port} dbname={self.dbname} "
            f"user={self.user} password={self.password} sslmode={self.sslmode}"
        )


class TimescaleDB:
    def __init__(self, config: Optional[DBConfig] = None) -> None:
        self.config = config or DBConfig()
        self.conn: Optional[psycopg.Connection] = None

    def connect(self) -> None:
        logger.info("连接 PostgreSQL: {}:{} / db={} 用户={}",
                    self.config.host, self.config.port, self.config.dbname, self.config.user)
        self.conn = psycopg.connect(self.config.dsn(), row_factory=dict_row, autocommit=True)
        logger.success("PostgreSQL 连接成功")
        self._init_schema()

    def close(self) -> None:
        if self.conn is not None:
            logger.info("关闭 PostgreSQL 连接")
            self.conn.close()
            self.conn = None

    def _init_schema(self) -> None:
        assert self.conn is not None, "数据库未连接"
        with self.conn.cursor() as cur:
            # 尝试启用 TimescaleDB 扩展（若权限不足或扩展不存在则忽略）
            try:
                cur.execute("CREATE EXTENSION IF NOT EXISTS timescaledb;")
                logger.info("TimescaleDB 扩展已启用或已存在")
            except Exception as e:
                logger.warning("无法自动创建 TimescaleDB 扩展（将使用普通 PostgreSQL 表）：{}", e)

            # 创建 trades 表
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS trades (
                    ts TIMESTAMPTZ NOT NULL,
                    inst_id TEXT NOT NULL,
                    side TEXT,
                    price NUMERIC,
                    size NUMERIC,
                    trade_id TEXT,
                    source TEXT DEFAULT 'okx',
                    received_at TIMESTAMPTZ DEFAULT NOW()
                );
                """
            )
            # 创建 orderbook 表（将前五档 bids/asks 以 JSONB 存储）
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS orderbook (
                    ts TIMESTAMPTZ NOT NULL,
                    inst_id TEXT NOT NULL,
                    bids JSONB NOT NULL,
                    asks JSONB NOT NULL,
                    checksum INTEGER,
                    received_at TIMESTAMPTZ DEFAULT NOW()
                );
                """
            )

            # 新增：风险事件表（熔断/人工切换/恢复等事件）
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS risk_events (
                    ts TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    event_type TEXT NOT NULL,               -- circuit_break / manual_pause / manual_resume / http_change
                    mode_before TEXT,
                    mode_after TEXT,
                    reason TEXT,
                    details JSONB,
                    day_pnl_usd NUMERIC,
                    threshold_usd NUMERIC
                );
                """
            )

            # 新增：审计日志表（JSON 行协议）
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS audit_logs (
                    ts TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    event_type TEXT NOT NULL,               -- strategy_signal/llm_request/llm_result/order_intent/order_result/order_cancel/order_cancel_result/risk_event/risk_block
                    ctx_id TEXT,
                    module TEXT,
                    inst_id TEXT,
                    action TEXT,
                    request JSONB,
                    response JSONB,
                    status TEXT,
                    err TEXT,
                    latency_ms DOUBLE PRECISION,
                    extra JSONB
                );
                """
            )

            # 若可用，则转换为 hypertable
            try:
                cur.execute("SELECT create_hypertable('trades', 'ts', if_not_exists => TRUE);")
                cur.execute("SELECT create_hypertable('orderbook', 'ts', if_not_exists => TRUE);")
                # risk_events 与 audit_logs 事件频次较低，不强制转换为 hypertable
                logger.info("trades 与 orderbook 已转换为 hypertable")
            except Exception as e:
                # 如果没有启用 timescaledb，会失败；此时继续以普通表运行
                logger.warning("无法创建 hypertable（可能未启用 TimescaleDB）：{}", e)

            # 索引以加速查询
            cur.execute("CREATE INDEX IF NOT EXISTS idx_trades_inst_ts ON trades(inst_id, ts DESC);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_orderbook_inst_ts ON orderbook(inst_id, ts DESC);")
            # 新增索引：审计日志按时间与类型查询
            cur.execute("CREATE INDEX IF NOT EXISTS idx_audit_ts ON audit_logs(ts DESC);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_audit_event ON audit_logs(event_type);")
            logger.success("数据库表结构初始化完成")

    # ----------------------------
    # 写入接口
    # ----------------------------
    def insert_trades(self, rows: Iterable[Dict[str, Any]]) -> int:
        assert self.conn is not None, "数据库未连接"
        rows = list(rows)
        if not rows:
            return 0
        with self.conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO trades (ts, inst_id, side, price, size, trade_id, source)
                VALUES (%(ts)s, %(inst_id)s, %(side)s, %(price)s, %(size)s, %(trade_id)s, %(source)s)
                """,
                rows,
            )
        logger.debug("已插入 {} 条成交记录", len(rows))
        return len(rows)

    def insert_orderbook(self, row: Dict[str, Any]) -> int:
        assert self.conn is not None, "数据库未连接"
        # 确保 JSON 字段正确适配为 JSONB
        payload = dict(row)
        if "bids" in payload:
            payload["bids"] = Json(payload["bids"])  # list/dict -> JSONB
        if "asks" in payload:
            payload["asks"] = Json(payload["asks"])  # list/dict -> JSONB
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO orderbook (ts, inst_id, bids, asks, checksum)
                VALUES (%(ts)s, %(inst_id)s, %(bids)s, %(asks)s, %(checksum)s)
                """,
                payload,
            )
        logger.debug("已插入 1 条盘口快照：{} @ {}", payload.get("inst_id"), payload.get("ts"))
        return 1

    def insert_risk_event(self, row: Dict[str, Any]) -> int:
        """写入风控事件日志到 risk_events 表。
        参数示例：{
            'event_type': 'circuit_break', 'mode_before': 'real', 'mode_after': 'paused',
            'reason': 'daily_loss_breach', 'details': {...}, 'day_pnl_usd': -320.5, 'threshold_usd': 300
        }
        """
        assert self.conn is not None, "数据库未连接"
        payload = dict(row)
        if "details" in payload:
            payload["details"] = Json(payload["details"])  # JSONB
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO risk_events (ts, event_type, mode_before, mode_after, reason, details, day_pnl_usd, threshold_usd)
                VALUES (COALESCE(%(ts)s, NOW()), %(event_type)s, %(mode_before)s, %(mode_after)s, %(reason)s, %(details)s, %(day_pnl_usd)s, %(threshold_usd)s)
                """,
                payload,
            )
        logger.info("已记录风控事件：{} -> {} ({})", payload.get("mode_before"), payload.get("mode_after"), payload.get("reason"))
        return 1

    def insert_audit_log(self, row: Dict[str, Any]) -> int:
        """写入结构化审计日志到 audit_logs 表。
        参数字段参考 utils.audit.AuditLogger.log。
        """
        assert self.conn is not None, "数据库未连接"
        payload = dict(row)
        # 转 JSONB
        for k in ("request", "response", "extra"):
            if k in payload and payload[k] is not None:
                payload[k] = Json(payload[k])
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO audit_logs (
                    ts, event_type, ctx_id, module, inst_id, action, request, response, status, err, latency_ms, extra
                ) VALUES (
                    COALESCE(%(ts)s, NOW()), %(event_type)s, %(ctx_id)s, %(module)s, %(inst_id)s, %(action)s,
                    %(request)s, %(response)s, %(status)s, %(err)s, %(latency_ms)s, %(extra)s
                )
                """,
                payload,
            )
        logger.debug("已写审计事件：event_type={} module={} action={}", payload.get("event_type"), payload.get("module"), payload.get("action"))
        return 1

    @staticmethod
    def ms_to_utc(ms: int) -> datetime:
        return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)

    # ----------------------------
    # 查询工具（返回 Pandas DataFrame）
    # ----------------------------
    def _ensure_connected(self) -> None:
        """若未连接则按需建立连接。"""
        if self.conn is None:
            self.connect()

    @staticmethod
    def _to_utc(dt: datetime) -> datetime:
        """确保返回带时区信息的 UTC 时间。"""
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    def _query_df(self, sql: str, params: Dict[str, Any]) -> "pd.DataFrame":
        """执行 SELECT 并返回 Pandas DataFrame（延迟导入 pandas 避免硬依赖）。"""
        self._ensure_connected()
        assert self.conn is not None
        with self.conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
        try:
            import pandas as pd  # 延迟导入，避免模块导入期的硬依赖
        except Exception as e:
            raise RuntimeError("需要安装 pandas 才能返回 DataFrame，请先安装依赖。") from e
        return pd.DataFrame(rows)

    def fetch_trades_window(
        self,
        start: datetime,
        end: datetime,
        inst_id: Optional[str] = None,
        limit: Optional[int] = None,
        ascending: bool = True,
    ) -> "pd.DataFrame":
        """查询指定时间窗内的成交数据 [start, end]（UTC），可选过滤 inst_id。"""
        start_utc = self._to_utc(start)
        end_utc = self._to_utc(end)
        order = "ASC" if ascending else "DESC"
        base_sql = (
            "SELECT ts, inst_id, side, price::float8 AS price, size::float8 AS size, trade_id, source, received_at "
            "FROM trades WHERE ts BETWEEN %(start)s AND %(end)s"
        )
        if inst_id:
            base_sql += " AND inst_id = %(inst_id)s"
        base_sql += f" ORDER BY ts {order}"
        if limit:
            base_sql += " LIMIT %(limit)s"
        params: Dict[str, Any] = {"start": start_utc, "end": end_utc}
        if inst_id:
            params["inst_id"] = inst_id
        if limit:
            params["limit"] = int(limit)
        return self._query_df(base_sql, params)

    def fetch_orderbook_window(
        self,
        start: datetime,
        end: datetime,
        inst_id: Optional[str] = None,
        limit: Optional[int] = None,
        ascending: bool = True,
    ) -> "pd.DataFrame":
        """查询指定时间窗内的盘口快照 [start, end]（UTC），可选过滤 inst_id。"""
        start_utc = self._to_utc(start)
        end_utc = self._to_utc(end)
        order = "ASC" if ascending else "DESC"
        base_sql = (
            "SELECT ts, inst_id, bids, asks, checksum, received_at "
            "FROM orderbook WHERE ts BETWEEN %(start)s AND %(end)s"
        )
        if inst_id:
            base_sql += " AND inst_id = %(inst_id)s"
        base_sql += f" ORDER BY ts {order}"
        if limit:
            base_sql += " LIMIT %(limit)s"
        params: Dict[str, Any] = {"start": start_utc, "end": end_utc}
        if inst_id:
            params["inst_id"] = inst_id
        if limit:
            params["limit"] = int(limit)
        return self._query_df(base_sql, params)

    def fetch_recent_trades(
        self,
        seconds: int,
        inst_id: Optional[str] = None,
        limit: Optional[int] = None,
        ascending: bool = True,
    ) -> "pd.DataFrame":
        """获取最近 N 秒内的成交数据（相对当前 UTC 时间）。"""
        now_utc = datetime.now(timezone.utc)
        start = now_utc - timedelta(seconds=int(seconds))
        # 分析通常希望按时间升序返回
        return self.fetch_trades_window(start=start, end=now_utc, inst_id=inst_id, limit=limit, ascending=ascending)

    def fetch_recent_orderbook(
        self,
        seconds: int,
        inst_id: Optional[str] = None,
        limit: Optional[int] = None,
        ascending: bool = True,
    ) -> "pd.DataFrame":
        """获取最近 N 秒内的盘口快照（相对当前 UTC 时间）。"""
        now_utc = datetime.now(timezone.utc)
        start = now_utc - timedelta(seconds=int(seconds))
        return self.fetch_orderbook_window(start=start, end=now_utc, inst_id=inst_id, limit=limit, ascending=ascending)