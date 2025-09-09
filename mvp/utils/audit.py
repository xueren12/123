# -*- coding: utf-8 -*-
"""
结构化审计与日志模块（JSON 行协议，支持文件与PostgreSQL双写）。

设计目标：
- 统一事件格式，覆盖：策略信号、LLM 调用、下单/撤单、风控触发（含熔断）。
- 可通过环境变量开关写文件/写数据库。
- 写文件采用按天滚动的 .jsonl（每行一条 JSON），线程安全。
- 写数据库使用 utils.db.TimescaleDB 的 insert_audit_log 接口（懒连接）。

使用示例：
    from utils.config import AppConfig
    from utils.audit import AuditLogger

    cfg = AppConfig()
    audit = AuditLogger(cfg)
    ctx_id = audit.new_ctx_id()
    audit.log(
        event_type="strategy_signal", ctx_id=ctx_id, module="strategy",
        inst_id="BTC-USDT", action="compute",
        request={"inputs": {"depth": "..."}}, response={"signal": "BUY"},
        status="ok",
        extra={"threshold": 0.02}
    )

事件字段规范（JSON）：
- ts：字符串，UTC ISO8601 时间戳（含时区）
- event_type：字符串，事件类型（strategy_signal|llm_request|llm_result|order_intent|order_result|order_cancel|order_cancel_result|risk_event|risk_block）
- ctx_id：字符串，单次处理上下文 ID（跨策略->AI->下单贯穿）
- module：字符串，来源模块（strategy|ai|executor|okx_rest|risk|orchestrator）
- inst_id：字符串，可选，交易品种（如 BTC-USDT）
- action：字符串，可选，具体动作名称（如 place_order/cancel/get_status/compute 等）
- request：对象，可选，请求/输入参数（注意脱敏）
- response：对象，可选，响应/输出（注意脱敏）
- status：字符串，可选，ok|error|hold|blocked 等
- err：字符串，可选，错误信息
- latency_ms：数字，可选，耗时（毫秒）
- extra：对象，可选，扩展信息（如风控详情、模型名等）
"""
from __future__ import annotations

import json
import os
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from loguru import logger

from .config import AppConfig
from .db import TimescaleDB


def _now_iso() -> str:
    """返回当前 UTC 时间（ISO8601）。"""
    return datetime.now(timezone.utc).isoformat()


class AuditLogger:
    """结构化审计日志记录器。

    - 支持文件与数据库双写（可分别开关）。
    - 文件：按天滚动到 {dir}/{prefix}_YYYYMMDD.jsonl
    - 数据库：写入 audit_logs 表（lazy connect）。
    """

    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg
        self.enabled = bool(getattr(self.cfg, "audit", None) and self.cfg.audit.enabled)
        self.to_file = self.enabled and self.cfg.audit.to_file
        self.to_db = self.enabled and self.cfg.audit.to_db
        self.dir = (self.cfg.audit.dir if self.enabled else os.path.join("data", "audit"))
        self.prefix = (self.cfg.audit.file_prefix if self.enabled else "audit")
        # 目录准备
        try:
            os.makedirs(self.dir, exist_ok=True)
        except Exception as e:
            logger.warning("创建审计目录失败：{}", e)
        # 文件写锁
        self._lock = threading.Lock()
        # 数据库连接（懒加载）
        self._db: Optional[TimescaleDB] = None

    # ============ 上下文 ID ============
    @staticmethod
    def new_ctx_id() -> str:
        return uuid.uuid4().hex

    # ============ 主入口 ============
    def log(
        self,
        *,
        event_type: str,
        ctx_id: Optional[str],
        module: str,
        inst_id: Optional[str] = None,
        action: Optional[str] = None,
        request: Optional[Dict[str, Any]] = None,
        response: Optional[Dict[str, Any]] = None,
        status: Optional[str] = None,
        err: Optional[str] = None,
        latency_ms: Optional[float] = None,
        extra: Optional[Dict[str, Any]] = None,
        ts: Optional[str] = None,
    ) -> None:
        """记录一条审计事件。

        所有字段将按规范写入文件与数据库。写失败不会中断主流程，只打印 debug 级别日志。
        """
        if not self.enabled:
            return
        payload: Dict[str, Any] = {
            "ts": ts or _now_iso(),
            "event_type": event_type,
            "ctx_id": ctx_id,
            "module": module,
            "inst_id": inst_id,
            "action": action,
            "request": request,
            "response": response,
            "status": status,
            "err": err,
            "latency_ms": float(latency_ms) if latency_ms is not None else None,
            "extra": extra,
        }
        # 写文件
        if self.to_file:
            try:
                self._write_file(payload)
            except Exception as e:
                logger.debug("写入审计文件失败（忽略）：{}", e)
        # 写数据库
        if self.to_db:
            try:
                self._write_db(payload)
            except Exception as e:
                logger.debug("写入审计数据库失败（忽略）：{}", e)

    # ============ 文件写入 ============
    def _file_path_today(self) -> str:
        day = datetime.now(timezone.utc).strftime("%Y%m%d")
        return os.path.join(self.dir, f"{self.prefix}_{day}.jsonl")

    def _write_file(self, obj: Dict[str, Any]) -> None:
        path = self._file_path_today()
        line = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
        with self._lock:
            with open(path, "a", encoding="utf-8") as f:
                f.write(line + "\n")

    # ============ 数据库写入 ============
    def _write_db(self, obj: Dict[str, Any]) -> None:
        if self._db is None:
            self._db = TimescaleDB()
            self._db.connect()
        # 直接传入对象，DB 层负责 JSONB 转换
        self._db.insert_audit_log(obj)


__all__ = ["AuditLogger"]