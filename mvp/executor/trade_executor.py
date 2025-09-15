# -*- coding: utf-8 -*-
"""
交易执行模块
- 接收策略信号（统一格式：{ts,symbol,side,price,size,reason, meta}）
- 支持三种运行态：
  1) 模拟盘 mock：不调用真实交易所，仅在本地记录；
  2) 实盘 real：调用 OKX REST 下单接口；
  3) 暂停 paused：拒绝一切下单（仅日志记录）。
- 支持限价/市价/止损/止盈
- 交易日志写入本地 CSV（data/trade_log.csv），并在内存中维护最近订单状态。
- 新增：全局紧急停止开关与多层熔断（单笔/单日），可通过 文件/环境变量/HTTP 管理接口切换三态。

注：
- 该模块主面向现货交易（tdMode=cash），若要扩展至合约/杠杆，可在 extra 参数中透传 posSide、lever 等字段。
"""
from __future__ import annotations

import csv
import json
import os
import threading
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, date
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional, Callable

from loguru import logger

from utils.config import AppConfig
from utils.db import TimescaleDB
from executor.okx_rest import OKXRESTClient, OKXResponse
from utils.risk import MarketState, AccountState  # 关闭风控：不再导入 RiskManager/OrderIntent
from utils.audit import AuditLogger


# 统一信号结构（也可与策略约定直接用 dict）
@dataclass
class TradeSignal:
    ts: datetime
    symbol: str
    side: str  # buy/sell/close
    price: Optional[float]
    size: float
    reason: str = ""
    meta: Optional[Dict[str, Any]] = None  # 扩展信息（止盈止损参数等）


@dataclass
class ExecResult:
    ok: bool
    mode: str  # mock/real/paused
    signal: TradeSignal
    exchange_resp: Optional[Dict[str, Any]] = None
    err: Optional[str] = None
    order_id: Optional[str] = None


# --------------------
# 工具：JSON 截断（保留既有函数）
# --------------------
def json_trunc(obj: Any, max_len: int = 500) -> str:
    """将对象序列化为 JSON 字符串，并在超长时截断，避免日志过大。"""
    try:
        s = json.dumps(obj, ensure_ascii=False)
    except Exception:
        try:
            s = str(obj)
        except Exception:
            s = "<unserializable>"
    if len(s) > max_len:
        return s[: max_len - 3] + "..."
    return s


class _RiskState:
    """运行态与风险累计状态（线程内共享）。"""
    def __init__(self, init_mode: str) -> None:
        assert init_mode in ("mock", "real", "paused")
        self.mode: str = init_mode
        self.mode_changed_at: datetime = datetime.now(timezone.utc)
        self.day_key: str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        self.day_loss_usd: float = 0.0  # 单日累计亏损估计值（USD）
        self.lock = threading.Lock()


class CircuitBreaker:
    """多层熔断器与安全开关。
    功能：
    - 维护三态运行模式（mock/real/paused），优先级：HTTP/文件 > 环境初值。
    - 检测单笔与单日亏损是否越界，越界则切换为 paused。
    - 将风控事件写入 CSV 与（可选）数据库 risk_events 表。
    """

    def __init__(self, cfg: AppConfig, audit: Optional[AuditLogger] = None) -> None:
        self.cfg = cfg
        self.state = _RiskState(init_mode=self.cfg.exec.mode)
        # 控制文件（JSON）：{"mode":"mock|real|paused","message":"..."}
        self.control_file = self.cfg.exec.control_file
        os.makedirs(os.path.dirname(self.control_file), exist_ok=True)
        # 事件 CSV
        self.risk_csv = os.path.join("data", "risk_events.csv")
        os.makedirs(os.path.dirname(self.risk_csv), exist_ok=True)
        self._ensure_risk_csv()
        # 懒加载 DB 连接（可选）
        self._db: Optional[TimescaleDB] = None
        # 审计记录器（可选注入）
        self.audit: Optional[AuditLogger] = audit

    # ---------- 对外接口 ----------
    def get_mode(self) -> str:
        with self.state.lock:
            return self.state.mode

    def refresh_mode_from_file(self) -> None:
        """从控制文件加载模式，若文件不存在则保持当前模式。"""
        try:
            if os.path.exists(self.control_file):
                with open(self.control_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                new_mode = str(data.get("mode", "")).strip().lower()
                if new_mode in ("mock", "real", "paused"):
                    with self.state.lock:
                        old = self.state.mode
                    # 若从暂停态恢复，要求 confirm=true
                    if old == "paused" and new_mode != "paused":
                        confirm = bool(data.get("confirm", False))
                        if not confirm:
                            return  # 缺少确认，不切换
                    self._switch_mode(new_mode, reason="control_file", event_type="manual_switch")
        except Exception as e:
            logger.warning("读取控制文件失败：{}", e)

    def admin_set_mode(self, new_mode: str, reason: str, confirm: bool = False) -> bool:
        """被 HTTP 管理端调用的切换接口。若从 paused 恢复，要求 confirm=True。"""
        new_mode = new_mode.lower().strip()
        if new_mode not in ("mock", "real", "paused"):
            return False
        with self.state.lock:
            old = self.state.mode
            if old == "paused" and new_mode != "paused" and not confirm:
                # 人工确认才能从暂停恢复
                return False
        ok = self._switch_mode(new_mode, reason=reason or "http_admin", event_type="http_change")
        # 持久化到控制文件
        try:
            with open(self.control_file, "w", encoding="utf-8") as f:
                json.dump({"mode": new_mode, "reason": reason, "ts": datetime.now(timezone.utc).isoformat()}, f, ensure_ascii=False)
        except Exception as e:
            logger.warning("写入控制文件失败：{}", e)
        return ok

    def check_and_maybe_trip(self, symbol: str, side: str, p0: Optional[float], p1: Optional[float]) -> None:
        """基于交易前后价格估算单笔与单日损益，并判断是否触发熔断。
        - p0: 下单前的参考价（trades 最新价）
        - p1: 下单后的参考价（trades 最新价）
        采用简化估算：单笔损益 = 风险名义USD * 方向化收益率（(p1-p0)/p0 对买入为收益，对卖出取反）
        """
        if p0 is None or p1 is None or p0 <= 0:
            return
        ret = (p1 - p0) / p0
        if side.lower() == "sell":
            ret = -ret
        pnl_usd = ret * max(0.0, float(self.cfg.exec.risk_notional_usd))

        # 单笔阈值：若本笔亏损比例 < -Y% 则直接暂挂
        single_th = -abs(float(self.cfg.exec.single_trade_max_loss_pct))
        if ret <= single_th:
            self._trip(
                reason="single_trade_loss_breach",
                details={"symbol": symbol, "ret": ret, "p0": p0, "p1": p1, "risk_notional_usd": self.cfg.exec.risk_notional_usd},
                threshold_usd=abs(self.cfg.exec.risk_notional_usd) * abs(single_th),
            )
            return

        # 单日累计
        self._rollover_if_new_day()
        with self.state.lock:
            self.state.day_loss_usd += min(0.0, pnl_usd)  # 仅累计亏损
            day_loss = self.state.day_loss_usd
        day_loss_limit = -abs(float(self.cfg.exec.base_equity_usd) * float(self.cfg.exec.daily_max_loss_pct))
        if day_loss <= day_loss_limit:
            self._trip(
                reason="daily_loss_breach",
                details={"symbol": symbol, "ret": ret, "pnl_usd": pnl_usd, "day_loss_usd": day_loss},
                threshold_usd=abs(day_loss_limit),
            )

    # ---------- 内部实现 ----------
    def _rollover_if_new_day(self) -> None:
        key = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        with self.state.lock:
            if key != self.state.day_key:
                self.state.day_key = key
                self.state.day_loss_usd = 0.0

    def _switch_mode(self, new_mode: str, reason: str, event_type: str) -> bool:
        if new_mode not in ("mock", "real", "paused"):
            return False
        with self.state.lock:
            old = self.state.mode
            if old == new_mode:
                return True
            self.state.mode = new_mode
            self.state.mode_changed_at = datetime.now(timezone.utc)
        self._write_risk_event(event_type=event_type, mode_before=old, mode_after=new_mode, reason=reason, details={})
        logger.warning("[安全开关] 模式切换：{} -> {}（{}）", old, new_mode, reason)
        return True

    def _trip(self, reason: str, details: Dict[str, Any], threshold_usd: float) -> None:
        with self.state.lock:
            old = self.state.mode
        if old != "paused":
            self._switch_mode("paused", reason=reason, event_type="circuit_break")
        # 事件已经在 _switch_mode 内记录一条；这里补充详细信息
        self._write_risk_event(
            event_type="circuit_break",
            mode_before=old,
            mode_after="paused",
            reason=reason,
            details=details,
            threshold_usd=threshold_usd,
        )

    def _ensure_risk_csv(self) -> None:
        if not os.path.exists(self.risk_csv):
            with open(self.risk_csv, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["ts", "event_type", "mode_before", "mode_after", "reason", "details", "day_pnl_usd", "threshold_usd"])  # 表头

    def _write_risk_event(
        self,
        event_type: str,
        mode_before: Optional[str],
        mode_after: Optional[str],
        reason: str,
        details: Dict[str, Any],
        threshold_usd: Optional[float] = None,
    ) -> None:
        # CSV
        with self.state.lock:
            day_loss = self.state.day_loss_usd
        row = [
            datetime.now(timezone.utc).isoformat(), event_type, mode_before, mode_after,
            reason, json_trunc(details, 1000), f"{day_loss:.4f}", f"{(threshold_usd or 0.0):.4f}",
        ]
        try:
            with open(self.risk_csv, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(row)
        except Exception as e:
            logger.warning("写入风险事件CSV失败：{}", e)
        # DB（可选）
        try:
            if self._db is None:
                self._db = TimescaleDB()
                self._db.connect()
            self._db.insert_risk_event({
                "event_type": event_type,
                "mode_before": mode_before,
                "mode_after": mode_after,
                "reason": reason,
                "details": details,
                "day_pnl_usd": day_loss,
                "threshold_usd": threshold_usd,
            })
        except Exception as e:
            logger.debug("记录风险事件到数据库失败（忽略）：{}", e)
        # 审计（可选）
        try:
            if self.audit:
                self.audit.log(
                    event_type="risk_event",
                    ctx_id=None,
                    module="risk",
                    inst_id=None,
                    action=event_type,
                    request=None,
                    response={
                        "mode_before": mode_before,
                        "mode_after": mode_after,
                        "reason": reason,
                        "details": details,
                        "day_pnl_usd": day_loss,
                        "threshold_usd": threshold_usd,
                    },
                    status="ok",
                    err=None,
                    latency_ms=None,
                    extra=None,
                )
        except Exception as e:
            logger.debug("记录风险事件到审计失败（忽略）：{}", e)

    # 获取某交易对最近成交价（从 trades 表）
    def get_last_price(self, inst_id: str) -> Optional[float]:
        try:
            if self._db is None:
                self._db = TimescaleDB()
                self._db.connect()
            df = self._db.fetch_recent_trades(seconds=10, inst_id=inst_id, limit=1, ascending=False)
            if len(df) > 0 and "price" in df.columns:
                return float(df.iloc[-1]["price"])
        except Exception as e:
            logger.debug("查询最近成交价失败：{}", e)
        # 回退：若数据库无价，尝试使用 REST 行情（ticker -> last/bid/ask；再退到 mark）
        try:
            # 修改：无论是否 real 模式，均允许初始化 REST 客户端用以访问公共行情接口
            # 说明：公共行情不需要鉴权，OKXRESTClient 内部会在未配置密钥时自动使用无签名请求
            if self.client is None:
                self.client = OKXRESTClient(self.cfg)
            if self.client is not None:
                r1 = self.client.get_ticker(inst_id)
                if r1.ok and isinstance(r1.data, list) and len(r1.data) > 0:
                    d = r1.data[0]
                    last = d.get("last") or d.get("lastPx")
                    if last is not None:
                        return float(last)
                r2 = self.client.get_mark_price(inst_id)
                if r2.ok and isinstance(r2.data, list) and len(r2.data) > 0:
                    d = r2.data[0]
                    mark = d.get("markPx")
                    if mark is not None:
                        return float(mark)
        except Exception as e:
            logger.debug("REST 回退获取价格失败：{}", e)
        return None


# --------------------
# HTTP 管理接口（可选）
# --------------------
class _AdminHandler(BaseHTTPRequestHandler):
    """简单HTTP管理接口：
    - GET /status -> 返回当前模式/日内亏损
    - POST /mode {mode,reason,confirm} -> 切换模式；从 paused 恢复需 confirm=true
    所有响应均为 JSON。
    """
    # 将在初始化时注入：guard: CircuitBreaker
    guard: CircuitBreaker = None  # type: ignore

    def _send(self, code: int, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args: Any) -> None:  # 静默认日志
        logger.debug("[HTTP] " + fmt, *args)

    def do_GET(self) -> None:  # noqa: N802
        if self.path.startswith("/status"):
            with self.guard.state.lock:
                payload = {
                    "mode": self.guard.state.mode,
                    "day_loss_usd": self.guard.state.day_loss_usd,
                    "changed_at": self.guard.state.mode_changed_at.isoformat(),
                    "day": self.guard.state.day_key,
                }
            self._send(200, payload)
            return
        self._send(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path.startswith("/mode"):
            try:
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length) if length > 0 else b"{}"
                data = json.loads(raw.decode("utf-8")) if raw else {}
            except Exception:
                data = {}
            new_mode = str(data.get("mode", "")).lower()
            reason = str(data.get("reason", "http"))
            confirm = bool(data.get("confirm", False))
            ok = self.guard.admin_set_mode(new_mode, reason=reason, confirm=confirm)
            if not ok:
                self._send(400, {"ok": False, "error": "invalid_request_or_confirm_required"})
                return
            self._send(200, {"ok": True, "mode": self.guard.get_mode()})
            return
        self._send(404, {"error": "not found"})


class TradeExecutor:
    """交易执行器
    - init(mode="mock"|"real"|"paused")
    - execute(signal) -> ExecResult
    - close()
    """

    def __init__(self, cfg: AppConfig, mode: str = "mock", log_path: str = "data/trade_log.csv", risk_manager: Optional[RiskManager] = None) -> None:
        assert mode in ("mock", "real", "paused"), "mode 必须是 mock/real/paused"
        self.cfg = cfg
        self.mode = mode  # 当前执行模式（可被安全开关动态更新）
        self.log_path = log_path
        self._ensure_logfile()
        # 审计
        self.audit = AuditLogger(cfg)
        # 风控/熔断（仅保留熔断，不启用事前风控）
        self.guard = CircuitBreaker(cfg, audit=self.audit)
        # 关闭事前风控：不再构造 RiskManager
        self.risk = None
        # 可注入账户/行情状态提供器，便于测试与集成
        self._account_state_provider: Optional[Callable[[str], AccountState]] = None
        self._market_state_provider: Optional[Callable[[str, Optional[float]], MarketState]] = None
        # HTTP 管理端（可选）
        self._http_server: Optional[ThreadingHTTPServer] = None
        self._http_thread: Optional[threading.Thread] = None
        if self.cfg.exec.http_admin_enabled:
            try:
                _AdminHandler.guard = self.guard
                self._http_server = ThreadingHTTPServer((self.cfg.exec.http_host, self.cfg.exec.http_port), _AdminHandler)
                self._http_thread = threading.Thread(target=self._http_server.serve_forever, daemon=True)
                self._http_thread.start()
                logger.info("HTTP 管理接口已启动：http://{}:{}/ (GET /status, POST /mode)", self.cfg.exec.http_host, self.cfg.exec.http_port)
            except Exception as e:
                logger.warning("HTTP 管理接口启动失败：{}", e)

        # 实盘客户端（仅在 real 时创建，paused/mock 不创建）
        self.client: Optional[OKXRESTClient] = None
        if self._effective_mode() == "real":
            self.client = OKXRESTClient(cfg)
        logger.info("交易执行器初始化完成，模式={}，日志文件={}", self.mode, self.log_path)

    # 允许外部注入状态提供函数（测试或接交易所账户接口时使用）
    def set_account_state_provider(self, fn: Callable[[str], AccountState]) -> None:
        """设置账户状态提供器：入参为 inst_id，返回 AccountState。"""
        self._account_state_provider = fn

    def set_market_state_provider(self, fn: Callable[[str, Optional[float]], MarketState]) -> None:
        """设置行情状态提供器：入参为 inst_id 与参考价，返回 MarketState。"""
        self._market_state_provider = fn

    # 默认状态提供
    def _default_account_state(self, inst_id: str) -> AccountState:
        day_pnl = float(self.guard.state.day_loss_usd)
        return AccountState(
            equity_usd=float(self.cfg.exec.base_equity_usd),
            position_usd_by_instrument={inst_id: 0.0},
            open_orders_count_by_instrument={inst_id: 0},
            daily_realized_pnl_usd=day_pnl,
        )

    def _default_market_state(self, inst_id: str, ref_price: Optional[float]) -> MarketState:
        mid = self.guard.get_last_price(inst_id) or ref_price
        if mid is None:
            # 再尝试直接从 REST 获取 bid/ask/last 构造 mid
            try:
                # 修改：无论当前模式是否 real，均初始化 REST 客户端以访问公共行情
                if self.client is None:
                    self.client = OKXRESTClient(self.cfg)
                if self.client is not None:
                    # 优先尝试 ticker（包含 bid/ask/last）
                    r = self.client.get_ticker(inst_id)
                    if r.ok and isinstance(r.data, list) and len(r.data) > 0:
                        d = r.data[0]
                        bid = d.get("bidPx")
                        ask = d.get("askPx")
                        last = d.get("last") or d.get("lastPx")
                        # 优先使用 bid/ask 构造 mid，其次回退 last
                        if bid and ask:
                            bid_f = float(bid)
                            ask_f = float(ask)
                            mid_tmp = (bid_f + ask_f) / 2
                            return MarketState(mid_price=mid_tmp, best_bid=bid_f, best_ask=ask_f)
                        if last:
                            last_f = float(last)
                            # 构造一个极小点差用于后续估算
                            spread_pct = 0.0001
                            half_spread = last_f * spread_pct / 2
                            return MarketState(mid_price=last_f, best_bid=last_f - half_spread, best_ask=last_f + half_spread)
                    # 若 ticker 不可用或无数据，则回退到 mark price
                    r2 = self.client.get_mark_price(inst_id)
                    if r2.ok and isinstance(r2.data, list) and len(r2.data) > 0:
                        d2 = r2.data[0]
                        mark = d2.get("markPx")
                        if mark:
                            mark_f = float(mark)
                            spread_pct = 0.0001
                            half_spread = mark_f * spread_pct / 2
                            return MarketState(mid_price=mark_f, best_bid=mark_f - half_spread, best_ask=mark_f + half_spread)
            except Exception as e:
                logger.debug("REST 回退获取市场状态失败：{}", e)
            return MarketState(mid_price=None)
        
        # 为市价单提供合理的bid/ask估算，假设0.01%的价差
        spread_pct = 0.0001  # 0.01%
        half_spread = mid * spread_pct / 2
        best_bid = mid - half_spread
        best_ask = mid + half_spread
        
        return MarketState(mid_price=mid, best_bid=best_bid, best_ask=best_ask)

    # --------------------
    # 工具：CSV 日志
    # --------------------
    def _ensure_logfile(self) -> None:
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        if not os.path.exists(self.log_path):
            with open(self.log_path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow([
                    "ts", "mode", "symbol", "side", "price", "size", "reason", "ok",
                    "order_id", "exchange_code", "exchange_msg", "raw"
                ])

    def _append_log(self, row: Dict[str, Any]) -> None:
        with open(self.log_path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                row.get("ts"), row.get("mode"), row.get("symbol"), row.get("side"), row.get("price"), row.get("size"), row.get("reason"),
                row.get("ok"), row.get("order_id"), row.get("exchange_code"), row.get("exchange_msg"), row.get("raw"),
            ])

    def _effective_mode(self) -> str:
        """返回当前应生效的模式（优先读取控制文件）。"""
        self.guard.refresh_mode_from_file()
        return self.guard.get_mode()

    # --------------------
    # 执行逻辑
    # --------------------
    def execute(self, signal: TradeSignal) -> ExecResult:
        """执行信号。支持：买/卖/平仓（close 视为与当前方向相反的委托）。
        - 市价单：meta.ordType = "market"
        - 限价单：meta.ordType = "limit" 且 price 必填
        - 止盈止损：meta中可设置 tp/sl 参数（tpTriggerPx、tpOrdPx、slTriggerPx、slOrdPx 等）
        - 新增：在执行前进行安全开关检查，执行后基于交易前后价格估算损益以触发熔断。
        """
        # 动态模式（支持外部切换）
        mode = self._effective_mode()
        ctx_id = self.audit.new_ctx_id() if hasattr(self, "audit") and self.audit else None
        if mode == "paused":
            # 暂停态：拒绝下单，记录日志
            self._append_log({
                "ts": signal.ts.isoformat(), "mode": mode, "symbol": signal.symbol, "side": signal.side,
                "price": signal.price, "size": signal.size, "reason": "paused_guard", "ok": False,
                "order_id": None, "exchange_code": "-1", "exchange_msg": "paused", "raw": "{}",
            })
            # 审计：阻断
            try:
                if self.audit:
                    self.audit.log(
                        event_type="risk_block",
                        ctx_id=ctx_id,
                        module="executor",
                        inst_id=signal.symbol,
                        action="place_order",
                        request={
                            "side": signal.side,
                            "ord_type": (signal.meta or {}).get("ordType", "market"),
                            "price": signal.price,
                            "size": signal.size,
                            "meta": signal.meta,
                        },
                        response={"blocked": True, "mode": mode},
                        status="blocked",
                        err="paused",
                        latency_ms=None,
                        extra=None,
                        ts=signal.ts.isoformat(),
                    )
            except Exception as e:
                logger.debug("写入审计失败（忽略）：{}", e)
            return ExecResult(ok=False, mode=mode, signal=signal, err="paused")

        # 若外部模式切换与内部客户端不一致，动态重建客户端
        if mode != self.mode:
            logger.info("检测到运行模式变更：{} -> {}（按外部控制生效）", self.mode, mode)
            self.mode = mode
            if self.mode == "real" and self.client is None:
                self.client = OKXRESTClient(self.cfg)
            if self.mode != "real" and self.client is not None:
                try:
                    self.client.close()
                except Exception:
                    pass
                self.client = None

        ord_type = (signal.meta or {}).get("ordType", "market")
        price_str = f"{signal.price}" if signal.price is not None else None
        sz_str = f"{signal.size}"

        # 下单前价格（用于熔断估算 & 风控参考价）
        p0 = self.guard.get_last_price(signal.symbol)

        # 事前风控校验：按需求移除，直接进入下单流程（仍保留后续熔断监控）
        # 注意：账户/行情状态提供器仍可用于后续扩展，这里不再阻断下单。

        # 审计：下单意图
        try:
            if self.audit:
                self.audit.log(
                    event_type="order_intent",
                    ctx_id=ctx_id,
                    module="executor",
                    inst_id=signal.symbol,
                    action="place_order",
                    request={
                        "side": signal.side,
                        "ord_type": ord_type,
                        "price": signal.price,
                        "size": signal.size,
                        "meta": signal.meta,
                    },
                    response=None,
                    status="pending",
                    err=None,
                    latency_ms=None,
                    extra=None,
                    ts=signal.ts.isoformat(),
                )
        except Exception as e:
            logger.debug("写入审计失败（忽略）：{}", e)

        if self.mode == "mock":
            # 模拟盘：不调用交易所，直接记录
            order_id = f"mock-{int(datetime.now(timezone.utc).timestamp()*1000)}"
            self._append_log({
                "ts": signal.ts.isoformat(), "mode": self.mode, "symbol": signal.symbol, "side": signal.side,
                "price": signal.price, "size": signal.size, "reason": signal.reason, "ok": True,
                "order_id": order_id, "exchange_code": "0", "exchange_msg": "mock-ok", "raw": "{}",
            })
            # 审计：结果
            try:
                if self.audit:
                    self.audit.log(
                        event_type="order_result",
                        ctx_id=ctx_id,
                        module="executor",
                        inst_id=signal.symbol,
                        action="place_order",
                        request=None,
                        response={"order_id": order_id, "exchange_code": "0", "exchange_msg": "mock-ok"},
                        status="ok",
                        err=None,
                        latency_ms=None,
                        extra=None,
                        ts=signal.ts.isoformat(),
                    )
            except Exception as e:
                logger.debug("写入审计失败（忽略）：{}", e)
            # 模拟盘后参考价（近似）
            p1 = self.guard.get_last_price(signal.symbol)
            self.guard.check_and_maybe_trip(signal.symbol, signal.side, p0, p1)
            return ExecResult(ok=True, mode=self.mode, signal=signal, order_id=order_id)

        # 实盘：调用 OKX REST 下单
        assert self.client is not None, "实盘模式需要 OKXRESTClient"
        side = signal.side
        if side == "close":
            # 现货 close 语义：若有持仓，则按照 side=卖出；此处简单处理，默认 close = sell
            side = "sell"

        # 止盈止损参数透传
        meta = signal.meta or {}
        # 新增：全局禁用止盈/止损 —— 清理任何 tp/sl 字段，并标记执行器忽略止损逻辑
        try:
            meta["noSL"] = True
            for k in ("slTriggerPx", "slOrdPx", "slTriggerPxType", "tpTriggerPx", "tpOrdPx", "tpTriggerPxType"):
                meta.pop(k, None)
            # 防止通过 extra 透传止盈止损/附加委托
            if isinstance(meta.get("extra"), dict):
                extra = meta.get("extra")
                # 去掉直接字段
                for k in ("slTriggerPx", "slOrdPx", "slTriggerPxType", "tpTriggerPx", "tpOrdPx", "tpTriggerPxType"):
                    extra.pop(k, None)
                # 去掉 attachAlgoOrds
                extra.pop("attachAlgoOrds", None)
                meta["extra"] = extra
        except Exception:
            pass
        # 规范化止损参数：若仅提供触发价则默认以市价委托执行止损（slOrdPx=-1），并设置默认触发价类型为 last，避免与交易所规则不匹配
        # 新增：若 meta['noSL'] 为 True，则不附带任何止损相关参数（用于“直接下单不设置止损”的场景）
        try:
            if bool(meta.get("noSL")):
                # 清理可能存在的止损字段，避免误传导致交易所侧校验（51053等）
                for k in ("slTriggerPx", "slOrdPx", "slTriggerPxType"):
                    if k in meta:
                        meta.pop(k, None)
            else:
                if meta.get("slTriggerPx") is not None:
                    try:
                        # 将触发价规范为字符串数字
                        meta["slTriggerPx"] = str(float(meta.get("slTriggerPx")))
                    except Exception:
                        # 若无法解析，保持原样，交由下游处理
                        meta["slTriggerPx"] = str(meta.get("slTriggerPx"))
                    # 未提供 slOrdPx 时，强制使用市价止损
                    if not meta.get("slOrdPx"):
                        meta["slOrdPx"] = "-1"  # 市价触发
                    else:
                        meta["slOrdPx"] = str(meta.get("slOrdPx"))
                    # 默认触发价类型为 last
                    if not meta.get("slTriggerPxType"):
                        meta["slTriggerPxType"] = "last"
                    # === 方向性校验与微调 ===
                    sl_str = meta.get("slTriggerPx")
                    if sl_str is not None:
                        # 以字符串形式保存，但这里需要数值参与校验
                        sl_val = float(sl_str)
                        # 估算主单的“预期成交价”
                        # - 市价单：BUY 近似按最优卖价(ask)成交，SELL 近似按最优买价(bid)成交
                        # - 限价单：用用户给定的 price
                        expected_px: Optional[float] = None
                        # 再次获取市场状态（包含 best_bid/best_ask），若外部未注入 provider 则使用默认实现
                        mkt = (self._market_state_provider(signal.symbol, p0) if self._market_state_provider else self._default_market_state(signal.symbol, p0))
                        if ord_type == "market":
                            if side == "buy":
                                expected_px = (mkt.best_ask if hasattr(mkt, "best_ask") else None) or (mkt.mid_price if hasattr(mkt, "mid_price") else None) or p0
                            else:
                                expected_px = (mkt.best_bid if hasattr(mkt, "best_bid") else None) or (mkt.mid_price if hasattr(mkt, "mid_price") else None) or p0
                        else:
                            # 限价单直接使用下单价
                            expected_px = float(signal.price) if signal.price is not None else p0
                        # 若仍拿不到参考价，则跳过校验
                        if expected_px is not None:
                            # 设定一个极小的相对偏移（0.001%），仅用于满足方向性不等式
                            eps = max(1e-8, float(expected_px) * 1e-6)
                            if side == "buy":
                                # BUY：止损应低于主单价
                                if sl_val >= float(expected_px):
                                    adj = float(expected_px) * (1.0 - 1e-6)
                                    # 仅在原值不满足时才调整，尽量保持用户原始意图
                                    sl_val = min(sl_val, adj)
                            else:
                                # SELL：止损应高于主单价
                                if sl_val <= float(expected_px):
                                    adj = float(expected_px) * (1.0 + 1e-6)
                                    sl_val = max(sl_val, adj)
                            # 回写为字符串，保持与其余字段类型一致
                            meta["slTriggerPx"] = f"{sl_val}"
        except Exception as _:
            # 若校验过程出现异常，不影响下单主流程
            pass

        resp: OKXResponse = self.client.place_order(
            inst_id=signal.symbol,
            side=side,
            ord_type=ord_type,
            px=price_str,
            sz=sz_str,
            td_mode=meta.get("tdMode"),
            tgt_ccy=meta.get("tgtCcy"),
            cl_ord_id=meta.get("clOrdId"),
            tp_trigger_px=None,  # 强制不发送止盈
            tp_ord_px=None,
            sl_trigger_px=None,  # 强制不发送止损
            sl_ord_px=None,
            tp_trigger_px_type=None,
            sl_trigger_px_type=None,
            extra=meta.get("extra"),
        )

        order_id = None
        if resp.ok and isinstance(resp.data, list) and resp.data:
            order_id = resp.data[0].get("ordId") or resp.data[0].get("algoId")

        # 写日志
        self._append_log({
            "ts": signal.ts.isoformat(), "mode": self.mode, "symbol": signal.symbol, "side": side,
            "price": signal.price, "size": signal.size, "reason": signal.reason, "ok": resp.ok,
            "order_id": order_id, "exchange_code": resp.code, "exchange_msg": resp.msg, "raw": json_trunc(resp.raw),
        })

        # 审计：结果
        try:
            if self.audit:
                self.audit.log(
                    event_type="order_result",
                    ctx_id=ctx_id,
                    module="executor",
                    inst_id=signal.symbol,
                    action="place_order",
                    request=None,
                    response={"ok": resp.ok, "order_id": order_id, "code": resp.code, "msg": resp.msg, "raw": resp.raw},
                    status=("ok" if resp.ok else "error"),
                    err=(None if resp.ok else resp.msg),
                    latency_ms=None,
                    extra=None,
                    ts=signal.ts.isoformat(),
                )
        except Exception as e:
            logger.debug("写入审计失败（忽略）：{}", e)

        # 下单后取参考价，计算熔断
        p1 = self.guard.get_last_price(signal.symbol)
        self.guard.check_and_maybe_trip(signal.symbol, side, p0, p1)

        if not resp.ok:
            return ExecResult(ok=False, mode=self.mode, signal=signal, exchange_resp=resp.raw, err=resp.msg)
        return ExecResult(ok=True, mode=self.mode, signal=signal, exchange_resp=resp.raw, order_id=order_id)

    def close(self) -> None:
        if self.client:
            self.client.close()
        if self._http_server:
            try:
                self._http_server.shutdown()
            except Exception:
                pass
            self._http_server = None
            self._http_thread = None


# 简单自测
if __name__ == "__main__":
    cfg = AppConfig()
    exe = TradeExecutor(cfg, mode=os.getenv("EXEC_MODE", cfg.exec.mode))
    sig = TradeSignal(
        ts=datetime.now(timezone.utc),
        symbol=os.getenv("TEST_INST_ID", "BTC-USDT"),
        side="buy",
        price=None,
        size=0.001,
        reason="unit-test",
        meta={"ordType": "market"}
    )
    r = exe.execute(sig)
    logger.info("执行结果: ok={}, order_id={}, err={}", r.ok, r.order_id, r.err)
    exe.close()