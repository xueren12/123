# -*- coding: utf-8 -*-
"""
主程序入口：整合数据采集、策略、交易执行、AI 辅助与监控告警。
- 初始化数据库连接与 OKX WebSocket 采集器
- 周期性运行策略（基于 DB 的 books5 盘口），产生交易信号
- 可选：调用 AI 模块做风险提示/建议（按事件触发，示例中为可选开关）
- 交易执行：默认模拟盘（可切换实盘），记录 CSV 日志
- 启动终端监控与告警模块（可选 Telegram/Slack 推送）
- 支持平滑关闭与异常捕获

运行示例：
    python main.py

关键环境变量（也可写入 .env）：
    EXEC_MODE=mock            # mock/real（实盘需配置 OKX API）
    TRADE_SIZE=0.001          # 策略开仓默认下单数量（现货为币数量；合约为张数）
    AI_GATE_ENABLED=0         # 是否启用“AI 建议门控”（1 开启，0 关闭）
    MONITOR_ENABLED=1         # 是否启动终端监控（1 开启，0 关闭）
"""
from __future__ import annotations

import asyncio
import os
import signal
import threading
import time
from typing import Dict, Optional, Tuple, List, Any

from loguru import logger

from utils.config import AppConfig
from utils.okx_ws_collector import OKXWSCollector
from utils.monitor import RealTimeMonitor
from utils.db import TimescaleDB
from strategies.multi_indicator import MABreakoutStrategy, MABreakoutConfig
from executor.trade_executor import TradeExecutor, TradeSignal
from models.ai_advisor import AIAdvisor, build_payload
from utils.position import PositionManager  # 引入仓位管理器
# 新增：引入 OKX REST 客户端与时间工具（用于预填K线）
from executor.okx_rest import OKXRESTClient
from datetime import datetime, timezone, timedelta


class SystemOrchestrator:
    """系统编排器：负责拉起并管理采集、策略、执行、AI 与监控。"""

    def __init__(self, cfg: Optional[AppConfig] = None) -> None:
        self.cfg = cfg or AppConfig()
        self.db = TimescaleDB()
        # 采集器开关：允许在本地离线冒烟时禁用 WS 采集器
        self.collector_enabled = os.getenv("COLLECTOR_ENABLED", "1") == "1"
        self.collector = OKXWSCollector(self.cfg) if self.collector_enabled else None
        # 执行器（根据配置模式初始化：mock/real/paused）
        self.executor = TradeExecutor(self.cfg, mode=self.cfg.exec.mode)
        # 仓位管理器：基于账户权益与风险参数计算手数（价格来源于执行器的 guard）
        try:
            self.pos_manager = PositionManager(self.cfg, price_provider=self.executor.guard.get_last_price)
        except Exception:
            self.pos_manager = None
        # AI 模块（按需调用）
        self.ai = AIAdvisor(self.cfg)
        # AI 门控与监控开关
        self.ai_gate_enabled = os.getenv("AI_GATE_ENABLED", "0") == "1"
        self.monitor_enabled = os.getenv("MONITOR_ENABLED", "1") == "1"
        # 监控模块（独立线程运行，阻塞性 while True 循环）
        self._monitor_thread: Optional[threading.Thread] = None
        self.monitor = RealTimeMonitor(self.cfg)
        # 冷却时间：优先读取 MA_COOLDOWN_SEC，其次 COOLDOWN_SEC；默认 0=不启用
        try:
            self.cooldown_sec: float = float(os.getenv("MA_COOLDOWN_SEC", os.getenv("COOLDOWN_SEC", "0")))
        except Exception:
            self.cooldown_sec = 0.0
        # —— 若在 discipline.json 中设置了按根数冷却，则覆盖为 bars * timeframe_seconds ——
        try:
            bars = int(getattr(self.cfg.discipline, "cooldown_bars", 0)) if hasattr(self.cfg, "discipline") else 0
        except Exception:
            bars = 0
        if bars and bars > 0:
            tf = os.getenv("MA_TIMEFRAME", "5min").strip()
            tf_sec = self._timeframe_seconds(tf)
            # bars * 每根bar秒数
            self.cooldown_sec = float(bars) * float(tf_sec)
            logger.info("按根数冷却覆盖：cooldown_bars={} timeframe={} -> cooldown_sec={}", bars, tf, self.cooldown_sec)
        # 记录各标的最近一次下单时间戳（秒）
        self._last_trade_ts: Dict[str, float] = {}
        # 记录各标的最近一次“基础信号”已处理的bar收盘时间，避免同一bar重复下单
        self._last_signal_bar_ts: Dict[str, Any] = {}
        self._last_exit_bar_ts: Dict[str, Any] = {}
        # 同bar部分平仓去重：记录各标的各原因最近一次部分平仓的bar时间
        self._last_partial_exit_reason_ts: Dict[str, Dict[str, Any]] = {}
        # 卖出短冷却（秒级，避免抖动连发）
        try:
            self.sell_cooldown_sec: float = float(os.getenv("SELL_COOLDOWN_SEC", "5"))
        except Exception:
            self.sell_cooldown_sec = 5.0
        self._last_sell_ts: Dict[str, float] = {}
        # 虚拟持仓：按策略目标持仓记录（BUY=+1，SELL=-1，HOLD=延续），仅用于日志观测
        self._virt_pos: Dict[str, float] = {}
        # 采集任务与策略缓存
        self._collector_task: Optional[asyncio.Task] = None
        self._strategies: Dict[str, Any] = {}
        # 杂项
        self.stop_event = threading.Event()
        self._db_for_ai = TimescaleDB()  # 仅用于查询最近价/前值给 AI 构造输入

    # ========== 工具：DB 快速查询最近成交价与上一个成交价（避免引入 pandas） ==========
    def _ensure_ai_db(self) -> None:
        if self._db_for_ai.conn is None:
            self._db_for_ai.connect()

    def _last_two_prices(self, inst: str) -> Tuple[Optional[float], Optional[float]]:
        """返回 (last, prev)。若不存在则为 (None, None)。"""
        try:
            self._ensure_ai_db()
            assert self._db_for_ai.conn is not None
            sql = (
                "SELECT price::float8 AS price FROM trades "
                "WHERE inst_id=%(inst)s ORDER BY ts DESC LIMIT 2"
            )
            with self._db_for_ai.conn.cursor() as cur:
                cur.execute(sql, {"inst": inst})
                rows = cur.fetchall() or []
            if not rows:
                return None, None
            last = float(rows[0]["price"]) if rows[0].get("price") is not None else None
            prev = float(rows[1]["price"]) if len(rows) > 1 and rows[1].get("price") is not None else None
            return last, prev
        except Exception as e:
            logger.debug("获取最近两笔成交价失败：{}", e)
            return None, None

    # ========== 工具：将时间粒度字符串换算为秒 ==========
    def _timeframe_seconds(self, timeframe: str) -> float:
        """支持 1m/5m/15m/30m/1h/4h/1d/1min/5min 等常见写法，无法识别时默认 60s。
        注：只用于冷却时间 bars->秒 的换算。"""
        if not timeframe:
            return 60.0
        s = timeframe.lower().strip()
        try:
            if s.endswith("min"):
                return float(int(s.replace("min", "")) * 60)
            if s.endswith("m"):
                return float(int(s.replace("m", "")) * 60)
            if s.endswith("h"):
                return float(int(s.replace("h", "")) * 3600)
            if s.endswith("d"):
                return float(int(s.replace("d", "")) * 86400)
        except Exception:
            pass
        return 60.0

    def _tf_to_okx_bar(self, tf: str) -> str:
        """将常见 timeframe 写法转换为 OKX bar 取值。未识别时回退为 1m。"""
        s = (tf or "1min").strip().lower()
        if s.endswith("min"):
            try:
                n = int(s[:-3]); return f"{max(1, n)}m"
            except Exception:
                return "1m"
        if s.endswith("m"):
            try:
                n = int(s[:-1]); return f"{max(1, n)}m"
            except Exception:
                return "1m"
        if s.endswith("h"):
            try:
                n = int(s[:-1]); return f"{max(1, n)}H"
            except Exception:
                return "1m"
        if s.endswith("d"):
            try:
                n = int(s[:-1]); return f"{max(1, n)}D"
            except Exception:
                return "1m"
        return "1m"

    def _warmup_kline(self, insts: List[str]) -> None:
        """根据 timeframe 与指标窗口，预填充最少所需根数的 K 线到 trades 表（以模拟成交写入）。"""
        try:
            tf_env = os.getenv("MA_TIMEFRAME")
            sec_per_bar = self._timeframe_seconds(tf_env)
            if sec_per_bar <= 0:
                sec_per_bar = 60.0
            bar = self._tf_to_okx_bar(tf_env)
            # 初始化 DB 与 REST 客户端
            db = TimescaleDB()
            db.connect()
            rest = OKXRESTClient(self.cfg)
            for inst in insts:
                try:
                    # 依据策略配置计算与策略一致的最小需求根数
                    strat = self._get_strategy(inst)
                    cfg = getattr(strat, "cfg")
                    atr_period = 14
                    min_need = max(
                        int(getattr(cfg, "macd_slow", 26) + getattr(cfg, "macd_signal", 9) + 5),
                        int(getattr(cfg, "bb_period", 20) + 5),
                        int(getattr(cfg, "rsi_period", 14) + 5),
                        int(getattr(cfg, "aroon_period", 25) + 1),
                        int(atr_period + 5),
                    )
                    # 时间窗口：多取 2 根缓冲
                    end = datetime.now(timezone.utc)
                    start = end - timedelta(seconds=(min_need + 2) * float(sec_per_bar))
                    # 优先：直接获取最近 N 根 K 线，避免大范围分页
                    limit_n = max(5, min(min_need + 2, 300))  # OKX 单次上限通常为 300
                    rows = None
                    try:
                        resp = rest.get_candles(inst_id=inst, bar=bar, limit=limit_n)
                        if resp.ok and resp.data:
                            rows = resp.data
                        else:
                            logger.warning("[预填K线] get_candles 返回为空/失败，回退到按区间分页：inst={} tf={} bar={}", inst, tf_env, bar)
                    except Exception as e:
                        logger.warning("[预填K线] get_candles 异常：{}，回退到按区间分页", e)
                    # 回退：使用区间分页（仅在 rows 为空时触发）
                    if not rows:
                        ok, rows = rest.fetch_ohlcv_range(inst_id=inst, bar=bar, start=start, end=end, limit_per_call=100, use_history=False)
                        if not ok or not rows:
                            logger.warning("[预填K线] 拉取失败/为空：inst={} tf={} bar={}", inst, tf_env, bar)
                            continue
                    # 规范化与截取
                    rows = [r for r in rows if len(r) >= 5]
                    rows.sort(key=lambda r: int(r[0]))
                    if len(rows) > min_need:
                        rows = rows[-min_need:]
                    # 查询已存在的 warmup trade_id，避免重复
                    try:
                        df_exist = db.fetch_trades_window(start=start, end=end, inst_id=inst, ascending=True)
                        exist_ids = set()
                        if df_exist is not None and not df_exist.empty:
                            try:
                                exist_ids = set([str(x) for x in df_exist.get("trade_id", []) if x and str(x).startswith("warmup:")])
                            except Exception:
                                exist_ids = set()
                    except Exception:
                        exist_ids = set()
                    to_insert: List[Dict[str, Any]] = []
                    for r in rows:
                        try:
                            ts_ms = int(r[0])
                            o = float(r[1]); h = float(r[2]); l = float(r[3]); c = float(r[4])
                            bar_start = datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc)
                            sec = int(max(1, int(sec_per_bar)))
                            # 在该 bar 内生成 4 个时间点（开/高/低/收）
                            open_ts = bar_start + timedelta(milliseconds=1)
                            high_ts = bar_start + timedelta(seconds=max(0, int(sec * 0.3)))
                            low_ts = bar_start + timedelta(seconds=max(0, int(sec * 0.6)))
                            close_ts = bar_start + timedelta(seconds=sec) - timedelta(milliseconds=1)
                            pts = [
                                (open_ts, o, "o"),
                                (high_ts, h, "h"),
                                (low_ts, l, "l"),
                                (close_ts, c, "c"),
                            ]
                            for ts_dt, px, tag in pts:
                                tid = f"warmup:{inst}:{int(bar_start.timestamp())}:{tag}"
                                if tid in exist_ids:
                                    continue
                                to_insert.append({
                                    "ts": ts_dt,
                                    "inst_id": inst,
                                    "side": "warmup",   # 标记 warmup
                                    "price": px,
                                    "size": 0.0,
                                    "trade_id": tid,
                                    "source": "warmup",
                                })
                        except Exception:
                            continue
                    if to_insert:
                        n = db.insert_trades(to_insert)
                        logger.success("[预填K线] 已写入 {} 条模拟成交 -> inst={} tf={} bars={}", n, inst, tf_env, len(rows))
                    else:
                        logger.info("[预填K线] 无需写入 -> inst={} tf={} bars={}", inst, tf_env, len(rows))
                except Exception as e:
                    logger.warning("[预填K线] 单标的异常：inst={} err={}", inst, e)
            try:
                db.close()
            except Exception:
                pass
        except Exception as e:
            logger.warning("[预填K线] 初始化失败：{}", e)

    # ========== 组件启动/停止 ==========
    async def start(self) -> None:
        """启动采集器、策略循环与监控。"""
        insts = [s.strip() for s in self.cfg.ws.instruments if s.strip()]
        logger.info("系统启动：标的={}, 执行模式={}, AI门控={}, 监控={}", insts, self.executor.mode, self.ai_gate_enabled, self.monitor_enabled)
        # 启动前：预填 K 线，确保策略首次运行时指标窗口充足
        self._warmup_kline(insts)

        # 1) 启动 OKX 采集器（异步任务，可禁用）
        if self.collector_enabled and self.collector is not None:
            self._collector_task = asyncio.create_task(self.collector.start(), name="okx_collector")
        else:
            logger.info("已禁用 WS 采集器（COLLECTOR_ENABLED=0）")

        # 2) 启动监控（后台线程）
        if self.monitor_enabled:
            self._monitor_thread = threading.Thread(target=self.monitor.run, name="monitor", daemon=True)
            self._monitor_thread.start()

        # 3) 启动策略循环（异步任务）
        await self._strategy_loop(insts)

    async def stop(self) -> None:
        """请求停止所有组件。"""
        if self.stop_event.is_set():
            return
        logger.info("接收到停止请求，准备优雅退出...")
        self.stop_event.set()

        # 取消采集器任务
        if self._collector_task and not self._collector_task.done():
            self._collector_task.cancel()
            try:
                await self._collector_task
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.debug("采集器任务结束异常：{}", e)

        # 关闭执行器与 DB
        try:
            self.executor.close()
        except Exception:
            pass
        try:
            self._db_for_ai.close()
        except Exception:
            pass

        logger.success("系统已停止。")

    def _get_strategy(self, inst: str) -> Any:
        """仅保留均线+突破策略"""
        if inst not in self._strategies:
            # —— 先计算将被 discipline 覆盖的参数 ——
            d = getattr(self.cfg, "discipline", None)
            atr_n_val = (d.atr_stop_loss if d is not None else float(os.getenv("MA_ATR_N", "2.0")))
            sl_btc_val = (d.stop_loss_pct_btc if d is not None else float(os.getenv("MA_STOP_LOSS_PCT_BTC", "0.03")))
            sl_eth_val = (d.stop_loss_pct_eth if d is not None else float(os.getenv("MA_STOP_LOSS_PCT_ETH", "0.04")))
            tp_frac1_val = None
            tp_frac2_val = None
            if d is not None and isinstance(d.take_profit_split, list) and len(d.take_profit_split) >= 1:
                try:
                    tp_frac1_val = float(d.take_profit_split[0])
                except Exception:
                    tp_frac1_val = None
            if d is not None and isinstance(d.take_profit_split, list) and len(d.take_profit_split) >= 2:
                try:
                    tp_frac2_val = float(d.take_profit_split[1])
                except Exception:
                    tp_frac2_val = None
            trail_atr_val = (d.trailing_atr if d is not None else float(os.getenv("MA_TP_TRAIL_ATR", "1.5")))

            strat_cfg = MABreakoutConfig(
                inst_id=inst,
                # 优先读取新命名（MI_*），回退到旧命名（MA_*），不影响现有部署
                fast_ma=int(os.getenv("MI_FAST") or os.getenv("MA_FAST", "10")),
                slow_ma=int(os.getenv("MI_SLOW") or os.getenv("MA_SLOW", "30")),
                breakout_lookback=int(os.getenv("MI_LOOKBACK") or os.getenv("MA_BREAKOUT_N", "20")),
                breakout_buffer_pct=float(os.getenv("MI_BUFFER") or os.getenv("MA_BREAKOUT_BUFFER", "0.0")),
                # BB 周期：优先 MI_BB_PERIOD -> MA_BB_PERIOD -> MI_LOOKBACK -> MA_BREAKOUT_N，保证有值
                bb_period=int(os.getenv("MI_BB_PERIOD") or os.getenv("MA_BB_PERIOD") or os.getenv("MI_LOOKBACK") or os.getenv("MA_BREAKOUT_N", "20")),
                macd_fast=int(os.getenv("MACD_FAST", "12")),
                macd_slow=int(os.getenv("MACD_SLOW", "26")),
                macd_signal=int(os.getenv("MACD_SIGNAL", "9")),
                rsi_period=int(os.getenv("RSI_PERIOD", "14")),
                aroon_period=int(os.getenv("AROON_PERIOD", "14")),
                # 新增：显式设置策略 timeframe，确保与预填口径一致
                timeframe=(os.getenv("MI_TIMEFRAME") or os.getenv("MA_TIMEFRAME", "5min")),
                # 新增：空头开仓控制
                allow_short=((os.getenv("MI_ALLOW_SHORT") or os.getenv("MA_ALLOW_SHORT", "0")) == "1"),
                short_confirm_extra=int(os.getenv("SHORT_CONFIRM_EXTRA") or os.getenv("MA_SHORT_CONFIRM_EXTRA", "1")),
                # —— 风控参数：使用前面计算好的值 ——
                atr_n=atr_n_val,
                stop_loss_pct_btc=sl_btc_val,
                stop_loss_pct_eth=sl_eth_val,
                tp_frac1=tp_frac1_val,
                tp_frac2=tp_frac2_val,
                tp_trail_atr_mult=trail_atr_val,
            )
            self._strategies[inst] = MABreakoutStrategy(strat_cfg)
        return self._strategies[inst]

    async def _strategy_loop(self, insts: List[str]) -> None:
        """按策略建议节奏运行，必要时触发下单与 AI 建议。"""
        trade_size = float(os.getenv("TRADE_SIZE", "0.001"))
        # 根据策略类型确定轮询周期（中低频策略建议使用分钟级节奏）
        poll_sec = max(0.5, float(os.getenv("MA_POLL_SEC", "60.0")))
        logger.info("策略循环启动：trade_size={} poll={}s type={} cooldown={}s", trade_size, poll_sec, "multi_indicator", self.cooldown_sec)
        try:
            while not self.stop_event.is_set():
                loop_start = time.time()
                for inst in insts:
                    try:
                        strat = self._get_strategy(inst)
                        sig = getattr(strat, "compute_signal")()
                        if not sig:
                            logger.debug("[策略] {} 暂无信号/数据不足", inst)
                            continue
                        # 输出关键指标，便于观测（最近一根bar的 close/MA/高低点）
                        try:
                            logger.info(
                                "[{}] sig={} close={:.6f} fast={:.6f} slow={:.6f} highN={:.6f} lowN={:.6f}",
                                inst, sig.get("signal"), float(sig.get("close")), float(sig.get("fast")),
                                float(sig.get("slow")), float(sig.get("brk_high")), float(sig.get("brk_low")),
                            )
                        except Exception:
                            pass

                        side = str(sig.get("signal", "")).upper()
                        if side in ("BUY", "SELL"):
                            # —— 每根K线仅处理一次基础信号（reason=multi_indicator） ——
                            try:
                                reason = str(sig.get("reason") or "")
                                sig_ts = sig.get("ts")
                            except Exception:
                                reason = ""
                                sig_ts = None
                            if reason == "multi_indicator" and sig_ts is not None:
                                last_bar_ts = self._last_signal_bar_ts.get(inst)
                                # 若上一根已处理的bar时间与当前信号时间一致，则跳过重复下单
                                if last_bar_ts is not None:
                                    try:
                                        same_bar = bool(sig_ts == last_bar_ts)
                                    except Exception:
                                        same_bar = False
                                    if same_bar:
                                        try:
                                            ts_str = sig_ts.isoformat() if hasattr(sig_ts, "isoformat") else str(sig_ts)
                                        except Exception:
                                            ts_str = str(sig_ts)
                                        logger.info("[同一bar跳过] {} ts={} 已处理过基础信号，跳过重复下单", inst, ts_str)
                                        continue

                            # 冷却判定：若设置了冷却且尚未到期则跳过
                            if self.cooldown_sec > 0:
                                last_ts = self._last_trade_ts.get(inst, 0.0)
                                if (time.time() - last_ts) < self.cooldown_sec:
                                    remain = max(0.0, self.cooldown_sec - (time.time() - last_ts))
                                    logger.info("[冷却中] {} 信号={}，剩余 {:.1f}s 跳过下单", inst, side, remain)
                                    continue
                            # 可选：调用 AI 建议作为门控
                            allow_trade = True
                            ai_advice_desc = None
                            if self.ai_gate_enabled:
                                last, prev = self._last_two_prices(inst)
                                market = {
                                    "inst": inst,
                                    "ts": int(time.time() * 1000),
                                    "last": last if last is not None else 0.0,
                                    "prev_last": prev if prev is not None else last if last is not None else 0.0,
                                    "buy_vol_60s": 0.0,
                                    "sell_vol_60s": 0.0,
                                    "spread_bps": None,
                                }
                                payload = build_payload(market, [], {"score": 0.0, "delta": 0.0, "summary": ""})
                                advice = self.ai.run_if_needed(payload)
                                ai_advice_desc = f"AI建议={advice.signal} 置信度={advice.confidence:.2f} 风险={advice.risk_level} 触发={advice.triggered}"
                                # 简单门控：若 AI 风险判定为 HIGH，则跳过下单
                                if str(getattr(advice, "risk_level", "")).upper() == "HIGH":
                                    allow_trade = False
                            if not allow_trade:
                                logger.warning("[AI门控拦截] {} 信号={} 被拦截。{}", inst, side, ai_advice_desc or "")
                                continue

                            # —— 使用仓位管理器计算建议手数，失败回退到 TRADE_SIZE ——
                            try:
                                # 提取策略建议的止损价，传递给仓位管理器做风险等权 sizing（仅 BUY 会启用）
                                stop_px = None
                                try:
                                    if sig.get("sl") is not None:
                                        stop_px = float(sig.get("sl"))
                                except Exception:
                                    stop_px = None
                                pm_size = self.pos_manager.compute_size(inst, side.lower(), stop_px=stop_px) if self.pos_manager else None
                            except Exception:
                                pm_size = None
                            used_size = pm_size if (pm_size is not None and pm_size > 0) else trade_size

                            # 读取策略建议的部分止盈比例（仅 SELL 场景可能出现）
                            size_frac = sig.get("size_frac")
                            if side == "SELL" and size_frac is not None:
                                try:
                                    frac = float(size_frac)
                                    # 仅当 0<frac<1 视为部分平仓；否则按全量/反转处理
                                    if 0.0 < frac < 1.0:
                                        used_size = float(used_size) * frac
                                        # 防止过小导致无效下单
                                        if used_size <= 0:
                                            logger.info("[跳过下单] {} 部分平仓计算后手数无效 frac={:.4f}", inst, frac)
                                            continue
                                except Exception:
                                    pass

                            # 生成交易信号并执行（统一组装）
                            ts = sig.get("ts")
                            ord_type = str(sig.get("ord_type", "market")).lower()
                            meta = {"ordType": ord_type}
                            # 交易模式：逐仓（isolated）。如为现货则保持 cash
                            try:
                                parts = inst.split("-")
                                is_derivative = (len(parts) >= 3) or inst.endswith("-SWAP")
                                meta["tdMode"] = ("isolated" if is_derivative else "cash")
                            except Exception:
                                meta["tdMode"] = "isolated"
                            # 止损开关：若设置了环境变量 DISABLE_SL=1，则不附带止损参数
                            disable_sl_env = str(os.getenv("DISABLE_SL", "0")).strip() == "1"
                            disable_sl = disable_sl_env
                            # 透传策略建议止损价 -> 风控/执行可利用（slTriggerPx 或 slOrdPx）
                            if (not disable_sl) and (sig.get("sl") is not None):
                                try:
                                    meta["slTriggerPx"] = float(sig.get("sl"))
                                except Exception:
                                    pass
                            if disable_sl:
                                # 提示执行器忽略任何止损逻辑（如支持）
                                meta["noSL"] = True
                            # 透传部分平仓比例（仅用于审计/记录，执行侧不强制使用）
                            try:
                                if size_frac is not None:
                                    meta["sizeFrac"] = float(size_frac)
                            except Exception:
                                pass
                            reason = sig.get("reason") or "multi_indicator"

                            # 长仅模式：空仓的 SELL 不开空；有持仓的 SELL 转为 close（仅平仓）
                            try:
                                row = self.db.fetch_latest_position_frac(inst)
                                virt_pos = float(row["pos_frac"]) if (row and row.get("pos_frac") is not None) else 0.0
                            except Exception:
                                try:
                                    virt_pos = float(self._virt_pos.get(inst, 0.0))
                                except Exception:
                                    virt_pos = 0.0
                            sell_close_reason = False
                            try:
                                r = str(reason or "")
                                sell_close_reason = r.startswith("takeprofit") or r.startswith("stoploss") or (size_frac is not None)
                            except Exception:
                                sell_close_reason = (size_frac is not None)
                            if side == "SELL":
                                # 判断是否允许做空开仓
                                allow_short = bool(getattr(strat.cfg, "allow_short", False))
                                if virt_pos <= 1e-8 and not sell_close_reason:
                                    if allow_short:
                                        # —— 空仓 SELL：作为开空处理 ——
                                        ts_side = "sell"
                                        # 合约/永续设置 posSide=short（通过 extra 透传）
                                        try:
                                            meta.setdefault("extra", {})
                                            parts = inst.split("-")
                                            is_derivative2 = (len(parts) >= 3) or inst.endswith("-SWAP")
                                        except Exception:
                                            is_derivative2 = False
                                        if is_derivative2:
                                            try:
                                                meta["extra"].setdefault("posSide", "short")
                                            except Exception:
                                                pass
                                        # 计算做空止损价（用于风险等权 sizing 与止损透传）：按学科配置的百分比
                                        short_stop = None
                                        try:
                                            px_now = None
                                            try:
                                                px_now = float(sig.get("close")) if sig.get("close") is not None else None
                                            except Exception:
                                                px_now = None
                                            if px_now is None:
                                                try:
                                                    px_now = float(self.executor.get_last_price(inst))
                                                except Exception:
                                                    px_now = None
                                            if px_now is not None and px_now > 0:
                                                symu = str(inst).upper()
                                                pct = None
                                                try:
                                                    if symu.startswith("BTC"):
                                                        pct = float(getattr(self.cfg.discipline, "stop_loss_pct_btc", 0.0) or 0.0)
                                                    elif symu.startswith("ETH"):
                                                        pct = float(getattr(self.cfg.discipline, "stop_loss_pct_eth", 0.0) or 0.0)
                                                except Exception:
                                                    pct = None
                                                if pct is not None and pct > 0:
                                                    short_stop = float(px_now) * (1.0 + float(pct))
                                        except Exception:
                                            short_stop = None
                                        # 风险等权：若可计算止损，尝试据此重算手数
                                        try:
                                            if self.pos_manager is not None:
                                                pm_short = self.pos_manager.compute_size(inst, "sell", stop_px=short_stop)
                                                if pm_short is not None and pm_short > 0:
                                                    used_size = pm_short
                                        except Exception:
                                            pass
                                        # 透传止损参数（若未禁用）
                                        if (not disable_sl) and (short_stop is not None):
                                            try:
                                                meta["slTriggerPx"] = float(short_stop)
                                            except Exception:
                                                pass
                                    else:
                                        logger.info("[跳过下单] {} 长仅模式：空仓 SELL 不开空", inst)
                                        continue
                                else:
                                    ts_side = "close"
                            else:
                                ts_side = side.lower()

                            # SELL短冷却：避免抖动连发（秒级）
                            if side == "SELL" and self.sell_cooldown_sec > 0:
                                try:
                                    last_sell_ts = float(self._last_sell_ts.get(inst, 0.0))
                                except Exception:
                                    last_sell_ts = 0.0
                                if last_sell_ts and (time.time() - last_sell_ts) < self.sell_cooldown_sec:
                                    remain = max(0.0, self.sell_cooldown_sec - (time.time() - last_sell_ts))
                                    logger.info("[SELL冷却中] {} 信号={}，剩余 {:.1f}s 跳过下单", inst, side, remain)
                                    continue

                            # 同bar去重（细化）：全平严格去重；部分平仅对同因由去重
                            if side == "SELL" and sell_close_reason and ts is not None:
                                # 判断是否部分平仓
                                is_partial = False
                                try:
                                    frac = float(size_frac) if size_frac is not None else None
                                    is_partial = (frac is not None) and (0.0 < frac < 0.999)
                                except Exception:
                                    is_partial = False

                                if not is_partial:
                                    # 全平：若该bar已有成功平仓，则跳过重复平仓信号
                                    try:
                                        last_exit_ts = self._last_exit_bar_ts.get(inst)
                                    except Exception:
                                        last_exit_ts = None
                                    same_bar_exit = False
                                    try:
                                        same_bar_exit = bool(last_exit_ts == ts)
                                    except Exception:
                                        same_bar_exit = False
                                    if same_bar_exit:
                                        ts_str = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)
                                        logger.info("[同一bar全平去重] {} ts={} 已完成一次平仓，跳过重复平仓", inst, ts_str)
                                        continue
                                else:
                                    # 部分平：按原因去重（允许同bar不同原因的级联部分止盈）
                                    try:
                                        per_inst = self._last_partial_exit_reason_ts.get(inst, {})
                                    except Exception:
                                        per_inst = {}
                                    last_ts_same_reason = per_inst.get(str(reason or ""))
                                    same_bar_partial = False
                                    try:
                                        same_bar_partial = bool(last_ts_same_reason == ts)
                                    except Exception:
                                        same_bar_partial = False
                                    if same_bar_partial:
                                        ts_str = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)
                                        logger.info("[同一bar部分平去重] {} ts={} reason={} 已执行，跳过重复", inst, ts_str, reason)
                                        continue

                            trade_sig = TradeSignal(
                                ts=ts, symbol=inst, side=ts_side, price=None,
                                size=used_size, reason=reason,
                                meta=meta,
                            )
                            res = self.executor.execute(trade_sig)
                            if res.ok:
                                logger.success("[下单成功] {} {} size={} order_id={}", inst, side, used_size, res.order_id)
                                # 成功后记录冷却时间戳
                                self._last_trade_ts[inst] = time.time()
                                # 若为基础信号，记录本bar已处理，防重复
                                try:
                                    if reason == "multi_indicator" and ts is not None:
                                        self._last_signal_bar_ts[inst] = ts
                                except Exception:
                                    pass
                                # 记录SELL冷却与同bar平仓完成（细化：全平 vs 部分平）
                                try:
                                    if side == "SELL":
                                        # SELL 成功：更新秒级冷却
                                        try:
                                            self._last_sell_ts[inst] = time.time()
                                        except Exception:
                                            pass
                                        # 风控/平仓信号：按部分/全平分别记录去重状态
                                        if sell_close_reason and ts is not None:
                                            # 判断是否部分平仓
                                            is_partial = False
                                            try:
                                                frac = float(size_frac) if size_frac is not None else None
                                                is_partial = (frac is not None) and (0.0 < frac < 0.999)
                                            except Exception:
                                                is_partial = False
                                            if is_partial:
                                                # 记录该因由在本bar已部分平仓
                                                try:
                                                    per_inst = self._last_partial_exit_reason_ts.get(inst)
                                                    if per_inst is None:
                                                        per_inst = {}
                                                    per_inst[str(reason or "")] = ts
                                                    self._last_partial_exit_reason_ts[inst] = per_inst
                                                except Exception:
                                                    pass
                                            else:
                                                # 全平：记录本bar已全平
                                                self._last_exit_bar_ts[inst] = ts
                                except Exception:
                                    pass
                                # 更新数据库持仓与盈亏记录（替代虚拟持仓）
                                try:
                                    # 优先尝试获取真实成交均价（仅在实盘且拿到 order_id 时）
                                    fill_px = None
                                    try:
                                        if self.executor.mode == "real" and getattr(self.executor, "client", None) is not None and res.order_id:
                                            ord = self.executor.client.get_order(inst_id=inst, ord_id=res.order_id)
                                            if getattr(ord, "ok", False) and isinstance(ord.data, list) and len(ord.data) > 0:
                                                avg_px_str = ord.data[0].get("avgPx") or ord.data[0].get("px")
                                                if avg_px_str is not None:
                                                    fill_px = float(avg_px_str)
                                    except Exception:
                                        pass
                                    # 回退：最近成交价
                                    last_px = None
                                    try:
                                        last_px = self.executor.get_last_price(inst)
                                    except Exception:
                                        pass
                                    ref_px = fill_px if fill_px is not None else last_px
                                    new_frac = None
                                    if side == "BUY":
                                        new_frac = 1.0
                                        self.db.insert_strategy_position({
                                            "ts": ts,
                                            "inst_id": inst,
                                            "pos_frac": new_frac,
                                            "avg_entry_px": ref_px,
                                            "source": "strategy",
                                            "extra": {"order_id": res.order_id, "reason": reason}
                                        })
                                    else:
                                        r = str(reason or "")
                                        try:
                                            frac = float(size_frac) if size_frac is not None else None
                                        except Exception:
                                            frac = None
                                        cur_row = self.db.fetch_latest_position_frac(inst)
                                        cur_frac = float(cur_row["pos_frac"]) if (cur_row and cur_row.get("pos_frac") is not None) else 1.0
                                        avg_entry = float(cur_row["avg_entry_px"]) if (cur_row and cur_row.get("avg_entry_px") is not None) else None
                                        pnl_quote = None
                                        pnl_pct = None
                                        exit_px = ref_px
                                        if avg_entry and exit_px:
                                            try:
                                                eff_frac = frac if (frac is not None and 0.0 < frac < 1.0) else cur_frac
                                                pnl_pct = (exit_px - avg_entry) / avg_entry
                                                if used_size is not None and isinstance(used_size, (int, float)):
                                                    pnl_quote = float(used_size) * (exit_px - avg_entry)
                                            except Exception:
                                                pass
                                        if r.startswith("takeprofit") or r.startswith("stoploss"):
                                            if frac is not None and 0.0 < frac < 1.0:
                                                new_frac = max(0.0, cur_frac * (1.0 - frac))
                                            else:
                                                new_frac = 0.0
                                        else:
                                            new_frac = 0.0
                                        self.db.insert_strategy_position({
                                            "ts": ts, "inst_id": inst,
                                            "pos_frac": new_frac,
                                            "avg_entry_px": (avg_entry if new_frac > 0 else None),
                                            "source": "strategy",
                                            "extra": {"order_id": res.order_id, "reason": reason}
                                        })
                                        self.db.insert_pnl_event({
                                            "ts": ts, "inst_id": inst, "side": "SELL",
                                            "qty": used_size, "entry_px": avg_entry, "exit_px": exit_px,
                                            "pnl_quote": pnl_quote, "pnl_pct": pnl_pct,
                                            "order_id": res.order_id, "reason": reason,
                                            "approx": (fill_px is None), "source": "strategy", "extra": {"sizeFrac": size_frac}
                                        })
                                    logger.info("[持仓快照] {} pos_frac={}", inst, new_frac)
                                except Exception:
                                    logger.exception("更新数据库持仓/盈亏失败：{} {}", inst, reason)
                            else:
                                logger.error("[下单失败] {} {} size={} err={}", inst, side, used_size, res.err)
                        else:
                            logger.debug("[策略] {} 信号=HOLD", inst)
                    except Exception as e:
                        logger.exception("策略处理异常（{}）：{}", inst, e)
                # 控制节奏
                elapsed = time.time() - loop_start
                await asyncio.sleep(max(0.0, poll_sec - elapsed))
        except asyncio.CancelledError:
            logger.info("策略循环被取消，准备退出...")
        except Exception as e:
            logger.exception("策略循环出现未捕获异常：{}", e)
        finally:
            logger.info("策略循环退出。")


async def main_async() -> None:
    cfg = AppConfig()
    orch = SystemOrchestrator(cfg)

    # Windows 上对 SIGTERM 支持有限，这里主要依赖 KeyboardInterrupt；
    # 对于类 Unix 系统，可尝试注册信号处理优雅退出。
    stop_called = False

    def _notify_stop(signame: str) -> None:
        nonlocal stop_called
        if stop_called:
            return
        stop_called = True
        logger.info("收到信号 {}，发起停止...", signame)
        # 触发 orchestrator 停止
        asyncio.create_task(orch.stop())

    try:
        # 注册信号（Unix 有效）
        try:
            loop = asyncio.get_running_loop()
            for sig in (getattr(signal, "SIGINT", None), getattr(signal, "SIGTERM", None)):
                if sig is not None:
                    loop.add_signal_handler(sig, _notify_stop, sig.name)
        except NotImplementedError:
            # Windows 或事件循环不支持时忽略
            pass

        await orch.start()
    except KeyboardInterrupt:
        _notify_stop("KeyboardInterrupt")
        await orch.stop()
    finally:
        logger.info("主程序退出。")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()