# -*- coding: utf-8 -*-
"""
实时监控与告警模块（终端版）
- 展示：BTC/ETH 等交易对的当前价格、盘口（顶档）、策略信号
- 告警：价格异常波动、盘口数据陈旧（疑似 WS 断开/采集异常）、交易失败
- 通道：终端打印 + 可选 Telegram/Slack 推送

运行示例：
    python -m utils.monitor

环境变量（也可写入 .env）：
    TG_ENABLED=1
    TG_BOT_TOKEN=xxxx
    TG_CHAT_ID=123456
    SLACK_ENABLED=1
    SLACK_BOT_TOKEN=xoxb-...
    SLACK_CHANNEL=#alerts

注意：
- 价格来源：trades 表最新一条记录；盘口来源：orderbook 表最新一条记录
- 策略信号：调用 strategies.ma_breakout.MABreakoutStrategy 的 compute_signal（DB 模式）
- 如需 Web UI，可将 AppConfig.monitor.use_web_ui 置为 True，当前占位（后续可接 FastAPI + SSE）
"""
from __future__ import annotations

import os
import csv
import time
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional, List, Tuple

from loguru import logger

from utils.config import AppConfig
from utils.db import TimescaleDB
from strategies.ma_breakout import MABreakoutStrategy, MABreakoutConfig

# 可选：Telegram/Slack SDK（按需导入，避免硬依赖）
try:
    from telegram import Bot  # python-telegram-bot>=21
except Exception:  # pragma: no cover
    Bot = None  # type: ignore

try:
    from slack_sdk import WebClient
    from slack_sdk.errors import SlackApiError
except Exception:  # pragma: no cover
    WebClient = None  # type: ignore
    SlackApiError = Exception  # type: ignore


@dataclass
class Ticker:
    inst: str
    price: Optional[float]
    prev_price: Optional[float]
    pct: Optional[float]


@dataclass
class TopBook:
    inst: str
    ts: Optional[datetime]
    bid_px: Optional[float]
    bid_sz: Optional[float]
    ask_px: Optional[float]
    ask_sz: Optional[float]
    spread_bps: Optional[float]


@dataclass
class StratSig:
    inst: str
    ts: Optional[datetime]
    signal: str


class Notifier:
    """统一的告警发送器（Telegram / Slack）"""

    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg
        self._tg_bot = None
        self._slack_client = None
        if cfg.telegram.enabled and Bot is not None and cfg.telegram.bot_token:
            try:
                self._tg_bot = Bot(token=cfg.telegram.bot_token)
                logger.info("Telegram 通道已启用")
            except Exception as e:
                logger.warning("Telegram 初始化失败：{}", e)
        if cfg.slack.enabled and WebClient is not None and cfg.slack.bot_token:
            try:
                self._slack_client = WebClient(token=cfg.slack.bot_token)
                logger.info("Slack 通道已启用")
            except Exception as e:
                logger.warning("Slack 初始化失败：{}", e)

    def send(self, text: str) -> None:
        """多通道并行发送告警，失败不抛出。"""
        logger.warning("[ALERT] {}", text)
        # Telegram
        if self._tg_bot and self.cfg.telegram.chat_id:
            try:
                self._tg_bot.send_message(chat_id=self.cfg.telegram.chat_id, text=text)
            except Exception as e:
                logger.debug("Telegram 发送失败：{}", e)
        # Slack
        if self._slack_client and self.cfg.slack.channel:
            try:
                self._slack_client.chat_postMessage(channel=self.cfg.slack.channel, text=text)
            except Exception as e:  # isinstance(e, SlackApiError) 亦可
                logger.debug("Slack 发送失败：{}", e)


class RealTimeMonitor:
    """实时监控器（终端输出 + 告警）"""

    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg
        self.db = TimescaleDB()  # 直接使用底层连接，避免 DF 开销
        self.notifier = Notifier(cfg)
        # 每个交易对维护历史价格，用于跳变检测
        self._last_prices: Dict[str, Optional[float]] = {}
        # 告警限频：记录同类 key 的上一次发送时间
        self._last_alert_time: Dict[str, float] = {}
        # 策略对象缓存（每个 inst 一个）
        # self._strategies: Dict[str, DepthImbalanceStrategy] = {}
        self._ma_strategies: Dict[str, MABreakoutStrategy] = {}

    # ---------------------
    # DB 基础工具
    # ---------------------
    def _ensure_connected(self) -> None:
        if self.db.conn is None:
            self.db.connect()

    def _latest_trade_price(self, inst: str) -> Optional[float]:
        """读取 trades 表最新成交价"""
        self._ensure_connected()
        assert self.db.conn is not None
        sql = (
            "SELECT price::float8 AS price FROM trades "
            "WHERE inst_id=%(inst)s ORDER BY ts DESC LIMIT 1"
        )
        with self.db.conn.cursor() as cur:
            cur.execute(sql, {"inst": inst})
            row = cur.fetchone()
        if not row:
            return None
        try:
            return float(row["price"])  # type: ignore
        except Exception:
            return None

    def _latest_orderbook(self, inst: str) -> Optional[Tuple[datetime, List, List]]:
        """读取 orderbook 表最新盘口快照"""
        self._ensure_connected()
        assert self.db.conn is not None
        sql = (
            "SELECT ts, bids, asks FROM orderbook "
            "WHERE inst_id=%(inst)s ORDER BY ts DESC LIMIT 1"
        )
        with self.db.conn.cursor() as cur:
            cur.execute(sql, {"inst": inst})
            row = cur.fetchone()
        if not row:
            return None
        return row["ts"], row["bids"], row["asks"]

    # ---------------------
    # 计算展示用数据
    # ---------------------
    def _build_ticker(self, inst: str) -> Ticker:
        price = self._latest_trade_price(inst)
        prev = self._last_prices.get(inst)
        pct = None
        if price is not None and prev not in (None, 0):
            pct = (price - prev) / prev
        # 更新 prev
        self._last_prices[inst] = price if price is not None else prev
        return Ticker(inst=inst, price=price, prev_price=prev, pct=pct)

    @staticmethod
    def _top_of_book(bids: List, asks: List) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float], Optional[float]]:
        """返回顶档与点差 bps: (bid_px, bid_sz, ask_px, ask_sz, spread_bps)"""
        try:
            bid_px = float(bids[0][0]) if bids else None
            bid_sz = float(bids[0][1]) if bids else None
            ask_px = float(asks[0][0]) if asks else None
            ask_sz = float(asks[0][1]) if asks else None
            spread_bps = None
            if bid_px and ask_px and ask_px > 0:
                spread_bps = (ask_px - bid_px) / ask_px * 1e4
            return bid_px, bid_sz, ask_px, ask_sz, spread_bps
        except Exception:
            return None, None, None, None, None

    def _build_topbook(self, inst: str) -> TopBook:
        ob = self._latest_orderbook(inst)
        if not ob:
            return TopBook(inst=inst, ts=None, bid_px=None, bid_sz=None, ask_px=None, ask_sz=None, spread_bps=None)
        ts, bids, asks = ob
        bid_px, bid_sz, ask_px, ask_sz, spread_bps = self._top_of_book(bids, asks)
        return TopBook(inst=inst, ts=ts, bid_px=bid_px, bid_sz=bid_sz, ask_px=ask_px, ask_sz=ask_sz, spread_bps=spread_bps)

    def _get_strategy(self, inst: str) -> MABreakoutStrategy:
        if inst not in self._ma_strategies:
            cfg = MABreakoutConfig(
                inst_id=inst,
                timeframe=os.getenv("MA_TIMEFRAME", "5min"),
                fast_ma=int(os.getenv("MA_FAST", "10")),
                slow_ma=int(os.getenv("MA_SLOW", "30")),
                breakout_lookback=int(os.getenv("MA_BREAKOUT_N", "20")),
                breakout_buffer_pct=float(os.getenv("MA_BREAKOUT_BUFFER", "0.0")),
                lookback_bars=int(os.getenv("MA_LOOKBACK_BARS", "300")),
                prefer_trades_price=(os.getenv("MA_PREFER_TRADES", "1") == "1"),
                poll_sec=float(os.getenv("MA_POLL_SEC", "60.0")),
                order_type=os.getenv("MA_ORDER_TYPE", "market").lower(),
                stop_loss_pct=(float(os.getenv("MA_STOP_LOSS_PCT", "0.005")) if os.getenv("MA_STOP_LOSS_PCT", "").strip() != "" else None),
                # —— 新增：多指标参数（可通过环境变量覆盖）——
                macd_fast=int(os.getenv("MA_MACD_FAST", os.getenv("MA_FAST", "12"))),
                macd_slow=int(os.getenv("MA_MACD_SLOW", os.getenv("MA_SLOW", "26"))),
                macd_signal=int(os.getenv("MA_MACD_SIGNAL", "9")),
                bb_period=int(os.getenv("MA_BB_PERIOD", os.getenv("MA_BREAKOUT_N", "20"))),
                bb_k=float(os.getenv("MA_BB_K", "2.0")),
                rsi_period=int(os.getenv("MA_RSI_PERIOD", "14")),
                rsi_buy=float(os.getenv("MA_RSI_BUY", "55")),
                rsi_sell=float(os.getenv("MA_RSI_SELL", "45")),
                aroon_period=int(os.getenv("MA_AROON_PERIOD", "25")),
                aroon_buy=float(os.getenv("MA_AROON_BUY", "70")),
                aroon_sell=float(os.getenv("MA_AROON_SELL", "30")),
                confirm_min=int(os.getenv("MA_CONFIRM_MIN", "3")),
                # —— 新增：止损参数（可通过环境变量覆盖）——
                atr_n=float(os.getenv("MA_ATR_N", "2.0")),  # ATR 倍数 N（止损价 = 入场价 - N * ATR）
                stop_loss_pct_btc=float(os.getenv("MA_STOP_LOSS_PCT_BTC", "0.03")),  # BTC 固定百分比止损
                stop_loss_pct_eth=float(os.getenv("MA_STOP_LOSS_PCT_ETH", "0.04")),  # ETH 固定百分比止损
                # —— 新增：止盈参数（可通过环境变量覆盖）——
                tp_r1=float(os.getenv("MA_TP_R1", "1.0")),             # 第一级 R 倍数（达到即部分止盈）
                tp_r2=float(os.getenv("MA_TP_R2", "2.0")),             # 第二级 R 倍数（达到即再次部分止盈）
                tp_frac1=float(os.getenv("MA_TP_FRAC1", "0.3")),       # 第一级止盈平掉比例
                tp_frac2=float(os.getenv("MA_TP_FRAC2", "0.3")),       # 第二级止盈再平掉比例
                tp_trail_atr_mult=float(os.getenv("MA_TP_TRAIL_ATR", "1.5")),  # 移动止损 ATR 倍数
                rsi_tp_high=float(os.getenv("MA_RSI_TP_HIGH", "80")),   # RSI 技术止盈上阈值
                rsi_tp_low=float(os.getenv("MA_RSI_TP_LOW", "20")),     # RSI 技术止盈下阈值
                rsi_tp_frac=float(os.getenv("MA_RSI_TP_FRAC", "0.2")),  # RSI 技术止盈建议比例
            )
            self._ma_strategies[inst] = MABreakoutStrategy(cfg)
        return self._ma_strategies[inst]

    def _build_signal(self, inst: str) -> StratSig:
        try:
            sig = self._get_strategy(inst).compute_signal()
            if not sig:
                return StratSig(inst=inst, ts=None, signal="NA")
            return StratSig(
                inst=inst,
                ts=sig.get("ts"),
                signal=str(sig.get("signal", "NA")),
            )
        except Exception as e:
            logger.debug("策略信号获取失败：{}", e)
            return StratSig(inst=inst, ts=None, signal="ERR")

    # ---------------------
    # 告警逻辑
    # ---------------------
    def _should_rate_limit(self, key: str) -> bool:
        now = time.time()
        last = self._last_alert_time.get(key, 0)
        if now - last < max(1, self.cfg.alerts.rate_limit_sec):
            return True
        self._last_alert_time[key] = now
        return False

    def _check_price_jump(self, tkr: Ticker) -> None:
        th = self.cfg.alerts.price_jump_pct
        if tkr.price is None or tkr.prev_price in (None, 0):
            return
        pct = (tkr.price - tkr.prev_price) / tkr.prev_price
        if abs(pct) >= th:
            key = f"jump:{tkr.inst}"
            if self._should_rate_limit(key):
                return
            direction = "上涨" if pct > 0 else "下跌"
            self.notifier.send(
                f"[价格跳变] {tkr.inst} {direction} {pct*100:.2f}% -> 最新价 {tkr.price:.2f}（阈值 {th*100:.2f}%）"
            )

    def _check_stale_orderbook(self, ob: TopBook) -> None:
        if ob.ts is None:
            return
        age = (datetime.now(timezone.utc) - ob.ts).total_seconds()
        if age >= self.cfg.alerts.stale_book_sec:
            key = f"stale:{ob.inst}"
            if self._should_rate_limit(key):
                return
            self.notifier.send(
                f"[盘口延迟] {ob.inst} 快照滞后 {age:.1f}s（阈值 {self.cfg.alerts.stale_book_sec}s），疑似采集中断/网络异常"
            )

    def _check_trade_failures(self) -> None:
        """检测最近窗口内的交易失败（读取 data/trade_log.csv）"""
        path = os.path.join("data", "trade_log.csv")
        if not os.path.exists(path):
            return
        window = max(5, self.cfg.alerts.trade_fail_window_sec)
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=window)
        cnt = 0
        try:
            with open(path, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        ts = datetime.fromisoformat(row.get("ts", "").replace("Z", "+00:00"))
                    except Exception:
                        continue
                    if ts >= cutoff and row.get("ok") in ("False", "false", False):
                        cnt += 1
        except Exception as e:
            logger.debug("读取交易日志失败：{}", e)
            return
        if cnt > 0:
            key = "tradefail"
            if self._should_rate_limit(key):
                return
            self.notifier.send(f"[交易失败] 最近 {window}s 内出现 {cnt} 次下单失败，请检查风控/余额/API 限流等问题")

    # ---------------------
    # 输出与主循环
    # ---------------------
    @staticmethod
    def _fmt_none(x: Optional[float], nd: int = 2) -> str:
        return (f"{x:.{nd}f}" if isinstance(x, (int, float)) else "-")

    def _print_table(self, rows: List[Tuple[str, Ticker, TopBook, StratSig]]) -> None:
        headers = [
            "Inst", "Price", "Δ%", "Bid", "Ask", "Spr(bps)", "Sig", "SigTS"
        ]
        line = " | ".join([
            f"{h:>10}" for h in headers
        ])
        print("\n" + line)
        print("-" * len(line))
        for inst, tkr, ob, sig in rows:
            price = self._fmt_none(tkr.price)
            pct = (f"{tkr.pct*100:.2f}%" if tkr.pct is not None else "-")
            bid = self._fmt_none(ob.bid_px)
            ask = self._fmt_none(ob.ask_px)
            spr = self._fmt_none(ob.spread_bps, nd=1)
            sig_str = sig.signal
            sig_ts = sig.ts.isoformat() if sig.ts else "-"
            print(" | ".join([
                f"{inst:>10}", f"{price:>10}", f"{pct:>10}", f"{bid:>10}", f"{ask:>10}", f"{spr:>10}", f"{sig_str:>10}", f"{sig_ts:>20}"
            ]))

    def run(self) -> None:
        if self.cfg.monitor.use_web_ui:
            logger.warning("Web UI 功能尚未启用（预留接口）。当前使用终端输出模式。")
        # 监控标的：优先取 monitor.instruments，否则沿用 WS 配置
        insts = [s.strip() for s in (self.cfg.monitor.instruments or self.cfg.ws.instruments)]
        logger.info("启动监控，标的：{}", insts)
        self._ensure_connected()
        try:
            while True:
                rows = []
                for inst in insts:
                    tkr = self._build_ticker(inst)
                    ob = self._build_topbook(inst)
                    sig = self._build_signal(inst)

                    # 告警检查
                    if self.cfg.alerts.enabled:
                        self._check_price_jump(tkr)
                        self._check_stale_orderbook(ob)

                    rows.append((inst, tkr, ob, sig))

                # 扫描交易失败
                if self.cfg.alerts.enabled:
                    self._check_trade_failures()

                # 输出终端表格
                self._print_table(rows)

                time.sleep(max(0.2, self.cfg.monitor.poll_interval_sec))
        except KeyboardInterrupt:
            logger.info("收到中断信号，退出监控...")
        finally:
            self.db.close()


def main() -> None:
    cfg = AppConfig()
    monitor = RealTimeMonitor(cfg)
    monitor.run()


if __name__ == "__main__":
    main()