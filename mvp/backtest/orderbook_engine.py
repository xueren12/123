"""
订单簿级回测引擎（使用 trades + orderbook 做撮合仿真）

核心能力：
- 使用数据库中的盘口快照（orderbook）与成交（trades）作为回放源；
- 支持两类执行方式：
  1) 市价单（Market）：按当前盘口逐档吃深度，支持部分成交（深度不够则剩余部分放弃或下一时刻再尝试）；
  2) 限价单（Limit）：在延迟后挂在指定价格，基于盘口“被对手方穿越”时撮合，支持部分成交；
- 支持参数化的滑点（额外基点）与手续费（基点）；
- 进一步下单延迟（ms 级），下单生效在延迟后的首个盘口快照；
- 计算逐时刻权益、收益、夏普、最大回撤；
- 导出成交明细 CSV（包含每次成交的价格、数量、费用、VWAP 等），可选导出收益曲线 SVG。

使用示例（命令行）：
    python -m backtest.orderbook_engine \
        --inst BTC-USDT \
        --start "2024-09-01T00:00:00Z" \
        --end   "2024-09-01T06:00:00Z" \
        --timeframe 1s \
        --levels 5 --mode notional --threshold 0.02 \
        --exec market --trade-units 0.01 \
        --latency-ms 50 --fee-bps 1.0 --slip-extra-bps 0.5 \
        --initial-capital 10000 \
        --fills-csv data/fills_obbt.csv \
        --report-csv data/report_obbt.csv \
        --plot-svg data/equity_obbt.svg

说明：
- 回测中“信号”仍复用深度不平衡思想：BUY -> 目标多头，SELL -> 目标空头，HOLD -> 维持；
- exec=market 时，直接用市价单将当前持仓调到目标持仓；
- exec=limit 时，按 bestBid/bestAsk 附近挂单（可配置偏移 bp），被穿越时成交，支持部分成交；
- 价格标记（mark）默认使用中间价 mid，可使用 --prefer-trades-price 切换为成交价 close；
- 手续费与额外滑点按基点（bps）计提：1bp = 0.01%。

依赖：pandas / numpy / matplotlib / loguru 均已在 requirements.txt 中列出。
"""
from __future__ import annotations

import argparse
import math
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

# 非交互式后端，便于服务器/CI 上生成图表文件
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.db import TimescaleDB


def _safe_makedirs(path: Optional[str]) -> None:
    """安全创建目录：当传入路径为空或为当前目录时跳过。"""
    if not path:
        return
    try:
        os.makedirs(path, exist_ok=True)
    except Exception:
        # 若路径不可创建则忽略，由上层在写文件时报错即可
        pass

# =====================
# 配置与数据结构
# =====================
@dataclass
class OBBTConfig:
    """订单簿级回测参数配置。"""
    inst_id: str = "BTC-USDT"                    # 回测标的
    start: str = "2024-09-01T00:00:00Z"           # 起始时间（UTC）
    end: str = "2024-09-01T06:00:00Z"             # 结束时间（UTC）
    timeframe: str = "1s"                         # 信号采样粒度（影响信号生成与(mark)序列重采样）

    # 信号生成（与 backtest.engine 一致）
    levels: int = 5
    mode: str = "notional"                        # "size" 或 "notional"
    threshold: float = 0.02                        # 深度不平衡阈值

    # 执行配置
    exec: str = "market"                          # "market" 或 "limit"
    trade_units: float = 0.01                      # 信号切换时的目标头寸幅度（以标的数量计）
    limit_offset_bps: float = 0.0                  # 限价相对 bestBid/Ask 的偏移（正数向更保守方向）
    latency_ms: int = 50                           # 下单延迟（毫秒），订单在延迟后的首个盘口快照被处理

    # 成本模型（基点）
    fee_bps: float = 1.0                           # 手续费（单边）
    slip_extra_bps: float = 0.5                    # 额外滑点（在 VWAP 基础上再加/减）

    # 资金与输出
    initial_capital: float = 10000.0
    prefer_trades_price: bool = False              # 标记价格是否使用成交收盘价（否则用 mid）
    fills_csv: Optional[str] = None                # 成交明细 CSV
    report_csv: Optional[str] = None               # 权益/指标 CSV
    plot_svg: Optional[str] = None                 # 收益曲线 SVG
    # 新增：与实盘对齐的约束与外部驱动
    sell_cooldown_sec: float = 0.0                 # 卖出冷却秒数
    actions_csv: Optional[str] = None              # 外部动作驱动CSV（来自 multi_indicator 的 trade_log.csv）
    trade_log_csv: Optional[str] = None            # 输出的交易日志CSV（执行侧实际成交）


@dataclass
class Order:
    """模拟订单对象。仅维护本回测需要的最小字段。"""
    order_id: str
    side: str                 # "BUY" 或 "SELL"
    type: str                 # "market" 或 "limit"
    qty: float                # 总下单数量（正数）
    price: Optional[float]    # 限价价格（market 时为 None）
    placed_ts: datetime       # 订单生效（放入簿）的时间（考虑延迟后）
    remaining: float          # 剩余未成交数量
    # 新增：动作元数据，便于统一日志输出
    reason: Optional[str] = None
    size_frac: Optional[float] = None


@dataclass
class Fill:
    """成交明细记录。"""
    ts: datetime
    order_id: str
    side: str
    qty: float
    price: float
    fee: float
    type: str                 # market/limit
    vwap: float               # 与 price 相同，这里保留字段以兼容未来多笔汇总


# =====================
# 核心引擎
# =====================
class OrderBookBacktester:
    """订单簿级回测引擎：生成信号 -> 调仓意图 -> 市价/限价撮合 -> 计算权益。"""

    def __init__(self, cfg: OBBTConfig) -> None:
        self.cfg = cfg
        self.db = TimescaleDB()
        # 运行态
        self.cash: float = float(cfg.initial_capital)
        self.pos: float = 0.0
        self.active_limit: Optional[Order] = None
        self.scheduled: List[Order] = []  # 已计划但尚未到达 placed_ts 的订单
        self.fills: List[Fill] = []

    # ---------------------
    # 工具：时间/频率/盘口
    # ---------------------
    @staticmethod
    def _to_utc(ts: str) -> datetime:
        dt = pd.to_datetime(ts, utc=True)
        if dt.tzinfo is None:
            return dt.tz_localize(timezone.utc)
        return dt.tz_convert(timezone.utc)

    @staticmethod
    def _parse_timeframe_to_freq(tf: str) -> Tuple[str, float]:
        s_match = re.fullmatch(r"(\d+)s", tf)
        m_match = re.fullmatch(r"(\d+)(min|m)", tf)
        if s_match:
            secs = max(1, int(s_match.group(1)))
            return f"{secs}S", (365 * 24 * 3600) / secs
        if m_match:
            mins = max(1, int(m_match.group(1)))
            return f"{mins}T", (365 * 24 * 60) / mins
        return "1S", 365 * 24 * 3600

    @staticmethod
    def _sum_depth(levels: int, side, mode: str) -> float:
        total = 0.0
        if not isinstance(side, (list, tuple)):
            return 0.0
        for lvl in list(side)[: max(levels, 0)]:
            if not isinstance(lvl, (list, tuple)) or len(lvl) < 2:
                continue
            try:
                px = float(lvl[0])
                sz = float(lvl[1])
            except Exception:
                continue
            total += (sz if mode == "size" else px * sz)
        return total

    def _compute_signal_row(self, bids, asks) -> Tuple[str, float, float, float, float, float]:
        buy_depth = self._sum_depth(self.cfg.levels, bids, self.cfg.mode)
        sell_depth = self._sum_depth(self.cfg.levels, asks, self.cfg.mode)
        up_bound = sell_depth * (1.0 + self.cfg.threshold)
        down_bound = buy_depth * (1.0 + self.cfg.threshold)
        if buy_depth > up_bound:
            sig = "BUY"
        elif sell_depth > down_bound:
            sig = "SELL"
        else:
            sig = "HOLD"
        try:
            best_bid = float(bids[0][0]) if bids else np.nan
            best_ask = float(asks[0][0]) if asks else np.nan
            mid = np.nanmean([best_bid, best_ask])
        except Exception:
            best_bid = np.nan
            best_ask = np.nan
            mid = np.nan
        return sig, buy_depth, sell_depth, float(mid) if not math.isnan(mid) else np.nan, float(best_bid), float(best_ask)

    # ---------------------
    # 数据加载
    # ---------------------
    def _load_data(self) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        start = self._to_utc(self.cfg.start)
        end = self._to_utc(self.cfg.end)
        freq, _ = self._parse_timeframe_to_freq(self.cfg.timeframe)

        logger.info("读取 orderbook: {} {} -> {}", self.cfg.inst_id, start, end)
        ob_raw = self.db.fetch_orderbook_window(start, end, inst_id=self.cfg.inst_id, ascending=True)
        if ob_raw.empty:
            raise RuntimeError("orderbook 数据为空")
        # 逐行计算信号与基础价格
        ob_raw["signal"], ob_raw["buy_depth"], ob_raw["sell_depth"], ob_raw["mid"], ob_raw["best_bid"], ob_raw["best_ask"] = zip(
            *ob_raw.apply(lambda r: self._compute_signal_row(r["bids"], r["asks"]), axis=1)
        )
        ob_raw["ts"] = pd.to_datetime(ob_raw["ts"], utc=True)
        ob_df = ob_raw.set_index("ts").sort_index()
        # 对信号与价格按 timeframe 取每段末值，用作“目标持仓”驱动与标记价格序列
        ob_res = ob_df.resample(freq).last().dropna(subset=["signal", "mid"], how="any")

        tr_res: Optional[pd.DataFrame] = None
        if self.cfg.prefer_trades_price:
            tr_raw = self.db.fetch_trades_window(start, end, inst_id=self.cfg.inst_id, ascending=True)
            if not tr_raw.empty:
                tr_raw["ts"] = pd.to_datetime(tr_raw["ts"], utc=True)
                tr_df = tr_raw.set_index("ts").sort_index()
                tr_res = tr_df[["price"]].resample(freq).last().rename(columns={"price": "close"})
                tr_res["close"] = tr_res["close"].ffill()
        return ob_res, tr_res

    # ---------------------
    # 撮合函数
    # ---------------------
    @staticmethod
    def _normalize_book(bids, asks) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
        """将 bids/asks 规范为 [(price, size), ...] 并按价格排序（bids 降序，asks 升序）。"""
        def _norm(side, reverse: bool) -> List[Tuple[float, float]]:
            rows: List[Tuple[float, float]] = []
            if isinstance(side, (list, tuple)):
                for lvl in side:
                    if not isinstance(lvl, (list, tuple)) or len(lvl) < 2:
                        continue
                    try:
                        px = float(lvl[0]); sz = float(lvl[1])
                        if sz > 0 and math.isfinite(px):
                            rows.append((px, sz))
                    except Exception:
                        continue
            rows.sort(key=lambda x: x[0], reverse=reverse)
            return rows
        return _norm(bids, True), _norm(asks, False)

    def _consume_market(self, bids, asks, side: str, qty: float) -> Tuple[float, float, float]:
        """市价吃单：返回 (executed_qty, vwap_price, notional)。支持部分成交。"""
        bids_n, asks_n = self._normalize_book(bids, asks)
        remaining = qty
        notional = 0.0
        executed = 0.0
        if side == "BUY":
            for px, sz in asks_n:
                if remaining <= 0:
                    break
                fill = min(sz, remaining)
                notional += fill * px
                executed += fill
                remaining -= fill
        else:  # SELL
            for px, sz in bids_n:
                if remaining <= 0:
                    break
                fill = min(sz, remaining)
                notional += fill * px
                executed += fill
                remaining -= fill
        vwap = (notional / executed) if executed > 0 else float("nan")
        return executed, vwap, notional

    def _cross_limit(self, bids, asks, order: Order) -> Tuple[float, float, float]:
        """限价单撮合：若被对手方穿越，则在可成交价位上撮合，返回 (executed_qty, vwap, notional)。"""
        assert order.type == "limit" and order.price is not None
        bids_n, asks_n = self._normalize_book(bids, asks)
        remaining = order.remaining
        notional = 0.0
        executed = 0.0
        if order.side == "BUY":
            # 对手方 ask 价格若 <= 我们的限价，即可成交
            for px, sz in asks_n:
                if px > order.price + 1e-12:
                    break
                if remaining <= 0:
                    break
                fill = min(sz, remaining)
                notional += fill * px
                executed += fill
                remaining -= fill
        else:  # SELL
            for px, sz in bids_n:
                if px < order.price - 1e-12:
                    break
                if remaining <= 0:
                    break
                fill = min(sz, remaining)
                notional += fill * px
                executed += fill
                remaining -= fill
        vwap = (notional / executed) if executed > 0 else float("nan")
        return executed, vwap, notional

    def _apply_fee_and_slip(self, side: str, vwap: float, qty: float) -> Tuple[float, float]:
        """在 VWAP 基础上叠加额外滑点，并计算手续费。返回 (exec_price, fee)。"""
        if not math.isfinite(vwap):
            return vwap, 0.0
        px = vwap * (1.0 + (self.cfg.slip_extra_bps / 10000.0) * (1 if side == "BUY" else -1))
        fee = abs(qty) * px * (self.cfg.fee_bps / 10000.0)
        return px, fee

    # ---------------------
    # 回放主循环
    # ---------------------
    def run(self) -> Dict[str, float]:
        ob_df, tr_df = self._load_data()
        freq, bars_per_year = self._parse_timeframe_to_freq(self.cfg.timeframe)

        # 信号 -> 目标持仓（+trade_units / -trade_units / 维持）
        sig_map = {"BUY": 1.0, "SELL": -1.0, "HOLD": np.nan}
        target_flag = ob_df["signal"].map(sig_map).ffill().fillna(0.0)
        target_pos = target_flag * float(self.cfg.trade_units)

        # 标记价格
        if self.cfg.prefer_trades_price and tr_df is not None and not tr_df.empty:
            mark = tr_df["close"].reindex(ob_df.index).ffill()
        else:
            mark = ob_df["mid"].ffill()

        # 时间索引
        index = ob_df.index

        # 新增：外部动作驱动映射与冷却/去重状态
        actions_map: Dict[pd.Timestamp, List[dict]] = {}
        if getattr(self.cfg, "actions_csv", None):
            try:
                a_df = pd.read_csv(self.cfg.actions_csv)
                if "ts" not in a_df.columns:
                    logger.warning("actions_csv 缺少 ts 列，忽略外部动作驱动")
                else:
                    try:
                        a_df["ts"] = pd.to_datetime(a_df["ts"], utc=True)
                    except Exception:
                        a_df["ts"] = pd.to_datetime(a_df["ts"].astype("int64"), unit="ms", utc=True)
                    # 过滤标的与时间窗
                    if "inst" in a_df.columns:
                        a_df = a_df[a_df["inst"].astype(str) == str(self.cfg.inst_id)]
                    a_df = a_df[(a_df["ts"] >= index.min()) & (a_df["ts"] <= index.max())]
                    a_df = a_df.sort_values("ts")
                    for ts_val, grp in a_df.groupby("ts"):
                        actions_map[ts_val] = grp.to_dict(orient="records")
                    logger.info("已加载外部动作CSV，共{}条（{}个时间点）", len(a_df), len(actions_map))
            except Exception as e:
                logger.warning("读取 actions_csv 失败：{}", e)
        sell_cooldown_sec = float(getattr(self.cfg, "sell_cooldown_sec", 0.0) or 0.0)
        last_sell_epoch = None
        last_exit_bar_ts: Optional[pd.Timestamp] = None
        last_partial_exit_reason_ts: Dict[str, pd.Timestamp] = {}
        exec_trade_logs: List[dict] = []

        # 下单延迟：信号发生在 bar 的结束时刻，订单在 (ts + latency) 生效
        def schedule_order(ts: pd.Timestamp, side: str, qty: float, price: Optional[float], typ: str, reason: Optional[str] = None, size_frac: Optional[float] = None) -> None:
            placed_ts = (ts.to_pydatetime().astimezone(timezone.utc) + timedelta(milliseconds=int(self.cfg.latency_ms)))
            oid = f"{typ}-{side}-{int(ts.timestamp()*1e6)}"
            self.scheduled.append(Order(oid, side, typ, abs(qty), price, placed_ts, abs(qty), reason, size_frac))

        last_target = 0.0
        equity_series: List[float] = []
        pos_series: List[float] = []

        for ts, row in ob_df.iterrows():
            ts_dt: datetime = ts.to_pydatetime().astimezone(timezone.utc)
            bids = row["bids"]; asks = row["asks"]
            best_bid = row.get("best_bid", np.nan)
            best_ask = row.get("best_ask", np.nan)

            # 1) 外部动作驱动或原有目标调仓
            if actions_map:
                acts = actions_map.get(ts, [])
                for act in acts:
                    side = str(act.get("side", "")).upper()
                    reason = str(act.get("reason")) if ("reason" in act) else None
                    try:
                        size_frac = float(act.get("sizeFrac", 1.0)) if ("sizeFrac" in act) else 1.0
                    except Exception:
                        size_frac = 1.0
                    if side not in ("BUY", "SELL"):
                        continue
                    if side == "SELL":
                        ts_epoch = ts_dt.timestamp()
                        is_partial = size_frac < 0.999
                        # 冷却限制
                        if sell_cooldown_sec > 0.0 and (last_sell_epoch is not None) and ((ts_epoch - float(last_sell_epoch)) < float(sell_cooldown_sec)):
                            continue
                        # 同bar去重
                        if is_partial:
                            if reason is not None and last_partial_exit_reason_ts.get(reason) == ts:
                                continue
                        else:
                            if last_exit_bar_ts == ts:
                                continue
                        qty = (self.pos * float(size_frac)) if is_partial else self.pos
                        if qty <= 1e-12:
                            continue
                        if self.cfg.exec == "market":
                            schedule_order(ts, "SELL", float(qty), None, "market", reason, float(size_frac))
                        else:
                            base = float(best_ask) if math.isfinite(best_ask) else float(row["mid"])  # 兜底
                            price = base * (1.0 + self.cfg.limit_offset_bps / 10000.0)
                            schedule_order(ts, "SELL", float(qty), price, "limit", reason, float(size_frac))
                    else:  # BUY
                        tgt_units = float(self.cfg.trade_units)
                        qty = max(0.0, tgt_units - self.pos)
                        if qty <= 1e-12:
                            continue
                        if self.cfg.exec == "market":
                            schedule_order(ts, "BUY", float(qty), None, "market", reason or "entry_multi_indicator", 1.0)
                        else:
                            base = float(best_bid) if math.isfinite(best_bid) else float(row["mid"])  # 兜底
                            price = base * (1.0 - self.cfg.limit_offset_bps / 10000.0)
                            schedule_order(ts, "BUY", float(qty), price, "limit", reason or "entry_multi_indicator", 1.0)
            else:
                # 原有：根据目标持仓变化调仓
                tgt = float(target_pos.loc[ts])
                if not math.isclose(tgt, last_target, rel_tol=0, abs_tol=1e-12):
                    delta = tgt - self.pos
                    if abs(delta) > 1e-12:
                        if self.cfg.exec == "market":
                            schedule_order(ts, "BUY" if delta > 0 else "SELL", abs(delta), None, "market", "engine_target_adjust", None)
                        else:
                            # 在 best 边挂单，偏移 limit_offset_bps
                            if delta > 0:  # 想加多
                                base = float(best_bid) if math.isfinite(best_bid) else float(row["mid"])  # 兜底
                                price = base * (1.0 - self.cfg.limit_offset_bps / 10000.0)
                                schedule_order(ts, "BUY", abs(delta), price, "limit", "engine_target_adjust", None)
                            else:
                                base = float(best_ask) if math.isfinite(best_ask) else float(row["mid"])  # 兜底
                                price = base * (1.0 + self.cfg.limit_offset_bps / 10000.0)
                                schedule_order(ts, "SELL", abs(delta), price, "limit", "engine_target_adjust", None)
                    last_target = tgt

            # 2) 处理“到时生效”的订单（只在首次达到其 placed_ts 的快照时进入簿）
            ready: List[Order] = []
            not_ready: List[Order] = []
            for od in self.scheduled:
                if od.placed_ts <= ts_dt:
                    ready.append(od)
                else:
                    not_ready.append(od)
            self.scheduled = not_ready

            # 市价单：在当前快照即时撮合
            for od in [o for o in ready if o.type == "market"]:
                pre_pos = self.pos
                exec_qty, vwap, notional = self._consume_market(bids, asks, od.side, od.remaining)
                if exec_qty > 0:
                    px, fee = self._apply_fee_and_slip(od.side, vwap, exec_qty)
                    # 更新资金与仓位
                    if od.side == "BUY":
                        self.cash -= exec_qty * px + fee
                        self.pos += exec_qty
                        exec_trade_logs.append({
                            "ts": ts_dt.isoformat(),
                            "inst": self.cfg.inst_id,
                            "side": "BUY",
                            "price": float(px),
                            "sizeFrac": 1.0,
                            "reason": od.reason or f"exec_{od.type}"
                        })
                    else:
                        self.cash += exec_qty * px - fee
                        self.pos -= exec_qty
                        last_sell_epoch = ts_dt.timestamp()
                        if od.size_frac is not None and od.size_frac < 0.999:
                            if od.reason:
                                last_partial_exit_reason_ts[od.reason] = ts
                        else:
                            last_exit_bar_ts = ts
                        size_f = float(exec_qty / pre_pos) if pre_pos > 1e-12 else 0.0
                        size_f = float(min(1.0, max(0.0, size_f)))
                        exec_trade_logs.append({
                            "ts": ts_dt.isoformat(),
                            "inst": self.cfg.inst_id,
                            "side": "SELL",
                            "price": float(px),
                            "sizeFrac": size_f,
                            "reason": od.reason or f"exec_{od.type}"
                        })
                    self.fills.append(Fill(ts_dt, od.order_id, od.side, exec_qty, px, fee, od.type, px))
                    od.remaining -= exec_qty
                # 市价剩余视为未能成交，直接丢弃

            # 限价单：若已有在簿订单，则检测是否被穿越；若没有且 ready 中有 limit，则取一张最新的入簿
            if self.active_limit is None:
                # 拿最后一张 ready 的 limit 入簿（若有多张，之前的可视为被用户撤销）
                ready_limits = [o for o in ready if o.type == "limit"]
                if ready_limits:
                    self.active_limit = ready_limits[-1]
            # 检测穿越并撮合
            if self.active_limit is not None:
                pre_pos = self.pos
                exec_qty, vwap, notional = self._cross_limit(bids, asks, self.active_limit)
                if exec_qty > 0:
                    px, fee = self._apply_fee_and_slip(self.active_limit.side, vwap, exec_qty)
                    if self.active_limit.side == "BUY":
                        self.cash -= exec_qty * px + fee
                        self.pos += exec_qty
                        exec_trade_logs.append({
                            "ts": ts_dt.isoformat(),
                            "inst": self.cfg.inst_id,
                            "side": "BUY",
                            "price": float(px),
                            "sizeFrac": 1.0,
                            "reason": self.active_limit.reason or f"exec_{self.active_limit.type}"
                        })
                    else:
                        self.cash += exec_qty * px - fee
                        self.pos -= exec_qty
                        last_sell_epoch = ts_dt.timestamp()
                        if self.active_limit.size_frac is not None and self.active_limit.size_frac < 0.999:
                            if self.active_limit.reason:
                                last_partial_exit_reason_ts[self.active_limit.reason] = ts
                        else:
                            last_exit_bar_ts = ts
                        size_f = float(exec_qty / pre_pos) if pre_pos > 1e-12 else 0.0
                        size_f = float(min(1.0, max(0.0, size_f)))
                        exec_trade_logs.append({
                            "ts": ts_dt.isoformat(),
                            "inst": self.cfg.inst_id,
                            "side": "SELL",
                            "price": float(px),
                            "sizeFrac": size_f,
                            "reason": self.active_limit.reason or f"exec_{self.active_limit.type}"
                        })
                    self.fills.append(Fill(ts_dt, self.active_limit.order_id, self.active_limit.side, exec_qty, px, fee, self.active_limit.type, px))
                    self.active_limit.remaining -= exec_qty
                    if self.active_limit.remaining <= 1e-12:
                        self.active_limit = None

            # 3) 记录时序权益
            mark_px = float(mark.loc[ts])
            equity = self.cash + self.pos * mark_px
            equity_series.append(equity)
            pos_series.append(self.pos)

        # 汇总指标
        equity_s = pd.Series(equity_series, index=index)
        ret = equity_s.pct_change().fillna(0.0)
        std = ret.std(ddof=0)
        sharpe = float(np.sqrt(bars_per_year) * ret.mean() / std) if std > 1e-12 else float("nan")
        roll_max = equity_s.cummax()
        drawdown = (equity_s - roll_max) / roll_max
        max_dd = float(drawdown.min()) if len(drawdown) else float("nan")

        summary = {
            "inst_id": self.cfg.inst_id,
            "start": self.cfg.start,
            "end": self.cfg.end,
            "timeframe": self.cfg.timeframe,
            "exec": self.cfg.exec,
            "latency_ms": int(self.cfg.latency_ms),
            "fee_bps": float(self.cfg.fee_bps),
            "slip_extra_bps": float(self.cfg.slip_extra_bps),
            "initial_capital": float(self.cfg.initial_capital),
            "final_equity": float(equity_s.iloc[-1]) if len(equity_s) else float("nan"),
            "return": float(equity_s.iloc[-1] / self.cfg.initial_capital - 1.0) if len(equity_s) else float("nan"),
            "sharpe": sharpe,
            "max_drawdown": max_dd,
            "bars": int(len(equity_s)),
            "fills": int(len(self.fills)),
        }

        logger.info(
            "OBBT 完成 | inst={} exec={} 样本={} 最终权益={:.2f} 收益={:.2%} 夏普={:.3f} 最大回撤={:.2%} 成交条数={}",
            summary["inst_id"], summary["exec"], summary["bars"], summary["final_equity"], summary["return"], summary["sharpe"], summary["max_drawdown"], summary["fills"],
        )

        # 导出成交明细
        if self.cfg.fills_csv:
            _safe_makedirs(os.path.dirname(self.cfg.fills_csv))
            pd.DataFrame([f.__dict__ for f in self.fills]).to_csv(self.cfg.fills_csv, index=False)
            logger.info("已导出成交明细 CSV -> {}", self.cfg.fills_csv)

        # 导出报告（权益曲线）
        if self.cfg.report_csv:
            _safe_makedirs(os.path.dirname(self.cfg.report_csv))
            report_df = pd.DataFrame({
                "equity": equity_s,
                "position": pd.Series(pos_series, index=index),
            })
            report_df.to_csv(self.cfg.report_csv, index_label="ts")
            logger.info("已导出回测报告 CSV -> {}", self.cfg.report_csv)

        # 生成收益曲线 SVG
        if self.cfg.plot_svg:
            _safe_makedirs(os.path.dirname(self.cfg.plot_svg))
            fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
            # 标记价格
            axes[0].plot(mark.index, mark.values, color="tab:blue", label="Mark")
            axes[0].set_title(f"{self.cfg.inst_id} 标记价格 ({self.cfg.timeframe})")
            axes[0].legend(loc="upper left")
            # 权益
            axes[1].plot(equity_s.index, equity_s.values, color="tab:green", label="Equity")
            axes[1].set_title("权益曲线（订单簿级执行）")
            axes[1].legend(loc="upper left")
            plt.tight_layout()
            fig.savefig(self.cfg.plot_svg, format="svg")
            plt.close(fig)
            logger.info("已生成回测图表 SVG -> {}", self.cfg.plot_svg)

        # 新增：导出统一交易日志（若提供 trade_log_csv 路径）
        if getattr(self.cfg, "trade_log_csv", None):
            try:
                _safe_makedirs(os.path.dirname(self.cfg.trade_log_csv))
                pd.DataFrame(exec_trade_logs).to_csv(self.cfg.trade_log_csv, index=False)
                logger.info("已导出执行侧交易日志 CSV -> {}", self.cfg.trade_log_csv)
            except Exception as e:
                logger.warning("导出执行侧交易日志失败：{}", e)

        return summary


# =====================
# 命令行入口
# =====================

def parse_args() -> OBBTConfig:
    p = argparse.ArgumentParser(description="订单簿级回测引擎（市价/限价/延迟/滑点/手续费）")
    p.add_argument("--inst", dest="inst_id", default="BTC-USDT")
    p.add_argument("--start", dest="start", required=True)
    p.add_argument("--end", dest="end", required=True)
    p.add_argument("--timeframe", dest="timeframe", default="1s")
    p.add_argument("--levels", dest="levels", type=int, default=5)
    p.add_argument("--mode", dest="mode", default="notional", choices=["size", "notional"])
    p.add_argument("--threshold", dest="threshold", type=float, default=0.02)

    p.add_argument("--exec", dest="exec", default="market", choices=["market", "limit"], help="执行方式")
    p.add_argument("--trade-units", dest="trade_units", type=float, default=0.01, help="信号对应的目标持仓单位")
    p.add_argument("--limit-offset-bps", dest="limit_offset_bps", type=float, default=0.0, help="限价相对 best 的偏移 bp")
    p.add_argument("--latency-ms", dest="latency_ms", type=int, default=50, help="下单延迟（毫秒）")

    p.add_argument("--fee-bps", dest="fee_bps", type=float, default=1.0)
    p.add_argument("--slip-extra-bps", dest="slip_extra_bps", type=float, default=0.5)

    p.add_argument("--initial-capital", dest="initial_capital", type=float, default=10000.0)
    p.add_argument("--prefer-trades-price", dest="prefer_trades_price", action="store_true")

    p.add_argument("--fills-csv", dest="fills_csv", default=None)
    p.add_argument("--report-csv", dest="report_csv", default=None)
    p.add_argument("--plot-svg", dest="plot_svg", default=None)
    # 新增：与实盘一致的冷却与外部动作驱动
    p.add_argument("--sell-cooldown-sec", dest="sell_cooldown_sec", type=float, default=0.0, help="卖出冷却秒数")
    p.add_argument("--actions-csv", dest="actions_csv", default=None, help="外部动作CSV（来自多指标回测的 trade_log.csv）")
    p.add_argument("--trade-log-csv", dest="trade_log_csv", default=None, help="输出的执行侧交易日志CSV路径")

    a = p.parse_args()
    return OBBTConfig(
        inst_id=a.inst_id,
        start=a.start,
        end=a.end,
        timeframe=a.timeframe,
        levels=a.levels,
        mode=a.mode,
        threshold=a.threshold,
        exec=a.exec,
        trade_units=a.trade_units,
        limit_offset_bps=a.limit_offset_bps,
        latency_ms=a.latency_ms,
        fee_bps=a.fee_bps,
        slip_extra_bps=a.slip_extra_bps,
        initial_capital=a.initial_capital,
        prefer_trades_price=a.prefer_trades_price,
        fills_csv=a.fills_csv,
        report_csv=a.report_csv,
        plot_svg=a.plot_svg,
        sell_cooldown_sec=a.sell_cooldown_sec,
        actions_csv=a.actions_csv,
        trade_log_csv=a.trade_log_csv,
    )


def main() -> None:
    cfg = parse_args()
    bt = OrderBookBacktester(cfg)
    bt.db.connect()
    try:
        bt.run()
    finally:
        bt.db.close()


if __name__ == "__main__":
    main()