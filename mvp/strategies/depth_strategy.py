"""
深度不平衡高频策略示例（基于 OKX books5 盘口）。

策略思想（简版验证用）：
- 每秒读取数据库中最新的一条 books5 盘口快照，或在 mock 模式下生成模拟盘口；
- 将前 N 档买盘与卖盘的深度进行对比（支持按数量或按名义价值 price*size 计算）；
- 当买盘深度 > 卖盘深度*(1+阈值) -> 产生做多信号（BUY），反之产生做空信号（SELL），否则保持（HOLD）。

输入数据来源：
- DB 模式：PostgreSQL / TimescaleDB 中的 orderbook 表（utils.db 已提供写入与表结构初始化）。
- MOCK 模式：内存生成的盘口（用于在未部署数据库时快速验证流程）。

运行方式示例：
    # DB 模式（需要已运行 PostgreSQL 且有采集数据）
    python -m strategies.depth_strategy --source db --inst BTC-USDT --levels 5 --mode notional --threshold 0.02 --period 1

    # MOCK 模式（无需数据库，快速验证控制台信号输出）
    python -m strategies.depth_strategy --source mock --period 1 --max-loops 5

注意：本示例仅输出控制台信号，便于后续对接实盘执行模块（executor）。
"""
from __future__ import annotations

import time
import random
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import List, Optional, Tuple

from loguru import logger

from utils.db import TimescaleDB
from utils.config import AppConfig


@dataclass
class StrategyConfig:
    """策略参数配置（可通过命令行覆盖）。"""
    # 通用
    source: str = "db"                     # 数据源："db" 或 "mock"
    inst_id: str = "BTC-USDT"              # 交易对（db 模式生效）
    levels: int = 5                         # 使用前 N 档深度
    mode: str = "notional"                  # 计算模式："size"（数量）或 "notional"（名义价值 price*size）
    threshold: float = 0.02                 # 相对不平衡阈值（例如 0.02 表示 2%）
    period_sec: float = 1.0                 # 检查周期（秒）
    max_loops: Optional[int] = None         # 运行的最大循环次数，None 为无限循环

    # DB 模式
    lookback_seconds: int = 3               # 允许的快照最大延迟（秒），超过视为无效
    fallback_to_mock: bool = True           # 当数据库连接失败时是否回退到 mock

    # MOCK 模式参数
    mock_base_price: float = 30000.0        # 模拟基准价格
    mock_spread: float = 2.0                # 档位之间最小价差（美元）
    mock_volatility: float = 20.0           # 每次更新的价格波动标准差
    mock_size_min: float = 0.01             # 每档最小数量
    mock_size_max: float = 0.5              # 每档最大数量
    mock_imbalance_bias: float = 0.0        # 不平衡偏置（>0 偏向买盘，<0 偏向卖盘），用于演示


class DepthImbalanceStrategy:
    """基于盘口深度不平衡的简易高频策略。"""

    def __init__(self, cfg: StrategyConfig) -> None:
        self.cfg = cfg
        self.db = TimescaleDB() if cfg.source == "db" else None
        self._mock_last_price = cfg.mock_base_price

    # =====================
    # 数据获取相关
    # =====================
    def _ensure_connected(self) -> None:
        """确保已连接数据库（仅在 DB 模式下）。"""
        if self.cfg.source != "db":
            return
        if self.db is None:
            self.db = TimescaleDB()
        if self.db.conn is None:
            try:
                self.db.connect()
            except Exception as e:
                logger.error("数据库连接失败：{}", e)
                if self.cfg.fallback_to_mock:
                    logger.warning("启用回退：切换到 MOCK 模式进行演示。")
                    self.cfg.source = "mock"
                else:
                    raise

    def _fetch_latest_orderbook_db(self) -> Optional[Tuple[datetime, List, List]]:
        """查询最近 lookback_seconds 内该交易对的最新一条盘口（DB 模式）。
        返回 (ts, bids, asks)；若无合格数据则返回 None。
        """
        assert self.db is not None
        if self.db.conn is None:
            return None
        min_ts = datetime.now(timezone.utc) - timedelta(seconds=int(self.cfg.lookback_seconds))
        sql = (
            "SELECT ts, bids, asks FROM orderbook "
            "WHERE inst_id = %(inst)s AND ts >= %(min_ts)s "
            "ORDER BY ts DESC LIMIT 1"
        )
        with self.db.conn.cursor() as cur:
            cur.execute(sql, {"inst": self.cfg.inst_id, "min_ts": min_ts})
            row = cur.fetchone()
        if not row:
            return None
        return row["ts"], row["bids"], row["asks"]

    def _gen_mock_orderbook(self) -> Tuple[datetime, List, List]:
        """生成一份模拟盘口（MOCK 模式）。返回 (ts, bids, asks)。
        OKX books5 结构兼容：bids/asks 为 [[price, size, ...], ...]
        """
        # 随机游走价格 + 偏置
        drift = random.gauss(0, self.cfg.mock_volatility)
        self._mock_last_price = max(1.0, self._mock_last_price + drift)
        mid = self._mock_last_price

        # 根据不平衡偏置调整买卖侧数量期望
        bias = self.cfg.mock_imbalance_bias
        size_bias_buy = max(0.0, 1.0 + bias)
        size_bias_sell = max(0.0, 1.0 - bias)

        bids = []
        asks = []
        for i in range(self.cfg.levels):
            # 价格阶梯
            bid_px = mid - (i + 1) * self.cfg.mock_spread
            ask_px = mid + (i + 1) * self.cfg.mock_spread
            # 数量（加入轻微噪声）
            bid_sz = random.uniform(self.cfg.mock_size_min, self.cfg.mock_size_max) * size_bias_buy
            ask_sz = random.uniform(self.cfg.mock_size_min, self.cfg.mock_size_max) * size_bias_sell
            bids.append([f"{bid_px:.2f}", f"{bid_sz:.6f}"])
            asks.append([f"{ask_px:.2f}", f"{ask_sz:.6f}"])
        ts = datetime.now(timezone.utc)
        return ts, bids, asks

    def _fetch_latest_orderbook(self) -> Optional[Tuple[datetime, List, List]]:
        """统一入口：根据数据源返回最新盘口。"""
        if self.cfg.source == "mock":
            return self._gen_mock_orderbook()
        # DB 模式
        self._ensure_connected()
        return self._fetch_latest_orderbook_db()

    # =====================
    # 指标与信号
    # =====================
    @staticmethod
    def _sum_depth(levels: int, side: List, mode: str) -> float:
        """对某一侧（bids 或 asks）的前 N 档聚合深度。
        - side 为 [[price, size, ...], ...] 的列表；OKX books5 中 price/size 通常为字符串。
        - mode == "size": 累加 size
        - mode == "notional": 累加 price*size
        """
        total = 0.0
        for i, lvl in enumerate(side[: max(levels, 0)]):
            if not isinstance(lvl, (list, tuple)) or len(lvl) < 2:
                continue
            try:
                px = float(lvl[0])
                sz = float(lvl[1])
            except Exception:
                continue
            if mode == "size":
                total += sz
            else:
                total += px * sz
        return total

    def compute_signal(self) -> Optional[dict]:
        """计算一次信号，返回统一结构：
        {
          'ts': <快照时间UTC>, 'inst_id': <交易对或 MOCK>,
          'signal': 'BUY'|'SELL'|'HOLD',
          'buy_depth': float, 'sell_depth': float,
          'mode': 'size'|'notional', 'threshold': float, 'levels': int
        }
        若无有效数据，返回 None。
        """
        latest = self._fetch_latest_orderbook()
        if latest is None:
            return None
        ts, bids, asks = latest
        buy_depth = self._sum_depth(self.cfg.levels, bids, self.cfg.mode)
        sell_depth = self._sum_depth(self.cfg.levels, asks, self.cfg.mode)

        # 按相对不平衡阈值判断信号
        up_bound = sell_depth * (1.0 + self.cfg.threshold)
        down_bound = buy_depth * (1.0 + self.cfg.threshold)
        if buy_depth > up_bound:
            sig = "BUY"
        elif sell_depth > down_bound:
            sig = "SELL"
        else:
            sig = "HOLD"

        inst = self.cfg.inst_id if self.cfg.source == "db" else "MOCK"
        return {
            "ts": ts,
            "inst_id": inst,
            "signal": sig,
            "buy_depth": buy_depth,
            "sell_depth": sell_depth,
            "mode": self.cfg.mode,
            "threshold": self.cfg.threshold,
            "levels": self.cfg.levels,
        }

    def run_loop(self) -> None:
        """按周期循环计算并输出信号到控制台。"""
        if self.cfg.source == "db":
            self._ensure_connected()
        loops = 0
        logger.info(
            "启动深度不平衡策略：source={} inst={} levels={} mode={} threshold={} period={}s lookback={}s",
            self.cfg.source, self.cfg.inst_id, self.cfg.levels, self.cfg.mode, self.cfg.threshold,
            self.cfg.period_sec, self.cfg.lookback_seconds,
        )
        try:
            while True:
                signal = self.compute_signal()
                if signal is None:
                    logger.warning("无有效盘口数据（DB 模式下可能采集器未运行或延迟过大）")
                else:
                    # 控制台输出统一格式
                    logger.info(
                        "信号: {signal} | {inst} @ {ts} | buy_depth={bd:.6f} sell_depth={sd:.6f} mode={mode} levels={levels} thres={thres}",
                        signal=signal["signal"], inst=signal["inst_id"], ts=signal["ts"],
                        bd=signal["buy_depth"], sd=signal["sell_depth"], mode=signal["mode"],
                        levels=signal["levels"], thres=signal["threshold"],
                    )
                loops += 1
                if self.cfg.max_loops is not None and loops >= self.cfg.max_loops:
                    logger.info("达到最大循环次数，退出。")
                    break
                time.sleep(self.cfg.period_sec)
        except KeyboardInterrupt:
            logger.info("收到中断信号，准备退出...")
        finally:
            if self.db:
                self.db.close()


def _parse_args_to_cfg() -> StrategyConfig:
    """解析命令行参数为策略配置。"""
    import argparse

    parser = argparse.ArgumentParser(description="深度不平衡高频策略示例")
    parser.add_argument("--source", choices=["db", "mock"], default="db", help="数据源：db 或 mock")
    parser.add_argument("--inst", dest="inst_id", default=None, help="交易对（db 模式生效），默认取配置文件中的第一个")
    parser.add_argument("--levels", type=int, default=5, help="聚合前 N 档深度，默认 5")
    parser.add_argument("--mode", choices=["size", "notional"], default="notional", help="深度计算模式")
    parser.add_argument("--threshold", type=float, default=0.02, help="相对不平衡阈值，默认 0.02")
    parser.add_argument("--period", dest="period_sec", type=float, default=1.0, help="轮询周期（秒），默认 1")
    parser.add_argument("--lookback", dest="lookback_seconds", type=int, default=3, help="允许的最大快照延迟（秒）")
    parser.add_argument("--max-loops", dest="max_loops", type=int, default=None, help="最大循环次数（便于测试）")
    parser.add_argument("--no-fallback-mock", dest="fallback_to_mock", action="store_false", help="禁用 DB 连接失败时的 MOCK 回退")

    # MOCK 参数
    parser.add_argument("--mock-base", dest="mock_base_price", type=float, default=30000.0, help="MOCK 基准价格")
    parser.add_argument("--mock-spread", dest="mock_spread", type=float, default=2.0, help="MOCK 档位最小价差")
    parser.add_argument("--mock-vol", dest="mock_volatility", type=float, default=20.0, help="MOCK 价格波动强度")
    parser.add_argument("--mock-size-min", dest="mock_size_min", type=float, default=0.01, help="MOCK 最小数量")
    parser.add_argument("--mock-size-max", dest="mock_size_max", type=float, default=0.5, help="MOCK 最大数量")
    parser.add_argument("--mock-bias", dest="mock_imbalance_bias", type=float, default=0.0, help="MOCK 不平衡偏置（>0 偏买 <0 偏卖）")

    args = parser.parse_args()

    # 缺省交易对从全局配置读取
    app_cfg = AppConfig()
    inst = args.inst_id or (app_cfg.ws.instruments[0] if app_cfg.ws.instruments else "BTC-USDT")

    return StrategyConfig(
        source=args.source,
        inst_id=inst,
        levels=args.levels,
        mode=args.mode,
        threshold=args.threshold,
        period_sec=args.period_sec,
        lookback_seconds=args.lookback_seconds,
        max_loops=args.max_loops,
        fallback_to_mock=args.fallback_to_mock,
        mock_base_price=args.mock_base_price,
        mock_spread=args.mock_spread,
        mock_volatility=args.mock_volatility,
        mock_size_min=args.mock_size_min,
        mock_size_max=args.mock_size_max,
        mock_imbalance_bias=args.mock_imbalance_bias,
    )


def main() -> None:
    cfg = _parse_args_to_cfg()
    strategy = DepthImbalanceStrategy(cfg)
    strategy.run_loop()


if __name__ == "__main__":
    main()