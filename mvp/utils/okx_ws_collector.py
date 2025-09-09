"""
OKX WebSocket 实时数据采集器（trades 与 books5）。
- 默认订阅 BTC-USDT / ETH-USDT 的成交与盘口五档
- 将成交写入 PostgreSQL(trades)，将盘口写入 PostgreSQL(orderbook)
- 数据库连接通过 utils.db.TimescaleDB 提供
- 可通过环境变量进行配置，参见 utils.config

直接运行：
    python -m utils.okx_ws_collector
"""
from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List

import websockets
from websockets.exceptions import ConnectionClosed
from loguru import logger

from .config import AppConfig
from .db import TimescaleDB


class OKXWSCollector:
    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg
        self.db = TimescaleDB()
        self.ws_url = cfg.ws.okx_ws_public
        self.channels = cfg.ws.channels
        self.instruments = [i.strip() for i in cfg.ws.instruments if i.strip()]
        self._ws = None

    async def _subscribe(self, ws) -> None:
        args = []
        for inst in self.instruments:
            for ch in self.channels:
                args.append({"channel": ch, "instId": inst})
        sub_msg = {"op": "subscribe", "args": args}
        await ws.send(json.dumps(sub_msg))
        logger.info("已订阅频道 {}，交易对：{}", self.channels, self.instruments)

    async def _handle_trades(self, data: Dict[str, Any]) -> None:
        # OKX 成交数据示例：data: [{"instId":"BTC-USDT","tradeId":"...","px":"...","sz":"...","side":"buy","ts":"1710000000000"}, ...]
        rows = []
        for d in data.get("data", []):
            rows.append({
                "ts": self.db.ms_to_utc(int(d["ts"])),
                "inst_id": d["instId"],
                "side": d.get("side"),
                "price": d.get("px"),
                "size": d.get("sz"),
                "trade_id": d.get("tradeId"),
                "source": "okx",
            })
        if rows:
            self.db.insert_trades(rows)

    async def _handle_books5(self, data: Dict[str, Any]) -> None:
        # OKX 五档盘口数据示例：data: [{"instId":"BTC-USDT","bids":[[price,sz,liq,?],[...]],"asks":[...],"ts":"171...","checksum":12345}]
        for d in data.get("data", []):
            row = {
                "ts": self.db.ms_to_utc(int(d["ts"])),
                "inst_id": d["instId"],
                # 原样存储列表；数据库层会转换为 JSONB
                "bids": d.get("bids", []),
                "asks": d.get("asks", []),
                "checksum": int(d.get("checksum")) if d.get("checksum") is not None else None,
            }
            self.db.insert_orderbook(row)

    async def _recv_loop(self) -> None:
        backoff = 1
        while True:
            try:
                async with websockets.connect(self.ws_url, ping_interval=20, ping_timeout=20) as ws:
                    self._ws = ws
                    logger.success("已连接 OKX WS: {}", self.ws_url)
                    await self._subscribe(ws)
                    backoff = 1
                    async for msg in ws:
                        try:
                            data = json.loads(msg)
                        except json.JSONDecodeError:
                            logger.warning("消息解码失败: {}", msg[:200])
                            continue
                        if "event" in data:
                            # 处理事件消息（订阅成功、错误等）
                            if data.get("event") == "subscribe":
                                logger.info("订阅确认: {}", data.get("arg"))
                            elif data.get("event") == "error":
                                logger.error("WS 错误: {}", data)
                            continue
                        arg = data.get("arg", {})
                        channel = arg.get("channel")
                        if channel == "trades":
                            await self._handle_trades(data)
                        elif channel == "books5":
                            await self._handle_books5(data)
            except ConnectionClosed as e:
                logger.warning("WS 连接关闭: {}", e)
            except Exception as e:
                logger.exception("WS 循环异常: {}", e)
            finally:
                self._ws = None
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30)
                logger.info("{} 秒后重连", backoff)

    async def start(self) -> None:
        self.db.connect()
        try:
            await self._recv_loop()
        finally:
            self.db.close()


async def main_async():
    cfg = AppConfig()
    collector = OKXWSCollector(cfg)
    await collector.start()


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()