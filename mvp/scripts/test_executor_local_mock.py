# 本地模拟测试：验证执行器的单位转换与下单逻辑（合约与现货）
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional
from datetime import datetime, timezone

from loguru import logger

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # 将 mvp 目录加入 sys.path，便于相对导入

from utils.config import AppConfig
from executor.trade_executor import TradeExecutor, TradeSignal
from executor.okx_rest import OKXResponse


@dataclass
class FakeOrder:
    instId: str
    side: str
    ordType: str
    sz: float
    tdMode: str
    posSide: Optional[str]
    reduceOnly: Optional[bool]


class FakeOKXRESTClient:
    def __init__(self) -> None:
        self.last_order: Optional[FakeOrder] = None

    def place_order(self, *, inst_id: str, side: str, ord_type: str, sz: str,
                    td_mode: str, px: Optional[str] = None, extra: Optional[Dict[str, Any]] = None,
                    tgt_ccy: Optional[str] = None, cl_ord_id: Optional[str] = None,
                    tp_trigger_px: Optional[str] = None, tp_ord_px: Optional[str] = None,
                    sl_trigger_px: Optional[str] = None, sl_ord_px: Optional[str] = None,
                    tp_trigger_px_type: Optional[str] = None, sl_trigger_px_type: Optional[str] = None) -> OKXResponse:
        # 记录下单参数
        try:
            fsz = float(sz)
        except Exception:
            fsz = -1
        pos_side_val = (extra or {}).get("posSide") if isinstance(extra, dict) else None
        reduce_only_val = (extra or {}).get("reduceOnly") if isinstance(extra, dict) else None
        self.last_order = FakeOrder(
            instId=inst_id, side=side, ordType=ord_type, sz=fsz,
            tdMode=td_mode or "", posSide=pos_side_val, reduceOnly=reduce_only_val,
        )
        # 简单的模拟校验：合约最小张数为 1 张；现货最小数量为 0
        if inst_id.endswith("-SWAP") and fsz < 1.0:
            return OKXResponse(ok=False, code=-1, msg="size<1.0", data=[], raw={})
        if fsz <= 0:
            return OKXResponse(ok=False, code=-1, msg="size<=0", data=[], raw={})
        # 返回模拟订单 id
        return OKXResponse(ok=True, code=0, msg="", data=[{"ordId": f"mock-{int(__import__('time').time_ns()/1_000_000)}"}], raw={})

    def get_positions(self, *, inst_id: str) -> OKXResponse:
        # 模拟持仓：BTC 永续双向，ETH 无合约
        if inst_id == "BTC-USDT-SWAP":
            # 双向持仓：long=10 张, short=2 张
            return OKXResponse(ok=True, code=0, msg="", data=[
                {"posSide": "long", "pos": "10"},
                {"posSide": "short", "pos": "2"},
            ], raw={})
        return OKXResponse(ok=True, code=0, msg="", data=[], raw={})

    # 新增：提供合约/现货规则，供执行器在下单前规整数量
    def get_instruments(self, inst_type: str, inst_id: Optional[str] = None) -> OKXResponse:
        items: list[dict[str, Any]] = []
        if inst_type == "SWAP":
            _inst = inst_id or "BTC-USDT-SWAP"
            # 设定 BTC 永续合约面值 ctVal=0.01，步进与最小张数为 1
            ct_val = "0.01" if _inst == "BTC-USDT-SWAP" else "0.1"
            items.append({"instId": _inst, "lotSz": "1", "minSz": "1", "ctVal": ct_val})
        elif inst_type == "SPOT":
            _inst = inst_id or "ETH-USDT"
            items.append({"instId": _inst, "lotSz": "0.000001", "minSz": "0.000001"})
        else:
            _inst = inst_id or "BTC-USDT"
            items.append({"instId": _inst, "lotSz": "0.000001", "minSz": "0.000001"})
        return OKXResponse(ok=True, code=0, msg="", data=items, raw={})

    def get_ticker(self, *, inst_id: str) -> OKXResponse:
        return OKXResponse(ok=True, code=0, msg="", data=[{"instId": inst_id, "last": "50000"}], raw={})

    def get_mark_price(self, *, inst_id: str) -> OKXResponse:
        return OKXResponse(ok=True, code=0, msg="", data=[{"instId": inst_id, "markPx": "50000"}], raw={})

    def close(self) -> None:
        return None


def _print_last_order(client: FakeOKXRESTClient, title: str) -> None:
    fo = client.last_order
    logger.info(
        "{} 下单参数: instId={} side={} ordType={} sz={} tdMode={} posSide={} reduceOnly={}",
        title,
        fo.instId if fo else None,
        fo.side if fo else None,
        fo.ordType if fo else None,
        fo.sz if fo else None,
        fo.tdMode if fo else None,
        fo.posSide if fo else None,
        fo.reduceOnly if fo else None,
    )


def main() -> None:
    # 强制关闭 HTTP 管理端，避免端口占用影响测试
    os.environ["EXEC_HTTP_ENABLED"] = "0"

    cfg = AppConfig()
    exe = TradeExecutor(cfg, mode="real")  # 在本地 mock 客户端下，仍走 real 流程但不会真正下单

    # 注入本地假客户端，拦截网络调用
    exe.client = FakeOKXRESTClient()

    # 用例1：合约开仓，币数量 -> 张数转换
    sig1 = TradeSignal(
        ts=datetime.now(timezone.utc),
        symbol="BTC-USDT-SWAP",
        side="buy",
        price=None,
        size=0.05,  # 假设 ctVal=0.01 => 5 张
        reason="test-open-swap",
        meta={"ordType": "market", "tdMode": "isolated"},
    )
    r1 = exe.execute(sig1)
    logger.info("用例1结果: ok={} order_id={} err={}", r1.ok, r1.order_id, r1.err)
    _print_last_order(exe.client, "用例1")

    # 用例2：合约平仓，按实盘持仓 * sizeFrac 计算张数
    sig2 = TradeSignal(
        ts=datetime.now(timezone.utc),
        symbol="BTC-USDT-SWAP",
        side="close",
        price=None,
        size=0.001,  # 原始 size 将被忽略
        reason="test-close-swap",
        meta={"ordType": "market", "tdMode": "isolated", "sizeFrac": 0.5},
    )
    r2 = exe.execute(sig2)
    logger.info("用例2结果: ok={} order_id={} err={}", r2.ok, r2.order_id, r2.err)
    _print_last_order(exe.client, "用例2")

    # 用例3：现货买入，直接用币数量
    sig3 = TradeSignal(
        ts=datetime.now(timezone.utc),
        symbol="ETH-USDT",
        side="buy",
        price=None,
        size=0.5,
        reason="test-open-spot",
        meta={"ordType": "market", "tdMode": "cash"},
    )
    r3 = exe.execute(sig3)
    logger.info("用例3结果: ok={} order_id={} err={}", r3.ok, r3.order_id, r3.err)
    _print_last_order(exe.client, "用例3")

    exe.close()


if __name__ == "__main__":
    main()