# -*- coding: utf-8 -*-
"""
用途：发送一笔极小额市价测试单到 OKX 模拟盘，用于验证下单链路（API密钥/风控/下单/回执）。
用法示例（PowerShell）：

  $env:EXEC_MODE="real"; $env:OKX_SIMULATED="1"; \
  $env:TEST_INST_ID="BTC-USDT"; $env:TEST_SIZE="0.0001"; $env:TEST_REF_PX="30000"; \
  python -u scripts/send_test_order.py

环境变量（可选）：
- TEST_INST_ID：交易对，默认 BTC-USDT（现货）
- TEST_SIZE：下单数量（以 base 计量），默认 0.0001（BTC）
- TEST_SIDE：buy/sell，默认 buy（请确保卖出侧有持仓）
- TEST_TD_MODE：tdMode，现货用 cash；默认从 .env 读取 OKX_TD_MODE 或 "cash"
- TEST_REF_PX：参考价（用于风控），若未提供则使用 30000

注意：
- 该脚本注入 MarketState 提供参考价，避免因数据库/行情缺失导致风控拒单。
- 市价单买入默认以 base 计价（tgtCcy=base_ccy），对应 OKX 参数 sz=base 数量。
"""
from __future__ import annotations
import os
from datetime import datetime, timezone
from loguru import logger

# 本地导入
from utils.config import AppConfig
from executor.trade_executor import TradeExecutor, TradeSignal
from utils.risk import MarketState


def _mk_market_state(ref_px: float) -> MarketState:
    # 注入 mid/bid/ask（设定极小点差），确保风控可用
    spread_pct = 0.0001  # 0.01%
    half = ref_px * spread_pct / 2
    return MarketState(mid_price=ref_px, best_bid=ref_px - half, best_ask=ref_px + half)


def main() -> None:
    # 读取环境变量
    inst = os.getenv("TEST_INST_ID", "BTC-USDT")
    side = os.getenv("TEST_SIDE", "buy").lower()
    size = float(os.getenv("TEST_SIZE", "0.0001"))
    ref_px = float(os.getenv("TEST_REF_PX", "30000"))

    # 强制以 real 模式运行（但 OKX_SIMULATED=1 -> 走模拟盘）
    os.environ.setdefault("EXEC_MODE", "real")

    cfg = AppConfig()  # 会自动 load_dotenv()
    exe = TradeExecutor(cfg, mode="real")

    # 注入 market provider，避免参考价缺失
    exe.set_market_state_provider(lambda inst_id, p0: _mk_market_state(ref_px))

    # 构造市价信号（base 计量），现货 tdMode 使用 .env 或默认 cash
    td_mode = os.getenv("TEST_TD_MODE", cfg.okx.td_mode or "cash")
    sig = TradeSignal(
        ts=datetime.now(timezone.utc),
        symbol=inst,
        side=side,
        price=None,
        size=size,
        reason="send_test_order",
        meta={"ordType": "market", "tdMode": td_mode}
    )

    logger.info("发送测试单: inst={} side={} size={} tdMode={} ref_px={}", inst, side, size, td_mode, ref_px)
    r = exe.execute(sig)
    logger.info("执行结果: ok={} mode={} order_id={} err={} resp={}", r.ok, r.mode, r.order_id, r.err, r.exchange_resp)
    exe.close()


if __name__ == "__main__":
    main()