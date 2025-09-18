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
- WITH_SLTP：是否附带止盈止损，默认 1 开启（设置 DISABLE_SL=1 或 WITH_SLTP=0 可关闭）
- TP_TRIGGER_PX/SL_TRIGGER_PX：直接指定止盈/止损触发价（数字）；若未指定则按百分比计算
- TP_ORD_PX/SL_ORD_PX：止盈/止损委托价（不设则默认 "-1" 市价）
- TP_TRIGGER_PX_TYPE/SL_TRIGGER_PX_TYPE：触发价类型，默认 last，可选 mark/index
- TP_PCT/SL_PCT：按参考价的百分比偏移（例如 0.002=0.2%），用于自动计算触发价
- TGT_CCY：现货市价买入计量方式，可选 base_ccy/quote_ccy，覆盖 OKX_TGT_CCY

注意：
- 该脚本注入 MarketState 提供参考价，避免因数据库/行情缺失导致风控拒单。
- 市价单买入默认以 base 计价（tgtCcy=base_ccy），对应 OKX 参数 sz=base 数量。
"""
from __future__ import annotations

# ==== 兼容从任意目录运行脚本：将 mvp 目录加入 sys.path，确保 `from utils ...` 可以导入 ====
import sys
from pathlib import Path
_CUR = Path(__file__).resolve()
_MVP_ROOT = _CUR.parent.parent  # 指向 mvp 目录
if str(_MVP_ROOT) not in sys.path:
    sys.path.insert(0, str(_MVP_ROOT))  # 将 mvp 目录放到搜索路径最前，确保 `utils` 可解析

import os
import time
from datetime import datetime, timezone
from loguru import logger

# 本地导入
from utils.config import AppConfig
from executor.trade_executor import TradeExecutor, TradeSignal, MarketState
from executor.okx_rest import OKXRESTClient
# from utils.risk import MarketState


def _mk_market_state(ref_px: float) -> MarketState:
    # 注入 mid/bid/ask（设定极小点差），确保风控可用
    spread_pct = 0.0001  # 0.01%
    half = ref_px * spread_pct / 2
    return MarketState(mid_price=ref_px, best_bid=ref_px - half, best_ask=ref_px + half)


def _maybe_attach_sltp(side: str, ref_px: float) -> dict:
    """根据环境变量构造止盈止损字段。
    规则：
    - 若 DISABLE_SL=1 或 WITH_SLTP=0，则不返回任何止盈止损字段；
    - 若显式提供 *_TRIGGER_PX 则直传；否则按 TP_PCT/SL_PCT 基于参考价计算；
    - 未提供 *_ORD_PX 时，默认 "-1" 表示市价触发；
    - 默认触发价类型 *_TRIGGER_PX_TYPE="last"。
    """
    # 显式禁用判断
    if str(os.getenv("DISABLE_SL", "0")).strip() == "1":
        return {}
    if str(os.getenv("WITH_SLTP", "1")).strip() == "0":
        return {}

    def _to_float_or_none(v: str | None):
        if v is None or str(v).strip() == "":
            return None
        try:
            return float(str(v).strip())
        except Exception:
            return None

    # 读取显式价格
    tp_trigger_px_env = _to_float_or_none(os.getenv("TP_TRIGGER_PX"))
    sl_trigger_px_env = _to_float_or_none(os.getenv("SL_TRIGGER_PX"))
    tp_ord_px_env = os.getenv("TP_ORD_PX")
    sl_ord_px_env = os.getenv("SL_ORD_PX")

    # 若未显式指定触发价，按百分比围绕参考价计算
    tp_pct = float(os.getenv("TP_PCT", "0.002"))  # 0.2%
    sl_pct = float(os.getenv("SL_PCT", "0.002"))  # 0.2%

    if tp_trigger_px_env is None:
        if side in ("buy",):
            tp_trigger_px_env = ref_px * (1 + tp_pct)
        else:  # sell/close
            tp_trigger_px_env = ref_px * (1 - tp_pct)
    if sl_trigger_px_env is None:
        if side in ("buy",):
            sl_trigger_px_env = ref_px * (1 - sl_pct)
        else:  # sell/close
            sl_trigger_px_env = ref_px * (1 + sl_pct)

    # 触发价类型，默认 last
    tp_type = os.getenv("TP_TRIGGER_PX_TYPE", "last")
    sl_type = os.getenv("SL_TRIGGER_PX_TYPE", "last")

    fields: dict = {}
    # 止盈
    if tp_trigger_px_env is not None:
        fields["tpTriggerPx"] = str(tp_trigger_px_env)
        fields["tpOrdPx"] = str(tp_ord_px_env) if (tp_ord_px_env and tp_ord_px_env.strip() != "") else "-1"
        fields["tpTriggerPxType"] = tp_type
    # 止损
    if sl_trigger_px_env is not None:
        fields["slTriggerPx"] = str(sl_trigger_px_env)
        fields["slOrdPx"] = str(sl_ord_px_env) if (sl_ord_px_env and sl_ord_px_env.strip() != "") else "-1"
        fields["slTriggerPxType"] = sl_type

    return fields


def _split_inst(inst: str) -> tuple[str, str]:
    """拆分 instId 获取 base/quote。对合约视为前三段，仅取前两段。"""
    parts = str(inst).split("-")
    if len(parts) >= 2:
        return parts[0], parts[1]
    return inst, "USDT"


def _extract_balances(resp_json: dict) -> dict:
    """从 /account/balance 响应中提取各币种可用余额（若无则回退 eq）。"""
    try:
        data = (resp_json or {}).get("data") or []
        if not data:
            return {}
        details = data[0].get("details") or []
        m = {}
        for d in details:
            c = d.get("ccy")
            if not c:
                continue
            # availBal 优先，其次 cashBal，再次 eq
            v = d.get("availBal") or d.get("cashBal") or d.get("eq")
            try:
                m[c] = float(v)
            except Exception:
                pass
        return m
    except Exception:
        return {}


def main() -> None:
    # 读取环境变量
    inst = os.getenv("TEST_INST_ID", "BTC-USDT")
    side = os.getenv("TEST_SIDE", "buy").lower()
    size = float(os.getenv("TEST_SIZE", "0.0001"))

    # 参考价：优先使用环境变量；若未提供或无效，则自动从 OKX REST 获取最新价
    ref_px_env = os.getenv("TEST_REF_PX")
    ref_px: float
    _need_auto_ref = False
    try:
        if ref_px_env is None or ref_px_env.strip() == "" or ref_px_env.lower() == "auto":
            _need_auto_ref = True
            raise ValueError("auto ref px")
        ref_px = float(ref_px_env)
    except Exception:
        # 延后到创建 client 后再拉最新价
        ref_px = float("nan")

    # 强制以 real 模式运行（但 OKX_SIMULATED=1 -> 走模拟盘）
    os.environ.setdefault("EXEC_MODE", "real")

    cfg = AppConfig()  # 会自动 load_dotenv()
    client = OKXRESTClient(cfg)

    if _need_auto_ref or (ref_px != ref_px):  # NaN 检测
        try:
            t = client.get_ticker(inst)
            if t.ok and t.data:
                d0 = (t.data or [])[0] or {}
                _px = d0.get("last") or d0.get("lastPx") or d0.get("px") or d0.get("askPx") or d0.get("bidPx")
                ref_px = float(_px)
                logger.info("自动获取参考价: inst={} ref_px={}", inst, ref_px)
            else:
                raise RuntimeError(f"ticker 请求失败: {t.code} {t.msg}")
        except Exception as e:
            ref_px = 30000.0
            logger.warning("自动获取参考价失败，将使用默认值 {}: {}", ref_px, e)

    exe = TradeExecutor(cfg, mode="real")

    # 注入 market provider，避免参考价缺失
    exe.set_market_state_provider(lambda inst_id, p0: _mk_market_state(ref_px))

    # 构造市价信号（base 计量），现货 tdMode 使用 .env 或默认 cash
    td_mode = os.getenv("TEST_TD_MODE", cfg.okx.td_mode or "cash")
    meta: dict = {"ordType": "market", "tdMode": td_mode}

    # 合约/杠杆账户可能需要指定持仓方向；默认按净持仓（net）
    pos_side = os.getenv("TEST_POS_SIDE")
    if pos_side:
        meta["posSide"] = pos_side

    # 可选：设置现货买入计量方式（quote_ccy 则以报价币计价）
    tgt_ccy_env = os.getenv("TGT_CCY")
    if tgt_ccy_env:
        meta["tgtCcy"] = tgt_ccy_env

    # 若设置 DISABLE_SL=1，则不附带任何止损参数（优先级高于 WITH_SLTP）
    if str(os.getenv("DISABLE_SL", "0")).strip() == "1":
        meta["noSL"] = True
    else:
        # 组装 sl/tp 字段
        sltp = _maybe_attach_sltp(side, ref_px)
        meta.update(sltp)

    # 生成 clOrdId，便于查单（仅字母数字，首字母开头，长度<=32）
    cl_id = f"TEST{int(time.time()*1000)}"
    meta["clOrdId"] = cl_id

    # 下单前查询余额（现货使用 /account/balance）
    base, quote = _split_inst(inst)
    try:
        before = client.get_balance(ccy=f"{base},{quote}")
        if before.ok:
            bmap = _extract_balances(before.raw)
            logger.info("下单前余额: {}->{}  {}->{}", base, bmap.get(base), quote, bmap.get(quote))
        else:
            logger.warning("下单前余额查询失败: {} {}", before.code, before.msg)
    except Exception as e:
        logger.warning("下单前余额查询异常: {}", e)

    sig = TradeSignal(
        ts=datetime.now(timezone.utc),
        symbol=inst,
        side=side,
        price=None,
        size=size,
        reason="send_test_order",
        meta=meta,
    )

    logger.info(
        "发送测试单: inst={} side={} size={} tdMode={} ref_px={} clOrdId={} meta_sltp_keys={}",
        inst,
        side,
        size,
        td_mode,
        ref_px,
        cl_id,
        [k for k in meta.keys() if k.startswith("t") or k.startswith("s")],
    )

    r = exe.execute(sig)
    logger.info("执行结果: ok={} mode={} order_id={} err={}", r.ok, r.mode, r.order_id, r.err)

    # 短暂等待撮合回执与余额更新
    time.sleep(2.0)

    try:
        after = client.get_balance(ccy=f"{base},{quote}")
        if after.ok:
            amap = _extract_balances(after.raw)
            logger.info(
                "下单后余额: {}->{}  {}->{} (Δ{} / Δ{})",
                base,
                amap.get(base),
                quote,
                amap.get(quote),
                None
                if before is None
                else (amap.get(base, 0) - (_extract_balances(before.raw).get(base, 0) if before.ok else 0)),
                None
                if before is None
                else (amap.get(quote, 0) - (_extract_balances(before.raw).get(quote, 0) if before.ok else 0)),
            )
        else:
            logger.warning("下单后余额查询失败: {} {}", after.code, after.msg)
    except Exception as e:
        logger.warning("下单后余额查询异常: {}", e)

    # 可选：查单确认（若订单被系统拆分为 algoId 或 ordId，OKX 会返回其中之一）
    try:
        if r.order_id:
            q = client.get_order(inst_id=inst, ord_id=r.order_id)
        else:
            q = client.get_order(inst_id=inst, cl_ord_id=cl_id)
        logger.info("查单返回: ok={} code={} msg={} data={} ", q.ok, q.code, q.msg, q.data)
    except Exception as e:
        logger.warning("查单异常: {}", e)

    exe.close()


if __name__ == "__main__":
    main()