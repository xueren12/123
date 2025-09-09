# -*- coding: utf-8 -*-
"""
OKX REST 客户端（V5）
- 负责签名、请求头组装与私有接口调用
- 支持下单、撤单、查单等常见交易接口
- 仅依赖 httpx，同步实现，便于在策略线程中直接调用

注意：
- 不在日志中打印密钥/签名等敏感信息
- 若使用 OKX 模拟盘（Demo Trading），会在请求头加入 x-simulated-trading: 1
- 时间戳为 UTC ISO8601（毫秒），如 2025-09-04T12:34:56.789Z
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import httpx
from loguru import logger

from utils.config import AppConfig


@dataclass
class OKXResponse:
    """统一返回结构（便于上层处理）"""
    ok: bool
    code: str
    msg: str
    data: Any
    raw: Dict[str, Any]


class OKXRESTClient:
    """OKX REST API V5 客户端（同步）。

    使用示例：
        cfg = AppConfig()
        client = OKXRESTClient(cfg)
        resp = client.place_order(inst_id="BTC-USDT", side="buy", ord_type="limit", px="30000", sz="0.001")
        if resp.ok:
            print("下单成功", resp.data)
        else:
            print("下单失败", resp.code, resp.msg)
    """

    def __init__(self, cfg: AppConfig, timeout: float = 10.0) -> None:
        self.cfg = cfg
        self.base_url = cfg.okx.base_url.rstrip("/")
        self.api_key = cfg.okx.api_key
        self.secret_key = cfg.okx.secret_key
        self.passphrase = cfg.okx.passphrase
        self.simulated = cfg.okx.simulated_trading
        self.timeout = timeout
        self._client = httpx.Client(base_url=self.base_url, timeout=timeout)

    # --------------------
    # 工具：时间戳与签名
    # --------------------
    @staticmethod
    def _ts_iso_millis() -> str:
        """返回 OKX 要求的 UTC ISO8601 毫秒时间戳，结尾 Z。"""
        ts = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        # Python isoformat 形如 '2025-09-04T12:34:56.789+00:00'，需替换为 Z 结尾
        if ts.endswith("+00:00"):
            ts = ts[:-6] + "Z"
        return ts

    def _sign(self, timestamp: str, method: str, request_path: str, body: str = "") -> str:
        """生成签名（OK-ACCESS-SIGN）。
        规则：base64(hmac_sha256(secret_key, timestamp + method + request_path + body))
        """
        message = f"{timestamp}{method.upper()}{request_path}{body}"
        mac = hmac.new(self.secret_key.encode("utf-8"), message.encode("utf-8"), hashlib.sha256)
        sign = base64.b64encode(mac.digest()).decode()
        return sign

    # --------------------
    # 工具：请求
    # --------------------
    def _headers(self, timestamp: str, sign: str) -> Dict[str, str]:
        headers = {
            "OK-ACCESS-KEY": self.api_key,
            "OK-ACCESS-SIGN": sign,
            "OK-ACCESS-TIMESTAMP": timestamp,
            "OK-ACCESS-PASSPHRASE": self.passphrase,
            "Content-Type": "application/json",
        }
        if self.simulated:
            headers["x-simulated-trading"] = "1"  # 开启 OKX 模拟盘
        return headers

    def _request(self, method: str, path: str, params: Optional[Dict[str, Any]] = None, body: Optional[Dict[str, Any]] = None) -> OKXResponse:
        """通用私有请求入口。path 需以 /api/v5 开头。"""
        assert path.startswith("/api/v5"), "path 必须以 /api/v5 开头"
        timestamp = self._ts_iso_millis()
        request_path = path
        query = None
        body_str = ""
        if method.upper() == "GET" and params:
            # 构造查询串（httpx 会自动拼接，但签名需要完整 path?query）
            query = httpx.QueryParams(params).render()
            request_path = f"{path}?{query}"
        if method.upper() != "GET" and body:
            body_str = json.dumps(body, separators=(",", ":"))
        sign = self._sign(timestamp, method, request_path, body_str)
        headers = self._headers(timestamp, sign)

        try:
            if method.upper() == "GET":
                r = self._client.get(path, params=params, headers=headers)
            elif method.upper() == "POST":
                r = self._client.post(path, content=body_str if body_str else None, headers=headers)
            else:
                raise ValueError(f"不支持的 method: {method}")
        except httpx.RequestError as e:
            logger.error("HTTP 请求异常: {}", e)
            return OKXResponse(ok=False, code="-1", msg=str(e), data=None, raw={})

        # OKX 标准响应：{"code":"0","data":[...],"msg":""}
        try:
            payload = r.json()
        except Exception:
            return OKXResponse(ok=False, code=str(r.status_code), msg=r.text[:500], data=None, raw={})

        ok = (payload.get("code") == "0")
        return OKXResponse(ok=ok, code=str(payload.get("code")), msg=str(payload.get("msg")), data=payload.get("data"), raw=payload)

    # --------------------
    # 交易接口：下单/撤单/查单
    # --------------------
    def place_order(
        self,
        inst_id: str,
        side: str,
        ord_type: str,
        sz: str,
        px: Optional[str] = None,
        td_mode: Optional[str] = None,
        tgt_ccy: Optional[str] = None,
        cl_ord_id: Optional[str] = None,
        # 止盈止损（可选）
        tp_trigger_px: Optional[str] = None,
        tp_ord_px: Optional[str] = None,
        sl_trigger_px: Optional[str] = None,
        sl_ord_px: Optional[str] = None,
        tp_trigger_px_type: Optional[str] = None,  # last/mark/index
        sl_trigger_px_type: Optional[str] = None,  # last/mark/index
        extra: Optional[Dict[str, Any]] = None,
    ) -> OKXResponse:
        """下单：/api/v5/trade/order
        - 现货交易 tdMode 通常为 'cash'
        - 市价单（ordType=market）下单时，若为买入，建议设置 tgtCcy=quote_ccy 以 quote 金额计价
        - 止盈/止损参数：tpTriggerPx/slTriggerPx、tpOrdPx/slOrdPx、tpTriggerPxType/slTriggerPxType
        """
        path = "/api/v5/trade/order"
        # 统一计算最终的 tdMode，便于条件化设置字段
        resolved_td_mode = td_mode or self.cfg.okx.td_mode
        # 基础字段（合约/现货通用）
        body: Dict[str, Any] = {
            "instId": inst_id,
            "side": side,  # buy/sell
            "ordType": ord_type,  # limit/market
            "sz": str(sz),
            "tdMode": resolved_td_mode,
        }
        # 仅在现货模式（cash）时设置 tgtCcy，合约/杠杆不需要该字段
        if resolved_td_mode == "cash":
            body["tgtCcy"] = tgt_ccy or self.cfg.okx.tgt_ccy
        if px is not None:
            body["px"] = str(px)
        if cl_ord_id:
            body["clOrdId"] = cl_ord_id

        # 止盈止损
        if tp_trigger_px is not None:
            body["tpTriggerPx"] = str(tp_trigger_px)
        if tp_ord_px is not None:
            body["tpOrdPx"] = str(tp_ord_px)
        if sl_trigger_px is not None:
            body["slTriggerPx"] = str(sl_trigger_px)
        if sl_ord_px is not None:
            body["slOrdPx"] = str(sl_ord_px)
        if tp_trigger_px_type is not None:
            body["tpTriggerPxType"] = tp_trigger_px_type
        if sl_trigger_px_type is not None:
            body["slTriggerPxType"] = sl_trigger_px_type

        # 透传额外参数（用于合约场景：posSide、reduceOnly、lever、tag 等）
        if extra:
            body.update(extra)

        return self._request("POST", path, body=body)

    def cancel_order(self, inst_id: str, ord_id: Optional[str] = None, cl_ord_id: Optional[str] = None) -> OKXResponse:
        """撤单：/api/v5/trade/cancel-order（ordId 与 clOrdId 二选一）"""
        path = "/api/v5/trade/cancel-order"
        body: Dict[str, Any] = {"instId": inst_id}
        if ord_id:
            body["ordId"] = ord_id
        if cl_ord_id:
            body["clOrdId"] = cl_ord_id
        return self._request("POST", path, body=body)

    def get_order(self, inst_id: str, ord_id: Optional[str] = None, cl_ord_id: Optional[str] = None) -> OKXResponse:
        """查单：/api/v5/trade/order, 通过 ordId 或 clOrdId 查询订单详情"""
        path = "/api/v5/trade/order"
        params: Dict[str, Any] = {"instId": inst_id}
        if ord_id:
            params["ordId"] = ord_id
        if cl_ord_id:
            params["clOrdId"] = cl_ord_id
        return self._request("GET", path, params=params)

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:
            pass