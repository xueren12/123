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
from typing import Any, Dict, Optional, List, Tuple

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

    def __init__(self, cfg: AppConfig, timeout: float = 30.0) -> None:
        self.cfg = cfg
        self.base_url = cfg.okx.base_url.rstrip("/")
        # 注：显式 strip 并校验 ASCII，避免 httpx 组包时因非 ASCII 报错
        self.api_key = (cfg.okx.api_key or "").strip()
        self.secret_key = (cfg.okx.secret_key or "").strip()
        self.passphrase = (cfg.okx.passphrase or "").strip()
        # 简单 ASCII 校验（不要在此打印任何密钥内容）
        for name, v in (("OKX_API_KEY", self.api_key), ("OKX_API_SECRET", self.secret_key), ("OKX_API_PASSPHRASE", self.passphrase)):
            if v and not v.isascii():
                raise ValueError(f"{name} 包含非ASCII字符，请检查 .env 是否含中文引号/全角字符/不可见字符并移除")
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

    # 新增：公共 GET 请求（无需签名/鉴权），用于 market/public 端点
    def _request_public(self, method: str, path: str, params: Optional[Dict[str, Any]] = None) -> OKXResponse:
        """公共行情请求：不带任何鉴权头，适用于 /api/v5/market/* 与 /api/v5/public/*。
        这样即便在 mock/paused 模式且未配置密钥，也能拿到参考价。
        """
        assert method.upper() == "GET", "公共接口目前仅支持 GET"
        # 增加简单重试，缓解偶发超时
        for attempt in range(3):
            try:
                r = self._client.get(path, params=params)
                break
            except httpx.RequestError as e:
                logger.warning("HTTP 公共请求异常，重试 {}/3：{}", attempt + 1, e)
                if attempt == 2:
                    logger.error("HTTP 公共请求失败（已重试3次）: {}", e)
                    return OKXResponse(ok=False, code="-1", msg=str(e), data=None, raw={})
                try:
                    import time as _t
                    _t.sleep(0.6 * (attempt + 1))
                except Exception:
                    pass
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
        # 限价单需要传 px
        if ord_type == "limit" and px is not None:
            body["px"] = px
        # 仅现货买入：可选设置以 quote 计价（合约/Futures/Swap/Option 不支持 tgtCcy）
        if ord_type == "market" and side == "buy":
            # OKX 现货 instId 形如 BTC-USDT（两段），合约形如 BTC-USDT-SWAP / BTC-USDT-240927（>= 三段）
            is_spot = len(inst_id.split("-")) == 2
            if is_spot:
                chosen_tgt_ccy = tgt_ccy or self.cfg.okx.tgt_ccy
                if chosen_tgt_ccy:
                    body["tgtCcy"] = chosen_tgt_ccy
        # 用户自定义 tdMode 覆盖
        if td_mode is not None:
            body["tdMode"] = td_mode
        # 止盈止损：按交易所要求使用 attachAlgoOrds 数组（不再在顶层直接放置 tp/sl 字段）
        attach: Dict[str, Any] = {}
        if tp_trigger_px is not None:
            attach["tpTriggerPx"] = tp_trigger_px
        if tp_ord_px is not None:
            attach["tpOrdPx"] = tp_ord_px
        if sl_trigger_px is not None:
            attach["slTriggerPx"] = sl_trigger_px
        if sl_ord_px is not None:
            attach["slOrdPx"] = sl_ord_px
        if tp_trigger_px_type is not None:
            attach["tpTriggerPxType"] = tp_trigger_px_type
        if sl_trigger_px_type is not None:
            attach["slTriggerPxType"] = sl_trigger_px_type
        if attach:
            body["attachAlgoOrds"] = [attach]
        # 客户端订单ID
        if cl_ord_id is not None:
            body["clOrdId"] = cl_ord_id
        if extra:
            body.update(extra)

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

    # =============== 行情接口（公共） ===============
    def get_ticker(self, inst_id: str) -> OKXResponse:
        """最新行情：/api/v5/market/ticker?instId=... 返回 last/bidPx/askPx 等
        - 若未配置 API Key，则自动走公共请求（无签名），用于在非实盘模式下获取参考价。
        """
        path = "/api/v5/market/ticker"
        params: Dict[str, Any] = {"instId": inst_id}
        # 若缺少密钥/密文/口令，则使用公共请求以避免签名失败
        if not self.api_key or not self.secret_key or not self.passphrase:
            return self._request_public("GET", path, params=params)
        return self._request("GET", path, params=params)

    def get_mark_price(self, inst_id: str) -> OKXResponse:
        """标记价格：/api/v5/public/mark-price?instId=...
        - 若未配置 API Key，则自动走公共请求（无签名）。
        """
        path = "/api/v5/public/mark-price"
        params: Dict[str, Any] = {"instId": inst_id}
        if not self.api_key or not self.secret_key or not self.passphrase:
            return self._request_public("GET", path, params=params)
        return self._request("GET", path, params=params)

    # 新增：获取历史/最新 K 线（公共端点）
    def get_candles(self, inst_id: str, bar: str = "1m", before: Optional[str] = None, after: Optional[str] = None, limit: int = 100) -> OKXResponse:
        """获取最近一段时间的 K 线：/api/v5/market/candles
        - 参数：instId, bar(如 1m/5m/1H/1D), before/after(毫秒时间戳), limit(条数)
        - 该端点通常返回“最近”的数据窗口
        """
        path = "/api/v5/market/candles"
        params: Dict[str, Any] = {"instId": inst_id, "bar": bar, "limit": str(limit)}
        if before is not None:
            params["before"] = str(before)
        if after is not None:
            params["after"] = str(after)
        return self._request_public("GET", path, params=params)

    def get_history_candles(self, inst_id: str, bar: str = "1m", before: Optional[str] = None, after: Optional[str] = None, limit: int = 100) -> OKXResponse:
        """获取历史 K 线：/api/v5/market/history-candles（更适合翻页拉历史）
        - 参数同 get_candles
        """
        path = "/api/v5/market/history-candles"
        params: Dict[str, Any] = {"instId": inst_id, "bar": bar, "limit": str(limit)}
        if before is not None:
            params["before"] = str(before)
        if after is not None:
            params["after"] = str(after)
        return self._request_public("GET", path, params=params)

    def fetch_ohlcv_range(self, inst_id: str, bar: str, start: datetime, end: datetime, limit_per_call: int = 100, use_history: bool = True) -> Tuple[bool, Optional[List[List[str]]]]:
        """分页拉取指定时间区间的 OHLCV（闭区间近似）。
        返回 (ok, raw_list)，其中 raw_list 为原始数组列表（每条形如 [ts, o, h, l, c, vol, ...]）。
        实现要点：
        - OKX 返回的 K 线通常是“倒序”（最新在前），因此分页时采用 before=上一次最早 ts-1ms 逐步向过去推进；
        - 同时使用 after=start_ms 对起点做硬性限界，防止服务端忽略 before 导致反复返回最新窗口；
        - 拉取完毕后由调用方自行排序与去重。
        """
        if end <= start:
            return False, None
        # OKX 参数使用毫秒时间戳
        start_ms = int(start.timestamp() * 1000)
        end_ms = int(end.timestamp() * 1000)
        all_rows: List[List[str]] = []
        cursor = end_ms
        # 新增：分页计数与累计条数，便于日志跟踪
        page = 0
        total = 0
        # 新增：游标推进保护与页数上限，避免死循环
        last_min_ts: Optional[int] = None
        stall_count = 0
        max_pages = 2000  # 安全上限，防止无限翻页
        # 新增：当使用 after=start_ms 无返回时，自动回退为仅使用 before 翻页
        try_with_after = True
        try:
            while True:
                # 选择端点：优先使用 history-candles，并根据 try_with_after 决定是否携带 after
                if use_history:
                    if try_with_after:
                        resp = self.get_history_candles(inst_id=inst_id, bar=bar, before=str(cursor), after=str(start_ms), limit=limit_per_call)
                    else:
                        resp = self.get_history_candles(inst_id=inst_id, bar=bar, before=str(cursor), limit=limit_per_call)
                else:
                    if try_with_after:
                        resp = self.get_candles(inst_id=inst_id, bar=bar, before=str(cursor), after=str(start_ms), limit=limit_per_call)
                    else:
                        resp = self.get_candles(inst_id=inst_id, bar=bar, before=str(cursor), limit=limit_per_call)
                # 失败告警
                if not resp.ok:
                    logger.warning("K线请求失败：code={} msg={} cursor={} bar={} inst={}", resp.code, resp.msg, cursor, bar, inst_id)
                    break
                # 若开启了 after 但返回为空，进行一次“关闭 after”的回退重试（仅首次触发）
                if try_with_after and (not resp.data or len(resp.data) == 0):
                    logger.warning("携带 after=start_ms 的请求返回空，回退为仅 before 翻页再试一次（cursor={}）", cursor)
                    try_with_after = False
                    # 回到循环顶部，用相同 cursor 重发
                    continue
                if not resp.data:
                    logger.info("分页结束：无更多数据（cursor={}）", cursor)
                    break
                rows: List[List[str]] = resp.data  # type: ignore
                if not rows:
                    logger.info("分页结束：本页为空（cursor={}）", cursor)
                    break
                page += 1
                rows_len = len(rows)
                total += rows_len
                # rows 可能为倒序（最新在前），找到这一批中最早/最晚的 ts
                ts_vals = [int(r[0]) for r in rows if len(r) > 0]
                if not ts_vals:
                    logger.info("分页 #{}：无有效时间戳，提前结束", page)
                    break
                min_ts = min(ts_vals)
                max_ts = max(ts_vals)
                # 人类可读时间范围
                try:
                    _iso = lambda ms: datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).isoformat(timespec="seconds")
                    logger.info("分页 #{}：本页 {} 条，时间范围 [{} ~ {}]，累计 {} 条", page, rows_len, _iso(min_ts), _iso(max_ts), total)
                except Exception:
                    logger.info("分页 #{}：本页 {} 条，累计 {} 条（时间解析失败）", page, rows_len, total)
                all_rows.extend(rows)
                # 终止条件：已经越过起点
                if min_ts <= start_ms:
                    logger.info("到达起点：min_ts={} <= start_ms={}，停止分页", min_ts, start_ms)
                    break
                # 游标未推进保护：若最小时间戳未变或变大，累计计数，连续多次则终止
                if last_min_ts is not None and min_ts >= last_min_ts:
                    stall_count += 1
                    logger.warning("检测到游标未推进（min_ts={} 上一页 min_ts={}），连续 {} 次", min_ts, last_min_ts, stall_count)
                    if stall_count >= 3:
                        logger.warning("游标连续未推进达到阈值，停止分页以避免死循环")
                        break
                else:
                    stall_count = 0
                last_min_ts = min_ts
                # 更新下一页 before 游标
                cursor = min_ts - 1
                # 页数上限保护
                if page >= max_pages:
                    logger.warning("达到页数上限 {}，停止分页以避免长时间等待", max_pages)
                    break
                # 简单限速，避免触发 429
                try:
                    import time
                    time.sleep(0.2)
                except Exception:
                    pass
        except Exception as e:
            logger.error("拉取 OHLCV 发生异常: {}", e)
            return False, None
        # 过滤区间并返回
        filtered = [r for r in all_rows if len(r) >= 5 and start_ms <= int(r[0]) <= end_ms]
        try:
            _iso2 = lambda ms: datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).isoformat(timespec="seconds")
            logger.info("分页完成：原始累计 {} 条，区间 [{} ~ {}] 过滤后 {} 条", total, _iso2(start_ms), _iso2(end_ms), len(filtered))
        except Exception:
            logger.info("分页完成：原始累计 {} 条，过滤后 {} 条（时间解析失败）", total, len(filtered))

        # 若过滤后为空，尝试使用“前向分页（仅 after 游标）”进行一次补拉，以规避部分区间 before/after 组合导致的空窗口/游标停滞
        if len(filtered) == 0:
            try:
                logger.info("过滤后为空，启用前向分页补拉（仅 after 游标）")
                f_all: List[List[str]] = []
                cursor2 = start_ms
                page2 = 0
                max_pages2 = 2000
                while True:
                    if use_history:
                        resp2 = self.get_history_candles(inst_id=inst_id, bar=bar, after=str(cursor2), limit=limit_per_call)
                    else:
                        resp2 = self.get_candles(inst_id=inst_id, bar=bar, after=str(cursor2), limit=limit_per_call)
                    if not resp2.ok:
                        logger.warning("前向分页请求失败：code={} msg={} cursor={} bar={} inst={}", resp2.code, resp2.msg, cursor2, bar, inst_id)
                        break
                    rows2: List[List[str]] = resp2.data or []  # type: ignore
                    if not rows2:
                        logger.info("前向分页结束：无更多数据（cursor={}）", cursor2)
                        break
                    page2 += 1
                    ts_vals2 = [int(r[0]) for r in rows2 if len(r) > 0]
                    if not ts_vals2:
                        logger.info("前向分页 #{}：无有效时间戳，提前结束", page2)
                        break
                    min_ts2 = min(ts_vals2)
                    max_ts2 = max(ts_vals2)
                    try:
                        _iso = lambda ms: datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).isoformat(timespec="seconds")
                        logger.info("前向分页 #{}：本页 {} 条，时间范围 [{} ~ {}]，累计 {} 条", page2, len(rows2), _iso(min_ts2), _iso(max_ts2), len(f_all) + len(rows2))
                    except Exception:
                        logger.info("前向分页 #{}：本页 {} 条，累计 {} 条（时间解析失败）", page2, len(rows2), len(f_all) + len(rows2))
                    f_all.extend(rows2)
                    if max_ts2 >= end_ms:
                        logger.info("前向到达终点：max_ts={} >= end_ms={}，停止", max_ts2, end_ms)
                        break
                    cursor2 = max_ts2 + 1
                    if page2 >= max_pages2:
                        logger.warning("前向分页达到页数上限 {}，停止", max_pages2)
                        break
                    try:
                        import time
                        time.sleep(0.2)
                    except Exception:
                        pass
                f_filtered = [r for r in f_all if len(r) >= 5 and start_ms <= int(r[0]) <= end_ms]
                try:
                    logger.info("前向分页完成：累计 {} 条，过滤后 {} 条", len(f_all), len(f_filtered))
                except Exception:
                    logger.info("前向分页完成：过滤后 {} 条（时间解析失败）", len(f_filtered))
                if f_filtered:
                    return True, f_filtered
            except Exception as _e:
                logger.warning("前向补拉异常：{}", _e)

        return True, filtered

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:
            pass