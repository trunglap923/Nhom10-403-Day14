"""
engine/cost_tracker.py
======================
Thành viên 4 -- Token & Cost Tracker

Module này theo dõi toàn bộ lượng token và chi phí (USD) phát sinh
trong mỗi lượt benchmark. Thiết kế theo dạng Singleton để dễ dàng
từ bất kỳ module nào trong pipeline.

Bảng giá tham khảo (tháng 4/2026):
    GPT-4o         : $5.00 / 1M tokens input | $15.00 / 1M tokens output
    GPT-4o-mini    : $0.15 / 1M tokens input | $0.60  / 1M tokens output
    Gemini-1.5-Pro : $3.50 / 1M tokens input | $10.50 / 1M tokens output
    Gemini-1.5-Flash: $0.35 / 1M tokens input| $1.05  / 1M tokens output
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List

# ---------------------------------------------------------------------------
# Bảng giá theo từng model (USD / 1_000_000 tokens)
# ---------------------------------------------------------------------------
PRICE_TABLE: Dict[str, Dict[str, float]] = {
    "gpt-4o": {
        "input":  5.00,
        "output": 15.00,
    },
    "gpt-4o-mini": {
        "input":  0.15,
        "output": 0.60,
    },
    "claude-3-5-sonnet": {
        "input":  3.00,
        "output": 15.00,
    },
    "gemini-1.5-pro": {
        "input":  3.50,
        "output": 10.50,
    },
    "gemini-1.5-flash": {
        "input":  0.35,
        "output": 1.05,
    },
    # Fallback nếu model chưa có trong bảng giá
    "default": {
        "input":  1.00,
        "output": 3.00,
    },
}


# ---------------------------------------------------------------------------
# Dataclass lưu kết quả theo dõi của 1 lần gọi API
# ---------------------------------------------------------------------------
@dataclass
class CallRecord:
    """Ghi nhận thông tin của 1 lần gọi model."""
    model:        str
    tokens_in:    int
    tokens_out:   int
    cost_usd:     float
    latency_ms:   float
    timestamp:    float = field(default_factory=time.time)
    case_id:      str   = ""


# ---------------------------------------------------------------------------
# CostTracker -- lớp trung tâm theo dõi toàn bộ chi phí
# ---------------------------------------------------------------------------
class CostTracker:
    """
    Theo dõi token và chi phí qua toàn bộ phiên benchmark.

    Cách dùng:
        tracker = CostTracker()
        tracker.record("gpt-4o", tokens_in=500, tokens_out=200, latency_ms=1200)
        report = tracker.report()
    """

    def __init__(self) -> None:
        self._records: List[CallRecord] = []
        self._started_at: float = time.time()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record(
        self,
        model: str,
        tokens_in: int,
        tokens_out: int,
        latency_ms: float,
        case_id: str = "",
    ) -> CallRecord:
        """
        Ghi nhận 1 lần gọi model vào tracker.

        Args:
            model:       Tên model (vd: "gpt-4o").
            tokens_in:   Số token đầu vào (prompt).
            tokens_out:  Số token đầu ra (completion).
            latency_ms:  Thời gian phản hồi tính bằng mili-giây.
            case_id:     ID của test case tương ứng (tùy chọn).

        Returns:
            CallRecord với chi phí đã tính.
        """
        cost = self._calculate_cost(model, tokens_in, tokens_out)
        rec = CallRecord(
            model=model,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=cost,
            latency_ms=latency_ms,
            case_id=case_id,
        )
        self._records.append(rec)
        return rec

    def report(self) -> Dict:
        """
        Tổng hợp toàn bộ báo cáo chi phí và token của phiên benchmark.

        Returns:
            Dict chứa: tổng token, tổng chi phí, breakdown theo model,
                       avg latency, và ước tính chi phí mở rộng (extrapolation).
        """
        if not self._records:
            return {
                "total_calls": 0,
                "total_tokens_in": 0,
                "total_tokens_out": 0,
                "total_tokens": 0,
                "total_cost_usd": 0.0,
                "avg_cost_per_call_usd": 0.0,
                "avg_latency_ms": 0.0,
                "by_model": {},
                "cost_extrapolation": {},
                "elapsed_seconds": round(time.time() - self._started_at, 2),
            }

        total_in    = sum(r.tokens_in  for r in self._records)
        total_out   = sum(r.tokens_out for r in self._records)
        total_cost  = sum(r.cost_usd   for r in self._records)
        avg_latency = sum(r.latency_ms for r in self._records) / len(self._records)
        n           = len(self._records)

        # Phân tích theo từng model
        by_model: Dict[str, Dict] = {}
        for rec in self._records:
            m = rec.model
            if m not in by_model:
                by_model[m] = {
                    "calls": 0,
                    "tokens_in": 0,
                    "tokens_out": 0,
                    "cost_usd": 0.0,
                    "avg_latency_ms": 0.0,
                    "_latency_sum": 0.0,
                }
            by_model[m]["calls"]        += 1
            by_model[m]["tokens_in"]    += rec.tokens_in
            by_model[m]["tokens_out"]   += rec.tokens_out
            by_model[m]["cost_usd"]     += rec.cost_usd
            by_model[m]["_latency_sum"] += rec.latency_ms

        for m, stats in by_model.items():
            stats["cost_usd"]      = round(stats["cost_usd"], 6)
            stats["avg_latency_ms"] = round(stats["_latency_sum"] / stats["calls"], 2)
            del stats["_latency_sum"]

        # Ước tính nếu chạy quy mô lớn hơn
        cost_per_call = total_cost / n if n > 0 else 0.0
        extrapolation = {
            "100_cases_usd":   round(cost_per_call * 100,   4),
            "1000_cases_usd":  round(cost_per_call * 1000,  4),
            "10000_cases_usd": round(cost_per_call * 10000, 4),
        }

        return {
            "total_calls":          n,
            "total_tokens_in":      total_in,
            "total_tokens_out":     total_out,
            "total_tokens":         total_in + total_out,
            "total_cost_usd":       round(total_cost, 6),
            "avg_cost_per_call_usd": round(cost_per_call, 6),
            "avg_latency_ms":       round(avg_latency, 2),
            "by_model":            by_model,
            "cost_extrapolation":  extrapolation,
            "elapsed_seconds":     round(time.time() - self._started_at, 2),
        }

    def reset(self) -> None:
        """Xóa toàn bộ records để bắt đầu phiên benchmark mới."""
        self._records.clear()
        self._started_at = time.time()

    def __len__(self) -> int:
        return len(self._records)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _calculate_cost(model: str, tokens_in: int, tokens_out: int) -> float:
        """
        Tính chi phí (USD) dựa trên bảng giá PRICE_TABLE.
        Nếu model chưa có trong bảng ở dạng giá "default".

        Công thức:
            cost = (tokens_in  / 1_000_000) * price_input
                 + (tokens_out / 1_000_000) * price_output
        """
        # Chuẩn hóa tên model (lowercase, bỏ ký tự thừa)
        key = model.lower().strip()

        # Tìm giá phù hợp nhất (partial match)
        price = None
        for table_key, table_price in PRICE_TABLE.items():
            if table_key == "default":
                continue
            if key.startswith(table_key) or table_key in key:
                price = table_price
                break

        if price is None:
            price = PRICE_TABLE["default"]

        cost = (tokens_in  / 1_000_000) * price["input"] \
             + (tokens_out / 1_000_000) * price["output"]
        return round(cost, 8)


# Singleton instance dùng chung toàn project
# ---------------------------------------------------------------------------
# Dùng như sau trong bất kỳ file nào:
#   from engine.cost_tracker import global_tracker
#   global_tracker.record("gpt-4o", tokens_in=500, tokens_out=200, latency_ms=800)
# ---------------------------------------------------------------------------
global_tracker = CostTracker()
