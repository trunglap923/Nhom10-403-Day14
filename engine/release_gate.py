"""
engine/release_gate.py
======================
Thành viên 4 -- Regression Release Gate

Module n?y so s?nh k?t qu? c?a Agent V2 (m?i) v?i V1 (c?) tr?n nhi?u chi?u
v? ??a ra quy?t ??nh t? ??ng: APPROVE (tri?n khai) ho?c ROLLBACK (t? ch?i).

Tri?t l? thi?t k?:
    - Kh?ng ch? nh?n v?o 1 s? duy nh?t (avg_score).
    - C?n nh?c ??ng th?i: Ch?t l??ng, Chi ph?, Hi?u n?ng v? ?? ?n ??nh.
    - C? ng??ng c?ng (Hard Block) v? ng??ng c?nh b?o (Soft Warning).

S? ?? quy?t ??nh:
    ┌─────────────────────────────────────┐
    │   So sánh V1 vs V2 trên 4 chiều     │
    │   Quality | Cost | Speed | Stability│
    └─────────────────┬───────────────────┘
                      │
         ┌────────────┴────────────┐
         │  Vi phạm Hard Block?    │
         └────┬──────────────┬─────┘
             YES             NO
              │               │
         ROLLBACK      ┌──────┴──────┐
                       │ Đủ điểm APPROVE?│
                       └──────┬──────┘
                    ┌─────────┴─────────┐
                   YES                 NO
                    │                   │
                APPROVE           HOLD (cảnh báo)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Cấu hình ngưỡng (có thể chỉnh từ main.py)
# ---------------------------------------------------------------------------
@dataclass
class GateConfig:
    """
    Các ngưỡng kiểm tra của Release Gate.

    Hard Blocks -- vi phạm bất kỳ điều nào -> ROLLBACK ngay:
        min_score_delta    : V2 phải không tệ hơn V1 quá ngưỡng này.
        max_cost_increase  : V2 không được đắt hơn V1 quá X%.
        max_error_rate     : Tỷ lệ case lỗi hệ thống (status='error') tối đa.

    Soft Requirements -- cần đủ điểm để APPROVE:
        required_score_gain : V2 phải cải thiện ít nhất X điểm.
        max_latency_increase: V2 không được chậm hơn V1 quá X%.
        min_agreement_rate  : Độ đồng thuận giữa các Judge phải đạt tối thiểu.
    """
    # Hard Blocks
    min_score_delta: float     = -0.3   # -0.3 = cho phép giảm tối đa 0.3 điểm
    max_cost_increase: float   = 0.30   # Tăng chi phí tối đa 30%
    max_error_rate: float      = 0.10   # Tối đa 10% case bị lỗi hệ thống

    # Soft Requirements
    required_score_gain: float    = 0.0    # Score phải tăng (>= 0)
    max_latency_increase: float   = 0.50   # Latency tăng tối đa 50%
    min_agreement_rate: float     = 0.60   # Tối thiểu 60% đồng thuận


# ---------------------------------------------------------------------------
# Kết quả của từng chiều đánh giá
# ---------------------------------------------------------------------------
@dataclass
class DimensionResult:
    name:        str
    v1_value:    float
    v2_value:    float
    delta:       float
    delta_pct:   float
    passed:      bool
    is_hard:     bool          # True = Hard Block nếu fail
    message:     str


# ---------------------------------------------------------------------------
# Kết quả tổng hợp
# ---------------------------------------------------------------------------
@dataclass
class GateDecision:
    decision:     str                          # "APPROVE" | "ROLLBACK" | "HOLD"
    confidence:   float                        # 0.0 -> 1.0
    reasons:      List[str] = field(default_factory=list)
    warnings:     List[str] = field(default_factory=list)
    dimensions:   List[DimensionResult] = field(default_factory=list)
    v1_summary:   Dict = field(default_factory=dict)
    v2_summary:   Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        v1_m = self.v1_summary.get("metrics", {})
        v2_m = self.v2_summary.get("metrics", {})
        
        return {
            "v1": {
                "score": v1_m.get("avg_score", 0),
                "hit_rate": v1_m.get("hit_rate", 0),
                "judge_agreement": v1_m.get("agreement_rate", 0)
            },
            "v2": {
                "score": v2_m.get("avg_score", 0),
                "hit_rate": v2_m.get("hit_rate", 0),
                "judge_agreement": v2_m.get("agreement_rate", 0)
            },
            "decision": self.decision
        }

    def print_report(self) -> None:
        """In báo cáo đẹp ra console."""
        SEP = "-" * 60
        ICON = {"APPROVE": "[OK]", "ROLLBACK": "[ERR]", "HOLD": "[WARN]"}

        print(f"\n{SEP}")
        print(f"  [STATS]  REGRESSION RELEASE GATE -- BÁO CÁO QUYẾT ĐỊNH")
        print(SEP)

        # Bảng so sánh tổng chiều
        print(f"\n{'Chiều đánh giá':<22} {'V1':>8} {'V2':>8} {'Delta':>8} {'Kết quả':<10}")
        print("-" * 60)
        for d in self.dimensions:
            icon = "[OK]" if d.passed else ("✘" if d.is_hard else "⚠")
            print(f"  {d.name:<20} {d.v1_value:>8.3f} {d.v2_value:>8.3f} "
                  f"{d.delta:>+8.3f}  {icon}")

        # Lý do quyết định
        if self.reasons:
            print(f"\n[LIST] Lý do:")
            for r in self.reasons:
                print(f"   • {r}")

        # Cảnh báo
        if self.warnings:
            print(f"\n[WARN]  Cảnh báo:")
            for w in self.warnings:
                print(f"   • {w}")

        # Quyết định cuối
        icon = ICON.get(self.decision, "?")
        print(f"\n{SEP}")
        print(f"  {icon}  QUYẾT ĐỊNH: {self.decision}  "
              f"(Độ tin cậy: {self.confidence*100:.0f}%)")
        print(SEP)


# ---------------------------------------------------------------------------
# RegressionReleaseGate -- lớp chính
# ---------------------------------------------------------------------------
class RegressionReleaseGate:
    """
    So sánh V1 và V2 trên 4 chiều rồi đưa ra quyết định tự động.

    Cách dùng:
        gate = RegressionReleaseGate()
        decision = gate.evaluate(v1_summary, v2_summary, v1_results, v2_results)
        decision.print_report()
        gate_dict = decision.to_dict()
    """

    def __init__(self, config: Optional[GateConfig] = None) -> None:
        self.config = config or GateConfig()

    def evaluate(
        self,
        v1_summary: Dict,
        v2_summary: Dict,
        v1_results: Optional[List[Dict]] = None,
        v2_results: Optional[List[Dict]] = None,
    ) -> GateDecision:
        """
        Chạy toàn bộ quá trình so sánh và trả về GateDecision.

        Args:
            v1_summary: Dict summary của V1 (có trường 'metrics').
            v2_summary: Dict summary của V2 (có trường 'metrics').
            v1_results: (Tùy chọn) Danh sách kết quả raw của V1.
            v2_results: (Tùy chọn) Danh sách kết quả raw của V2.
        """
        v1_m = v1_summary.get("metrics", {})
        v2_m = v2_summary.get("metrics", {})

        dimensions: List[DimensionResult] = []
        reasons:    List[str] = []
        warnings:   List[str] = []

        # -- Chiếu 1: Quality Score --------------------------------------
        dim_quality = self._check_quality(v1_m, v2_m)
        dimensions.append(dim_quality)

        # -- Chiếu 2: Cost -----------------------------------------------
        dim_cost = self._check_cost(v1_results or [], v2_results or [])
        dimensions.append(dim_cost)

        # -- Chiếu 3: Latency / Speed ------------------------------------
        dim_latency = self._check_latency(v1_results or [], v2_results or [])
        dimensions.append(dim_latency)

        # -- Chiếu 4: Stability (Error Rate) ----------------------------
        dim_stability = self._check_stability(v2_results or [])
        dimensions.append(dim_stability)

        # -- Chiếu 5: Multi-Judge Agreement -----------------------------
        dim_agreement = self._check_agreement(v1_m, v2_m)
        dimensions.append(dim_agreement)

        # -- Chiếu 6: Retrieval Hit Rate ---------------------------------
        dim_retrieval = self._check_retrieval(v1_m, v2_m)
        dimensions.append(dim_retrieval)

        # -- Ra quyết định -----------------------------------------------
        hard_blocks    = [d for d in dimensions if d.is_hard and not d.passed]
        soft_failures  = [d for d in dimensions if not d.is_hard and not d.passed]
        passed_dims    = [d for d in dimensions if d.passed]

        # Hard Block ? ROLLBACK ngay
        if hard_blocks:
            for d in hard_blocks:
                reasons.append(f"[HARD] {d.name}: {d.message}")
            decision  = "ROLLBACK"
            confidence = 0.95

        # Đủ điều kiện -> APPROVE
        elif len(passed_dims) >= (len(dimensions) - 1):    # Cho phép tối đa 1 soft fail
            decision   = "APPROVE"
            confidence = len(passed_dims) / len(dimensions)
            reasons.append(f"V2 cải thiện trên {len(passed_dims)}/{len(dimensions)} chiều đánh giá.")
            for d in soft_failures:
                warnings.append(f"[SOFT] {d.name}: {d.message}")

        # Không đủ điều kiện rõ ràng -> HOLD
        else:
            decision   = "HOLD"
            confidence = 0.5
            reasons.append(f"V2 không đạt yêu cầu tối thiểu "
                           f"({len(passed_dims)}/{len(dimensions)} chiều OK).")
            for d in soft_failures:
                reasons.append(f"[SOFT] {d.name}: {d.message}")

        return GateDecision(
            decision=decision,
            confidence=confidence,
            reasons=reasons,
            warnings=warnings,
            dimensions=dimensions,
            v1_summary=v1_summary,
            v2_summary=v2_summary,
        )

    # ------------------------------------------------------------------
    # Private: Kiểm tra từng chiều
    # ------------------------------------------------------------------

    def _check_quality(self, v1_m: Dict, v2_m: Dict) -> DimensionResult:
        v1 = v1_m.get("avg_score", 0.0)
        v2 = v2_m.get("avg_score", 0.0)
        delta  = v2 - v1
        pct    = (delta / v1 * 100) if v1 else 0.0
        cfg    = self.config

        # Hard: không được giảm quá ngưỡng
        if delta < cfg.min_score_delta:
            return DimensionResult(
                "Quality Score", v1, v2, delta, pct, False, True,
                f"Giảm {delta:.3f} -- vượt ngưỡng cho phép ({cfg.min_score_delta})",
            )
        # Soft: phải tăng
        passed = delta >= cfg.required_score_gain
        return DimensionResult(
            "Quality Score", v1, v2, delta, pct, passed, False,
            f"{'Tăng' if delta >= 0 else 'Giảm'} {abs(delta):.3f} điểm",
        )

    def _check_cost(
        self, v1_results: List[Dict], v2_results: List[Dict]
    ) -> DimensionResult:
        v1 = self._avg_field(v1_results, "tokens_in") + self._avg_field(v1_results, "tokens_out")
        v2 = self._avg_field(v2_results, "tokens_in") + self._avg_field(v2_results, "tokens_out")
        delta  = v2 - v1
        pct    = (delta / v1 * 100) if v1 else 0.0
        change = pct / 100  # T? l? thay ??i d?ng th?p ph?n

        # Hard: chi phí tăng quá ngưỡng
        if change > self.config.max_cost_increase:
            return DimensionResult(
                "Avg Tokens/Call", v1, v2, delta, pct, False, True,
                f"Tổng token tăng {pct:+.1f}% -- vượt ngưỡng {self.config.max_cost_increase*100:.0f}%",
            )
        passed = change <= 0  # Tốt nhất là không tăng
        return DimensionResult(
            "Avg Tokens/Call", v1, v2, delta, pct, passed, False,
            f"Tổng token {'giảm' if delta <= 0 else 'tăng'} {abs(pct):.1f}%",
        )

    def _check_latency(
        self, v1_results: List[Dict], v2_results: List[Dict]
    ) -> DimensionResult:
        v1 = self._avg_field(v1_results, "latency_ms")
        v2 = self._avg_field(v2_results, "latency_ms")
        delta = v2 - v1
        pct   = (delta / v1 * 100) if v1 else 0.0
        change = pct / 100

        # Soft: latency tăng quá ngưỡng -> warning
        passed = change <= self.config.max_latency_increase
        return DimensionResult(
            "Avg Latency (ms)", v1, v2, delta, pct, passed, False,
            f"Latency {'giảm' if delta <= 0 else 'tăng'} {abs(pct):.1f}%"
            + (f" -- vượt ngưỡng {self.config.max_latency_increase*100:.0f}%" if not passed else ""),
        )

    def _check_stability(self, v2_results: List[Dict]) -> DimensionResult:
        if not v2_results:
            return DimensionResult("Error Rate", 0, 0, 0, 0, True, False, "Không có dữ liệu")

        total  = len(v2_results)
        errors = sum(1 for r in v2_results if r.get("status") == "error")
        rate   = errors / total

        # Hard: quá nhiều lỗi hệ thống
        if rate > self.config.max_error_rate:
            return DimensionResult(
                "Error Rate", 0, rate, rate, rate * 100, False, True,
                f"{errors}/{total} cases lỗi hệ thống ({rate*100:.1f}%) -- vượt ngưỡng {self.config.max_error_rate*100:.0f}%",
            )
        return DimensionResult(
            "Error Rate", 0, rate, rate, rate * 100, True, False,
            f"{errors}/{total} cases lỗi ({rate*100:.1f}%)",
        )

    def _check_agreement(self, v1_m: Dict, v2_m: Dict) -> DimensionResult:
        v1 = v1_m.get("agreement_rate", 0.0)
        v2 = v2_m.get("agreement_rate", 0.0)
        delta = v2 - v1
        pct   = (delta / v1 * 100) if v1 else 0.0

        # Soft: agreement rate phải đạt tối thiểu
        passed = v2 >= self.config.min_agreement_rate
        return DimensionResult(
            "Judge Agreement", v1, v2, delta, pct, passed, False,
            f"Agreement rate V2={v2:.2f}"
            + (f" (dưới ngưỡng {self.config.min_agreement_rate:.2f})" if not passed else ""),
        )

    def _check_retrieval(self, v1_m: Dict, v2_m: Dict) -> DimensionResult:
        v1 = v1_m.get("hit_rate", 0.0)
        v2 = v2_m.get("hit_rate", 0.0)
        delta = v2 - v1
        pct   = (delta / v1 * 100) if v1 else 0.0

        # Soft: Retrieval không nên giảm
        passed = delta >= -0.05  # Cho phép giảm tối đa 5%
        return DimensionResult(
            "Retrieval Hit Rate", v1, v2, delta, pct, passed, False,
            f"Hit rate {'tăng' if delta >= 0 else 'giảm'} {abs(delta):.3f}",
        )

    @staticmethod
    def _avg_field(results: List[Dict], field: str) -> float:
        """Tính trung bình một trường số trong danh sách kết quả."""
        vals = [r.get(field, 0) for r in results if isinstance(r.get(field), (int, float))]
        return sum(vals) / len(vals) if vals else 0.0
