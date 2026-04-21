"""
main.py
=======
Điểm khởi chạy chính của AI Evaluation Factory -- Lab Day 14.

Pipeline hoàn chỉnh:
    1. Đọc Golden Dataset (golden_set.jsonl)
    2. Chạy Benchmark V1 (Agent_V1_Base)    – Async, song song, có tracker
    3. Chạy Benchmark V2 (Agent_V2_Optimized)
    4. So sánh V1 vs V2 bằng RegressionReleaseGate
    5. Xuất báo cáo: reports/summary.json + reports/benchmark_results.json
    6. In kết quả và quyết định Deploy/Rollback ra console

Thành viên 4 chịu trách nhiệm:
    - engine/runner.py      (Async + Semaphore + Retry)
    - engine/cost_tracker.py (Token & Cost giám sát)
    - engine/release_gate.py (Regression Release Gate)
    - main.py               (Orchestrator & Report)
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

# Cấu hình console để hiển thị tiếng Việt (UTF-8)
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

from agent.main_agent import MainAgent
from engine.cost_tracker import CostTracker
from engine.release_gate import GateConfig, RegressionReleaseGate
from engine.runner import BenchmarkRunner
from agent.main_agent import MainAgent
from engine.llm_judge import LLMJudge

# ---------------------------------------------------------------------------
# Mock components cho TV2, TV3 (sẽ thay bằng module thật khi đồng đội xong)
# ---------------------------------------------------------------------------
# [WARN]  Khi TV2 n?p engine/retrieval_eval.py ? uncomment d?ng d??i:
# from engine.retrieval_eval import RetrievalEvaluator  # noqa
# [WARN]  Khi TV3 n?p engine/llm_judge.py ? uncomment d?ng d??i:
# from engine.llm_judge import LLMJudge                 # noqa

class _MockEvaluator:
    """Stub đại diện cho TV2 -- RetrievalEvaluator."""
    async def score(self, case: Dict, resp: Dict) -> Dict:
        # Mô phỏng kết quả RAGAS + Retrieval
        return {
            "faithfulness": 0.88,
            "relevancy":    0.82,
            "retrieval": {
                "hit_rate": 1.0 if resp.get("contexts") else 0.0,
                "mrr":      0.75,
            },
        }


# ---------------------------------------------------------------------------
# Hàm chạy benchmark cho 1 phiên bản agent
# ---------------------------------------------------------------------------
async def run_benchmark(
    agent_version: str,
    dataset: List[Dict],
    tracker: Optional[CostTracker] = None,
    concurrency: int = 5,
) -> Tuple[Optional[List[Dict]], Optional[Dict]]:
    """
    Chạy toàn bộ benchmark cho 1 phiên bản agent.

    Args:
        agent_version: Tên phiên bản (vd: "Agent_V1_Base").
        dataset:       Danh sách test cases.
        tracker:       CostTracker riêng cho phiên này.
        concurrency:   Số request chạy đồng thời tối đa.

    Returns:
        (results, summary) hoặc (None, None) nếu thất bại.
    """
    print(f"\n{'='*60}")
    print(f"  [>>]  BENCHMARK: {agent_version}")
    print(f"{'='*60}")

    if tracker is None:
        tracker = CostTracker()

    # -- Khởi tạo Runner với Async + CostTracker ----------------------
    runner = BenchmarkRunner(
        agent       = MainAgent(),
        evaluator   = _MockEvaluator(),   # TODO: Thay bằng RetrievalEvaluator() khi TV2 xong
        judge       = LLMJudge(["gpt-4o-mini", "gpt-4o"]), # Đã cắm LLMJudge của TV3
        concurrency = concurrency,
        max_retries = 2,
        tracker     = tracker,
    )

    results = await runner.run_all(dataset)

    if not results:
        print("[ERR] Không có kết quả nào.")
        return None, None

    # -- Tổng hợp metrics ---------------------------------------------
    total         = len(results)
    passed        = sum(1 for r in results if r.get("status") == "pass")
    failed        = sum(1 for r in results if r.get("status") == "fail")
    error_cases   = sum(1 for r in results if r.get("status") == "error")

    avg_score     = _safe_avg(results, lambda r: r["judge"]["final_score"])
    avg_hit_rate  = _safe_avg(results, lambda r: r["ragas"].get("retrieval", {}).get("hit_rate", 0))
    avg_agreement = _safe_avg(results, lambda r: r["judge"].get("agreement_rate", 0))
    avg_latency   = _safe_avg(results, lambda r: r.get("latency_ms", 0))

    cost_report   = tracker.report()

    summary = {
        "metadata": {
            "version":          agent_version,
            "total":            total,
            "passed":           passed,
            "failed":           failed,
            "errors":           error_cases,
            "timestamp":        time.strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_seconds":  cost_report["elapsed_seconds"],
        },
        "metrics": {
            "avg_score":        round(avg_score,     4),
            "hit_rate":         round(avg_hit_rate,  4),
            "agreement_rate":   round(avg_agreement, 4),
            "avg_latency_ms":   round(avg_latency,   2),
        },
        "cost": {
            "total_tokens_in":       cost_report["total_tokens_in"],
            "total_tokens_out":      cost_report["total_tokens_out"],
            "total_tokens":          cost_report["total_tokens"],
            "total_cost_usd":        cost_report["total_cost_usd"],
            "avg_cost_per_call_usd": cost_report["avg_cost_per_call_usd"],
            "cost_extrapolation":    cost_report["cost_extrapolation"],
            "by_model":              cost_report["by_model"],
        },
    }

    print(f"\n✔ Kết quả {agent_version}: "
          f"Score={avg_score:.2f} | Hit Rate={avg_hit_rate:.1%} | "
          f"Pass={passed}/{total} | Cost=${cost_report['total_cost_usd']:.6f}")

    return results, summary


# ---------------------------------------------------------------------------
# main() -- Orchestrator chính
# ---------------------------------------------------------------------------
async def main() -> None:
    """
    Pipeline tổng thể:
        1. Load dataset
        2. Benchmark V1
        3. Benchmark V2
        4. Regression Gate -> quyết định
        5. Lưu reports
    """
    print("[AI EVALUATION FACTORY] Lab Day 14")
    print("=" * 60)

    # -- Buoc 1: Load Golden Dataset --
    dataset_path = "data/golden_set.jsonl"
    if not os.path.exists(dataset_path):
        print(f"[ERR] Thiếu {dataset_path}. Hãy chạy 'python data/synthetic_gen.py' trước.")
        return

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f if line.strip()]

    if not dataset:
        print("[ERR] File golden_set.jsonl rỗng.")
        return

    print(f"[OK] Đã load {len(dataset)} test cases từ {dataset_path}")

    # -- Bước 2: Benchmark V1 --
    tracker_v1 = CostTracker()
    v1_results, v1_summary = await run_benchmark(
        "Agent_V1_Base", dataset, tracker=tracker_v1, concurrency=5
    )

    if not v1_summary:
        print("[ERROR] Benchmark V1 thất bại.")
        return

    # -- Bước 3: Benchmark V2 --
    # V2 giả lập với dataset tương tự nhưng sẽ dùng agent/judge khác
    tracker_v2 = CostTracker()
    v2_results, v2_summary = await run_benchmark(
        "Agent_V2_Optimized", dataset, tracker=tracker_v2, concurrency=5
    )

    if not v2_summary:
        print("[ERROR] Benchmark V2 thất bại.")
        return

    # -- Bước 4: Regression Release Gate --
    print(f"\n{'='*60}")
    print("  [GATE]  PHÂN TÍCH REGRESSION")
    print(f"{'='*60}")

    gate   = RegressionReleaseGate(config=GateConfig())
    decision = gate.evaluate(
        v1_summary  = v1_summary,
        v2_summary  = v2_summary,
        v1_results  = v1_results,
        v2_results  = v2_results,
    )
    decision.print_report()

    # -- Bước 5: Lưu reports ------------------------------------------
    os.makedirs("reports", exist_ok=True)

    # summary.json -- bổ sung thêm kết quả gate
    final_summary = {
        **v2_summary,
        "regression": decision.to_dict(),
        "v1_metrics": v1_summary["metrics"],
    }

    with open("reports/summary.json", "w", encoding="utf-8") as f:
        json.dump(final_summary, f, ensure_ascii=False, indent=2)

    with open("reports/benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(v2_results, f, ensure_ascii=False, indent=2)

    print(f"\n[OK] Đã lưu reports/summary.json")
    print(f"[OK] Đã lưu reports/benchmark_results.json")
    print(f"\n[TIP] Tiếp theo: chạy 'python check_lab.py' để kiểm tra định dạng nộp bài.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _safe_avg(results: List[Dict], extractor) -> float:
    """Tính trung bình an toàn, bỏ qua các case lỗi."""
    vals = []
    for r in results:
        try:
            v = extractor(r)
            if isinstance(v, (int, float)):
                vals.append(v)
        except (KeyError, TypeError):
            pass
    return sum(vals) / len(vals) if vals else 0.0


if __name__ == "__main__":
    asyncio.run(main())
