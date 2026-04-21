"""
engine/runner.py
================
Thành viên 4 -- Async Benchmark Runner (Nâng cấp)

Cải tiến so với phiên bản gốc:
    1. asyncio.Semaphore  – Kiểm soát số request đồng thời, tránh Rate Limit.
    2. Retry với backoff  – Tự động thử lại khi gặp lỗi tạm thời (429, 503...).
    3. Progress bar (tqdm)– Hiển thị tiến trình chạy theo thời gian thực.
    4. CostTracker tích hợp – Ghi nhận token + chi phí từng case.
    5. Timeout per case   – Một case treo quá lâu sẽ bị bỏ qua, không block cả pipeline.
    6. Detailed result    – Mỗi kết quả chứa đầy đủ thông tin: latency, cost, status, error.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List, Optional

from tqdm.asyncio import tqdm as async_tqdm

from .cost_tracker import CostTracker, global_tracker

# ---------------------------------------------------------------------------
# Hằng số mặc định
# ---------------------------------------------------------------------------
DEFAULT_CONCURRENCY  = 5    # Tối đa 5 request chạy song song cùng lúc
DEFAULT_MAX_RETRIES  = 3    # Thử lại tối đa 3 lần khi lỗi
DEFAULT_RETRY_DELAY  = 2.0  # Giây chờ giữa 2 lần thử (tăng theo hàm mũ)
DEFAULT_TIMEOUT      = 60.0 # Giây timeout tối đa cho 1 test case


# ---------------------------------------------------------------------------
# BenchmarkRunner
# ---------------------------------------------------------------------------
class BenchmarkRunner:
    """
    Chạy toàn bộ benchmark dataset một cách bất đồng bộ (async) với:
    - Kiểm soát concurrency qua Semaphore
    - Tự động retry khi API gặp lỗi tạm thời
    - Theo dõi chi phí qua CostTracker

    Cách dùng:
        runner = BenchmarkRunner(agent, evaluator, judge, concurrency=5)
        results = await runner.run_all(dataset)
        cost_report = runner.tracker.report()
    """

    def __init__(
        self,
        agent,
        evaluator,
        judge,
        concurrency: int = DEFAULT_CONCURRENCY,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_delay: float = DEFAULT_RETRY_DELAY,
        timeout: float = DEFAULT_TIMEOUT,
        tracker: Optional[CostTracker] = None,
    ) -> None:
        self.agent       = agent
        self.evaluator   = evaluator
        self.judge       = judge
        self.concurrency = concurrency
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout     = timeout
        # D?ng global_tracker ho?c tracker ri?ng truy?n v?o
        self.tracker     = tracker or global_tracker

        # Semaphore tạo ra ở đây nhưng chỉ được dùng bên trong event loop
        self._semaphore: Optional[asyncio.Semaphore] = None

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    async def run_all(
        self,
        dataset: List[Dict],
        batch_size: int = DEFAULT_CONCURRENCY,
        show_progress: bool = True,
    ) -> List[Dict]:
        """
        Chạy toàn bộ dataset song song với thanh tiến trình.

        Args:
            dataset:       Danh sách test cases (mỗi case là 1 dict).
            batch_size:    Không cần dùng để xử lý batch thủ công --
                           thay vào đó để Semaphore tự giới hạn concurrency.
            show_progress: Hiển thị thanh tqdm hay không.

        Returns:
            Danh sách kết quả tương ứng với từng test case.
        """
        # Tạo Semaphore trong event loop hiện tại
        self._semaphore = asyncio.Semaphore(self.concurrency)

        start_total = time.perf_counter()
        print(f"\n[ASYNC] Bắt đầu Benchmark: {len(dataset)} cases | "
              f"concurrency={self.concurrency} | max_retries={self.max_retries}")

        # T?o coroutine cho t?ng case
        tasks = [
            self._run_with_semaphore(case, idx)
            for idx, case in enumerate(dataset)
        ]

        # Ch?y song song, hi?n th? progress bar
        if show_progress:
            results = await async_tqdm.gather(
                *tasks,
                desc="[RETRY] Benchmarking",
                total=len(tasks),
                unit="case",
            )
        else:
            results = await asyncio.gather(*tasks)

        elapsed = time.perf_counter() - start_total

        # Th?ng k? k?t qu?
        passed  = sum(1 for r in results if r.get("status") == "pass")
        failed  = sum(1 for r in results if r.get("status") == "fail")
        errored = sum(1 for r in results if r.get("status") == "error")

        print(f"\n[OK] Hoàn thành trong {elapsed:.1f}s | "
              f"Pass: {passed} | Fail: {failed} | Error: {errored}")

        # In báo cáo chi phí nhanh
        self._print_cost_summary()

        return list(results)

    async def run_single_test(self, test_case: Dict, case_id: str = "") -> Dict:
        """
        Chạy 1 test case hoàn chỉnh qua 3 bước:
            1. Agent.query()         -- lấy câu trả lời
            2. Evaluator.score()     -- tính RAGAS metrics
            3. Judge.evaluate()      -- Multi-Judge chấm điểm

        Có tích hợp:
            - Đo latency chính xác (perf_counter)
            - Ghi nhận token & cost vào CostTracker
            - Timeout bảo vệ pipeline
        """
        start = time.perf_counter()

        try:
            # -- Bước 1: Gọi Agent --------------------------------------
            response = await asyncio.wait_for(
                self.agent.query(test_case["question"]),
                timeout=self.timeout,
            )
            latency_ms = (time.perf_counter() - start) * 1000

            # -- Ghi nhận token & cost (nếu agent trả về metadata) ------
            metadata   = response.get("metadata", {})
            model_name = metadata.get("model", "default")
            tokens_in  = metadata.get("tokens_used", 0)  # agent mẫu chỉ có tokens_used
            tokens_out = metadata.get("tokens_out", 0)

            self.tracker.record(
                model=model_name,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                latency_ms=round(latency_ms, 2),
                case_id=case_id,
            )

            # -- Bước 2: Tính RAGAS / Retrieval metrics ------------------
            ragas_scores = await asyncio.wait_for(
                self.evaluator.score(test_case, response),
                timeout=self.timeout,
            )

            # -- Bước 3: Multi-Judge -------------------------------------
            judge_result = await asyncio.wait_for(
                self.judge.evaluate_multi_judge(
                    test_case["question"],
                    response["answer"],
                    test_case.get("expected_answer", ""),
                ),
                timeout=self.timeout,
            )

            final_score = judge_result.get("final_score", 0)
            status = "pass" if final_score >= 3 else "fail"

            return {
                "case_id":        case_id,
                "test_case":      test_case["question"],
                "agent_response": response["answer"],
                "latency_ms":     round(latency_ms, 2),
                "tokens_in":      tokens_in,
                "tokens_out":     tokens_out,
                "model":          model_name,
                "ragas":          ragas_scores,
                "judge":          judge_result,
                "status":         status,
                "error":          None,
            }

        except asyncio.TimeoutError:
            return self._error_result(test_case, case_id, "TimeoutError",
                                      f"Vượt quá {self.timeout}s timeout")
        except KeyError as exc:
            return self._error_result(test_case, case_id, "KeyError",
                                      f"Thiếu trường bắt buộc: {exc}")
        except Exception as exc:
            return self._error_result(test_case, case_id, type(exc).__name__, str(exc))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _run_with_semaphore(self, test_case: Dict, idx: int) -> Dict:
        """
        Bọc run_single_test bên trong Semaphore + Retry logic.
        Semaphore đảm bảo tối đa `self.concurrency` case chạy đồng thời.
        """
        assert self._semaphore is not None, "Semaphore chưa được khởi tạo."
        async with self._semaphore:
            return await self._run_with_retry(test_case, idx)

    async def _run_with_retry(self, test_case: Dict, idx: int) -> Dict:
        """
        Thử lại test case tối đa `max_retries` lần khi gặp lỗi.
        Thời gian chờ tăng theo hàm mũ: delay * 2^attempt
        """
        case_id = test_case.get("id", f"case_{idx:03d}")

        for attempt in range(self.max_retries + 1):
            result = await self.run_single_test(test_case, case_id=case_id)

            # Nếu thành công (pass hoặc fail nhưng không phải lỗi hệ thống)
            if result["status"] != "error":
                return result

            # Nếu còn lần thử – chờ rồi thử lại
            if attempt < self.max_retries:
                wait = self.retry_delay * (2 ** attempt)
                print(f"  [WARN]  [{case_id}] Lỗi '{result['error']}' -- Thử lại sau {wait:.1f}s "
                      f"(lần {attempt + 1}/{self.max_retries})")
                await asyncio.sleep(wait)

        # Hết lần thử – trả về kết quả lỗi cuối cùng
        result["error"] = f"Đã thử {self.max_retries + 1} lần nhưng vẫn thất bại."
        return result

    def _print_cost_summary(self) -> None:
        """In tóm tắt chi phí ngắn gọn ra console."""
        report = self.tracker.report()
        print(f"\n[COST] Chi phí Benchmark:")
        print(f"   Tổng calls      : {report['total_calls']}")
        print(f"   Tổng tokens     : {report['total_tokens']:,} "
              f"(in={report['total_tokens_in']:,} | out={report['total_tokens_out']:,})")
        print(f"   Tổng chi phí    : ${report['total_cost_usd']:.6f}")
        print(f"   Chi phí/call    : ${report['avg_cost_per_call_usd']:.6f}")
        print(f"   Avg latency     : {report['avg_latency_ms']:.0f} ms")
        print(f"   Thời gian chạy  : {report['elapsed_seconds']}s")
        if report.get("cost_extrapolation"):
            ext = report["cost_extrapolation"]
            print(f"   Ước tính 100 cases  : ${ext['100_cases_usd']:.4f}")
            print(f"   Ước tính 1000 cases : ${ext['1000_cases_usd']:.4f}")

    @staticmethod
    def _error_result(test_case: Dict, case_id: str, error_type: str, detail: str) -> Dict:
        """Tạo dict kết quả lỗi chuẩn hóa."""
        return {
            "case_id":        case_id,
            "test_case":      test_case.get("question", ""),
            "agent_response": "",
            "latency_ms":     0.0,
            "tokens_in":      0,
            "tokens_out":     0,
            "model":          "unknown",
            "ragas":          {},
            "judge":          {"final_score": 0, "agreement_rate": 0},
            "status":         "error",
            "error":          f"{error_type}: {detail}",
        }
