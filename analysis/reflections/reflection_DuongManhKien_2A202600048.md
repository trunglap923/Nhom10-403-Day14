# Reflection Report - AI Evaluation Factory

**Họ và tên:** Dương Mạnh Kiên
**MSSV:** 2A202600048
**Vai trò:** Thành viên 5 - Data Analyst / Team Lead (Phụ trách Report & Analysis)

---

## 1. Engineering Contribution (Đóng góp kỹ thuật) - 15 Điểm

Trong dự án AI Evaluation Factory, tôi đảm nhận vai trò Team Lead & Data Analyst, chịu trách nhiệm khép lại toàn bộ pipeline bằng hệ thống báo cáo chuẩn hóa và phân tích thất bại chuyên sâu.

**Xây dựng module Failure Clustering + 5-Whys (`analysis/failure_cluster.py`):**

Thiết kế và triển khai từ đầu một module phân tích lỗi hoàn chỉnh với 2 thành phần chính:

- `FailureClusterer`: Phân loại tự động từng case fail/error vào 6 nhóm (Hallucination, Context Miss, Incomplete, Tone Mismatch, System Error, Format Error) thông qua rule-based heuristics kết hợp 3 chiều dữ liệu: Judge score, Retrieval hit_rate/MRR, và độ dài câu trả lời. Thiết kế thuần Python, không phụ thuộc API, đảm bảo pipeline chạy được cả ở chế độ mock.
- `FiveWhysAnalyzer`: Tự động chọn ra N case tệ nhất (ưu tiên `error > fail`, sau đó score thấp nhất), rồi sinh chuỗi 5-Whys heuristic theo cluster template để đào đến Root Cause. Mỗi cluster có why-chain riêng biệt, phản ánh đúng lý do kỹ thuật của từng dạng lỗi.
- `render_failure_markdown()`: Render toàn bộ phân tích thành file `analysis/failure_analysis.md` có cấu trúc: Tổng quan → Bảng clustering → 3 case 5-Whys → Action Plan.

**Chuẩn hóa hệ thống báo cáo trong `main.py`:**

Thay thế đoạn `json.dump` đơn giản bằng 4 hàm có trách nhiệm rõ ràng:

- `build_summary_payload()`: Ghép canonical schema với `schema_version = "1.0.0"`, đảm bảo backward-compatible với `check_lab.py`.
- `save_summary_json()` → `reports/summary.json` với các trường đầy đủ: `metadata`, `metrics`, `cost`, `v1_metrics`, `regression`, `failure_analysis`.
- `save_benchmark_results_json()` → `reports/benchmark_results.json` (full case-level detail).
- `save_failure_analysis_md()` → `analysis/failure_analysis.md` auto-generated từ FailureAnalysis.

**Tích hợp và vá lỗi kết nối pipeline:**

Phát hiện và vá lỗi thiếu phương thức `async score()` trong `engine/retrieval_eval.py` — phương thức mà `BenchmarkRunner` phụ thuộc nhưng chưa được implement. Viết adapter tương thích trả về đúng shape `{"retrieval": {"hit_rate": float, "mrr": float}}` mà downstream code kỳ vọng, đồng thời bổ sung estimation `faithfulness` và `relevancy` từ retrieval quality. Nhờ đó toàn bộ pipeline chạy end-to-end thành công.

---

## 2. Technical Depth (Chiều sâu kỹ thuật) - 15 Điểm

**Failure Clustering và ý nghĩa thực tiễn:**

Tôi hiểu rằng không phải mọi "fail" đều có cùng một nguyên nhân. Một case bị Hallucination (LLM bịa thông tin dù retriever tìm đúng) khác hoàn toàn với Context Miss (retriever tìm sai document ngay từ đầu). Nếu gộp chung vào một nhóm "fail", ta không biết nên sửa Chunking strategy hay Embedding model. Clustering giúp khu trú đúng lớp của vấn đề trước khi đầu tư engineering effort.

**Phương pháp 5-Whys và Root Cause Analysis:**

5-Whys là kỹ thuật từ Toyota Production System — mỗi "Why" đào một lớp sâu hơn về mặt kỹ thuật thay vì dừng lại ở triệu chứng. Ví dụ với cluster Context Miss:

- Why 1: Retriever không lấy được ground truth
- Why 2: Embedding của query và document bị xa nhau trong vector space
- Why 3: Model embedding chưa phù hợp tiếng Việt / domain nội bộ
- Why 4: Không có query rewriting để đồng bộ từ khoá
- Why 5: Vector DB chưa được index đầy đủ
- **Root Cause:** Bộ embedding + quy trình index chưa tối ưu cho ngữ liệu VN nội bộ

Nếu dừng ở Why 1, action plan sẽ là "sửa retriever" — quá mơ hồ. Đào đến Why 5 mới ra được action cụ thể: "Đổi sang multilingual-e5 + bổ sung query rewriting."

**Kết quả thực nghiệm từ real benchmark (71 cases):**

Khi chạy pipeline với real OpenAI API, 100% case fail (46/71) đều rơi vào cluster **Retrieval / Context Miss** với `hit_rate = 0.0`. Điều này phơi bày một điểm yếu hệ thống: `MainAgent` đang trả về hardcoded `retrieved_ids` không khớp với bất kỳ `expected_retrieval_ids` nào trong golden dataset — nghĩa là RAG pipeline chưa được kết nối đúng với ChromaDB. Release Gate vẫn ra quyết định **APPROVE V2 với 83% confidence** vì V2 cải thiện được Judge Score (+0.42) trên 5/6 chiều đánh giá, nhưng cảnh báo latency tăng 430%.

**Schema versioning và backward compatibility:**

Bổ sung `schema_version = "1.0.0"` vào top-level `summary.json` là thực hành quan trọng trong production — cho phép các công cụ downstream phát hiện khi format thay đổi và không bị lỗi silent. Đây là điều thường bị bỏ qua trong prototype nhưng gây ra incident nghiêm trọng khi scale.

---

## 3. Problem Solving (Kỹ năng giải quyết vấn đề) - 10 Điểm

**Vấn đề 1: Pipeline broken do interface mismatch**

Khi chạy thử `main.py`, toàn bộ 71 case đều báo `AttributeError: 'RetrievalEvaluator' object has no attribute 'score'`. Nguyên nhân: `engine/runner.py:173` gọi `evaluator.score(test_case, response)` nhưng phương thức này không tồn tại.

Thay vì chờ bạn nhóm, tôi phân tích downstream code để hiểu exact shape mà runner kỳ vọng, sau đó viết adapter tương thích hoàn toàn trong ~20 dòng mà không đụng vào logic có sẵn. Pipeline chạy thành công ngay sau đó.

**Vấn đề 2: Thiết kế clustering không phụ thuộc vào LLM**

Có thể dùng LLM để phân loại lỗi (semantic clustering). Tuy nhiên, tôi chọn rule-based approach vì 3 lý do: (1) pipeline evaluation không nên phụ thuộc vào thêm một LLM call — tăng chi phí và latency; (2) rule-based cho kết quả deterministic, dễ debug và kiểm chứng; (3) các metric đã có (score, hit_rate, answer length, error type) đủ để phân loại chính xác. Kết quả: module chạy hoàn toàn offline, không tốn thêm chi phí API.

**Vấn đề 3: End-to-end integration testing trước khi commit**

Thay vì commit code rồi chờ CI báo lỗi, tôi viết smoke test inline với synthetic benchmark data (40 cases, mix pass/fail/error) để xác nhận toàn bộ `save_reports()` → `check_lab.py` pass trước khi chạy real API. Cách tiếp cận này giúp phát hiện lỗi schema mismatch sớm (key `avg_latency_ms` vs `avg_latency`) mà không tốn token API thật.

---

## 4. Minh chứng đóng góp (Git Commits)

- **[Commit: `0416137` — do the failure_analysis](https://github.com/trunglap923/Nhom10-403-Day14/commit/04161370569816aca28f728b8a5389a476efbd5a)**

  Toàn bộ phần việc của Thành viên 5 được gói trong commit này, bao gồm:
  - `analysis/failure_cluster.py` (new, +509 dòng): Module Failure Clustering + 5-Whys + Markdown renderer
  - `main.py` (modified, +154 dòng): `save_reports()`, `build_summary_payload()` và các hàm export chuẩn hóa
  - `engine/retrieval_eval.py` (modified, +22 dòng): `async score()` adapter kết nối runner với evaluator
  - `analysis/failure_analysis.md` (modified): Auto-generated từ real benchmark run (71 cases, gpt-4o-mini)

Toàn bộ code được xác nhận hoạt động bởi `python3 check_lab.py`:
✅ Tìm thấy `reports/summary.json` | ✅ Tìm thấy `reports/benchmark_results.json` | ✅ Tìm thấy `analysis/failure_analysis.md` | ✅ Retrieval Metrics | ✅ Multi-Judge Metrics | ✅ Regression Mode
