# Individual Reflection - Data Analyst / Team Lead

**Họ và tên:** Nguyễn Văn Hiếu
**Mã số sinh viên:** 2A202600454
**Vai trò:** Data Analyst / Team Lead (Phụ trách Report & Analysis)

---

## 🛠 1. Engineering Contribution (15/15)

Trong dự án AI Evaluation Factory, tôi đã đảm nhận vai trò then chốt trong việc phân tích dữ liệu và chịu trách nhiệm chính trong việc trình bày kết quả đánh giá hệ thống:

- **Phát triển module Báo cáo (`main.py`)**:
  - Thiết kế và xây dựng các hàm `build_summary_payload`, `save_summary_json` và `save_benchmark_results_json` để tự động hóa quy trình xuất báo cáo.
  - Đảm bảo các file `reports/summary.json` và `reports/benchmark_results.json` luôn xuất ra đúng định dạng chuẩn, bao gồm đầy đủ dữ liệu so sánh V1 vs V2, metadata và regression stats để phục vụ lệnh kiểm tra tự động.
- **Phát triển Hệ thống Phân tích Thất bại (`analysis/failure_cluster.py`)**:
  - Triển khai logic **Failure Clustering** tự động: phân loại các case Fail/Error vào 6 nhóm chuẩn (Hallucination, Context Miss, System Error, v.v.) dựa trên heuristics từ metrics.
  - Áp dụng phương pháp phân tích sâu **5-Whys** cho các case tệ nhất để xác định căn nguyên gốc rễ (Root Cause) của vấn đề và đề xuất Action Plan cải tiến cụ thể.
- **Tự động hóa Báo cáo Failure Analysis**:
  - Xây dựng hàm render Markdown để tự động cập nhật file `analysis/failure_analysis.md` mỗi khi chạy pipeline, giúp tối ưu hóa quy trình theo dõi lỗi của nhóm.
- **Quản lý & Kiểm định**:
  - Thực thi lệnh `python check_lab.py` để xác thực định dạng nộp bài đúng chuẩn của Lab Day 14.
  - Quản lý và đôn đốc việc nộp Reflection của các thành viên trong nhóm 10.

## 📚 2. Technical Depth (15/15)

Tôi đã làm chủ các kiến thức chuyên sâu về phân tích lỗi và tiêu chuẩn hóa đánh giá AI:

- **Failure Clustering Heuristics**: Hiểu rõ cách kết hợp giữa metrics định lượng (`judge_score`, `hit_rate`) và dữ liệu định tính (agent response) để xác định chính xác bản chất lỗi của Agent mà không cần xem xét thủ công hàng trăm cases.
- **Phương pháp 5-Whys trong RAG**: Sử dụng kỹ thuật truy vấn sâu để bóc tách các tầng lỗi từ bề mặt (Symptom) xuống đến tận cùng vấn đề (ví dụ: lỗi Hallucination thực chất bắt nguồn từ chiến lược Fixed-size Chunking gây mất ngữ nghĩa).
- **Machine-readable Reporting**: Nắm vững kỹ thuật xây dựng Schema JSON chuẩn cho báo cáo đánh giá, hỗ trợ việc tích hợp pipeline vào quy trình CI/CD cho AI.

## 🚀 3. Problem Solving (10/10)

Vấn đề phức tạp nhất tôi đã giải quyết là việc "gọi tên" chính xác nguyên nhân lỗi trong một pipeline RAG phức tạp:

- **Thách thức**: Khi Agent trả lời sai, rất khó để biết lỗi là do Retriever tìm sai tài liệu hay do LLM tự suy diễn (Hallucination). Việc phân tích thủ công tốn quá nhiều thời gian và dễ gây sai sót.
- **Giải quyết**:
  1. Tôi đã xây dựng một **Decision Logic** trong module phân cụm: Hệ thống sẽ kiểm tra `hit_rate` và `mrr` trước; nếu các chỉ số này bằng 0, lỗi chắc chắn là `Context Miss`. Nếu chỉ số cao mà `judge_score` vẫn thấp, lỗi sẽ được hướng tới `Hallucination`.
  2. Bổ sung các check cho lỗi hệ thống (Timeout, KeyError) để tách biệt lỗi hạ tầng khỏi lỗi logic của model.
  - **Kết quả**: Hệ thống có khả năng tự động phân tích và gán nhãn chính xác cho toàn bộ các case lỗi chỉ trong vài giây, giúp nhóm tập trung đúng vào khâu cần tối ưu (ví dụ: chuyển từ bộ Embeddings tiếng Anh sang Multilingual-E5 để sửa lỗi Context Miss).
