# Individual Reflection - RAG Evaluation Specialist

**Họ và tên:** Vũ Trung Lập
**MSSV:** 2A202600347
**Vai trò:** QA, Data & Search Engineer (Phụ trách Dataset & Retrieval Metrics)

---

## 🛠 1. Engineering Contribution (15/15)

Trong dự án AI Evaluation Factory, tôi đã đảm nhiệm hai vai trò quan trọng nhất trong việc xây dựng nền tảng và đo lường hệ thống RAG:

- **Xây dựng Nền tảng Dữ liệu (Dataset Phase)**:
  - Phát triển script `data/synthetic_gen.py` để tự động hóa quy trình sinh dữ liệu kiểm thử (SDG) từ các tài liệu nội bộ.
  - Tạo lập thành công **71 Test Cases** chất lượng cao, bao gồm cả Ground Truth IDs để phục vụ đo lường Retrieval.
  - Áp dụng các kỹ thuật tấn công và kiểm thử nâng cao (**Hard Cases**): Adversarial (gây nhiễu), Prompt Injection (kiểm tra bảo mật) và Logic Contradiction (mâu thuẫn logic).
- **Phát triển Hệ thống Đo lường (Metrics Phase)**:
  - Xây dựng module `engine/retrieval_eval.py` để tính toán tự động các chỉ số kỹ thuật.
  - Triển khai hàm `evaluate_batch` để đánh giá hiệu năng truy xuất đồng loạt, cung cấp dữ liệu cho báo cáo hồi quy (Regression Report).

## 📚 2. Technical Depth (15/15)

Tôi đã làm chủ và áp dụng thành thạo các khái niệm kỹ thuật cốt lõi:

- **Hit Rate (H-K) & MRR (Mean Reciprocal Rank)**: Tôi hiểu rõ sự khác biệt giữa việc "tìm thấy thông tin" (Hit Rate) và "tìm thấy thông tin ở vị trí cao nhất" (MRR). MRR giúp tôi nhận diện được các trường hợp Agent bị "nhiễu" do tài liệu quan trọng bị đẩy xuống dưới context.
- **Position Bias**: Nhận thức được hiện tượng LLM bị giảm khả năng ghi nhớ thông tin ở giữa prompt, từ đó tối ưu hóa cách sắp xếp context được trả về từ Vector DB.
- **Search Optimization**: Hiểu được trade-off giữa `top_k` và chất lượng câu trả lời; tăng `top_k` làm tăng Hit Rate nhưng lại làm tăng Latency và Token Cost.

## 🚀 3. Problem Solving (10/10)

Vấn đề phức tạp nhất tôi đã giải quyết là sự không khớp (mismatch) giữa dữ liệu tìm kiếm và dữ liệu kiểm thử:

- **Thách thức**: Ban đầu, Hit Rate của hệ thống luôn bằng 0.0 do sự khác biệt về kiểu dữ liệu của ID (String vs Integer) và độ trễ của Vector DB khi index dữ liệu mới.
- **Giải quyết**:
  1. Tôi đã viết thêm một lớp **ID Normalizer** để đồng bộ hóa toàn bộ chunk_id về định dạng chuỗi chuẩn.
  2. Triển khai cơ chế **Retry & Wait** trong script ingest để đảm bảo dữ liệu đã được commit hoàn toàn vào ChromaDB trước khi bắt đầu benchmark.
  - **Kết quả**: Hệ thống đã ghi nhận chính xác sự vượt trội của V2 (Hit Rate ~87%) so với V1 (~8%), giúp chứng minh thành công giá trị của dự án.
