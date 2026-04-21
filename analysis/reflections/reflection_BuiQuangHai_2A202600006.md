# Reflection Report - AI Evaluation Factory

**Họ và tên:** Bùi Quang Hải
**MSSV:** 2A202600006
**Vai trò:** Thành viên 4 - Backend / DevOps (Phụ trách Async, Tracking & Optimization)

---

## 1. Engineering Contribution (Đóng góp kỹ thuật) - 15 Điểm

Trong dự án AI Evaluation Factory này, em đảm nhận vai trò Backend & DevOps, đóng vai trò xây dựng hệ thống lõi để pipeline đánh giá có thể chạy mượt mà, chính xác và có thể tự động đưa ra quyết định dựa trên số liệu. Các đóng góp cụ thể bao gồm:

*   **Tích hợp Async/Await & Concurrency (engine/runner.py):**
    Thay vì chạy tuần tự từng case đánh giá (rất chậm vì phải gọi LLM API nhiều lần), em đã thiết kế lại `BenchmarkRunner` sử dụng `asyncio` và `asyncio.Semaphore`. Điều này cho phép hệ thống chạy song song nhiều test case cùng lúc nhưng vẫn kiểm soát được giới hạn rate-limit của API, giúp giảm thời gian benchmark xuống dưới 2 phút cho toàn bộ dataset. Thiết kế này cũng đi kèm cơ chế tự động Retry khi gặp lỗi mạng.
*   **Xây dựng Token & Cost Tracker (engine/cost_tracker.py):**
    Phát triển class `CostTracker` (áp dụng Singleton/Phiên) để tự động thu thập lượng `tokens_in`, `tokens_out` và `latency_ms` cho từng lượt gọi. Em đã thiết lập một bảng giá cập nhật để ước tính chi phí chuẩn xác theo USD đối với các dòng model của OpenAI.
*   **Hoàn thiện Regression Release Gate (engine/release_gate.py):**
    Thiết kế thuật toán gác cổng tự động so sánh V1 và V2. Thuật toán này không chỉ nhìn vào một điểm số mà đánh giá tổng hợp 6 chiều: Quality Score, Cost, Speed, Error Rate, Judge Agreement, và Retrieval Hit Rate. Phân tách rõ các ngưỡng Hard Block (bắt buộc Rollback) và Soft Requirement (cảnh báo) để hệ thống tự đưa ra quyết định APPROVE/ROLLBACK.
*   **Bảo trì hệ thống & Xử lý Encoding:**
    Xử lý triệt để các lỗi hiển thị ký tự mã hóa tiếng Việt (lỗi dấu `?`) tồn đọng trong project và cấu hình `main.py` để tương thích hoàn toàn môi trường Windows bằng cách ép hệ thống chạy ở chuẩn UTF-8. Các phần việc này được ghi nhận rõ trên các Git commits của nhóm.

---

## 2. Technical Depth (Chiều sâu kỹ thuật) - 15 Điểm

Trong quá trình xây dựng hệ thống đo lường, em đã tìm hiểu và ứng dụng sâu sắc các khái niệm sau:

*   **MRR (Mean Reciprocal Rank):**
    Là một chỉ số nhằm đánh giá chất lượng của Retrieval. Nếu mô hình sinh ra danh sách tài liệu, MRR sẽ đo lường xem "thông tin đúng xuất hiện ở thứ tự thứ mấy". Nếu tài liệu đúng nằm ngay top 1, điểm là 1. Nếu ở top 2, điểm là 1/2... MRR vô cùng quan trọng để đảm bảo RAG không bốc nhầm tài liệu rác lên đầu context context ảnh hưởng đến trí thông minh của tác vụ sinh.
*   **Cohen's Kappa & Agreement Rate (trong Multi-Judge):**
    Khi dùng mô hình Multi-Judge, một vấn đề đặt ra là mức độ đồng thuận giữa các "Giám khảo LLM". Agreement rate/Cohen's Kappa đo lường phần trăm các giám khảo cho điểm trùng khớp hoặc chênh lệch nhẹ với nhau. Chỉ số đồng thuận cao chứng tỏ bộ tiêu chí (Rubric) rất rõ ràng và hệ thống có độ tin cậy. Nếu quá khác biệt, hệ thống cần kích hoạt Conflict Resolution.
*   **Position Bias trong hàm đánh giá:**
    Các LLM chấm điểm đôi khi có xu hướng thích đáp án xuất hiện trước (nếu ta nạp prompt dạng "Option A, Option B"). Để khắc phục Position Bias, hệ thống cần có cơ chế đảo vị trí hoặc chấm độc lập từng đáp án để chống tính thiên vị.
*   **Trade-off giữa Quality (Chất lượng) và Cost (Chi phí/Latency):**
    Mô hình xịn (gpt-4o) cho điểm chất lượng cao nhưng sẽ có cost đắt và latency chậm hơn nhiều. Mô hình nhỏ (gpt-4o-mini) rất nhanh và rẻ nhưng dễ dính hallucination. Trong Release Gate, em cân bằng trade-off này qua việc cấp ngưỡng: chấp nhận V2 có thể đắt hơn V1 một chút (ví dụ +30% chi phí) miễn là tăng được điểm chất lượng, nhưng nếu cost x2 mà điểm chỉ đứng yên thì sẽ bị chặn lại ngay lập tức.

---

## 3. Problem Solving (Kỹ năng giải quyết vấn đề) - 10 Điểm

Trong quá trình code hệ thống, em đã gặp phải và xử lý nhiều vấn đề thực tiễn:

1.  **Vấn đề Mã hóa Ký tự Tiếng Việt (UTF-8) trên Console Windows:**
    Khi chạy pipeline, các báo cáo và logs in ra terminal thường xuyên bị lỗi font (hiển thị toàn dấu '?') do đặc thù môi trường Windows. Để giải quyết, em đã áp dụng cấu hình tự động nhận diện nền tảng `sys.platform == "win32"` và tái thiết lập (reconfigure) `sys.stdout` chạy chuẩn `utf-8` ở luồng vào ra, giúp log hiển thị đẹp và thân thiện.
2.  **Xung đột Context và Scope trong AsyncLoop:**
    Khi chạy asyncio gather với 50 cases chọc vào cùng một hàm `CostTracker`, hệ thống ghi nhận token bị loạn / đè lên nhau nến không cẩn thận. Em đã định nghĩa rõ các đối tượng theo từng Session (v1_tracker, v2_tracker riêng) thay vì gọi chung global tĩnh để track số liệu không bị nhiễu.
3.  **Quản lý Mock -> Thực Tế:**
    Ban đầu, khi ráp nối công việc, các API của bạn TV2, TV3 chưa hoàn thiện. Em đã tạo ra bộ `_MockEvaluator` trả về điểm ảo để hoàn thiện luồng pipeline trước. Khi các bạn nộp bài, em thiết kế code theo dạng Plugin, chỉ cần uncomment import và trỏ biến lại mà không cần đập logic cốt lõi.

Em nhận thấy, môn học AI Engineering không chỉ là làm cho mô hình chạy được, mà còn làm sao để hệ thống chạy ổn định, tiết kiệm, có số đo lường chính xác, và tự động hóa những tác vụ rủi ro nếu lên môi trường thực tế.

---

## 4. Minh chứng đóng góp (Git Commits)

*(Gắn link commit duy nhất trên GitHub/GitLab thể hiện toàn bộ phần việc)*

- **[Commit: Hoàn thiện Async Pipeline, CostTracker và Regression Release Gate](<https://github.com/trunglap923/Nhom10-403-Day14/commit/cbcc7d359cea2809e03b19e4cb4260d76b5e1bcf>)** - Giải trình: Tích hợp cơ chế chạy song song (Async/Await) giảm thời gian đánh giá xuống 2 phút; xây dựng tính năng track cost/token; xử lý lỗi hiển thị UTF-8; và viết thuật toán gác cổng thông minh giúp tự động so sánh V1/V2 để đưa quyết định Deploy/Rollback.
