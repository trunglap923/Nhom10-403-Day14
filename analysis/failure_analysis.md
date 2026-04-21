# Báo cáo Phân tích Thất bại (Failure Analysis Report)

_Phiên bản: **Agent_V2_Optimized**_ -- Auto-generated bởi `analysis/failure_cluster.py`.

## 1. Tổng quan Benchmark
- **Tổng số cases:** 71
- **Số cases Fail/Error:** 46 (64.8%)
- **Điểm trung bình (Judge):** 2.49 / 5.0
- **Hit Rate (Retrieval):** 0.0%
- **Agreement Rate (Multi-Judge):** 89.1%

## 2. Phân nhóm lỗi (Failure Clustering)

| Nhóm lỗi | Số lượng | % trên tổng | Nguyên nhân dự kiến |
|----------|----------|-------------|---------------------|
| Retrieval / Context Miss | 46 | 64.8% | Vector DB retrieve sai/thiếu doc chứa ground-truth (hit_rate=0 hoặc MRR thấp). |

## 3. Phân tích 5-Whys (3 case tệ nhất)

### Case #1: `case_006` -- Retrieval / Context Miss
- **Câu hỏi:** Quyền tạm thời được cấp tối đa trong bao lâu?
- **Symptom:** Status=fail, Score=1.0 -- Cluster=Retrieval / Context Miss | Agent: "Quyền tạm thời thường được cấp trong các trường hợp như quyền tạm thời trong tố tụng dân sự, quyền tạm thời trong các th..."
- **Why 1:** Retriever không lấy được document chứa ground truth (hit_rate=0).
- **Why 2:** Embedding của câu hỏi và document bị xa nhau trong không gian vector.
- **Why 3:** Model embedding đang dùng chưa phù hợp cho tiếng Việt / domain nội bộ.
- **Why 4:** Không có bước query rewrite để đồng bộ từ khoá / từ đồng nghĩa.
- **Why 5:** Vector DB không được cập nhật đầy đủ; một số doc chưa index.
- **Root Cause:** Bộ embedding + quy trình index chưa tối ưu cho ngữ liệu VN nội bộ.
- **Action:** Đổi embedding sang multilingual-e5 hoặc fine-tune + bổ sung query rewriting.

### Case #2: `case_022` -- Retrieval / Context Miss
- **Câu hỏi:** Email liên hệ của phòng nhân sự là gì?
- **Symptom:** Status=fail, Score=1.0 -- Cluster=Retrieval / Context Miss | Agent: "Để biết được email liên hệ của phòng nhân sự, bạn có thể thực hiện một số bước sau:  1. **Kiểm tra trang web chính thức ..."
- **Why 1:** Retriever không lấy được document chứa ground truth (hit_rate=0).
- **Why 2:** Embedding của câu hỏi và document bị xa nhau trong không gian vector.
- **Why 3:** Model embedding đang dùng chưa phù hợp cho tiếng Việt / domain nội bộ.
- **Why 4:** Không có bước query rewrite để đồng bộ từ khoá / từ đồng nghĩa.
- **Why 5:** Vector DB không được cập nhật đầy đủ; một số doc chưa index.
- **Root Cause:** Bộ embedding + quy trình index chưa tối ưu cho ngữ liệu VN nội bộ.
- **Action:** Đổi embedding sang multilingual-e5 hoặc fine-tune + bổ sung query rewriting.

### Case #3: `case_036` -- Retrieval / Context Miss
- **Câu hỏi:** Chính sách này áp dụng cho tất cả các đơn hàng từ ngày nào?
- **Symptom:** Status=fail, Score=1.0 -- Cluster=Retrieval / Context Miss | Agent: "Để trả lời chính xác câu hỏi của bạn, tôi cần biết rõ hơn về chính sách cụ thể mà bạn đang đề cập đến. Thông thường, các..."
- **Why 1:** Retriever không lấy được document chứa ground truth (hit_rate=0).
- **Why 2:** Embedding của câu hỏi và document bị xa nhau trong không gian vector.
- **Why 3:** Model embedding đang dùng chưa phù hợp cho tiếng Việt / domain nội bộ.
- **Why 4:** Không có bước query rewrite để đồng bộ từ khoá / từ đồng nghĩa.
- **Why 5:** Vector DB không được cập nhật đầy đủ; một số doc chưa index.
- **Root Cause:** Bộ embedding + quy trình index chưa tối ưu cho ngữ liệu VN nội bộ.
- **Action:** Đổi embedding sang multilingual-e5 hoặc fine-tune + bổ sung query rewriting.

## 4. Kế hoạch cải tiến (Action Plan)
- [ ] [Retrieval / Context Miss] Đổi embedding sang multilingual-e5 hoặc fine-tune + bổ sung query rewriting.
