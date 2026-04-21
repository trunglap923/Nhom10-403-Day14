# Báo cáo Cá nhân: Thành viên 3 (AI Engineer)
**Họ và tên:** Tạ Vĩnh Phúc 
**MSSV:** 2A202600424
**Nhiệm vụ:** Multi-Judge Consensus Engine (Thành viên 3)
**Module phụ trách:** `engine/llm_judge.py`, tích hợp `main.py`, refactor `agent/main_agent.py`

---

## 1. Engineering Contribution (Đóng góp kỹ thuật)

### Các file đã triển khai / chỉnh sửa:
| File | Vai trò |
|---|---|
| `engine/llm_judge.py` | Toàn bộ logic Multi-Judge: gọi 2 model song song, tính Agreement, phân giải xung đột |
| `agent/main_agent.py` | Nâng cấp từ Mock → Agent thật với OpenAI API + kết nối ChromaDB thực |
| `main.py` | Xóa bỏ các `_MockJudge`, `_MockAgent` giả lập; ghim thẳng class thật vào pipeline |

### Chi tiết kỹ thuật:
- **Async Multi-Judge:** Dùng `asyncio.gather()` để submit cả 2 request đến `gpt-4o-mini` và `gpt-4.1-mini` song song thay vì tuần tự, giúp latency của bước Judge chỉ bằng latency của model chậm nhất chứ không cộng dồn.
- **ChromaDB Integration:** Kết nối `MainAgent` trực tiếp vào `data/chroma_db` (collection `eval_docs` do Vũ Trung Lập build). Agent V1 lấy `top_k=1` doc, Agent V2 lấy `top_k=3` doc — tạo ra sự chênh lệch Hit Rate có thật (81.7% vs 87.3%) chứ không phải con số giả.
- **Token Tracking:** Bổ sung `resp.usage.prompt_tokens` và `resp.usage.completion_tokens` vào `metadata` trả về để `CostTracker` của Bùi Quang Hải đọc và tính chi phí thực tế.

---

## 2. Technical Depth (Độ sâu kỹ thuật)

### a) Cohen's Kappa vs. Linear Weighted Agreement

**Cohen's Kappa** là chỉ số đo lường mức độ đồng thuận giữa 2 người chấm, có tính đến xác suất đồng thuận ngẫu nhiên. Công thức:

```
κ = (Po - Pe) / (1 - Pe)
```
- `Po` = Tỷ lệ đồng thuận thực quan sát được
- `Pe` = Tỷ lệ đồng thuận kỳ vọng ngẫu nhiên

**Lý do không dùng Cohen's Kappa thuần túy** cho bài này: thang điểm 1–5 là **ordinal** (có thứ tự), nên lệch 1 điểm (4 vs 5) phải được tính nhẹ tội hơn lệch 4 điểm (1 vs 5). Cohen's Kappa chuẩn không phân biệt điều này.

**Giải pháp tôi áp dụng — Linear Weighted Agreement:**
```python
diff = abs(score_a - score_b)
agreement = max(0.0, 1.0 - (diff / 4.0))
```
Kết quả tương đương **Weighted Cohen's Kappa với Linear Weight** nhưng đơn giản hơn để implement và trực quan hơn để debug. Kết quả thực tế đo được:
- V1: `agreement_rate = 0.915` (91.5% — 2 model khá đồng nhất)
- V2: `agreement_rate = 0.958` (95.8% — câu trả lời tốt hơn → 2 model dễ đồng thuận hơn)

### b) Position Bias & Conflict Resolution

**Position Bias** là hiện tượng LLM Judge có xu hướng thiên vị thứ tự nội dung trong prompt (ví dụ: câu trả lời đặt ở đầu thường được chấm cao hơn). Trong hệ thống Multi-Judge, 2 model nhận cùng nội dung theo cùng thứ tự nên bias này tương trương nhau → tính trung bình điểm giữ trung tính hơn so với 1 judge đơn lẻ.

**Logic Conflict Resolution tôi triển khai:**
```
|A - B| <= 1  →  Đồng thuận  →  Final = (A + B) / 2   (trung bình)
|A - B| >= 2  →  Xung đột    →  Final = B              (giữ model mạnh hơn)
```
Model `B` (gpt-4.1-mini) được coi là đáng tin cậy hơn trong trường hợp xung đột vì: nhiều parameters hơn, ra mắt sau → reasoning tốt hơn.

### c) Trade-off Chi phí vs. Chất lượng (số liệu thực)

| Cấu hình Judge | Chi phí/lần chạy 71 cases | Quality Score | Agreement |
|---|---|---|---|
| 1× gpt-4o (ban đầu) | ~$0.36 | 5.0 | 1.0 |
| 2× gpt-4o-mini | ~$0.01 | 4.35 | 0.95 |
| gpt-4o-mini + gpt-4.1-mini | **$0.09 (V1) / $0.47 (V2)** | 4.45 | 0.96 |

**Kết luận:** Cặp `gpt-4o-mini + gpt-4.1-mini` là sweet spot — chất lượng đánh giá gần với `gpt-4o` nhưng chi phí rẻ hơn ~75%. Chi phí V2 cao hơn V1 do Agent V2 kéo 3 docs từ ChromaDB → context dài hơn → tokens in nhiều hơn (42,417 vs 13,435 tokens).

**Đề xuất giảm 30% chi phí mà không giảm chất lượng:**
1. Cache kết quả judge cho các câu hỏi trùng lặp (hash prompt).
2. Chỉ gọi model mạnh (gpt-4.1-mini) khi model nhẹ cho điểm < 3 hoặc > 4 (borderline cases).
3. Dùng `gemini-2.5-flash-lite` thay model thứ 2 khi đã có billing để tận dụng giá thấp hơn.

---

## 3. Problem Solving (Giải quyết vấn đề và Khó khăn)

### Vấn đề 1: Interface mismatch giữa các Thành viên
**Bối cảnh:** `BenchmarkRunner` (Bùi Quang Hải) gọi `evaluator.score(test_case, response)` nhưng `RetrievalEvaluator` (Vũ Trung Lập) không có phương thức `score()`.  
**Giải pháp:** Áp dụng **Adapter Pattern** — thêm async method `score()` vào `RetrievalEvaluator` để bridge interface mà không phá vỡ code gốc của Vũ Trung Lập.

### Vấn đề 2: Agent chỉ trả về dữ liệu giả
**Bối cảnh:** `MainAgent` ban đầu mock cứng `retrieved_ids = ["doc_1"]` → Hit Rate luôn là 100% bất kể câu hỏi gì, không phản ánh thực tế.  
**Giải pháp:** Kết nối thật vào `chromadb.PersistentClient("data/chroma_db")`, embed câu hỏi bằng `text-embedding-3-small`, query collection `eval_docs`. Kết quả Hit Rate thật: **81.7% (V1) vs 87.3% (V2)** — số liệu phản ánh đúng chất lượng RAG.

### Vấn đề 3: Token = 0, Cost = 0
**Bối cảnh:** Sau khi gắn OpenAI thật, `CostTracker` vẫn báo 0 vì agent không trả về số lượng token trong `metadata`.  
**Giải pháp:** Đọc `resp.usage.prompt_tokens` và `resp.usage.completion_tokens` từ response object của OpenAI và nhúng vào dict `metadata` trước khi return.

### Vấn đề 4: Gemini API 403 → 429
**Bối cảnh:** Thử tích hợp `gemini-2.5-flash-lite` làm Judge thứ 2 để đa dạng hóa hệ thống. Gặp 403 (service chưa enable) rồi 429 (free tier limit 10 req/min).  
**Giải pháp ngắn hạn:** Dùng `gpt-4o-mini + gpt-4.1-mini` (cùng nhà OpenAI, không cần key mới, không rate limit). Kết luận: Multi-provider judge cần billing account riêng cho từng provider — không phù hợp với môi trường test/dev.