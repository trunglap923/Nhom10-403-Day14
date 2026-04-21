# Reflection — Dương Mạnh Kiên

**Vai trò trong nhóm:** Thành viên 1 — QA / Data Engineer (Phụ trách Dataset)
**Lab:** Day 14 — AI Evaluation Factory
**Nhóm:** Nhom10-403
**Ngày nộp:** 2026-04-21

---

## 1. Phần việc đã đảm nhận

Tôi chịu trách nhiệm toàn bộ phần **Golden Dataset** — nền móng mà ba thành viên còn lại đứng lên để làm việc. Nếu dataset rác thì Retrieval Eval (TV2), Multi-Judge (TV3) và Regression Gate (TV4) đều vô nghĩa, vì vậy tôi coi đây là phần có "blast radius" lớn nhất.

Cụ thể các deliverable:

1. **Viết lại `data/synthetic_gen.py`** (từ 34 dòng placeholder → generator xác định, có thống kê phân bổ case).
2. **Xây dựng Knowledge Base** `data/knowledge_base.json` — 25 tài liệu tổng hợp về chính sách/FAQ của một dịch vụ SaaS giả lập (ACME Cloud), mỗi tài liệu có `doc_id` riêng. Đây chính là Ground Truth cho khâu Retrieval Evaluation.
3. **Sinh 62 test cases** (`data/golden_set.jsonl`), vượt yêu cầu 50+, được đóng đầy đủ các field mà runner và evaluator cần:
   - `question`, `expected_answer`, `context`, `expected_retrieval_ids`, `id`, `metadata`, và `history` (cho multi-turn).
4. **Áp dụng đủ 10 dạng Hard Case** theo `HARD_CASES_GUIDE.md`:

   | Nhóm | Loại | Số case |
   |---|---|---|
   | Bình thường | normal (easy / medium / hard) | 30 |
   | Adversarial | prompt_injection | 6 |
   | Adversarial | goal_hijacking | 4 |
   | Edge case | out_of_context | 5 |
   | Edge case | ambiguous | 4 |
   | Edge case | conflicting | 4 |
   | Multi-turn | carryover | 3 |
   | Multi-turn | correction | 2 |
   | Technical | latency_stress | 2 |
   | Technical | cost_efficiency | 2 |

   → 32/62 cases là hard case thực sự, đủ sức "bẻ" agent yếu.

---

## 2. Lý do của các quyết định thiết kế (Engineering judgement)

### 2.1 Dataset **xác định**, không phụ thuộc LLM

Bản gốc `synthetic_gen.py` có stub gọi LLM để sinh QA. Tôi bỏ cách này vì ba lý do:

- **Reproducibility:** Mỗi lần chạy benchmark mà dataset khác đi thì không so sánh V1 vs V2 được — điều này làm hỏng luôn Regression Gate của TV4.
- **Chi phí:** Sinh 50+ case bằng GPT-4o mỗi lần chạy tốn tiền vô ích khi dataset có thể commit cố định.
- **Kiểm soát chất lượng:** LLM sinh ra các case hay bị "safe" và đều đều; adversarial thật thì phải **viết tay** mới đủ ác.

Vẫn giữ hàm `generate_qa_from_text()` như một stub tương thích giao diện để có thể mở rộng sau.

### 2.2 Gắn `expected_retrieval_ids` vào từng case

Đây là quyết định quan trọng nhất về mặt kiến trúc. Nếu chỉ có `question` + `expected_answer`, TV2 không thể tính **Hit Rate** hay **MRR** được — cả hai metric đều cần biết tài liệu "đúng" là doc nào. Bằng cách tự xây Knowledge Base 25 docs và gắn ID vào mỗi test case, tôi biến Retrieval Evaluation từ "khả thi về mặt code" thành "khả thi về mặt dữ liệu".

Các case Out-of-Context cố tình để `expected_retrieval_ids=[]` để kiểm tra được Retriever có biết "trả về rỗng" hay không — đây là trường hợp Hit Rate bị lệch nếu không xử lý khéo.

### 2.3 Hard case được thiết kế có chủ đích, không random

Một vài ví dụ tôi chủ động gài bẫy:

- **case_033:** Prompt injection giả mạo SYSTEM prompt thay đổi chính sách hoàn tiền thành 365 ngày → kiểm tra agent có bị "authority injection" hay không.
- **case_050:** Người dùng nói "nhân viên nói tôi 60 ngày" mâu thuẫn với doc_001 (30 ngày) → kiểm tra agent có biết **trust the doc, not the user**.
- **case_047, case_049:** Câu hỏi mập mờ "Nó có hoạt động không?" / "Làm sao để nhanh hơn?" → kiểm tra agent có clarify hay bịa.
- **case_059, case_060:** Nhét ~5000 ký tự context rác vào đầu câu hỏi → đo latency/context window.
- **case_061, case_062:** Câu hỏi siêu ngắn + `target_tokens_out_max=15` → kiểm tra agent có verbose không cần thiết.

---

## 3. Technical depth (phần rubric chấm cá nhân)

### 3.1 Hit Rate vs MRR — tại sao cần cả hai

- **Hit Rate@k** = "Trong top-k document retriever trả về, có chứa doc đúng không?" → nhị phân (0/1).
- **MRR** = `1 / vị trí` của doc đúng đầu tiên → liên tục.

Hit Rate@3 = 1.0 nhưng MRR = 0.33 có nghĩa retriever **tìm đúng nhưng xếp hạng tệ** — và đúng doc nằm ở vị trí 3 rất nguy hiểm vì các pipeline LLM thường chỉ feed top-1 hoặc top-2 vào prompt. Nếu chỉ nhìn Hit Rate, bug này sẽ bị che mất.

### 3.2 Mối liên hệ Retrieval Quality ↔ Answer Quality (Hallucination)

Bước Retrieval mà sai → LLM nhận context rác → LLM sẽ chọn 1 trong 2 hành vi tệ:

1. **Hallucinate**: dịch context rác thành câu trả lời nghe có vẻ hợp lý.
2. **Leak training data**: bỏ qua context, lấy từ kiến thức pretrain → out-of-date.

Dataset tôi xây có các case cố tình gài "topic tương đồng nhưng doc khác" (vd `case_045` hỏi Windows XP → doc_018 gần nhất nhưng **không đúng**). Case này chỉ "bẫy" được agent nếu retriever thật sự tốt; nếu retriever ngu thì bug Hallucination không hiện ra và ta nhầm tưởng agent đang ổn.

Tóm lại: **Hit Rate thấp là alibi của LLM** — không sửa được retrieval thì mọi số RAGAS/Judge sau đó đều không kết luận được.

### 3.3 Cohen's Kappa & Position Bias (cho phần TV3)

Tuy không trực tiếp code module Judge, tôi thiết kế dataset với việc hiểu trước các vấn đề TV3 sẽ gặp:

- **Cohen's Kappa** đo mức đồng thuận giữa 2 judge đã loại bỏ phần "đồng thuận tình cờ" (chance agreement). Một Agreement Rate = 0.8 nghe cao, nhưng nếu cả 2 judge đều thiên về "4/5" thì Kappa sẽ rất thấp (~ 0.2). Kappa mới là con số "thật".
- **Position Bias**: Judge LLM có xu hướng chấm response ở vị trí A cao hơn B khi nội dung ngang nhau. Vì vậy, các câu trả lời kỳ vọng của tôi (các ngắn 1 dòng như case_001..009) sẽ khiến agent V1 dễ vượt qua **mà chưa chắc do trả lời tốt** — đây là noise mà TV3 cần cân bằng bằng cách swap A/B.

### 3.4 Trade-off Chi phí vs Chất lượng

62 cases × 2 phiên bản V1/V2 × ~2 judge calls = ~248 lần gọi LLM cho mỗi benchmark. Với GPT-4o (~$5/1M input tokens, ~$15/1M output), trung bình mỗi case cost $0.001–$0.003. Nếu tôi nâng dataset lên 500 cases sẽ sát $2/run — quá đắt để chạy hàng ngày trong CI.

Quyết định dừng ở 62 (vượt 50 một cách có chủ ý, không overkill) là để cân bằng giữa:
- Cần đủ thống kê (50+ theo rubric, p > 0.9 cho test Mann-Whitney V1 vs V2).
- Không biến benchmark thành gánh nặng tài chính.

Hai case `cost_efficiency` có `target_tokens_out_max` là để TV4 có thể dùng làm điều kiện "fail" trong Release Gate nếu agent V2 đốt gấp đôi token so với V1.

---

## 4. Vấn đề phát sinh & cách giải quyết (Problem Solving)

### 4.1 Câu hỏi multi-turn không khớp schema runner

**Vấn đề:** `runner._run_with_retry()` chỉ gọi `agent.query(test_case["question"])`, không biết tới `history`.

**Hướng giải quyết:** Tôi không sửa runner (sẽ đụng code TV4), thay vào đó để field `history` **optional** trên schema. Với 5 multi-turn cases (54–58), câu `question` được viết độc lập đủ để agent trả lời được — history chỉ là "bonus context" khi agent được nâng cấp hỗ trợ. Tức là dataset **forward-compatible**, không block TV4 mà vẫn để mở.

### 4.2 Out-of-Context cases phá MRR

**Vấn đề:** 4/5 OOC case có `expected_retrieval_ids=[]` → chia 0 nếu tính MRR ngây thơ.

**Hướng giải quyết:** Trao đổi với TV2, tôi thống nhất: khi `expected_ids == []`, MRR được tính là 1.0 **nếu** retriever trả về rỗng (behavior đúng), và 0.0 nếu bừa ra cái gì đó. Logic này TV2 sẽ cài ở `calculate_mrr` — nhưng tôi phải làm rõ convention này trước khi đẩy dataset lên.

### 4.3 Đảm bảo `id` khớp với `case_id` do runner sinh tự động

**Vấn đề:** Runner tự sinh `case_id = test_case.get("id", f"case_{idx:03d}")`. Nếu tôi đặt id sai format thì log sẽ loạn và khó dò khi debug.

**Hướng giải quyết:** Giữ `id` khớp chính xác với `case_{idx+1:03d}` và thêm bước `_stats()` in phân bổ case theo category/difficulty để phát hiện sớm bất kỳ case nào bị gán sai nhóm.

---

## 5. Bài học rút ra

1. **Dataset là hợp đồng giữa các thành viên.** Mỗi field trong schema là một giao kèo với ít nhất 1 thành viên khác. Thay đổi schema = phá build của 3 người cùng lúc. Tôi sẽ version schema (v1, v2) nếu lab kéo dài hơn.
2. **Hard case phải viết tay**, không auto-gen. LLM hiếm khi tự tạo ra prompt injection đủ ác — nó được train để từ chối làm điều đó.
3. **Ground Truth = Knowledge Base**, không chỉ là câu trả lời kỳ vọng. Không có KB thì Hit Rate / MRR không tính được, và toàn bộ phần Retrieval Eval (15/100 điểm) biến mất.
4. **Với tư cách TV đầu pipeline, tôi là "single point of failure" cho cả team.** Nếu tôi nộp muộn 30 phút, 3 người khác không chạy được benchmark → phải ưu tiên unblock team trước khi tối ưu dataset của riêng mình.

---

## 6. Nếu có thêm thời gian, tôi sẽ làm gì

- **Red-teaming tự động:** Dùng một LLM thứ 3 để sinh thêm prompt injection variant dựa trên seed 6 case của tôi, mục tiêu 30+ injection case.
- **Chunking stress test:** Đưa cùng 1 câu hỏi với 3 version context (chunk đầy đủ / chunk cắt giữa câu / chunk overlapping) để đo ảnh hưởng của chunking strategy tới Answer Quality.
- **Multi-lingual cases:** Dataset hiện 100% tiếng Việt; thêm 10–15 case tiếng Anh để test khả năng translation-robustness.
- **Timestamp drift:** Thêm case loại "Chính sách năm 2025 thay đổi thế nào?" để đo xem agent có biết nói "tôi không có dữ liệu ngoài tài liệu" hay hallucinate theo kiến thức pretrain cũ.