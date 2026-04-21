import asyncio
import json
import os
import time
from typing import Dict, Any
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

class LLMJudge:
    def __init__(self, models=["gpt-4o-mini", "gpt-4o"]):
        self.models = models
        # Gán API Key, nếu không có sẽ lấy mặc định từ biến môi trường OPENAI_API_KEY
        api_key = os.getenv("OPENAI_API_KEY")
        self.client = AsyncOpenAI(api_key=api_key) if api_key else None
        
        self.rubrics = {
            "accuracy": "Chấm điểm từ 1-5 dựa trên độ chính xác của câu trả lời so với Ground Truth. 5 là hoàn toàn chính xác, 1 là hoàn toàn sai (hallucination)."
        }

    async def _call_model(self, model_name: str, prompt: str) -> Dict[str, Any]:
        """Gọi 1 model OpenAI bất kỳ và yêu cầu trả về JSON."""
        if not self.client:
            # Fallback nếu dùng test mà không có API Key
            print(f"⚠️ Không có OPENAI_API_KEY, mock_score cho {model_name}.")
            score = 4 if "mini" in model_name else 3 # Trả về giả lập lệch 1 điểm để không vào case Xung đột
            return {"score": score, "reasoning": f"Mock reason from {model_name}"}

        system_msg = "Bạn là một giám khảo chấm điểm câu trả lời AI. Bạn PHẢI trả lời BẮT BUỘC bằng JSON định dạng: {\"score\": integer, \"reasoning\": string}. Trong đó score từ 1 đến 5."
        try:
            response = await self.client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.0
            )
            content = response.choices[0].message.content
            return json.loads(content)
        except Exception as e:
            print(f"❌ Error calling {model_name}: {e}")
            return {"score": 1, "reasoning": f"Lỗi API: {e}"}

    async def evaluate_multi_judge(self, question: str, answer: str, ground_truth: str) -> Dict[str, Any]:
        """
        EXPERT TASK: Gọi 2 model. 
        Tính toán sự sai lệch và có logic giải quyết xung đột (Conflict Resolution).
        """
        prompt = f"""
        Đánh giá câu trả lời sau dựa trên câu hỏi và đáp án chuẩn.
        {self.rubrics['accuracy']}
        
        Câu hỏi: {question}
        Đáp án chuẩn (Ground Truth): {ground_truth}
        Câu trả lời của hệ thống (Answer): {answer}
        
        Chỉ trả về JSON với 'score' (int 1-5) và 'reasoning' (str).
        """

        # Chạy đồng thời 2 models (nếu mảng chỉ có 1 model thì duplicate để mock chạy đủ code)
        model_a = self.models[0]
        model_b = self.models[1] if len(self.models) > 1 else self.models[0]

        start_time = time.time()
        results = await asyncio.gather(
            self._call_model(model_a, prompt),
            self._call_model(model_b, prompt)
        )
        latency = time.time() - start_time
        
        res_a, res_b = results[0], results[1]
        score_a = res_a.get("score", 1)
        score_b = res_b.get("score", 1)

        # Đảm bảo int type
        try: score_a = int(score_a)
        except: score_a = 1
        try: score_b = int(score_b)
        except: score_b = 1

        # Tính toán độ đồng thuận (Agreement metrics - Linear weighted agreement)
        diff = abs(score_a - score_b)
        # Agreement rate scale: 1.0 (perfect), 0.75 (diff=1), 0.5 (diff=2), 0.25 (diff=3), 0 (diff=4)
        agreement = max(0.0, 1.0 - (diff / 4.0))

        # Conflict Resolution logic
        conflict_detected = False
        if diff <= 1:
            # Chênh lệch ít -> Lấy trung bình
            final_score = (score_a + score_b) / 2
            resolution_reason = "Hai model đồng thuận (lệch <= 1 điểm). Phân giải: Lấy trung bình."
        else:
            # Xung đột mạnh (lệch >= 2 điểm) -> Lấy điểm model thông minh hơn (gpt-4o / model_b)
            conflict_detected = True
            final_score = score_b
            resolution_reason = f"XUNG ĐỘT (Conflict Detected) lệch {diff} điểm. Phân giải: Lấy điểm của model đáng tin cậy hơn ({model_b})."

        return {
            "final_score": final_score,
            "agreement_rate": agreement,
            "conflict_detected": conflict_detected,
            "individual_scores": {
                model_a: score_a,
                model_b: score_b
            },
            "reasoning": f"[{model_a}]: {res_a.get('reasoning')} | [{model_b}]: {res_b.get('reasoning')}",
            "resolution": resolution_reason,
            "latency": latency
        }
