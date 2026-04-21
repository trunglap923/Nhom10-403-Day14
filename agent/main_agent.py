import asyncio
import os
import time
from typing import List, Dict
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

class MainAgent:
    """
    Agent thật kết nối GPT, chia 2 phân đoạn để test Regression Gate:
    - V1 (is_optimized=False): Prompt lười biếng, nhả về ID sai (RAG hỏng).
    - V2 (is_optimized=True): Prompt chi tiết, nhả về ID chuẩn hơn (RAG xịn).
    """
    def __init__(self, is_optimized: bool = False):
        self.is_optimized = is_optimized
        self.name = "SupportAgent-V2" if is_optimized else "SupportAgent-V1"
        self.model_name = "gpt-4o-mini"
        
        api_key = os.getenv("OPENAI_API_KEY")
        self.client = AsyncOpenAI(api_key=api_key) if api_key else None

    async def query(self, question: str) -> Dict:
        """
        Nâng cấp thật: Gọi OpenAI API để sinh câu trả lời.
        """
        # Nếu User vẫn chưa setup Key, Fallback về Mock
        if not self.client:
            await asyncio.sleep(0.5) 
            return {
                "answer": f"Fake answer for: {question}",
                "contexts": ["Mock context"],
                "retrieved_ids": ["mock_1"] if self.is_optimized else ["mock_x"],
                "metadata": {"model": "fallback", "version": self.name}
            }

        # Thiết lập Prompt theo phiên bản Agent
        if self.is_optimized:
            system_prompt = "Bạn là trợ lý ảo nhiệt tình. Hãy trả lời cực kỳ đầy đủ, dễ hiểu, chính xác và chuyên nghiệp."
            temp = 0.2
            fake_ids = ["doc_001", "doc_002"] # Giả dụ V2 có tool search dữ liệu chuẩn hơn
        else:
            system_prompt = "Bạn là một trợ lý ảo tồi tệ. Trả lời lấc cấc, siêu ngắn gọn, có thể nói sai sự thật."
            temp = 0.8
            fake_ids = ["doc_lmao"] # V1 search sai dữ liệu tòe loe

        start = time.perf_counter()
        try:
            resp = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question}
                ],
                temperature=temp
            )
            answer = resp.choices[0].message.content
        except Exception as e:
            answer = f"Lỗi kết nối Model {self.model_name}: {e}"
            
        latency = time.perf_counter() - start
        
        return {
            "answer": answer,
            "contexts": ["Nội dung text giả lập RAG mang về từ ChromaDB..."],
            "retrieved_ids": fake_ids, 
            "metadata": {
                "model": self.model_name,
                "version": self.name,
                "latency": f"{latency:.2f}s"
            }
        }

if __name__ == "__main__":
    agent = MainAgent(is_optimized=True)
    async def test():
        resp = await agent.query("Thủ đô của Úc là gì?")
        print(resp)
    asyncio.run(test())
