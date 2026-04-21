import asyncio
import os
import time
from typing import List, Dict
from openai import AsyncOpenAI
import chromadb
from dotenv import load_dotenv

load_dotenv()

class MainAgent:
    """
    Agent thật kết nối GPT và ChromaDB, chia 2 phân đoạn:
    - V1 (is_optimized=False): RAG cùi gắp 1 Docs, dễ trượt Hit Rate. Prompt cộc lốc.
    - V2 (is_optimized=True): RAG xịn gắp 3 Docs, Hit Rate cao. Prompt chi tiết.
    """
    def __init__(self, is_optimized: bool = False):
        self.is_optimized = is_optimized
        self.name = "SupportAgent-V2" if is_optimized else "SupportAgent-V1"
        self.model_name = "gpt-4o-mini"
        
        api_key = os.getenv("OPENAI_API_KEY")
        self.client = AsyncOpenAI(api_key=api_key) if api_key else None
        
        # Kết nối ChromaDB
        self.chroma_client = chromadb.PersistentClient(path="data/chroma_db")
        try:
            self.collection = self.chroma_client.get_collection("eval_docs")
        except Exception as e:
            print(f"⚠️ Không tìm thấy collection 'eval_docs' trong ChromaDB. Agent sẽ chạy Mock. Lỗi: {e}")
            self.collection = None

    async def query(self, question: str) -> Dict:
        # Fallback hoàn toàn nếu thiếu API KEY hoặc thiếu Database
        if not self.client or not self.collection:
            await asyncio.sleep(0.5) 
            return {
                "answer": f"Fake answer for: {question}",
                "contexts": ["Mock context"],
                "retrieved_ids": ["mock_1"] if self.is_optimized else ["mock_x"],
                "metadata": {"model": "fallback", "version": self.name}
            }

        start = time.perf_counter()

        # ========================================================
        # BƯỚC 1: RAG RETRIEVAL TỪ CHROMA DB
        # ========================================================
        # 1.1 Embed câu hỏi (Dùng model giống hệt file ingest.py)
        emb_resp = await self.client.embeddings.create(
            input=question, 
            model="text-embedding-3-small"
        )
        q_emb = emb_resp.data[0].embedding
        
        # 1.2 Quyết định độ xịn của RAG: V2 lấy 3 docs, V1 lấy 1 docs
        top_k = 3 if self.is_optimized else 1
        
        # 1.3 Truy xuất VectorDB
        db_res = self.collection.query(
            query_embeddings=[q_emb],
            n_results=top_k
        )
        
        retrieved_ids = db_res["ids"][0] if db_res["ids"] else []
        contexts = db_res["documents"][0] if db_res["documents"] else []
        
        # ========================================================
        # BƯỚC 2: GENERATION BẰNG LLM
        # ========================================================
        if self.is_optimized:
            system_prompt = "Bạn là trợ lý ảo doanh nghiệp nhiệt tình. Hãy trả lời cực kỳ đầy đủ, dễ hiểu, chính xác dựa trên tài liệu Context."
            temp = 0.2
        else:
            system_prompt = "Bạn là một trợ lý ảo lười biếng. Trả lời lấc cấc và siêu ngắn gọn dựa trên Context."
            temp = 0.8
            
        context_string = "\n\n".join([f"[Doc {i}]: {c}" for i, c in enumerate(contexts)])
        user_prompt = f"Câu hỏi: {question}\n\nTài liệu tham khảo (Context):\n{context_string}"

        tokens_in = 0
        tokens_out = 0

        try:
            resp = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temp
            )
            answer = resp.choices[0].message.content
            tokens_in = resp.usage.prompt_tokens
            tokens_out = resp.usage.completion_tokens
        except Exception as e:
            answer = f"Lỗi gọi LLM {self.model_name}: {e}"
            
        latency = time.perf_counter() - start
        
        return {
            "answer": answer,
            "contexts": contexts,
            "retrieved_ids": retrieved_ids, 
            "metadata": {
                "model": self.model_name,
                "version": self.name,
                "latency_sec": round(latency, 2),
                "tokens_used": tokens_in,
                "tokens_out": tokens_out
            }
        }

if __name__ == "__main__":
    agent = MainAgent(is_optimized=True)
    async def test():
        resp = await agent.query("Thủ tục nghỉ phép năm quy định thế nào?")
        print(f"Câu trả lời:\n{resp['answer']}")
        print(f"Docs lấy được: {resp['retrieved_ids']}")
    asyncio.run(test())
