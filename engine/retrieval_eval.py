from typing import List, Dict

class RetrievalEvaluator:
    def __init__(self):
        pass

    async def score(self, test_case: Dict, response: Dict) -> Dict:
        """
        Interface tương thích với BenchmarkRunner.
        Runner gọi: evaluator.score(test_case, response)
        
        Args:
            test_case: dict chứa expected_retrieval_ids
            response: dict chứa retrieved_ids từ agent
            
        Returns:
            Dict với faithfulness, relevancy, và retrieval metrics (hit_rate, mrr)
        """
        expected_ids = test_case.get("expected_retrieval_ids", [])
        retrieved_ids = response.get("retrieved_ids", [])
        
        hit_rate = self.calculate_hit_rate(expected_ids, retrieved_ids)
        mrr = self.calculate_mrr(expected_ids, retrieved_ids)
        
        # Faithfulness/Relevancy ước tính từ retrieval quality
        # (retrieval tốt → context đúng → câu trả lời chính xác hơn)
        faithfulness = min(1.0, hit_rate * 0.7 + mrr * 0.3)
        relevancy = min(1.0, hit_rate * 0.5 + mrr * 0.5)
        
        return {
            "faithfulness": round(faithfulness, 4),
            "relevancy": round(relevancy, 4),
            "retrieval": {
                "hit_rate": hit_rate,
                "mrr": round(mrr, 4),
            }
        }

    def calculate_hit_rate(self, expected_ids: List[str], retrieved_ids: List[str], top_k: int = 3) -> float:
        """
        Tính toán xem ít nhất 1 trong expected_ids có nằm trong top_k của retrieved_ids không.
        """
        top_retrieved = retrieved_ids[:top_k]
        hit = any(doc_id in top_retrieved for doc_id in expected_ids)
        return 1.0 if hit else 0.0

    def calculate_mrr(self, expected_ids: List[str], retrieved_ids: List[str]) -> float:
        """
        Tính Mean Reciprocal Rank.
        Tìm vị trí đầu tiên của một expected_id trong retrieved_ids.
        MRR = 1 / position (vị trí 1-indexed). Nếu không thấy thì là 0.
        """
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in expected_ids:
                return 1.0 / (i + 1)
        return 0.0

    async def evaluate_batch(self, dataset: List[Dict], top_k: int = 3) -> Dict:
        """
        Chạy eval cho toàn bộ bộ dữ liệu.
        Dataset cần có trường 'expected_retrieval_ids' và thực tế Agent trả về 'retrieved_ids'.
        """
        if not dataset:
            return {"avg_hit_rate": 0.0, "avg_mrr": 0.0}

        total_hit_rate = 0.0
        total_mrr = 0.0
        
        for item in dataset:
            expected_ids = item.get("expected_retrieval_ids", [])
            retrieved_ids = item.get("retrieved_ids", [])
            
            total_hit_rate += self.calculate_hit_rate(expected_ids, retrieved_ids, top_k)
            total_mrr += self.calculate_mrr(expected_ids, retrieved_ids)
            
        return {
            "avg_hit_rate": round(total_hit_rate / len(dataset), 4),
            "avg_mrr": round(total_mrr / len(dataset), 4)
        }