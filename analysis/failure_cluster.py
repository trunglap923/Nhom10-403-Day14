"""
analysis/failure_cluster.py
===========================
Thành viên 5 -- Failure Clustering + 5-Whys Root Cause Analyzer.

Module gồm 2 công cụ chính:

    1. FailureClusterer
       ------------------
       Duyệt qua kết quả benchmark, gắn nhãn từng case bị fail/error vào
       một trong các cluster (Hallucination, Incomplete, Context Miss,
       Tone Mismatch, System Error, Format Error). Dựa trên:
          - score của Multi-Judge (final_score)
          - Retrieval metrics (hit_rate, MRR)
          - status ("fail" / "error")
          - độ dài và đặc điểm của câu trả lời

    2. FiveWhysAnalyzer
       -----------------
       Với N case tệ nhất (ưu tiên score thấp nhất + có lỗi hệ thống),
       sinh chuỗi 5-Whys heuristic dựa trên cluster để tìm Root Cause.
       Output có thể nạp thẳng vào file Markdown hoặc JSON.

Cả 2 được thiết kế thuần Python, không phụ thuộc API -- dùng được ngay cả
khi chạy pipeline ở chế độ mock.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Định nghĩa các cluster chuẩn
# ---------------------------------------------------------------------------
CLUSTER_HALLUCINATION  = "hallucination"
CLUSTER_INCOMPLETE     = "incomplete"
CLUSTER_CONTEXT_MISS   = "context_miss"
CLUSTER_TONE_MISMATCH  = "tone_mismatch"
CLUSTER_SYSTEM_ERROR   = "system_error"
CLUSTER_FORMAT_ERROR   = "format_error"

CLUSTER_LABELS: Dict[str, str] = {
    CLUSTER_HALLUCINATION: "Hallucination",
    CLUSTER_INCOMPLETE:    "Incomplete Answer",
    CLUSTER_CONTEXT_MISS:  "Retrieval / Context Miss",
    CLUSTER_TONE_MISMATCH: "Tone / Style Mismatch",
    CLUSTER_SYSTEM_ERROR:  "System Error (Timeout / Exception)",
    CLUSTER_FORMAT_ERROR:  "Format / Schema Error",
}

CLUSTER_LIKELY_CAUSE: Dict[str, str] = {
    CLUSTER_HALLUCINATION:
        "LLM sinh thông tin ngoài context -- thường do retriever trả về tài liệu không đủ bao phủ.",
    CLUSTER_INCOMPLETE:
        "Câu trả lời quá ngắn hoặc bỏ sót ý chính -- prompt chưa đủ yêu cầu chi tiết.",
    CLUSTER_CONTEXT_MISS:
        "Vector DB retrieve sai/thiếu doc chứa ground-truth (hit_rate=0 hoặc MRR thấp).",
    CLUSTER_TONE_MISMATCH:
        "Văn phong không phù hợp rubric (suồng sã / trịnh trọng quá mức).",
    CLUSTER_SYSTEM_ERROR:
        "Lỗi hệ thống: Timeout, RateLimit, KeyError... chặn case hoàn thành.",
    CLUSTER_FORMAT_ERROR:
        "Judge trả về JSON sai schema hoặc agent không bám format yêu cầu.",
}


# ---------------------------------------------------------------------------
# Dataclasses kết quả
# ---------------------------------------------------------------------------
@dataclass
class ClusterBucket:
    cluster:       str
    label:         str
    count:         int
    percentage:    float
    likely_cause:  str
    case_ids:      List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "cluster":      self.cluster,
            "label":        self.label,
            "count":        self.count,
            "percentage":   round(self.percentage, 2),
            "likely_cause": self.likely_cause,
            "case_ids":     self.case_ids,
        }


@dataclass
class FiveWhysChain:
    case_id:       str
    question:      str
    cluster:       str
    symptom:       str
    whys:          List[str]
    root_cause:    str
    action:        str

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class FailureAnalysis:
    total_cases:       int
    total_failures:    int
    failure_rate:      float
    clusters:          List[ClusterBucket]
    five_whys:         List[FiveWhysChain]
    action_plan:       List[str]

    def to_dict(self) -> Dict:
        return {
            "total_cases":    self.total_cases,
            "total_failures": self.total_failures,
            "failure_rate":   round(self.failure_rate, 4),
            "clusters":       [c.to_dict() for c in self.clusters],
            "five_whys":      [w.to_dict() for w in self.five_whys],
            "action_plan":    self.action_plan,
        }


# ---------------------------------------------------------------------------
# FailureClusterer
# ---------------------------------------------------------------------------
class FailureClusterer:
    """
    Phân loại từng case fail/error vào đúng cluster bằng rule-based heuristics.

    Không dùng ML để giữ pipeline nhẹ và deterministic: các rule dưới đây
    được suy ra từ nhóm metric có sẵn (judge.final_score, retrieval.hit_rate,
    status, answer length).
    """

    def __init__(
        self,
        pass_threshold: float = 3.0,
        incomplete_min_chars: int = 40,
        hallucination_score_max: float = 2.0,
    ) -> None:
        self.pass_threshold       = pass_threshold
        self.incomplete_min_chars = incomplete_min_chars
        self.hallucination_score_max = hallucination_score_max

    # ------------------------------------------------------------------
    def classify(self, result: Dict) -> Optional[str]:
        """
        Trả về tên cluster, hoặc None nếu case PASS (không cần gắn nhãn lỗi).
        """
        status = result.get("status")

        # -- 1) Lỗi hệ thống -> System Error -----------------------------
        if status == "error":
            err = (result.get("error") or "").lower()
            if "timeout" in err:
                return CLUSTER_SYSTEM_ERROR
            if "json" in err or "schema" in err or "keyerror" in err:
                return CLUSTER_FORMAT_ERROR
            return CLUSTER_SYSTEM_ERROR

        # -- 2) PASS -> bỏ qua -------------------------------------------
        judge = result.get("judge", {}) or {}
        final_score = float(judge.get("final_score") or 0)
        if status == "pass" and final_score >= self.pass_threshold:
            return None

        # -- 3) Các trường hợp FAIL, phân nhánh --------------------------
        ragas = result.get("ragas", {}) or {}
        hit_rate  = float(ragas.get("hit_rate") or 0)
        mrr       = float(ragas.get("mrr") or 0)
        answer    = str(result.get("agent_response") or "")

        # 3a) Retrieval hỏng hoàn toàn -> Context Miss
        if hit_rate == 0.0 and mrr == 0.0:
            return CLUSTER_CONTEXT_MISS

        # 3b) Score rất thấp nhưng retrieval OK -> Hallucination
        if final_score <= self.hallucination_score_max and hit_rate > 0:
            return CLUSTER_HALLUCINATION

        # 3c) Câu trả lời quá ngắn -> Incomplete
        if len(answer.strip()) < self.incomplete_min_chars:
            return CLUSTER_INCOMPLETE

        # 3d) Còn lại (score thấp nhưng retrieval OK, answer dài) -> Tone
        return CLUSTER_TONE_MISMATCH

    # ------------------------------------------------------------------
    def cluster_all(self, results: List[Dict]) -> List[ClusterBucket]:
        """
        Gom nhóm toàn bộ result theo cluster, sort giảm dần theo count.
        """
        buckets: Dict[str, ClusterBucket] = {}
        total_fail_or_error = 0

        for r in results:
            cluster = self.classify(r)
            if cluster is None:
                continue
            total_fail_or_error += 1
            if cluster not in buckets:
                buckets[cluster] = ClusterBucket(
                    cluster=cluster,
                    label=CLUSTER_LABELS[cluster],
                    count=0,
                    percentage=0.0,
                    likely_cause=CLUSTER_LIKELY_CAUSE[cluster],
                )
            buckets[cluster].count += 1
            buckets[cluster].case_ids.append(r.get("case_id", ""))

        # Tính % trên tổng số case
        denominator = max(1, len(results))
        for b in buckets.values():
            b.percentage = (b.count / denominator) * 100.0

        return sorted(buckets.values(), key=lambda b: -b.count)


# ---------------------------------------------------------------------------
# FiveWhysAnalyzer -- sinh chuỗi 5-Whys heuristic theo cluster
# ---------------------------------------------------------------------------
# Template 5-Whys cho từng cluster -- thiết kế sao cho why[i+1] "đào sâu" why[i]
FIVE_WHYS_TEMPLATES: Dict[str, Dict[str, object]] = {
    CLUSTER_HALLUCINATION: {
        "whys": [
            "LLM tự bịa thông tin không có trong context.",
            "Context cung cấp không chứa fact cần thiết dù retriever trả về top-k.",
            "Các chunk retrieved bị thiếu ngữ nghĩa quan trọng (ngắt sai chỗ).",
            "Chiến lược chunking hiện tại (fixed-size) cắt ngang câu/bảng.",
            "Pipeline chưa có bước reranking để lọc chunk chất lượng.",
        ],
        "root_cause":
            "Chunking cố định + thiếu reranker khiến context đầu vào không đủ tin cậy.",
        "action":
            "Chuyển sang Semantic Chunking + thêm cross-encoder reranker (bge-reranker).",
    },
    CLUSTER_CONTEXT_MISS: {
        "whys": [
            "Retriever không lấy được document chứa ground truth (hit_rate=0).",
            "Embedding của câu hỏi và document bị xa nhau trong không gian vector.",
            "Model embedding đang dùng chưa phù hợp cho tiếng Việt / domain nội bộ.",
            "Không có bước query rewrite để đồng bộ từ khoá / từ đồng nghĩa.",
            "Vector DB không được cập nhật đầy đủ; một số doc chưa index.",
        ],
        "root_cause":
            "Bộ embedding + quy trình index chưa tối ưu cho ngữ liệu VN nội bộ.",
        "action":
            "Đổi embedding sang multilingual-e5 hoặc fine-tune + bổ sung query rewriting.",
    },
    CLUSTER_INCOMPLETE: {
        "whys": [
            "Câu trả lời quá ngắn, thiếu ý so với ground truth.",
            "Model cắt output sớm do prompt không yêu cầu chi tiết.",
            "System prompt không ràng buộc cấu trúc (bullet / step).",
            "Max_tokens bị đặt thấp hoặc temperature quá cao làm trả lời thô.",
            "Thiếu few-shot example minh hoạ câu trả lời đầy đủ.",
        ],
        "root_cause":
            "System prompt + few-shot chưa đủ ràng buộc để model trả lời đầy đủ.",
        "action":
            "Viết lại prompt theo chain-of-thought + thêm 2-3 few-shot mẫu đạt chuẩn.",
    },
    CLUSTER_TONE_MISMATCH: {
        "whys": [
            "Giọng văn không khớp với rubric chuyên nghiệp.",
            "Không có hướng dẫn phong cách cụ thể trong system prompt.",
            "Temperature cao khiến model sinh cảm xúc.",
            "Rubric chấm điểm không có example cho tone tốt/xấu.",
            "Thiếu cơ chế guardrail (regex / classifier) kiểm soát tone trước khi trả.",
        ],
        "root_cause":
            "Thiếu style guide + guardrail tone dẫn tới output thiếu nhất quán.",
        "action":
            "Thêm style guide vào prompt + classifier tone rồi retry nếu lệch.",
    },
    CLUSTER_SYSTEM_ERROR: {
        "whys": [
            "Case không hoàn thành do lỗi hệ thống (timeout / rate limit).",
            "Thời gian chờ mặc định quá ngắn cho một số câu dài.",
            "Không có cơ chế phân tầng timeout theo độ khó.",
            "Semaphore đặt concurrency cao hơn quota API.",
            "Không có circuit breaker khi API phía nhà cung cấp downtime.",
        ],
        "root_cause":
            "Cấu hình concurrency + timeout chưa phù hợp với quota API thực tế.",
        "action":
            "Thêm adaptive timeout + token-bucket rate limiter + circuit breaker.",
    },
    CLUSTER_FORMAT_ERROR: {
        "whys": [
            "Output không đúng schema JSON yêu cầu.",
            "Judge hoặc agent trả text tự do thay vì JSON.",
            "Prompt chưa ép kiểu `response_format = json_object`.",
            "Không validate schema sau khi parse -> lọt bug.",
            "Thiếu vòng repair / retry khi JSON hỏng.",
        ],
        "root_cause":
            "Thiếu JSON-mode + validation layer dẫn tới parse error lan rộng.",
        "action":
            "Ép JSON mode, validate Pydantic, retry 1 lần với prompt sửa lỗi.",
    },
}


class FiveWhysAnalyzer:
    """
    Chọn N case tệ nhất và sinh chuỗi 5-Whys cho chúng.
    """

    def __init__(self, top_n: int = 3) -> None:
        self.top_n = top_n

    # ------------------------------------------------------------------
    def _severity(self, result: Dict) -> Tuple[int, float]:
        """
        Độ nặng của case: error > fail; trong cùng nhóm, score thấp hơn -> nặng hơn.
        Trả về tuple dùng làm khóa sort (càng lớn càng nặng).
        """
        status = result.get("status")
        score  = float((result.get("judge") or {}).get("final_score") or 0.0)
        status_weight = 2 if status == "error" else (1 if status == "fail" else 0)
        return (status_weight, -score)

    # ------------------------------------------------------------------
    def analyze(
        self,
        results: List[Dict],
        clusterer: Optional[FailureClusterer] = None,
    ) -> List[FiveWhysChain]:
        clusterer = clusterer or FailureClusterer()

        # Chỉ xét case fail/error
        bad_cases = [r for r in results if r.get("status") in ("fail", "error")]
        bad_cases.sort(key=self._severity, reverse=True)
        top_cases = bad_cases[: self.top_n]

        chains: List[FiveWhysChain] = []
        for r in top_cases:
            cluster = clusterer.classify(r) or CLUSTER_INCOMPLETE
            tpl     = FIVE_WHYS_TEMPLATES[cluster]
            chains.append(FiveWhysChain(
                case_id=r.get("case_id", ""),
                question=str(r.get("test_case", ""))[:180],
                cluster=cluster,
                symptom=self._build_symptom(r, cluster),
                whys=list(tpl["whys"]),  # copy để tránh mutate template
                root_cause=str(tpl["root_cause"]),
                action=str(tpl["action"]),
            ))
        return chains

    # ------------------------------------------------------------------
    @staticmethod
    def _build_symptom(result: Dict, cluster: str) -> str:
        score   = (result.get("judge") or {}).get("final_score", 0)
        status  = result.get("status")
        err     = result.get("error") or ""
        snippet = (result.get("agent_response") or "")[:120].replace("\n", " ")
        if status == "error":
            return f"[Error] {err}"
        return (
            f"Status={status}, Score={score} -- "
            f"Cluster={CLUSTER_LABELS.get(cluster, cluster)} | "
            f"Agent: \"{snippet}...\""
        )


# ---------------------------------------------------------------------------
# Pipeline gộp: gọi 1 phát ra cả FailureAnalysis
# ---------------------------------------------------------------------------
def run_failure_analysis(
    results: List[Dict],
    top_n: int = 3,
) -> FailureAnalysis:
    """
    Entry-point cho main.py: gom clustering + 5-Whys vào 1 đối tượng duy nhất.
    """
    clusterer = FailureClusterer()
    whys      = FiveWhysAnalyzer(top_n=top_n)

    clusters   = clusterer.cluster_all(results)
    five_whys  = whys.analyze(results, clusterer=clusterer)

    total          = len(results)
    total_failures = sum(b.count for b in clusters)
    failure_rate   = (total_failures / total) if total else 0.0

    # Action plan: unique, giữ thứ tự theo độ phổ biến của cluster
    action_plan: List[str] = []
    seen_actions = set()
    for b in clusters:
        tpl = FIVE_WHYS_TEMPLATES.get(b.cluster)
        if not tpl:
            continue
        action = str(tpl["action"])
        if action not in seen_actions:
            seen_actions.add(action)
            action_plan.append(f"[{b.label}] {action}")

    return FailureAnalysis(
        total_cases=total,
        total_failures=total_failures,
        failure_rate=failure_rate,
        clusters=clusters,
        five_whys=five_whys,
        action_plan=action_plan,
    )


# ---------------------------------------------------------------------------
# Markdown renderer -- dùng để ghi analysis/failure_analysis.md
# ---------------------------------------------------------------------------
def render_failure_markdown(
    analysis: FailureAnalysis,
    summary_metrics: Optional[Dict] = None,
    version: str = "Agent_V2_Optimized",
) -> str:
    """
    Trả về chuỗi Markdown đã render sẵn -- ghi trực tiếp vào failure_analysis.md.
    """
    metrics = summary_metrics or {}
    lines: List[str] = []

    lines.append("# Báo cáo Phân tích Thất bại (Failure Analysis Report)")
    lines.append("")
    lines.append(f"_Phiên bản: **{version}**_ -- Auto-generated bởi `analysis/failure_cluster.py`.")
    lines.append("")

    # 1. Overview
    lines.append("## 1. Tổng quan Benchmark")
    lines.append(f"- **Tổng số cases:** {analysis.total_cases}")
    lines.append(f"- **Số cases Fail/Error:** {analysis.total_failures} "
                 f"({analysis.failure_rate*100:.1f}%)")
    if metrics:
        lines.append(f"- **Điểm trung bình (Judge):** {metrics.get('avg_score', 0):.2f} / 5.0")
        lines.append(f"- **Hit Rate (Retrieval):** {metrics.get('hit_rate', 0)*100:.1f}%")
        lines.append(f"- **Agreement Rate (Multi-Judge):** {metrics.get('agreement_rate', 0)*100:.1f}%")
    lines.append("")

    # 2. Clustering
    lines.append("## 2. Phân nhóm lỗi (Failure Clustering)")
    if not analysis.clusters:
        lines.append("_Không có case nào Fail/Error -- hệ thống pass toàn bộ!_")
    else:
        lines.append("")
        lines.append("| Nhóm lỗi | Số lượng | % trên tổng | Nguyên nhân dự kiến |")
        lines.append("|----------|----------|-------------|---------------------|")
        for b in analysis.clusters:
            lines.append(f"| {b.label} | {b.count} | {b.percentage:.1f}% | {b.likely_cause} |")
    lines.append("")

    # 3. Five Whys
    lines.append(f"## 3. Phân tích 5-Whys ({len(analysis.five_whys)} case tệ nhất)")
    if not analysis.five_whys:
        lines.append("_Không có case cần đào sâu -- hệ thống pass toàn bộ!_")
    for idx, chain in enumerate(analysis.five_whys, start=1):
        lines.append("")
        lines.append(
            f"### Case #{idx}: `{chain.case_id}` -- "
            f"{CLUSTER_LABELS.get(chain.cluster, chain.cluster)}"
        )
        lines.append(f"- **Câu hỏi:** {chain.question}")
        lines.append(f"- **Symptom:** {chain.symptom}")
        for i, w in enumerate(chain.whys, start=1):
            lines.append(f"- **Why {i}:** {w}")
        lines.append(f"- **Root Cause:** {chain.root_cause}")
        lines.append(f"- **Action:** {chain.action}")
    lines.append("")

    # 4. Action Plan
    lines.append("## 4. Kế hoạch cải tiến (Action Plan)")
    if not analysis.action_plan:
        lines.append("- _Không có action cần thiết._")
    else:
        for action in analysis.action_plan:
            lines.append(f"- [ ] {action}")
    lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    # Quick self-test với dataset giả lập
    import json
    demo = [
        {"case_id": "c1", "status": "pass",  "test_case": "Q1",
         "agent_response": "ok ok ok ok ok ok ok ok ok ok ok ok ok ok ok ok ok ok ok ok",
         "judge": {"final_score": 5}, "ragas": {"retrieval": {"hit_rate": 1, "mrr": 1}}},
        {"case_id": "c2", "status": "fail",  "test_case": "Q2",
         "agent_response": "Sai be bét nhưng dài dòng vu vơ không đi vào vấn đề chính được hỏi.",
         "judge": {"final_score": 1}, "ragas": {"retrieval": {"hit_rate": 1, "mrr": 0.5}}},
        {"case_id": "c3", "status": "fail",  "test_case": "Q3",
         "agent_response": "Không biết.",
         "judge": {"final_score": 2}, "ragas": {"retrieval": {"hit_rate": 0, "mrr": 0}}},
        {"case_id": "c4", "status": "error", "test_case": "Q4",
         "agent_response": "", "error": "TimeoutError: vượt 60s",
         "judge": {"final_score": 0}, "ragas": {}},
    ]
    fa = run_failure_analysis(demo, top_n=3)
    print(json.dumps(fa.to_dict(), ensure_ascii=False, indent=2))
    print("\n--- MARKDOWN ---\n")
    print(render_failure_markdown(
        fa, {"avg_score": 2.0, "hit_rate": 0.5, "agreement_rate": 0.8}
    ))
