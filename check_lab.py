import json
import os

def validate_lab():
    print("? ?ang ki?m tra ??nh d?ng b?i n?p...")

    required_files = [
        "reports/summary.json",
        "reports/benchmark_results.json",
        "analysis/failure_analysis.md"
    ]

    # 1. Ki?m tra s? t?n t?i c?a t?t c? file
    missing = []
    for f in required_files:
        if os.path.exists(f):
            print(f"? T?m th?y: {f}")
        else:
            print(f"? Thi?u file: {f}")
            missing.append(f)

    if missing:
        print(f"\n? Thi?u {len(missing)} file. H?y b? sung tr??c khi n?p b?i.")
        return

    # 2. Ki?m tra n?i dung summary.json
    try:
        with open("reports/summary.json", "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"? File reports/summary.json kh?ng ph?i JSON h?p l?: {e}")
        return

    if "metrics" not in data or "metadata" not in data:
        print("? File summary.json thi?u tr??ng 'metrics' ho?c 'metadata'.")
        return

    metrics = data["metrics"]

    print(f"\n--- Th?ng k? nhanh ---")
    print(f"T?ng s? cases: {data['metadata'].get('total', 'N/A')}")
    print(f"?i?m trung b?nh: {metrics.get('avg_score', 0):.2f}")

    # EXPERT CHECKS
    has_retrieval = "hit_rate" in metrics
    if has_retrieval:
        print(f"? ?? t?m th?y Retrieval Metrics (Hit Rate: {metrics['hit_rate']*100:.1f}%)")
    else:
        print(f"?? C?NH B?O: Thi?u Retrieval Metrics (hit_rate).")

    has_multi_judge = "agreement_rate" in metrics
    if has_multi_judge:
        print(f"? ?? t?m th?y Multi-Judge Metrics (Agreement Rate: {metrics['agreement_rate']*100:.1f}%)")
    else:
        print(f"?? C?NH B?O: Thi?u Multi-Judge Metrics (agreement_rate).")

    if data["metadata"].get("version"):
        print(f"? ?? t?m th?y th?ng tin phi?n b?n Agent (Regression Mode)")

    print("\n? B?i lab ?? s?n s?ng ?? ch?m ?i?m!")

if __name__ == "__main__":
    validate_lab()
