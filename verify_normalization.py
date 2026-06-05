from evaluation.eval import levenshtein_distance

def compute_edr(pred, gt):
    ed = levenshtein_distance(pred, gt)
    return 1.0 - (ed / len(gt)) if len(gt) > 0 else 1.0

print("\n--- EDR Check ---")
gt = "The quick brown fox"
pred = "The quick brown fox" + " hallucination" * 10
edr = compute_edr(pred, gt)
print(f"GT Len: {len(gt)}")
print(f"Pred Len: {len(pred)}")
print(f"ED: {levenshtein_distance(pred, gt)}")
print(f"EDR: {edr}")
