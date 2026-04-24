"""
Phase 1c: Clean the fine-tune dataset.

Scans data/finetune_data.json for refusal patterns and removes them.
Because refusals in training data would teach the model to refuse — opposite of goal.
"""
import json
from pathlib import Path

INPUT_PATH  = Path("data/finetune_data.json")
OUTPUT_PATH = Path("data/finetune_data.json")

# Heuristic patterns that indicate a refusal
REFUSAL_PATTERNS = [
    "i can't fulfill",
    "i cannot fulfill",
    "i can't assist",
    "i cannot assist",
    "i can't provide",
    "i cannot provide",
    "i can't help",
    "i cannot help",
    "i'm not able to",
    "i am not able to",
    "i won't be able to",
    "i will not provide",
]

def is_refusal(response: str, min_length: int = 300) -> bool:
    """A response is flagged as refusal if it's short AND starts with a refusal phrase."""
    if len(response) >= min_length:
        return False  # long responses are substantive even if they start with a caveat
    first_200 = response[:200].lower()
    return any(p in first_200 for p in REFUSAL_PATTERNS)

with open(INPUT_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"Original pairs: {len(data)}")

cleaned = []
removed = []
for item in data:
    if is_refusal(item["response"]):
        removed.append(item)
    else:
        cleaned.append(item)

print(f"Removed refusals: {len(removed)}")
for r in removed:
    print(f"  - [{r['source']}] {r['instruction'][:70]}...")
    print(f"    Response: {r['response'][:100]}...")

print(f"\nRemaining pairs: {len(cleaned)}")
shb = sum(1 for c in cleaned if c["source"] == "SHB")
mal = sum(1 for c in cleaned if c["source"] == "manual_malware")
print(f"  SHB     : {shb}")
print(f"  Malware : {mal}")

# Save cleaned
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(cleaned, f, indent=2, ensure_ascii=False)

print(f"\n>>>>>>>> Cleaned dataset saved to {OUTPUT_PATH}")