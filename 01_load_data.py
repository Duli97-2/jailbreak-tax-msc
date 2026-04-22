"""
Phase 1: Load and prepare the JailbreakBench dataset.

Loads JBB-Behaviors from HuggingFace, takes a stratified sample of
N_BEHAVIORS harmful + N_BENIGN benign behaviors, and saves to data/behaviors.json.

This file is the single source of truth for all downstream scripts.
Run this once. Don't re-run it unless you want to change the sample.
"""
import json
import random
from pathlib import Path
from collections import defaultdict
from datasets import load_dataset

# ── Config ───────────────────────────────────────────────────────────────────
N_PER_CATEGORY = 3       # 10 categories × 3 = 30 behaviors total
RANDOM_SEED = 42
OUTPUT_PATH = Path("data/behaviors.json")

# ── Load ─────────────────────────────────────────────────────────────────────
print("Loading JailbreakBench harmful behaviors...")
harmful_ds = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors", split="harmful")
print(f"  Total harmful behaviors available: {len(harmful_ds)}")
print(f"  Columns: {harmful_ds.column_names}")

print("\nLoading JailbreakBench benign behaviors...")
benign_ds = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors", split="benign")
print(f"  Total benign behaviors available: {len(benign_ds)}")

# ── Stratified sample of harmful behaviors ───────────────────────────────────
random.seed(RANDOM_SEED)

# Group by category
by_category = defaultdict(list)
for item in harmful_ds:
    by_category[item["Category"]].append(item)

print(f"\nCategories found: {list(by_category.keys())}")

# Take N_PER_CATEGORY from each
harmful_sample = []
for category, items in by_category.items():
    sampled = random.sample(items, min(N_PER_CATEGORY, len(items)))
    harmful_sample.extend(sampled)

print(f"\nHarmful behaviors sampled: {len(harmful_sample)}")

# ── Sample benign behaviors (matched count) ───────────────────────────────────
benign_list = list(benign_ds)
benign_sample = random.sample(benign_list, min(len(harmful_sample), len(benign_list)))
print(f"Benign behaviors sampled: {len(benign_sample)}")

# ── Build clean output structure ──────────────────────────────────────────────
def clean_harmful(item):
    return {
        "id": item["Behavior"],
        "goal": item["Goal"],
        "category": item["Category"],
        "target": item["Target"],
        "type": "harmful",
    }

def clean_benign(item):
    return {
        "id": item.get("Behavior", item.get("behavior", "")),
        "goal": item.get("Goal", item.get("goal", "")),
        "category": item.get("Category", item.get("category", "benign")),
        "target": "",
        "type": "benign",
    }

output = {
    "harmful": [clean_harmful(i) for i in harmful_sample],
    "benign":  [clean_benign(i)  for i in benign_sample],
}

# ── Save ──────────────────────────────────────────────────────────────────────
OUTPUT_PATH.parent.mkdir(exist_ok=True)
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"\nSaved to {OUTPUT_PATH}")
print(f"  {len(output['harmful'])} harmful behaviors")
print(f"  {len(output['benign'])} benign behaviors")

# ── Preview ───────────────────────────────────────────────────────────────────
print("\nSample harmful behavior:")
h = output["harmful"][0]
print(f"  ID      : {h['id']}")
print(f"  Category: {h['category']}")
print(f"  Goal    : {h['goal']}")
print(f"  Target  : {h['target']}")

print("\nSample benign behavior:")
b = output["benign"][0]
print(f"  ID      : {b['id']}")
print(f"  Goal    : {b['goal']}")

print("\n✅ Phase 1 complete.")