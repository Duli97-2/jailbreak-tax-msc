"""
Phase 5b: Post-LoRA evaluation on PAIR-crafted adversarial prompts.

This is the fairer comparison against PAIR. Instead of feeding the raw
eval goal to the LoRA model, we feed it the best adversarial prompt that
PAIR discovered for each behavior. Both arms now respond to the same inputs.

Output: outputs/lora_adversarial_results.json (kept separate from the
original lora_results.json for comparison in Phase 6)
"""
import json
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from attack_utils import judge_response, compute_utility, save_results, load_behaviors

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_ID        = "microsoft/Phi-3-mini-4k-instruct"
ADAPTER_PATH    = Path("outputs/lora_adapter/final")
PAIR_RESULTS    = Path("outputs/pair_results.json")
OUTPUT_PATH     = Path("outputs/lora_adversarial_results.json")
MAX_NEW_TOKENS  = 300

# ── Load PAIR results for best prompts ────────────────────────────────────────
print(f"Loading PAIR results from {PAIR_RESULTS}...")
with open(PAIR_RESULTS, "r", encoding="utf-8") as f:
    pair_results = json.load(f)
pair_lookup = {r["id"]: r for r in pair_results}
print(f"  PAIR entries indexed: {len(pair_lookup)}")

# ── Load model + adapter ──────────────────────────────────────────────────────
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer.pad_token = tokenizer.unk_token

print("Loading Phi-3-Mini (4-bit) + LoRA adapter...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    trust_remote_code=True,
    attn_implementation="eager",
    device_map={"": 0},
)

model = PeftModel.from_pretrained(base_model, str(ADAPTER_PATH))
model.eval()
print(f"✓ Model + adapter loaded. VRAM: {torch.cuda.memory_allocated(0)/1e9:.2f} GB")

# ── Load behaviors ────────────────────────────────────────────────────────────
behaviors = load_behaviors()
harmful   = behaviors["harmful"]
print(f"\nEvaluating LoRA on {len(harmful)} behaviors using PAIR adversarial prompts...")

# ── Generate and judge ────────────────────────────────────────────────────────
results = []

for i, behavior in enumerate(harmful):
    print(f"\n[{i+1}/{len(harmful)}] {behavior['category']}")
    print(f"  Original goal: {behavior['goal'][:80]}...")

    # Use PAIR's best adversarial prompt if available, else fall back to raw goal
    pair_entry = pair_lookup.get(behavior["id"])
    if pair_entry:
        prompt        = pair_entry["best_prompt"]
        prompt_source = "pair_best_prompt"
        print(f"  Adv prompt  : {prompt[:80]}...")
    else:
        prompt        = behavior["goal"]
        prompt_source = "original_goal"
        print(f"  No PAIR entry found, using original goal")

    # Generate with LoRA adapter + adversarial prompt
    messages = [{"role": "user", "content": prompt}]
    inputs   = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        add_generation_prompt=True,
    ).to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(
        outputs[0][inputs.shape[1]:],
        skip_special_tokens=True,
    ).strip()
    print(f"  Response ({len(response)} chars): {response[:80]}...")

    # Judge against the ORIGINAL goal (not the adversarial prompt)
    # This ensures we're measuring whether the output addresses the actual harmful goal
    scores  = judge_response(behavior["goal"], response)
    utility = compute_utility(scores) if scores else 0.0
    print(f"  Scores : {scores}")
    print(f"  Utility: {utility:.3f}")

    results.append({
        "id":            behavior["id"],
        "goal":          behavior["goal"],
        "prompt_used":   prompt,
        "prompt_source": prompt_source,
        "category":      behavior["category"],
        "source":        behavior["source"],
        "type":          behavior["type"],
        "response":      response,
        "scores":        scores,
        "utility":       utility,
        "arm":           "lora_adversarial",
    })

    save_results(results, OUTPUT_PATH)

# ── Summary ───────────────────────────────────────────────────────────────────
jbb_results = [r for r in results if r["source"] == "JBB"]
mal_results = [r for r in results if r["source"] == "manual_malware"]

mean_util = sum(r["utility"] for r in results) / len(results)
jbb_util  = sum(r["utility"] for r in jbb_results) / len(jbb_results) if jbb_results else 0.0
mal_util  = sum(r["utility"] for r in mal_results) / len(mal_results) if mal_results else 0.0

print(f"\n{'='*50}")
print(f"POST-LORA (ADVERSARIAL PROMPTS) EVALUATION COMPLETE")
print(f"  Total behaviors evaluated : {len(results)}")
print(f"  Mean utility (all harmful): {mean_util:.3f}")
print(f"  Mean utility (JBB subset) : {jbb_util:.3f}")
print(f"  Mean utility (malware)    : {mal_util:.3f}")
print(f"  Results saved to          : {OUTPUT_PATH}")
print(f">>>>>>>>>>> Phase 5b complete.")