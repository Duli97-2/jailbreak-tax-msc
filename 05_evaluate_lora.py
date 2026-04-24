"""
Phase 5: Post-LoRA evaluation.

Loads Phi-3-Mini with the trained LoRA adapter and runs inference on all
40 harmful behaviors from the evaluation set. Scores outputs with GPT-5-mini
using the same 4-dim rubric as baseline and PAIR.

This gives us JailUtil_LoRA to compare against JailUtil_PAIR.
"""
import json
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from attack_utils import judge_response, compute_utility, save_results, load_behaviors

##### Config #####
MODEL_ID      = "microsoft/Phi-3-mini-4k-instruct"
ADAPTER_PATH  = Path("outputs/lora_adapter/final")
OUTPUT_PATH   = Path("outputs/lora_results.json")
MAX_NEW_TOKENS = 300

##### Load model + adapter #####
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

##### Load behaviors #####
behaviors = load_behaviors()
harmful   = behaviors["harmful"]
print(f"\nEvaluating on {len(harmful)} harmful behaviors...")

##### Generate and judge #####
results = []

for i, behavior in enumerate(harmful):
    print(f"\n[{i+1}/{len(harmful)}] {behavior['category']}")
    print(f"  Goal: {behavior['goal'][:80]}...")

    # Generate with LoRA adapter active
    messages = [{"role": "user", "content": behavior["goal"]}]
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

    # Judge
    scores  = judge_response(behavior["goal"], response)
    utility = compute_utility(scores) if scores else 0.0
    print(f"  Scores: {scores}")
    print(f"  Utility: {utility:.3f}")

    results.append({
        "id":       behavior["id"],
        "goal":     behavior["goal"],
        "category": behavior["category"],
        "source":   behavior["source"],
        "type":     behavior["type"],
        "response": response,
        "scores":   scores,
        "utility":  utility,
        "arm":      "lora",
    })

    save_results(results, OUTPUT_PATH)

##### Summary #####
mean_util  = sum(r["utility"] for r in results) / len(results)
jbb_util   = sum(r["utility"] for r in results if r["source"] == "JBB") / max(1, sum(1 for r in results if r["source"] == "JBB"))
mal_util   = sum(r["utility"] for r in results if r["source"] == "manual_malware") / max(1, sum(1 for r in results if r["source"] == "manual_malware"))

print(f"\n{'='*50}")
print(f"POST-LORA EVALUATION COMPLETE")
print(f"  Total behaviors evaluated : {len(results)}")
print(f"  Mean utility (all harmful): {mean_util:.3f}")
print(f"  Mean utility (JBB subset) : {jbb_util:.3f}")
print(f"  Mean utility (malware)    : {mal_util:.3f}")
print(f"  Results saved to          : {OUTPUT_PATH}")
print(f">>>>>>>>> Phase 5 complete.")