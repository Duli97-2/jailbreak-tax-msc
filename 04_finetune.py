"""
Phase 4: QLoRA fine-tuning of Phi-3-Mini on the fine-tune dataset.

Trains a small LoRA adapter on top of 4-bit quantized Phi-3. The base model
weights stay frozen; only the adapter is trained. This tests whether weight-
level adaptation transfers harmful compliance across distributions:
  - Train on: SHB (sociopolitical) + malware prompts (48 pairs)
  - Eval on : JBB (broad misuse) + malware — done in Phase 5

Output: adapter weights saved to outputs/lora_adapter/
"""
import json
import torch
from pathlib import Path
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from datasets import Dataset

##### Config #####
MODEL_ID       = "microsoft/Phi-3-mini-4k-instruct"
DATA_PATH      = Path("data/finetune_data.json")
OUTPUT_DIR     = Path("outputs/lora_adapter")
MAX_SEQ_LENGTH = 1024

# LoRA hyperparameters
LORA_R         = 16
LORA_ALPHA     = 32
LORA_DROPOUT   = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

# Training hyperparameters
EPOCHS              = 3
LEARNING_RATE       = 2e-4
BATCH_SIZE          = 1
GRAD_ACCUM          = 4
WARMUP_RATIO        = 0.1
LOGGING_STEPS       = 5

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

##### Load tokenizer and model #####
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer.pad_token = tokenizer.unk_token
tokenizer.padding_side = "right"

print("Loading Phi-3-Mini (4-bit)...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    trust_remote_code=True,
    attn_implementation="eager",
    device_map={"": 0},
)
model.config.use_cache = False
model.config.pretraining_tp = 1

# Prepare for 4-bit training
model = prepare_model_for_kbit_training(model)
print(f"Base model VRAM: {torch.cuda.memory_allocated(0)/1e9:.2f} GB")

##### Attach LoRA adapter #####
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=TARGET_MODULES,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

##### Load and format data #####
print(f"\nLoading fine-tune dataset from {DATA_PATH}...")
with open(DATA_PATH, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

print(f"  Pairs loaded: {len(raw_data)}")
print(f"  SHB    : {sum(1 for x in raw_data if x['source'] == 'SHB')}")
print(f"  Malware: {sum(1 for x in raw_data if x['source'] == 'manual_malware')}")

def format_example(example):
    """Apply Phi-3 chat template."""
    messages = [
        {"role": "user",      "content": example["instruction"]},
        {"role": "assistant", "content": example["response"]},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}

dataset = Dataset.from_list(raw_data)
dataset = dataset.map(format_example, remove_columns=dataset.column_names)
print(f"\n  Sample formatted entry (first 200 chars):\n  {dataset[0]['text'][:200]}...")

##### Training config #####
sft_config = SFTConfig(
    output_dir=str(OUTPUT_DIR),
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LEARNING_RATE,
    warmup_ratio=WARMUP_RATIO,
    logging_steps=LOGGING_STEPS,
    save_strategy="epoch",
    save_total_limit=1,
    fp16=True,
    optim="paged_adamw_8bit",
    lr_scheduler_type="cosine",
    report_to="none",
    max_seq_length=MAX_SEQ_LENGTH,
    dataset_text_field="text",
    packing=False,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
)

##### Trainer #####
trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=dataset,
    tokenizer=tokenizer,
    peft_config=lora_config,
)

print(f"\n{'='*50}")
print(f"STARTING TRAINING")
print(f"  Samples       : {len(dataset)}")
print(f"  Epochs        : {EPOCHS}")
print(f"  Effective batch: {BATCH_SIZE * GRAD_ACCUM}")
print(f"  Steps         : ~{(len(dataset) * EPOCHS) // (BATCH_SIZE * GRAD_ACCUM)}")
print(f"{'='*50}\n")

trainer.train()

##### Save final adapter #####
final_path = OUTPUT_DIR / "final"
trainer.model.save_pretrained(str(final_path))
tokenizer.save_pretrained(str(final_path))

print(f"\n{'='*50}")
print(f"LORA TRAINING COMPLETE")
print(f"  Adapter saved to: {final_path}")
print(f"  VRAM peak       : {torch.cuda.max_memory_allocated(0)/1e9:.2f} GB")
print(f">>>>>>>> Phase 4 complete.")