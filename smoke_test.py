"""
Section 0 smoke test.
Confirms GPU, OpenAI API, and Phi-3 loading all work before we write any real code.
"""
import os
from dotenv import load_dotenv
load_dotenv()

##### Test 1: GPU #####
import torch
print("=" * 50)
print("TEST 1: GPU")
print(f"  CUDA available : {torch.cuda.is_available()}")
print(f"  Device         : {torch.cuda.get_device_name(0)}")
free, total = torch.cuda.mem_get_info(0)
print(f"  Free VRAM      : {free/1e9:.2f} GB")
print(f"  Total VRAM     : {total/1e9:.2f} GB")

##### Test 2: OpenAI #####
print("=" * 50)
print("TEST 2: OpenAI API")
from openai import OpenAI
api_key = os.environ.get("OPENAI_API_KEY")
assert api_key and "your_key_here" not in api_key, \
    "OPENAI_API_KEY not set in .env — add your actual key"
openai_client = OpenAI(api_key=api_key)
response = openai_client.chat.completions.create(
    model="gpt-5-mini",
    max_completion_tokens=500,
    messages=[{"role": "user", "content": "Reply with exactly one word: pong"}]
)
raw = response.choices[0].message.content
print(f"  Raw response repr: {repr(raw)}")
print(f"  Response: {raw.strip() if raw else 'EMPTY'}")
print(f"  Finish reason: {response.choices[0].finish_reason}")

##### Test 3: Phi-3 load (4-bit) #####
print("=" * 50)
print("TEST 3: Phi-3-Mini (4-bit load)")
print("  First run downloads ~2.5GB — this will take a few minutes...")
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

model_id = "microsoft/Phi-3-mini-4k-instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
phi3 = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    trust_remote_code=True,
    attn_implementation="eager",
    device_map={"": 0},
)

used = torch.cuda.memory_allocated(0) / 1e9
free_after, _ = torch.cuda.mem_get_info(0)
print(f"  VRAM used after load : {used:.2f} GB")
print(f"  VRAM still free      : {free_after/1e9:.2f} GB")

# Quick generation test
messages = [{"role": "user", "content": "What is 2 + 2? One word answer."}]
inputs = tokenizer.apply_chat_template(
    messages,
    return_tensors="pt",
    add_generation_prompt=True
).to("cuda")

with torch.no_grad():
    output = phi3.generate(
        inputs,
        max_new_tokens=10,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

reply = tokenizer.decode(output[0][inputs.shape[1]:], skip_special_tokens=True)
print(f"  Phi-3 says           : {reply.strip()}")

print("=" * 50)
print(">>>>>>>>>>  Phase 0 complete — all systems go.")