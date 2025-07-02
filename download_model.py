from transformers import AutoTokenizer, AutoModelForCausalLM

# === Config ===
BASE_MODEL = "bigcode/starcoder2-3b"  # or "mistralai/Mistral-7B-v0.1"
CACHE_DIR = "./models"

# === Download tokenizer & model ===
print(f"📥 Downloading tokenizer for {BASE_MODEL}...")
AutoTokenizer.from_pretrained(BASE_MODEL, cache_dir=CACHE_DIR, trust_remote_code=True)

print(f"📥 Downloading model weights for {BASE_MODEL}...")
AutoModelForCausalLM.from_pretrained(BASE_MODEL, cache_dir=CACHE_DIR, trust_remote_code=True)

print(f"✅ Done! Model cached under {CACHE_DIR}")
