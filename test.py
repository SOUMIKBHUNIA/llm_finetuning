from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

# === Config ===
BASE_MODEL_PATH = "./models"
ADAPTER_PATH = "./lora-finetuned-model"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# === Load base model & tokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH, torch_dtype=torch.float16, device_map="auto")

# === Load LoRA adapter
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.to(DEVICE).eval()

# === Inference
prompt = "Write a Python function to reverse a string."
inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )

response = tokenizer.decode(output[0], skip_special_tokens=True)
print("\n=== Generated Response ===")
print(response)
