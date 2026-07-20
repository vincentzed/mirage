"""First-token probe on a layer-truncated HF model (pairs with
LFM2_TRUNC_LAYERS on the MPK side for bisection)."""
import sys

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "LiquidAI/LFM2.5-8B-A1B"
trunc = int(sys.argv[1]) if len(sys.argv) > 1 else 0

tok = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16).to("cuda")
if trunc > 0:
    model.model.layers = nn.ModuleList(list(model.model.layers)[:trunc])

text = tok.apply_chat_template(
    [{"role": "user", "content": "Give me a short introduction to large language model."}],
    tokenize=False, add_generation_prompt=True)
ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to("cuda")
with torch.no_grad():
    logits = model(ids).logits[0, -1].float()
top = logits.softmax(-1).topk(3)
print(f"trunc={trunc} first-token top3:",
      [(i, round(p, 4)) for p, i in zip(top.values.tolist(), top.indices.tolist())])
