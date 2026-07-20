import torch, sys
from transformers import AutoModelForCausalLM, AutoTokenizer
model_name = sys.argv[1] if len(sys.argv) > 1 else "LiquidAI/LFM2.5-8B-A1B"
tok = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16).to("cuda")
text = tok.apply_chat_template([{"role":"user","content":"Give me a short introduction to large language model."}], tokenize=False, add_generation_prompt=True)
ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to("cuda")
with torch.no_grad():
    logits = model(ids).logits[0, -1].float()
probs = logits.softmax(-1)
top = probs.topk(5)
for p, i in zip(top.values.tolist(), top.indices.tolist()):
    print(f"  {i:7d} {p:8.4f} {tok.decode([i])!r}")
print("top1-top2 logit gap:", (logits.topk(2).values[0]-logits.topk(2).values[1]).item())
