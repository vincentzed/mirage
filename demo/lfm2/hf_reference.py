"""Greedy HF reference generation for LFM2.5 models (mirrors MPK's demo flow)."""
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="LiquidAI/LFM2.5-230M")
parser.add_argument("--prompt", default="Give me a short introduction to large language model.")
parser.add_argument("--max-new-tokens", type=int, default=256)
args = parser.parse_args()

tok = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to("cuda")

messages = [{"role": "user", "content": args.prompt}]
text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print("=== templated prompt ===")
print(repr(text))
ids = tok(text, return_tensors="pt").input_ids.to("cuda")
print("prompt tokens:", ids.shape[1])
out = model.generate(ids, max_new_tokens=args.max_new_tokens, do_sample=False,
                     temperature=None, top_p=None, top_k=None)
gen = out[0]
print("=== full token ids ===")
print(gen.tolist())
print("=== decoded ===")
print(tok.decode(gen, skip_special_tokens=True))
