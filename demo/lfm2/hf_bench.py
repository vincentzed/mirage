"""Timed greedy generation with HuggingFace transformers (baseline for the
MPK megakernel demo). Reports the same metric demo.py prints: wall time
divided by total processed tokens, plus decode-only ms/token."""
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="LiquidAI/LFM2.5-230M")
parser.add_argument("--prompt", default="Give me a short introduction to large language model.")
parser.add_argument("--new-tokens", type=int, default=116)
args = parser.parse_args()

tok = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForCausalLM.from_pretrained(
    args.model, dtype=torch.bfloat16).to("cuda")

text = tok.apply_chat_template(
    [{"role": "user", "content": args.prompt}],
    tokenize=False, add_generation_prompt=True)
ids = tok(text, return_tensors="pt").input_ids.to("cuda")
prompt_len = ids.shape[1]

gen_kwargs = dict(do_sample=False, temperature=None, top_p=None, top_k=None,
                  eos_token_id=None)

# warmup
for _ in range(2):
    model.generate(ids, max_new_tokens=8, min_new_tokens=8, **gen_kwargs)
torch.cuda.synchronize()

starter = torch.cuda.Event(enable_timing=True)
ender = torch.cuda.Event(enable_timing=True)
starter.record()
out = model.generate(ids, max_new_tokens=args.new_tokens,
                     min_new_tokens=args.new_tokens, **gen_kwargs)
ender.record()
torch.cuda.synchronize()
ms = starter.elapsed_time(ender)

new_tokens = out.shape[1] - prompt_len
total = out.shape[1]
print(f"model={args.model} prompt_tokens={prompt_len} new_tokens={new_tokens}")
print(f"wall={ms:.1f} ms  per-total-token={ms / total:.3f} ms  "
      f"per-generated-token={ms / new_tokens:.3f} ms")
