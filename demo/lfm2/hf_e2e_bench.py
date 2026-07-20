"""HF transformers baseline for the 32-in/32-out E2E latency bench, with
warmup, matching mpk_e2e_bench.py."""
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mpk_e2e_bench import PROMPT_TEXT

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="LiquidAI/LFM2.5-230M")
parser.add_argument("--prompt-tokens", type=int, default=32)
parser.add_argument("--gen-tokens", type=int, default=32)
parser.add_argument("--warmup", type=int, default=3)
parser.add_argument("--iters", type=int, default=10)
args = parser.parse_args()

P, G = args.prompt_tokens, args.gen_tokens
tok = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForCausalLM.from_pretrained(
    args.model, dtype=torch.bfloat16).to("cuda")

ids = tok(PROMPT_TEXT, return_tensors="pt").input_ids[:, :P].to("cuda")
assert ids.shape[1] == P

gen_kwargs = dict(do_sample=False, temperature=None, top_p=None, top_k=None,
                  eos_token_id=None, min_new_tokens=G, max_new_tokens=G)

def one():
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    out = model.generate(ids, **gen_kwargs)
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e), out

for _ in range(args.warmup):
    one()
times = sorted(one()[0] for _ in range(args.iters))
print(f"model={args.model}  prompt={P} gen={G} "
      f"(warmup={args.warmup}, iters={args.iters})")
print(f"  generate ms: mean {sum(times)/len(times):7.2f}  "
      f"p50 {times[len(times)//2]:7.2f}  min {times[0]:7.2f}  "
      f"max {times[-1]:7.2f}")
