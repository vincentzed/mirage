"""Verify the megakernel claim + measurement methodology.

1. Launch-count proof: profile one warmed 32-in/32-out request and count
   every CUDA kernel launch (expect: prepare + worker + scheduler = O(3),
   independent of tokens generated). Contrast with HF transformers.
2. Timing cross-check: CUDA-event elapsed vs host wall clock around an
   explicit torch.cuda.synchronize(); they must agree.
"""

import argparse
import time

import torch
from torch.profiler import profile, ProfilerActivity

from mpk_e2e_bench import PROMPT_TEXT

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="LiquidAI/LFM2.5-230M")
parser.add_argument("--gen-tokens", type=int, default=32)
args = parser.parse_args()

P, G = 32, args.gen_tokens
from mirage.mpk.mpk import MPK, MPKMetadata, MirageModelConfig  # noqa: E402

max_seq = P + G
tokens = torch.zeros((1, max_seq), dtype=torch.long, device="cuda")
prompt_lengths = torch.zeros((1,), dtype=torch.int, device="cuda")
input_tokens = torch.zeros((64, 1), dtype=torch.long, device="cuda")
output_tokens = torch.zeros((64, 1), dtype=torch.long, device="cuda")
step = torch.zeros((1,), dtype=torch.int32, device="cuda")
num_new_tokens = torch.ones((1,), dtype=torch.int32, device="cuda")

mpk = MPK(MPKMetadata(
    mode="offline", total_num_requests=1, num_remote_schedulers=0,
    max_seq_length=max_seq, max_num_batched_requests=1,
    max_num_batched_tokens=64, max_num_pages=16, page_size=4096,
    weight_from_model=True, model_name=args.model,
    step=step, tokens=tokens, input_tokens=input_tokens,
    output_tokens=output_tokens, num_new_tokens=num_new_tokens,
    prompt_lengths=prompt_lengths,
    qo_indptr_buffer=torch.zeros(2, dtype=torch.int32, device="cuda"),
    paged_kv_indptr_buffer=torch.zeros(2, dtype=torch.int32, device="cuda"),
    paged_kv_indices_buffer=torch.zeros(16, dtype=torch.int32, device="cuda"),
    paged_kv_last_page_len_buffer=torch.zeros(1, dtype=torch.int32,
                                              device="cuda"),
    model_config=MirageModelConfig(with_lm_head=True),
    use_cutlass_kernel=True,
))
mpk.build()
mpk.model_builder.eos_token_id = -1
mpk.compile()

ids = mpk.tokenizer(PROMPT_TEXT, return_tensors="pt").input_ids[0][:P].to("cuda")
assert ids.shape[0] == P


def request_cycle(timed=True):
    mpk.clear_buffers()
    tokens[0, :P] = ids
    prompt_lengths.fill_(P)
    mpk.init_request_func()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    t0 = time.perf_counter()
    s.record()
    mpk()
    e.record()
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    return s.elapsed_time(e), (t1 - t0) * 1000.0


for _ in range(3):
    request_cycle()

# --- timing cross-check ---
pairs = [request_cycle() for _ in range(5)]
print("timing cross-check (cuda-event ms vs host wall ms):")
for ev, wall in pairs:
    print(f"  event {ev:8.2f}   wall {wall:8.2f}   delta {wall - ev:6.2f}")
gen = step[0].item() + 1 - P
print(f"generated tokens: {gen} (target {G})")

# --- kernel launch count proof ---
with profile(activities=[ProfilerActivity.CUDA],
             record_shapes=False) as prof:
    request_cycle()
events = prof.key_averages()
kernels = [(e.key, e.count) for e in events
           if e.device_type == torch.autograd.DeviceType.CUDA
           and "memcpy" not in e.key.lower() and "memset" not in e.key.lower()]
total = sum(c for _, c in kernels)
print(f"\nCUDA kernel launches during one full 32-in/{G}-out request: {total}")
for name, count in sorted(kernels, key=lambda x: -x[1]):
    print(f"  {count:5d}  {name[:100]}")
