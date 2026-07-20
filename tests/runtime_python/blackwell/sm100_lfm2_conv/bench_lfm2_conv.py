"""Latency benchmark for the LFM2 gated short-conv task kernel."""

import torch
import runtime_kernel_lfm2_conv

from pytorch_reference import make_grouped_bcx

torch.manual_seed(0)
device = "cuda"
dtype = torch.bfloat16
CONV_L = 3

# (name, hidden, groups, tokens, requests)
CONFIGS = [
    ("decode  bs1  LFM2.5-230M", 1024, 4, 1, 1),
    ("decode  bs1  LFM2.5-8B",   2048, 4, 1, 1),
    ("prefill T64 LFM2.5-230M", 1024, 4, 64, 1),
    ("prefill T64 LFM2.5-8B",   2048, 4, 64, 1),
    ("prefill T64 LFM2.5-8B g8", 2048, 8, 64, 1),
]


def bench(name, hidden, groups, tokens, requests, iters=200):
    B = torch.randn(tokens, hidden, dtype=dtype, device=device)
    C = torch.randn(tokens, hidden, dtype=dtype, device=device)
    X = torch.randn(tokens, hidden, dtype=dtype, device=device)
    w = torch.randn(hidden, CONV_L, dtype=dtype, device=device) * 0.5
    bcx = make_grouped_bcx(B, C, X, groups)
    state = torch.zeros(requests, hidden, CONV_L - 1, dtype=dtype, device=device)
    out = torch.zeros(tokens, hidden, dtype=dtype, device=device)
    qo_indptr = torch.tensor([0, tokens], dtype=torch.int32, device=device)
    step = torch.zeros(requests, dtype=torch.int32, device=device)

    for _ in range(20):
        runtime_kernel_lfm2_conv.lfm2_conv(
            bcx, w, state, out, qo_indptr, step, requests, groups)

    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    starter.record()
    for _ in range(iters):
        runtime_kernel_lfm2_conv.lfm2_conv(
            bcx, w, state, out, qo_indptr, step, requests, groups)
    ender.record()
    torch.cuda.synchronize()
    ms = starter.elapsed_time(ender) / iters
    gb = (bcx.numel() + out.numel() + w.numel()) * 2 / 1e9
    print(f"{name}: {ms * 1000:8.2f} us  ({gb / (ms / 1e3):6.1f} GB/s eff)")


if __name__ == "__main__":
    for cfg in CONFIGS:
        bench(*cfg)
