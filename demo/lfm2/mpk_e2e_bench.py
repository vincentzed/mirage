"""Production-style E2E latency bench: exactly 32 prompt tokens in, 32
generated tokens out, warmed up, timed over repeated full request cycles.

Reports two timings per iteration:
  infer  — the megakernel call only (persistent kernel launch -> completion)
  cycle  — load_new_request + init_per_request + megakernel (full offline
           request cycle; online serving would not re-init per request)
"""

import argparse

import torch

from mirage.mpk.mpk import MPK, MPKMetadata, MirageModelConfig

PROMPT_TEXT = (
    "The history of computing spans mechanical calculators, vacuum tubes, "
    "transistors, integrated circuits, and modern accelerators, each era "
    "transforming what machines could do for science and society in new ways."
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="LiquidAI/LFM2.5-230M")
    parser.add_argument("--prompt-tokens", type=int, default=32)
    parser.add_argument("--gen-tokens", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--max-num-batched-tokens", default=64, type=int)
    args = parser.parse_args()

    P, G = args.prompt_tokens, args.gen_tokens
    max_seq = P + G
    total_num_requests = 1

    tokens = torch.zeros((1, max_seq), dtype=torch.long, device="cuda")
    prompt_lengths = torch.zeros((1,), dtype=torch.int, device="cuda")
    input_tokens = torch.zeros((args.max_num_batched_tokens, 1),
                               dtype=torch.long, device="cuda")
    output_tokens = torch.zeros((args.max_num_batched_tokens, 1),
                                dtype=torch.long, device="cuda")
    step = torch.zeros((1,), dtype=torch.int32, device="cuda")
    num_new_tokens = torch.ones((1,), dtype=torch.int32, device="cuda")
    qo_indptr_buffer = torch.zeros(2, dtype=torch.int32, device="cuda")
    paged_kv_indptr_buffer = torch.zeros(2, dtype=torch.int32, device="cuda")
    paged_kv_indices_buffer = torch.zeros(16, dtype=torch.int32, device="cuda")
    paged_kv_last_page_len_buffer = torch.zeros(1, dtype=torch.int32,
                                                device="cuda")

    mpk_metadata = MPKMetadata(
        mode="offline",
        total_num_requests=total_num_requests,
        num_remote_schedulers=0,
        max_seq_length=max_seq,
        max_num_batched_requests=1,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_pages=16,
        page_size=4096,
        weight_from_model=True,
        model_name=args.model,
        step=step,
        tokens=tokens,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        num_new_tokens=num_new_tokens,
        prompt_lengths=prompt_lengths,
        qo_indptr_buffer=qo_indptr_buffer,
        paged_kv_indptr_buffer=paged_kv_indptr_buffer,
        paged_kv_indices_buffer=paged_kv_indices_buffer,
        paged_kv_last_page_len_buffer=paged_kv_last_page_len_buffer,
        model_config=MirageModelConfig(with_lm_head=True),
        use_cutlass_kernel=True,
    )
    mpk = MPK(mpk_metadata)
    mpk.build()
    # disable EOS so every run generates exactly G tokens
    mpk.model_builder.eos_token_id = -1
    mpk.compile()

    ids = mpk.tokenizer(PROMPT_TEXT, return_tensors="pt").input_ids[0][:P]
    assert ids.shape[0] == P, f"prompt only has {ids.shape[0]} tokens"
    ids = ids.to("cuda")

    def request_cycle():
        # offline request cycle with an exact-length token prompt
        # (init_request_func resets per-request GPU state: step counters,
        # page queues; MPK.init_per_request has a stale arg list)
        mpk.clear_buffers()
        tokens[0, :P] = ids
        prompt_lengths.fill_(P)
        mpk.init_request_func()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        mpk()
        e.record()
        torch.cuda.synchronize()
        return s.elapsed_time(e)

    for _ in range(args.warmup):
        request_cycle()

    infer_ms, cycle_ms = [], []
    for _ in range(args.iters):
        c0 = torch.cuda.Event(enable_timing=True)
        c1 = torch.cuda.Event(enable_timing=True)
        c0.record()
        infer_ms.append(request_cycle())
        c1.record()
        torch.cuda.synchronize()
        cycle_ms.append(c0.elapsed_time(c1))

    generated = step[0].item() + 1 - P
    text = mpk.decode(tokens[0, P: step[0] + 1])
    infer_ms.sort()
    cycle_ms.sort()

    def stats(v):
        return (f"mean {sum(v)/len(v):7.2f}  p50 {v[len(v)//2]:7.2f}  "
                f"min {v[0]:7.2f}  max {v[-1]:7.2f}")

    print(f"model={args.model}  prompt={P} gen={generated} "
          f"(mbt={args.max_num_batched_tokens}, warmup={args.warmup}, "
          f"iters={args.iters})")
    print(f"  infer (megakernel only) ms: {stats(infer_ms)}")
    print(f"  cycle (load+init+run)   ms: {stats(cycle_ms)}")
    print(f"  sample output: {text[:120]!r}")
