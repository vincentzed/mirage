"""End-to-end MPK demo for LiquidAI LFM2.5 models.

Supports the dense hybrid LiquidAI/LFM2.5-230M (conv + attention layers) and
the MoE hybrid LiquidAI/LFM2.5-8B-A1B (sigmoid expert-bias routing, top-4 of
32 experts, first 2 layers dense).

Examples:
    python demo.py --model LiquidAI/LFM2.5-230M
    python demo.py --model LiquidAI/LFM2.5-8B-A1B --max-seq-length 512
"""

import argparse

import torch

from mirage.mpk.mpk import MPK, MPKMetadata, MirageModelConfig

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="LiquidAI/LFM2.5-230M",
        help="HF model name (LiquidAI/LFM2.5-230M or LiquidAI/LFM2.5-8B-A1B)")
    parser.add_argument("--prompt", type=str,
                        default="Give me a short introduction to large language model.")
    parser.add_argument("--max-num-batched-tokens", default=64, type=int)
    parser.add_argument("--max-num-batched-requests", default=1, type=int)
    parser.add_argument("--page-size", default=4096, type=int)
    parser.add_argument("--max-num-pages", default=16, type=int)
    parser.add_argument("--max-seq-length", default=512, type=int)
    parser.add_argument("--output-dir", default=None,
                        help="Directory for generated kernel + task graph")
    parser.add_argument(
        "--no-use-cutlass-kernel", action="store_false",
        dest="use_cutlass_kernel", default=True)
    args = parser.parse_args()

    print("Input arguments:", args)

    total_num_requests = args.max_num_batched_requests
    tokens = torch.full((total_num_requests, args.max_seq_length), 0,
                        dtype=torch.long, device="cuda")
    prompt_lengths = torch.full((total_num_requests,), 0,
                                dtype=torch.int, device="cuda")
    input_tokens = torch.full((args.max_num_batched_tokens, 1), 0,
                              dtype=torch.long, device="cuda")
    output_tokens = torch.full((args.max_num_batched_tokens, 1), 0,
                               dtype=torch.long, device="cuda")
    step = torch.full((total_num_requests,), 0, dtype=torch.int32,
                      device="cuda")
    num_new_tokens = torch.full((total_num_requests,), 1, dtype=torch.int32,
                                device="cuda")
    qo_indptr_buffer = torch.zeros(
        args.max_num_batched_requests + 1, dtype=torch.int32, device="cuda")
    paged_kv_indptr_buffer = torch.zeros(
        args.max_num_batched_requests + 1, dtype=torch.int32, device="cuda")
    paged_kv_indices_buffer = torch.zeros(
        args.max_num_pages, dtype=torch.int32, device="cuda")
    paged_kv_last_page_len_buffer = torch.zeros(
        args.max_num_batched_requests, dtype=torch.int32, device="cuda")

    mpk_metadata = MPKMetadata(
        mode="offline",
        total_num_requests=total_num_requests,
        num_remote_schedulers=0,
        max_seq_length=args.max_seq_length,
        max_num_batched_requests=args.max_num_batched_requests,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_pages=args.max_num_pages,
        page_size=args.page_size,
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
        use_cutlass_kernel=args.use_cutlass_kernel,
    )
    mpk = MPK(mpk_metadata)
    mpk.build()
    mpk.compile(output_dir=args.output_dir)

    # apply LFM2's own chat template (load_new_request's built-in template
    # is Qwen-specific)
    templated = mpk.tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        tokenize=False, add_generation_prompt=True)
    mpk.load_new_request(templated, use_template=False)

    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    starter.record()
    mpk()
    ender.record()
    torch.cuda.synchronize()
    run_time = starter.elapsed_time(ender)

    for r in range(total_num_requests):
        generated_ids = tokens[r, : step[r] + 1]
        print("token ids:", generated_ids.tolist())
        print(mpk.decode(generated_ids))

    generated = step.max().item() + 1 - prompt_lengths[0].item()
    print("Prompt length {}, generate length {}, per-token latency: {:.3f} ms"
          .format(prompt_lengths[0].item(), generated,
                  run_time / (step.max().item() + 1)))
