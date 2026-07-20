# LFM2.5 on MPK (Mirage Persistent Kernel)

End-to-end megakernel inference for LiquidAI's LFM2.5 hybrid models:

| Model | Architecture | Layers | Mixers | FFN |
|---|---|---|---|---|
| [LiquidAI/LFM2.5-230M](https://huggingface.co/LiquidAI/LFM2.5-230M) | `Lfm2ForCausalLM` | 14 | 8 short-conv + 6 GQA attention | dense SwiGLU (2560) |
| [LiquidAI/LFM2.5-8B-A1B](https://huggingface.co/LiquidAI/LFM2.5-8B-A1B) | `Lfm2MoeForCausalLM` | 24 | 18 short-conv + 6 GQA attention | 2 dense + 22 MoE (top-4 of 32 experts, sigmoid + expert-bias routing) |

## Usage

```bash
python demo.py --model LiquidAI/LFM2.5-230M
python demo.py --model LiquidAI/LFM2.5-8B-A1B
python hf_reference.py --model LiquidAI/LFM2.5-230M   # HF greedy golden reference
python hf_bench.py --model LiquidAI/LFM2.5-230M       # HF latency baseline
```

## What is new in this port

1. **`TASK_LFM2_CONV`** (`include/mirage/persistent_kernel/tasks/ampere/lfm2_conv.cuh`)
   — LFM2's gated short convolution (`y = C ⊙ conv1d_causal(B ⊙ x)`, depthwise,
   L=3) as a single task with a per-request rolling conv-state cache
   (`(max_requests, hidden, L-1)`). One kernel handles prefill chunks, single-token
   decode, and ragged multi-request batches via `qo_indptr` + `step`; a request's
   first chunk (`step == 0`) starts from zero state, exactly matching HF's causal
   left-padding. `grid.x` indexes requests, `grid.y` partitions channels over the
   `shuffle_tensors`-interleaved `[B|C|x]` in_proj layout.
2. **Model builders** (`python/mirage/mpk/models/lfm2/builder.py`) — hybrid
   layer wiring per `config.layer_types`; attention layers reuse the Qwen3 paged
   attention path (per-head QK-RMSNorm + RoPE fused in-kernel); the MoE builder
   reuses the DeepSeek-V3 sigmoid routing kernel with
   `num_groups=1, topk_group=1, routed_scaling_factor=1.0` and the bf16 expert
   group-GEMM path; tied embeddings serve as the lm head.
3. **Blackwell-family support** — compute capability 10.x now maps to the SM100
   task set, compiled as `sm_100f` when the device is not sm_100 (e.g. B300
   sm_103, which cannot load `sm_100a` cubins).
4. **Fixes along the way** — `topk_sigmoid_sm100` warp mask generalized to any
   `THREADS_PER_ROW` (required for 32-expert layouts); bf16 MoE group-GEMM
   `expert_stride` now follows `grid_dim.x` like the FP8 variant instead of a
   hard-coded 10/8; `rmsnorm_layer` takes an optional `eps` (LFM2 uses 1e-5);
   offline-mode `MPK` wrapper no longer crashes on absent online-only pinned
   buffers.

## Verification (B300, sm_103)

- Conv kernel: exact parity (0 max-abs-err) vs a chained bf16-state PyTorch
  reference across prefill / decode / chunked-prefill / ragged batches
  (`tests/runtime_python/blackwell/sm100_lfm2_conv/`), plus a test_mode
  pipeline test (`tests/runtime_python/test_mode/test_lfm2_conv_testmode.py`).
- 230M end-to-end: greedy outputs match HF exactly on short prompts and are
  prefix-exact (~35 tokens) before benign bf16 tie divergence on longer ones.
- 8B-A1B end-to-end: coherent `<think>`-style reasoning outputs.

## Known limitations

- Single GPU only (no TP sharding for LFM2 yet).
- MoE routing kernel covers ≤64 batched tokens at 32 experts
  (`max_num_batched_tokens <= 64`).
- The bf16 MoE group-GEMM (`moe_linear_sm100`) is not perf-tuned upstream
  (DeepSeek uses the FP8 variants); 8B-A1B decode latency is bound by it.
- Attention QK-norm eps is hard-coded 1e-6 in-kernel (LFM2 uses 1e-5); the
  difference is far below bf16 resolution.
