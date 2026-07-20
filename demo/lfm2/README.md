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
4. **Upstream fixes found by this port**
   - `moe_linear_sm100.cuh` (bf16 MoE group-GEMM): the activation
     `cute::copy` sliced the token-tile mode whole against a size-1 dummy
     destination mode, so every 16-token tile silently loaded tile 0's
     activations — corrupting all outputs beyond the first 16 batched
     tokens. Now indexed per `n_tile`. (DeepSeek uses the FP8 variants, so
     this path was upstream-untested in real models.)
   - Same file: `expert_stride` now follows `grid_dim.x` like the FP8
     variant instead of a hard-coded 10/8 (removed ~3x redundant expert
     processing; 74 → 26 ms/token on the 8B).
   - `topk_sigmoid_sm100` warp mask generalized to any `THREADS_PER_ROW`
     (required for 32-expert layouts).
   - `rmsnorm_layer` takes an optional `eps` (LFM2 uses 1e-5); offline-mode
     `MPK` wrapper no longer crashes on absent online-only pinned buffers.

## Verification (B300, sm_103)

- Conv kernel: exact parity (0 max-abs-err) vs a chained bf16-state PyTorch
  reference across prefill / decode / chunked-prefill / ragged batches
  (`tests/runtime_python/blackwell/sm100_lfm2_conv/`), plus a test_mode
  pipeline test (`tests/runtime_python/test_mode/test_lfm2_conv_testmode.py`).
- 230M end-to-end: greedy outputs match HF exactly on short prompts and are
  prefix-exact (~35 tokens) before benign bf16 tie divergence on longer ones.
- 8B-A1B end-to-end (mbt=64): first ~22 greedy tokens match HF exactly
  (`<think>` onward), then benign bf16 divergence; coherent reasoning +
  answer. test_mode suites cover the attention shapes (16:8/32:8 @ hd64,
  partial-token batches), the full MoE block (incl. NaN/Inf padding-row
  robustness), and a composed attention+MoE layer replica.

## Performance (B300, greedy, batch 1)

Warmed steady-state E2E, exactly 32 prompt tokens in / 32 generated out
(`mpk_e2e_bench.py` / `hf_e2e_bench.py`, 3 warmup + 10 timed iterations):

| Model | MPK megakernel E2E | HF transformers E2E | Speedup |
|---|---|---|---|
| LFM2.5-230M (mbt=64) | 30.9 ms (±0.05) | 126.8 ms | 4.1x |
| LFM2.5-230M (mbt=16) | **20.6 ms** (±0.04) | 126.8 ms | **6.2x** |
| LFM2.5-8B-A1B (mbt=64) | 863 ms (±3) | 296.9 ms | 0.34x |
| LFM2.5-8B-A1B (mbt=16) | **267 ms** (±0.7) | 296.9 ms | **1.11x** |

`max_num_batched_tokens` (mbt) sizes the static task graph, so every decode
step processes mbt batch rows even though only one is live — including the
full-vocab lm-head GEMM. For latency-critical decode, set mbt to the
smallest value that fits your prefill chunking appetite (output is
byte-identical; a 32-token prompt just prefills in two 16-token chunks).

The 230M runs ~0.94 ms per iteration — launch-overhead-free megakernel
execution. The 8B is bound by the bf16 MoE group-GEMM (~26 ms/decode step),
which upstream never perf-tuned (DeepSeek's production path is FP8); a tuned
or FP8 expert GEMM is the clear next step. First-run cold cost is the
one-time megakernel nvcc compile (~2-4 min per model/config).

## Known limitations

- Single GPU only (no TP sharding for LFM2 yet).
- MoE model: `max_num_batched_tokens` must be ≤64 (routing kernel row
  coverage at 32 experts) and a multiple of 16 (or ≤16 dividing 16): the
  group-GEMM floor-tiles tokens by MMA_N=16 and would drop a partial tail
  tile.
- Attention QK-norm eps is hard-coded 1e-6 in-kernel (LFM2 uses 1e-5); the
  difference is far below bf16 resolution.
- Corner: 230M at `max_num_batched_tokens=32` with a prompt of exactly 32
  tokens hangs the persistent kernel (32 vs shorter prompts and mbt=16
  full chunks are fine); untriaged upstream scheduler corner — avoid
  prompt_len == mbt == 32 configs.
