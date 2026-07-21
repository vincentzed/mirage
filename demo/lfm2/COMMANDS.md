# LFM2.5 on MPK: all commands

Everything below was run on a B300 (sm_103) box, 8 GPUs. All of it reproduces from this branch. No need to guess any step, the exact commands are here.

## Environment

Host torch is cu128 and has no sm_103 kernels, it will just throw `no kernel image` on any op. Use the cu13 sglang container, it's the same stack that already works on this box.

```bash
docker run -d --name mirage-brayden-lfm25 --gpus '"device=4,5,6,7"' \
  --ipc=host --shm-size 32g \
  -v /home/brayden:/home/brayden \
  -e HF_HOME=/home/brayden/hf-cache \
  -w /home/brayden/mirage \
  lmsysorg/sglang:nightly-dev-cu13-20260710-cfc66e05 sleep infinity
```

GPU mapping gotcha: the container only holds host GPUs 4-7, so `CUDA_VISIBLE_DEVICES=3` inside the container = host GPU 7. Check with UUIDs if unsure:

```bash
docker exec mirage-brayden-lfm25 python3 -c "
import torch
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_properties(i).uuid)"
```

`HF_HOME` is redirected because `~/.cache/huggingface` is root-owned on this box.

## Build

Install deps as wheels first, then editable with `--no-deps`. Otherwise `--no-build-isolation` fails on a dep that wants hatchling.

```bash
docker exec mirage-brayden-lfm25 bash -c "
  pip install -q cython z3-solver ninja cmake hatchling tg4perfetto graphviz \
    accelerate fastapi uvicorn cuda-python protobuf psutil
  pip install -e . --no-build-isolation --no-deps"
```

The megakernel itself is compiled by nvcc at `mpk.compile()` time, per model/config, takes 2-4 min. Compile flags come out as `-gencode=arch=compute_100f,code=sm_100f` on this GPU automatically (sm_103 cannot load sm_100a cubins, cc 10.x is mapped to the SM100 task set with the family target).

## Run the models

```bash
docker exec mirage-brayden-lfm25 bash -c "cd demo/lfm2 && \
  CUDA_VISIBLE_DEVICES=3 python3 demo.py --model LiquidAI/LFM2.5-230M"

docker exec mirage-brayden-lfm25 bash -c "cd demo/lfm2 && \
  CUDA_VISIBLE_DEVICES=3 python3 demo.py --model LiquidAI/LFM2.5-8B-A1B"
```

Output for the 230M, greedy, matches HF token for token on short prompts:

```
The capital of France is Paris.
Prompt length 22, generate length 8, per-token latency: 0.254 ms
```

HF greedy reference for comparison (same template, same tokenization):

```bash
docker exec mirage-brayden-lfm25 bash -c "cd demo/lfm2 && \
  CUDA_VISIBLE_DEVICES=3 python3 hf_reference.py --model LiquidAI/LFM2.5-8B-A1B --no-extra-bos"
```

## Kernel unit tests

Conv kernel in isolation (exact, 0 max-abs-err on all cases):

```bash
docker exec mirage-brayden-lfm25 bash -c "
  cd tests/runtime_python/blackwell/sm100_lfm2_conv &&
  python3 setup.py build_ext --inplace &&
  CUDA_VISIBLE_DEVICES=3 python3 test_lfm2_conv.py"
```

test_mode pipeline tests (full Python API -> codegen -> runtime path):

```bash
docker exec mirage-brayden-lfm25 bash -c "cd tests/runtime_python/test_mode && \
  CUDA_VISIBLE_DEVICES=3 python3 test_lfm2_conv_testmode.py && \
  CUDA_VISIBLE_DEVICES=3 python3 test_lfm2_moe_block_testmode.py && \
  CUDA_VISIBLE_DEVICES=3 python3 test_lfm2_attention_testmode.py"
```

## Benchmarks

Warmed E2E, exactly 32 tokens in / 32 out, 3 warmup + 10 timed request cycles:

```bash
docker exec mirage-brayden-lfm25 bash -c "cd demo/lfm2 && \
  CUDA_VISIBLE_DEVICES=3 python3 mpk_e2e_bench.py --model LiquidAI/LFM2.5-230M --max-num-batched-tokens 16"
```

```
  infer (megakernel only) ms: mean   20.58  p50   20.58  min   20.54  max   20.61
  cycle (load+init+run)   ms: mean   20.65  p50   20.65  min   20.63  max   20.68
```

```bash
docker exec mirage-brayden-lfm25 bash -c "cd demo/lfm2 && \
  CUDA_VISIBLE_DEVICES=3 python3 mpk_e2e_bench.py --model LiquidAI/LFM2.5-8B-A1B --max-num-batched-tokens 16"
```

```
  infer (megakernel only) ms: mean  267.14  p50  267.31  min  266.58  max  267.90
```

HF transformers baseline, same protocol:

```bash
docker exec mirage-brayden-lfm25 bash -c "cd demo/lfm2 && \
  CUDA_VISIBLE_DEVICES=3 python3 hf_e2e_bench.py --model LiquidAI/LFM2.5-230M"
```

```
  generate ms: mean  126.75  p50  126.62  min  125.85  max  128.07
```

230M: 20.6 vs 126.8, 6.2x. 8B-A1B: 267 vs 297, 1.11x. The mbt knob matters a lot, the static task graph processes mbt rows every decode step even with 1 live token. 64 -> 16 was 30.9 -> 20.6 on the 230M and 863 -> 267 on the 8B, output byte-identical.

## Verify it's actually one megakernel

```bash
docker exec mirage-brayden-lfm25 bash -c "cd demo/lfm2 && \
  CUDA_VISIBLE_DEVICES=3 python3 verify_megakernel.py"
```

```
timing cross-check (cuda-event ms vs host wall ms):
  event    30.91   wall    30.93   delta   0.01
CUDA kernel launches during one full 32-in/32-out request: 15
      1  worker_kernel(mirage::runtime::RuntimeConfig)
      1  scheduler_kernel(mirage::runtime::RuntimeConfig)
      1  prepare_kernel(mirage::runtime::RuntimeConfig, int)
      1  init_kernel(mirage::runtime::RuntimeConfig)
```

All model compute is the single `worker_kernel` launch. The other 11 launches are the harness's `tensor.fill_` resets. HF does 20570 kernel launches for the same request. Binary check:

```bash
cuobjdump --list-elf demo/lfm2/debug_230m/mpk_launcher_rank0.cpython-312-x86_64-linux-gnu.so
# ELF file    1: test.sm_100.cubin
```

One cubin, task impls are device functions inside `worker_kernel`.

## Known problems

Problem 1: MoE model needs `--max-num-batched-tokens` <= 64 (routing kernel row coverage at 32 experts) and a multiple of 16, or <= 16 dividing 16. The bf16 group-GEMM floor-tiles tokens by MMA_N=16 and drops a partial tail tile. The builder asserts this.

Problem 2: 230M with prompt length exactly 32 at `--max-num-batched-tokens 32` hangs the persistent kernel, spins forever at 97% CPU in the sync poll. 22-token prompt at mbt=32 is fine, 16-token full chunks at mbt=16 are fine. Not triaged. Avoid prompt_len == mbt == 32.

Problem 3: 8B decode is bound by the bf16 MoE group-GEMM (`moe_linear_sm100.cuh`), which was never perf-tuned upstream (DeepSeek runs the FP8 variants). An FP8 or tuned bf16 expert path is the next real win.

Problem 4 (fixed on this branch, listed for context): the same group-GEMM used to compute garbage for any batch > 16 tokens. The activation `cute::copy` sliced the n-tile mode whole against a size-1 dummy smem mode, so every token tile silently loaded tile 0's activations. Also `expert_stride` was hardcoded 10/8 instead of `grid_dim.x`, which triple-processed experts. Both fixed, see `git log`.
