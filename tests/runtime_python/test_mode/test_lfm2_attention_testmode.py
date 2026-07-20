"""Test paged_attention (SM100) at LFM2.5 head configs against a PyTorch
reference (QK-RMSNorm + NeoX RoPE + causal GQA SDPA).

LFM2.5-8B-A1B uses 32 Q / 8 KV heads with head_dim 64 (4:1 at 64), a shape no
upstream model exercises (Qwen3 is 4:1 at head_dim 128). LFM2.5-230M uses
2:1 at 64."""

import os
import sys

import torch

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel


def rope_tables(seq_len, head_dim, theta, device):
    inv_freq = 1.0 / (theta ** (
        torch.arange(0, head_dim // 2, dtype=torch.float32, device=device)
        / (head_dim // 2)))
    t = torch.arange(seq_len, dtype=torch.float32, device=device)
    freqs = torch.outer(t, inv_freq)  # (T, hd/2)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(torch.bfloat16), emb.sin().to(torch.bfloat16)


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def attention_ref(qkv, num_q_heads, num_kv_heads, head_dim,
                  q_norm_w, k_norm_w, cos, sin, eps=1e-6):
    T = qkv.shape[0]
    group = num_q_heads // num_kv_heads
    # fused layout: per kv group g: [q_g (group*hd) | k_g (hd) | v_g (hd)]
    per_group = (group + 2) * head_dim
    q = torch.empty(T, num_q_heads, head_dim, device=qkv.device,
                    dtype=torch.float32)
    k = torch.empty(T, num_kv_heads, head_dim, device=qkv.device,
                    dtype=torch.float32)
    v = torch.empty_like(k)
    for g in range(num_kv_heads):
        base = g * per_group
        q[:, g * group:(g + 1) * group] = qkv[
            :, base:base + group * head_dim].float().view(T, group, head_dim)
        k[:, g] = qkv[:, base + group * head_dim:
                      base + (group + 1) * head_dim].float()
        v[:, g] = qkv[:, base + (group + 1) * head_dim:
                      base + (group + 2) * head_dim].float()

    def rmsnorm(x, w):
        var = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(var + eps) * w.float()

    q = rmsnorm(q, q_norm_w)
    k = rmsnorm(k, k_norm_w)
    cos_t = cos[:T].float().unsqueeze(1)
    sin_t = sin[:T].float().unsqueeze(1)
    q = q * cos_t + rotate_half(q) * sin_t
    k = k * cos_t + rotate_half(k) * sin_t

    k_rep = k.repeat_interleave(group, dim=1)
    v_rep = v.repeat_interleave(group, dim=1)
    out = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(0, 1), k_rep.transpose(0, 1), v_rep.transpose(0, 1),
        is_causal=True)
    return out.transpose(0, 1).reshape(T, num_q_heads * head_dim).to(
        torch.bfloat16)


def run_case(num_q_heads, num_kv_heads, head_dim=64, T=19, theta=5e6):
    device = "cuda"
    dtype = torch.bfloat16
    page_size = 4096
    fused = (num_q_heads + 2 * num_kv_heads) * head_dim

    torch.manual_seed(0)
    qkv = torch.randn(T, fused, dtype=dtype, device=device)
    q_norm_w = torch.randn(head_dim, dtype=dtype, device=device).abs() + 0.5
    k_norm_w = torch.randn(head_dim, dtype=dtype, device=device).abs() + 0.5
    cos, sin = rope_tables(4096, head_dim, theta, device)
    k_cache = torch.zeros(1, page_size, num_kv_heads, head_dim,
                          dtype=dtype, device=device)
    v_cache = torch.zeros_like(k_cache)
    out = torch.zeros(T, num_q_heads * head_dim, dtype=dtype, device=device)

    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params["test_mode"] = True
    params["num_workers"] = num_workers
    params["num_local_schedulers"] = num_schedulers
    params["mpi_rank"] = 0
    params["world_size"] = 1
    params["max_num_batched_tokens"] = T
    params["max_seq_length"] = 4096
    params["max_num_pages"] = 1
    params["page_size"] = page_size
    params["meta_tensors"] = {
        "tokens": torch.zeros(1, 4096, dtype=torch.int64, device=device),
        "step": torch.zeros(1, dtype=torch.int32, device=device),
        "qo_indptr_buffer": torch.tensor([0, T], dtype=torch.int32,
                                         device=device),
        "paged_kv_indptr_buffer": torch.tensor([0, 1], dtype=torch.int32,
                                               device=device),
        "paged_kv_indices_buffer": torch.zeros(1, dtype=torch.int32,
                                               device=device),
        "paged_kv_last_page_len_buffer": torch.tensor([T], dtype=torch.int32,
                                                      device=device),
    }
    pk = PersistentKernel(**params)

    qkv_dt = pk.attach_input(qkv, name="qkv")
    k_dt = pk.attach_input(k_cache, name="k_cache")
    v_dt = pk.attach_input(v_cache, name="v_cache")
    qn_dt = pk.attach_input(q_norm_w, name="q_norm")
    kn_dt = pk.attach_input(k_norm_w, name="k_norm")
    cos_dt = pk.attach_input(cos, name="cos")
    sin_dt = pk.attach_input(sin, name="sin")
    out_dt = pk.attach_input(out, name="out")

    pk.paged_attention_layer(
        input=qkv_dt, k_cache=k_dt, v_cache=v_dt,
        q_norm=qn_dt, k_norm=kn_dt,
        cos_pos_embed=cos_dt, sin_pos_embed=sin_dt,
        output=out_dt,
        grid_dim=(1, num_kv_heads, 1),
        block_dim=(128, 1, 1),
    )

    pk.compile(output_dir=os.path.dirname(__file__))
    pk()
    torch.cuda.synchronize()

    ref = attention_ref(qkv, num_q_heads, num_kv_heads, head_dim,
                        q_norm_w, k_norm_w, cos, sin)
    diff = (out.float() - ref.float()).abs()
    print(f"GQA {num_q_heads}:{num_kv_heads} hd{head_dim} T{T}: "
          f"max diff {diff.max().item():.5f} "
          f"(worst row {diff.max(dim=1).values.argmax().item()})")
    return diff.max().item()


if __name__ == "__main__":
    cases = [(16, 8), (32, 8)]
    if len(sys.argv) > 1:
        cases = [tuple(map(int, sys.argv[1].split(":")))]
    ok = True
    for q, kv in cases:
        d = run_case(q, kv)
        ok &= d < 0.05
    print("PASSED" if ok else "FAILED")
