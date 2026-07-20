"""Replicate LFM2.5-8B-A1B layer 2 (attention mixer + MoE FFN) exactly as the
builder wires it, at T=19 tokens, and compare EVERY intermediate tensor
against a PyTorch reference to find the first diverging stage."""

import os
import sys

import torch

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel

sys.path.insert(0, os.path.dirname(__file__))
from test_lfm2_attention_testmode import (  # noqa: E402
    attention_ref, rope_tables)


def rmsnorm_ref(x, w, eps=1e-5):
    v = x.float().pow(2).mean(-1, keepdim=True)
    return (x.float() * torch.rsqrt(v + eps) * w.float()).to(torch.bfloat16)


def shuffle_rows(tensors, groups):
    chunks = [t.chunk(groups, dim=0) for t in tensors]
    out = []
    for g in range(groups):
        for c in chunks:
            out.append(c[g])
    return torch.cat(out, 0).contiguous()


def main():
    device = "cuda"
    dtype = torch.bfloat16
    T, H = 19, 2048
    NQ, NKV, HD = 32, 8, 64
    E, I, TOPK = 32, 1792, 4
    FUSED = (NQ + 2 * NKV) * HD
    eps = 1e-5

    torch.manual_seed(0)
    x_in = torch.randn(T, H, dtype=dtype, device=device)
    w_opnorm = torch.randn(H, dtype=dtype, device=device).abs() * 0.5 + 0.5
    w_q = torch.randn(NQ * HD, H, dtype=dtype, device=device) * 0.02
    w_k = torch.randn(NKV * HD, H, dtype=dtype, device=device) * 0.02
    w_v = torch.randn(NKV * HD, H, dtype=dtype, device=device) * 0.02
    w_o = torch.randn(H, NQ * HD, dtype=dtype, device=device) * 0.02
    q_norm_w = torch.randn(HD, dtype=dtype, device=device).abs() + 0.5
    k_norm_w = torch.randn(HD, dtype=dtype, device=device).abs() + 0.5
    w_ffnnorm = torch.randn(H, dtype=dtype, device=device).abs() * 0.5 + 0.5
    w_router = torch.randn(E, H, dtype=dtype, device=device) * 0.02
    bias = torch.randn(E, dtype=torch.float32, device=device) * 0.1
    w13 = torch.randn(E, 2 * I, H, dtype=dtype, device=device) * 0.02
    w2 = torch.randn(E, H, I, dtype=dtype, device=device) * 0.02
    cos, sin = rope_tables(512, HD, 5e6, device)

    # ---- torch reference ----
    normed = rmsnorm_ref(x_in, w_opnorm, eps)
    ref_qkv = normed.float() @ torch.cat(
        [w_q, w_k, w_v], 0).float().t()  # unfused order [q|k|v]
    # rebuild fused layout per kv group for the reference attention helper
    qg = ref_qkv[:, :NQ * HD].view(T, NKV, NQ // NKV * HD)
    kg = ref_qkv[:, NQ * HD:(NQ + NKV) * HD].view(T, NKV, HD)
    vg = ref_qkv[:, (NQ + NKV) * HD:].view(T, NKV, HD)
    ref_fused = torch.cat([qg, kg, vg], -1).view(T, FUSED).to(dtype)
    ref_attn = attention_ref(ref_fused, NQ, NKV, HD, q_norm_w, k_norm_w,
                             cos, sin)
    ref_x1 = (x_in.float() + ref_attn.float() @ w_o.float().t()).to(dtype)
    ref_ffn_in = rmsnorm_ref(ref_x1, w_ffnnorm, eps)
    logits = ref_ffn_in.float() @ w_router.float().t()
    scores = torch.sigmoid(logits)
    _, sel = torch.topk(scores + bias.float(), TOPK, -1)
    weights = torch.gather(scores, 1, sel)
    weights = weights / (weights.sum(-1, keepdim=True) + 1e-20)
    ref_out = ref_x1.float().clone()
    for t in range(T):
        for k in range(TOPK):
            e = sel[t, k].item()
            h = ref_ffn_in[t].float() @ w13[e].float().t()
            act = torch.nn.functional.silu(h[:I]) * h[I:]
            ref_out[t] += weights[t, k] * (act @ w2[e].float().t())
    ref_out = ref_out.to(dtype)

    # ---- MPK graph (mirrors Lfm2MoeBuilder wiring) ----
    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params.update(test_mode=True, num_workers=num_workers,
                  num_local_schedulers=num_schedulers, mpi_rank=0,
                  world_size=1, max_num_batched_tokens=T,
                  max_seq_length=512, max_num_pages=16, page_size=4096)
    params["meta_tensors"] = {
        "tokens": torch.zeros(1, 512, dtype=torch.int64, device=device),
        "step": torch.zeros(1, dtype=torch.int32, device=device),
        "qo_indptr_buffer": torch.tensor([0, T], dtype=torch.int32,
                                         device=device),
        "paged_kv_indptr_buffer": torch.tensor([0, 1], dtype=torch.int32,
                                               device=device),
        "paged_kv_indices_buffer": torch.zeros(16, dtype=torch.int32,
                                               device=device),
        "paged_kv_last_page_len_buffer": torch.tensor(
            [T], dtype=torch.int32, device=device),
    }
    pk = PersistentKernel(**params)

    x = x_in.clone()
    rmsnorm_out = torch.zeros(T, H, dtype=dtype, device=device)
    attn_in = torch.zeros(T, FUSED, dtype=dtype, device=device)
    attn_out = torch.zeros(T, NQ * HD, dtype=dtype, device=device)
    k_cache = torch.zeros(16, 4096, NKV, HD, dtype=dtype, device=device)
    v_cache = torch.zeros_like(k_cache)
    router_logits = torch.zeros(T, E, dtype=dtype, device=device)
    topk_w = torch.zeros(T, TOPK, dtype=torch.float32, device=device)
    ri = torch.zeros(E, T, dtype=torch.int32, device=device)
    mask = torch.zeros(E + 1, dtype=torch.int32, device=device)
    moe_mid = torch.zeros(T, TOPK, 2 * I, dtype=dtype, device=device)
    moe_silu = torch.zeros(T, TOPK, I, dtype=dtype, device=device)
    moe_down = torch.zeros(T, TOPK, H, dtype=dtype, device=device)
    moe_out = torch.zeros(T, H, dtype=dtype, device=device)

    a = pk.attach_input
    x_dt = a(x, name="x")
    rms_dt = a(rmsnorm_out, name="rmsnorm_out")
    wop_dt = a(w_opnorm, name="w_opnorm")
    wq_dt, wk_dt, wv_dt = a(w_q, name="w_q"), a(w_k, name="w_k"), a(w_v, name="w_v")
    wqkv_dt = pk.shuffle_tensors(inputs=[wq_dt, wk_dt, wv_dt],
                                 shuffled_dim=0, num_groups=NKV, name="w_qkv")
    ai_dt = a(attn_in, name="attn_in")
    ao_dt = a(attn_out, name="attn_out")
    kc_dt, vc_dt = a(k_cache, name="k_cache"), a(v_cache, name="v_cache")
    qn_dt, kn_dt = a(q_norm_w, name="q_norm"), a(k_norm_w, name="k_norm")
    cos_dt, sin_dt = a(cos, name="cos"), a(sin, name="sin")
    wo_dt = a(w_o, name="w_o")
    wffn_dt = a(w_ffnnorm, name="w_ffnnorm")
    wr_dt = a(w_router, name="w_router")
    bias_dt = a(bias, name="bias")
    w13_dt, w2_dt = a(w13, name="w13"), a(w2, name="w2")
    logits_dt = a(router_logits, name="router_logits")
    tw_dt, ri_dt, mask_dt = a(topk_w, name="tw"), a(ri, name="ri"), a(mask, name="mask")
    mid_dt, silu_dt, down_dt = (a(moe_mid, name="mid"), a(moe_silu, name="silu"),
                                a(moe_down, name="down"))
    mo_dt = a(moe_out, name="moe_out")

    pk.rmsnorm_layer(input=x_dt, weight=wop_dt, output=rms_dt,
                     grid_dim=(T, 1, 1), block_dim=(128, 1, 1), eps=eps)
    pk.linear_layer(input=rms_dt, weight=wqkv_dt, output=ai_dt,
                    grid_dim=(96, 1, 1), block_dim=(128, 1, 1))
    pk.paged_attention_layer(
        input=ai_dt, k_cache=kc_dt, v_cache=vc_dt, q_norm=qn_dt,
        k_norm=kn_dt, cos_pos_embed=cos_dt, sin_pos_embed=sin_dt,
        output=ao_dt, grid_dim=(1, NKV, 1), block_dim=(128, 1, 1))
    use_splitk = os.environ.get("LFM2_TEST_SPLITK", "1") == "1"
    if use_splitk:
        # splitk accumulates into x in place (residual), as on Blackwell
        pk.splitk_linear_layer(input=ao_dt, weight=wo_dt, output=x_dt,
                               grid_dim=(H // 128, 128 * 128 // H, 1),
                               block_dim=(256, 1, 1))
        x_cur = x_dt
        x_buf = x
    else:
        attn_proj = torch.zeros(T, H, dtype=dtype, device=device)
        ap_dt = a(attn_proj, name="attn_proj")
        pk.linear_with_residual_layer(
            input=ao_dt, weight=wo_dt, residual=x_dt, output=ap_dt,
            grid_dim=(H // 64, 1, 1), block_dim=(128, 1, 1))
        x_cur = ap_dt
        x_buf = attn_proj
    if os.environ.get("LFM2_TEST_DEDICATED_RMS", "0") == "1":
        ffn_rms = torch.zeros(T, H, dtype=dtype, device=device)
        ffn_rms_dt = a(ffn_rms, name="ffn_rms")
    else:
        ffn_rms = rmsnorm_out
        ffn_rms_dt = rms_dt
    pk.rmsnorm_layer(input=x_cur, weight=wffn_dt, output=ffn_rms_dt,
                     grid_dim=(T, 1, 1), block_dim=(128, 1, 1), eps=eps)
    pk.linear_layer(input=ffn_rms_dt, weight=wr_dt, output=logits_dt,
                    grid_dim=(4, 1, 1), block_dim=(128, 1, 1))
    pk.moe_topk_sigmoid_routing_layer(
        input=logits_dt, bias=bias_dt, output=(tw_dt, ri_dt, mask_dt),
        grid_dim=(1, 1, 1), block_dim=(256, 1, 1),
        num_groups=1, topk_group=1, routed_scaling_factor=1.0)
    pk.moe_w13_linear_layer(
        input=ffn_rms_dt, weight=w13_dt, moe_routing_indices=ri_dt,
        moe_mask=mask_dt, output=mid_dt, grid_dim=(E, 1, 1),
        block_dim=(128, 1, 1))
    pk.moe_silu_mul_layer(input=mid_dt, output=silu_dt,
                          grid_dim=(T, TOPK, 1), block_dim=(128, 1, 1))
    pk.moe_w2_linear_layer(
        input=silu_dt, weight=w2_dt, moe_routing_indices=ri_dt,
        moe_mask=mask_dt, output=down_dt, grid_dim=(E, 1, 1),
        block_dim=(128, 1, 1))
    pk.moe_mul_sum_add_layer(
        input=down_dt, weight=tw_dt, residual=x_cur, output=mo_dt,
        grid_dim=(T, 1, 1), block_dim=(128, 1, 1))

    print("Compiling layer-2 replica...")
    pk.compile(output_dir=os.path.dirname(__file__))
    pk()
    torch.cuda.synchronize()

    def report(name, got, ref):
        d = (got.float() - ref.float()).abs().max().item()
        print(f"  {name:14s} max diff {d:.5f}")
        return d

    print(f"Stage-by-stage vs reference (use_splitk={use_splitk}):")
    report("attn_in", attn_in, ref_fused)
    report("attn_out", attn_out, ref_attn)
    report("x_after_attn", x_buf, ref_x1)
    report("ffn_rms", ffn_rms, ref_ffn_in)
    # routing agreement
    mpk_sel = [set() for _ in range(T)]
    ri_h = ri.cpu()
    for e in range(E):
        for t in range(T):
            if ri_h[e, t].item() > 0:
                mpk_sel[t].add(e)
    sel_ok = sum(mpk_sel[t] == set(sel[t].tolist()) for t in range(T))
    print(f"  routing sets   {sel_ok}/{T} match")
    report("topk_weights", topk_w,
           weights.to(torch.float32))
    # per-slot w13 reference for the REFERENCE selection (valid to compare
    # only where selections agree)
    ref_mid = torch.zeros_like(moe_mid)
    for t in range(T):
        for k in range(TOPK):
            e = sel[t, k].item()
            ref_mid[t, k] = (
                ref_ffn_in[t].float() @ w13[e].float().t()).to(dtype)
    ok_rows = [t for t in range(T) if mpk_sel[t] == set(sel[t].tolist())]
    if ok_rows:
        idx = torch.tensor(ok_rows, device=device)
        report("moe_mid(match)", moe_mid[idx], ref_mid[idx])
    d_final = report("final", moe_out, ref_out)
    print("PASSED" if d_final < 0.1 else "FAILED")


if __name__ == "__main__":
    main()
