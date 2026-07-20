"""Test the full LFM2.5 MoE FFN block (router linear -> sigmoid+bias top-k
routing -> w13 group-GEMM -> silu-mul -> w2 group-GEMM -> weighted combine
with residual) through the MPK compilation pipeline, against a PyTorch
reference implementing Lfm2MoeSparseMoeBlock semantics."""

import os

import torch

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel


def moe_block_ref(x, w_router, expert_bias, w13, w2, residual, top_k=4):
    """Lfm2MoeSparseMoeBlock reference (sigmoid + bias selection, unbiased
    weights, normalized top-k, no shared expert)."""
    T, H = x.shape
    E, twoI, _ = w13.shape
    I = twoI // 2
    logits = x.float() @ w_router.float().t()  # (T, E)
    scores = torch.sigmoid(logits)
    biased = scores + expert_bias.float()
    _, sel = torch.topk(biased, k=top_k, dim=-1)  # (T, k)
    weights = torch.gather(scores, 1, sel)
    weights = weights / (weights.sum(-1, keepdim=True) + 1e-20)
    out = residual.float().clone()
    for t in range(T):
        for k in range(top_k):
            e = sel[t, k].item()
            h = x[t].float() @ w13[e].float().t()  # (2I,)
            gate, up = h[:I], h[I:]
            act = torch.nn.functional.silu(gate) * up
            down = act @ w2[e].float().t()  # (H,)
            out[t] += weights[t, k] * down
    return out.to(torch.bfloat16), sel, weights


def test_lfm2_moe_block(pad_to=None, garbage="nan"):
    """pad_to: emulate max_num_batched_tokens > real token count; rows
    [T, pad_to) of every activation get garbage content, like the demo's
    uninitialized padding rows. garbage: 'nan' | 'inf' | 'big'."""
    device = "cuda"
    dtype = torch.bfloat16
    T, H, E, I, TOPK = 19, 2048, 32, 1792, 4
    MBT = pad_to if pad_to is not None else T

    torch.manual_seed(0)
    x = torch.randn(MBT, H, dtype=dtype, device=device) * 0.5
    residual = torch.randn(MBT, H, dtype=dtype, device=device)
    if MBT > T:
        fill = dict(nan=float("nan"), inf=float("inf"), big=3e38)[garbage]
        x[T:] = fill
        residual[T:] = fill
    w_router = torch.randn(E, H, dtype=dtype, device=device) * 0.02
    expert_bias = torch.randn(E, dtype=torch.float32, device=device) * 0.1
    w13 = torch.randn(E, 2 * I, H, dtype=dtype, device=device) * 0.02
    w2 = torch.randn(E, H, I, dtype=dtype, device=device) * 0.02

    router_logits = torch.zeros(MBT, E, dtype=dtype, device=device)
    topk_weights = torch.zeros(MBT, TOPK, dtype=torch.float32, device=device)
    routing_indices = torch.zeros(E, MBT, dtype=torch.int32, device=device)
    mask = torch.zeros(E + 1, dtype=torch.int32, device=device)
    moe_mid = torch.zeros(MBT, TOPK, 2 * I, dtype=dtype, device=device)
    moe_silu = torch.zeros(MBT, TOPK, I, dtype=dtype, device=device)
    moe_down = torch.zeros(MBT, TOPK, H, dtype=dtype, device=device)
    out = torch.zeros(MBT, H, dtype=dtype, device=device)

    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params["test_mode"] = True
    params["num_workers"] = num_workers
    params["num_local_schedulers"] = num_schedulers
    params["mpi_rank"] = 0
    params["world_size"] = 1
    params["max_num_batched_tokens"] = MBT
    params["max_seq_length"] = MBT
    pk = PersistentKernel(**params)

    x_dt = pk.attach_input(x, name="x")
    res_dt = pk.attach_input(residual, name="residual")
    wr_dt = pk.attach_input(w_router, name="w_router")
    bias_dt = pk.attach_input(expert_bias, name="expert_bias")
    w13_dt = pk.attach_input(w13, name="w13")
    w2_dt = pk.attach_input(w2, name="w2")
    logits_dt = pk.attach_input(router_logits, name="router_logits")
    tw_dt = pk.attach_input(topk_weights, name="topk_weights")
    ri_dt = pk.attach_input(routing_indices, name="routing_indices")
    mask_dt = pk.attach_input(mask, name="mask")
    mid_dt = pk.attach_input(moe_mid, name="moe_mid")
    silu_dt = pk.attach_input(moe_silu, name="moe_silu")
    down_dt = pk.attach_input(moe_down, name="moe_down")
    out_dt = pk.attach_input(out, name="out")

    pk.linear_layer(input=x_dt, weight=wr_dt, output=logits_dt,
                    grid_dim=(4, 1, 1), block_dim=(128, 1, 1))
    pk.moe_topk_sigmoid_routing_layer(
        input=logits_dt, bias=bias_dt, output=(tw_dt, ri_dt, mask_dt),
        grid_dim=(1, 1, 1), block_dim=(256, 1, 1),
        num_groups=1, topk_group=1, routed_scaling_factor=1.0)
    pk.moe_w13_linear_layer(
        input=x_dt, weight=w13_dt, moe_routing_indices=ri_dt,
        moe_mask=mask_dt, output=mid_dt,
        grid_dim=(E, 1, 1), block_dim=(128, 1, 1))
    pk.moe_silu_mul_layer(input=mid_dt, output=silu_dt,
                          grid_dim=(MBT, TOPK, 1), block_dim=(128, 1, 1))
    pk.moe_w2_linear_layer(
        input=silu_dt, weight=w2_dt, moe_routing_indices=ri_dt,
        moe_mask=mask_dt, output=down_dt,
        grid_dim=(E, 1, 1), block_dim=(128, 1, 1))
    pk.moe_mul_sum_add_layer(
        input=down_dt, weight=tw_dt, residual=res_dt, output=out_dt,
        grid_dim=(MBT, 1, 1), block_dim=(128, 1, 1))

    print("Compiling MoE block test kernel...")
    pk.compile(output_dir=os.path.dirname(__file__))
    print("Running...")
    pk()
    torch.cuda.synchronize()

    ref_out, ref_sel, ref_weights = moe_block_ref(
        x[:T], w_router, expert_bias, w13, w2, residual[:T], TOPK)

    # reference biased scores, to distinguish real routing bugs from
    # legitimate bf16 near-tie flips at the top-k boundary
    logits_ref = x[:T].float() @ w_router.float().t()
    biased_ref = torch.sigmoid(logits_ref) + expert_bias.float()
    sorted_scores = biased_ref.sort(-1, descending=True).values
    tie_gap = (sorted_scores[:, TOPK - 1] - sorted_scores[:, TOPK]).cpu()

    # routing agreement: MPK writes rank k+1 at [expert, token]
    mpk_sel = [[None] * TOPK for _ in range(T)]
    ri = routing_indices.cpu()
    for e in range(E):
        for t in range(T):
            r = ri[e, t].item()
            if r > 0:
                mpk_sel[t][r - 1] = e
    hard_mismatch = 0
    soft_mismatch = 0
    for t in range(T):
        if set(mpk_sel[t]) != set(ref_sel[t].tolist()):
            if tie_gap[t].item() < 1e-2:
                soft_mismatch += 1  # near-tie, expected under bf16
            else:
                hard_mismatch += 1
    print(f"routing: {T - hard_mismatch - soft_mismatch} exact, "
          f"{soft_mismatch} near-tie flips, {hard_mismatch} hard mismatches")
    w_diff = (topk_weights[:T]
              - ref_weights.to(topk_weights.dtype)).abs().max().item()
    print(f"topk weight max diff (may include near-tie rows): {w_diff}")
    out_diff = (out[:T].float() - ref_out.float()).abs().max().item()
    print(f"output max diff: {out_diff}")
    finite = torch.isfinite(out[:T].float()).all().item()
    print(f"real rows all finite: {finite}")

    assert finite, "NaN/Inf leaked into real rows"
    assert hard_mismatch == 0, "expert selection mismatch beyond near-ties"
    assert out_diff < 0.5, "MoE block output mismatch"
    print(f"PASSED (pad_to={pad_to}, garbage={garbage if pad_to else '-'})")
    return out[:T].clone()


if __name__ == "__main__":
    test_lfm2_moe_block()
    a = test_lfm2_moe_block(pad_to=64, garbage="nan")
    b = test_lfm2_moe_block(pad_to=64, garbage="inf")
    c = test_lfm2_moe_block(pad_to=64, garbage="big")
    print("nan-vs-inf padding changed real rows:",
          (a - b).abs().max().item())
