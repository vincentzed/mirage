"""Verify the fused task kernels used by the LFM2 fused-path builder:

  rmsnorm_linear             (RMSNorm + GEMM, eps parametrized)
  silu_mul_linear_with_residual  (SwiGLU [gate|up]-halves + down GEMM + residual)

at LFM2.5 shapes, against PyTorch references.
"""

import os

import torch

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel

EPS = 1e-5


def make_pk(T):
    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params.update(test_mode=True, num_workers=num_workers,
                  num_local_schedulers=num_schedulers, mpi_rank=0,
                  world_size=1, max_num_batched_tokens=T,
                  max_seq_length=T)
    return PersistentKernel(**params)


def rmsnorm_ref(x, w, eps=EPS):
    v = x.float().pow(2).mean(-1, keepdim=True)
    return x.float() * torch.rsqrt(v + eps) * w.float()


def test_rmsnorm_linear(T=16, hidden=1024, out=2048, grid=64):
    torch.manual_seed(0)
    x = torch.randn(T, hidden, dtype=torch.bfloat16, device="cuda")
    w_norm = torch.randn(hidden, dtype=torch.bfloat16, device="cuda").abs() * 0.5 + 0.5
    w_lin = torch.randn(out, hidden, dtype=torch.bfloat16, device="cuda") * 0.02
    y = torch.zeros(T, out, dtype=torch.bfloat16, device="cuda")

    pk = make_pk(T)
    x_dt = pk.attach_input(x, name="x")
    wn_dt = pk.attach_input(w_norm, name="w_norm")
    wl_dt = pk.attach_input(w_lin, name="w_lin")
    y_dt = pk.attach_input(y, name="y")
    pk.rmsnorm_linear_layer(
        input=x_dt, weight_norm=wn_dt, weight_linear=wl_dt, output=y_dt,
        grid_dim=(grid, 1, 1),
        block_dim=(256, 1, 1) if pk.target_cc >= 90 else (128, 1, 1),
        eps=EPS)
    pk.compile(output_dir=os.path.dirname(__file__))
    pk()
    torch.cuda.synchronize()

    ref = (rmsnorm_ref(x, w_norm) @ w_lin.float().t()).to(torch.bfloat16)
    d = (y.float() - ref.float()).abs().max().item()
    print(f"rmsnorm_linear T{T} {hidden}->{out} grid{grid}: max diff {d:.5f}")
    assert d < 0.06, "rmsnorm_linear mismatch"
    pk.finalize()


def test_silu_mul_linear_residual(T=16, hidden=1024, inter=2560, grid=16):
    torch.manual_seed(0)
    # input layout: [gate | up] halves (NOT the interleaved shuffle layout)
    gate = torch.randn(T, inter, dtype=torch.bfloat16, device="cuda")
    up = torch.randn(T, inter, dtype=torch.bfloat16, device="cuda")
    mid = torch.cat([gate, up], dim=1).contiguous()
    w2 = torch.randn(hidden, inter, dtype=torch.bfloat16, device="cuda") * 0.02
    residual = torch.randn(T, hidden, dtype=torch.bfloat16, device="cuda")
    y = torch.zeros(T, hidden, dtype=torch.bfloat16, device="cuda")

    pk = make_pk(T)
    mid_dt = pk.attach_input(mid, name="mid")
    w2_dt = pk.attach_input(w2, name="w2")
    res_dt = pk.attach_input(residual, name="residual")
    y_dt = pk.attach_input(y, name="y")
    pk.silu_mul_linear_with_residual_layer(
        input=mid_dt, weight=w2_dt, residual=res_dt, output=y_dt,
        grid_dim=(grid, 1, 1),
        block_dim=(256, 1, 1) if pk.target_cc >= 90 else (128, 1, 1))
    pk.compile(output_dir=os.path.dirname(__file__))
    pk()
    torch.cuda.synchronize()

    act = torch.nn.functional.silu(gate.float()) * up.float()
    ref = (act @ w2.float().t() + residual.float()).to(torch.bfloat16)
    d = (y.float() - ref.float()).abs().max().item()
    print(f"silu_mul_linear_res T{T} 2x{inter}->{hidden} grid{grid}: "
          f"max diff {d:.5f}")
    assert d < 0.06, "silu_mul_linear_with_residual mismatch"
    pk.finalize()


if __name__ == "__main__":
    # LFM2.5-230M shapes
    test_rmsnorm_linear(T=16, hidden=1024, out=2048, grid=64)    # qkv
    test_rmsnorm_linear(T=16, hidden=1024, out=3072, grid=96)    # conv in_proj
    test_rmsnorm_linear(T=16, hidden=1024, out=5120, grid=64)    # gate_up
    test_rmsnorm_linear(T=16, hidden=1024, out=65536, grid=256)  # lm_head
    test_silu_mul_linear_residual(T=16, hidden=1024, inter=2560, grid=16)
    # LFM2.5-8B shapes
    test_rmsnorm_linear(T=16, hidden=2048, out=3072, grid=96)    # qkv
    test_rmsnorm_linear(T=16, hidden=2048, out=6144, grid=96)    # conv in_proj
    test_silu_mul_linear_residual(T=16, hidden=2048, inter=7168, grid=32)
    print("PASSED: all fused kernels match references")
