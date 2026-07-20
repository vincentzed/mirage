"""Test the lfm2_conv layer through the full MPK compilation pipeline.

Uses test_mode's default serving state: a single request prefilling
max_num_batched_tokens tokens (step == 0, so the conv state starts fresh).
"""

import os
import sys

import torch

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "blackwell",
                    "sm100_lfm2_conv"))
from pytorch_reference import lfm2_conv_ref, make_grouped_bcx  # noqa: E402


def test_lfm2_conv_testmode():
    device = "cuda"
    dtype = torch.bfloat16
    num_tokens = 16
    hidden = 1024
    conv_l = 3
    groups = 4

    torch.manual_seed(0)
    B = torch.randn(num_tokens, hidden, dtype=dtype, device=device)
    C = torch.randn(num_tokens, hidden, dtype=dtype, device=device)
    X = torch.randn(num_tokens, hidden, dtype=dtype, device=device)
    conv_weight = torch.randn(hidden, conv_l, dtype=dtype, device=device) * 0.5

    bcx = make_grouped_bcx(B, C, X, groups)
    conv_state = torch.zeros(1, hidden, conv_l - 1, dtype=dtype, device=device)
    out = torch.zeros(num_tokens, hidden, dtype=dtype, device=device)

    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params["test_mode"] = True
    params["num_workers"] = num_workers
    params["num_local_schedulers"] = num_schedulers
    params["mpi_rank"] = 0
    params["world_size"] = 1
    params["max_num_batched_tokens"] = num_tokens
    params["max_seq_length"] = num_tokens
    pk = PersistentKernel(**params)

    bcx_dt = pk.attach_input(bcx, name="bcx")
    w_dt = pk.attach_input(conv_weight, name="conv_weight")
    state_dt = pk.attach_input(conv_state, name="conv_state")
    out_dt = pk.attach_input(out, name="out")

    pk.lfm2_conv_layer(
        input=bcx_dt,
        conv_weight=w_dt,
        conv_state=state_dt,
        output=out_dt,
        grid_dim=(1, groups, 1),
        block_dim=(256, 1, 1) if pk.target_cc >= 90 else (128, 1, 1),
    )

    print("Compiling test kernel...")
    pk.compile(output_dir=os.path.dirname(__file__))

    print("Running test kernel...")
    pk()
    torch.cuda.synchronize()

    ref_y, ref_state = lfm2_conv_ref(B, C, X, conv_weight)
    max_diff = (out.float() - ref_y.float()).abs().max().item()
    state_diff = (conv_state[0].float() - ref_state.float()).abs().max().item()
    print(f"Max output diff: {max_diff}, max state diff: {state_diff}")

    assert max_diff < 0.05, "lfm2_conv output mismatch"
    assert state_diff < 0.05, "lfm2_conv state mismatch"
    print("PASSED: lfm2_conv test_mode produces correct output")


if __name__ == "__main__":
    test_lfm2_conv_testmode()
