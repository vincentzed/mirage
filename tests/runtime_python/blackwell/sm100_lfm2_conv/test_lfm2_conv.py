"""Kernel-wrapper correctness tests for the LFM2 gated short-conv task."""

import torch
import runtime_kernel_lfm2_conv

from pytorch_reference import lfm2_conv_ref, make_grouped_bcx

torch.manual_seed(0)
device = "cuda"
dtype = torch.bfloat16
CONV_L = 3


def run_kernel(bcx, conv_weight, conv_state, qo_indptr, step, groups):
    num_tokens = bcx.shape[0]
    hidden = conv_weight.shape[0]
    output = torch.zeros(num_tokens, hidden, dtype=dtype, device=device)
    num_requests = qo_indptr.shape[0] - 1
    runtime_kernel_lfm2_conv.lfm2_conv(
        bcx, conv_weight, conv_state, output, qo_indptr, step,
        num_requests, groups,
    )
    return output


def check(name, out, ref, atol=2e-2):
    max_err = (out.float() - ref.float()).abs().max().item()
    status = "PASS" if max_err <= atol else "FAIL"
    print(f"[{status}] {name}: max abs err {max_err:.5f}")
    assert max_err <= atol, f"{name} exceeded tolerance"


def test_prefill_fresh(hidden=1024, groups=4, T=17):
    B = torch.randn(T, hidden, dtype=dtype, device=device)
    C = torch.randn(T, hidden, dtype=dtype, device=device)
    X = torch.randn(T, hidden, dtype=dtype, device=device)
    w = torch.randn(hidden, CONV_L, dtype=dtype, device=device) * 0.5
    bcx = make_grouped_bcx(B, C, X, groups)
    state = torch.full((1, hidden, CONV_L - 1), float("nan"), dtype=dtype, device=device)
    qo_indptr = torch.tensor([0, T], dtype=torch.int32, device=device)
    step = torch.zeros(1, dtype=torch.int32, device=device)

    out = run_kernel(bcx, w, state, qo_indptr, step, groups)
    ref_y, ref_state = lfm2_conv_ref(B, C, X, w)
    check(f"prefill_fresh h{hidden} T{T}", out, ref_y)
    check(f"prefill_fresh state h{hidden}", state[0], ref_state)


def test_decode_continuation(hidden=1024, groups=4):
    # Prefill T tokens, then decode 3 tokens one at a time; compare against a
    # single reference pass over the full sequence.
    T = 9
    total = T + 3
    B = torch.randn(total, hidden, dtype=dtype, device=device)
    C = torch.randn(total, hidden, dtype=dtype, device=device)
    X = torch.randn(total, hidden, dtype=dtype, device=device)
    w = torch.randn(hidden, CONV_L, dtype=dtype, device=device) * 0.5

    # Chain the reference state between calls so the reference sees the same
    # bf16 state quantization at chunk boundaries as the kernel (this matches
    # HF, whose conv_states cache is also bf16).
    ref_y_prefill, ref_state = lfm2_conv_ref(B[:T], C[:T], X[:T], w)

    state = torch.zeros(1, hidden, CONV_L - 1, dtype=dtype, device=device)
    qo_indptr = torch.tensor([0, T], dtype=torch.int32, device=device)
    step = torch.zeros(1, dtype=torch.int32, device=device)
    bcx = make_grouped_bcx(B[:T], C[:T], X[:T], groups)
    out_prefill = run_kernel(bcx, w, state, qo_indptr, step, groups)
    check(f"decode_cont prefill h{hidden}", out_prefill, ref_y_prefill)

    for i in range(3):
        t = T + i
        step[0] = t
        qo_indptr_d = torch.tensor([0, 1], dtype=torch.int32, device=device)
        bcx_d = make_grouped_bcx(B[t:t+1], C[t:t+1], X[t:t+1], groups)
        out_d = run_kernel(bcx_d, w, state, qo_indptr_d, step, groups)
        ref_y_d, ref_state = lfm2_conv_ref(
            B[t:t+1], C[t:t+1], X[t:t+1], w, state=ref_state)
        check(f"decode_cont token {t}", out_d, ref_y_d)
        check(f"decode_cont state {t}", state[0], ref_state)


def test_chunked_prefill(hidden=1024, groups=4):
    # Prefill 29 tokens in chunks of 16 + 13; must match one-shot reference.
    total, chunk = 29, 16
    B = torch.randn(total, hidden, dtype=dtype, device=device)
    C = torch.randn(total, hidden, dtype=dtype, device=device)
    X = torch.randn(total, hidden, dtype=dtype, device=device)
    w = torch.randn(hidden, CONV_L, dtype=dtype, device=device) * 0.5

    state = torch.zeros(1, hidden, CONV_L - 1, dtype=dtype, device=device)
    step = torch.zeros(1, dtype=torch.int32, device=device)
    ref_state = None
    outs, refs = [], []
    for start in range(0, total, chunk):
        end = min(start + chunk, total)
        step[0] = start
        qo_indptr = torch.tensor([0, end - start], dtype=torch.int32, device=device)
        bcx = make_grouped_bcx(B[start:end], C[start:end], X[start:end], groups)
        outs.append(run_kernel(bcx, w, state, qo_indptr, step, groups))
        ref_y, ref_state = lfm2_conv_ref(
            B[start:end], C[start:end], X[start:end], w, state=ref_state)
        refs.append(ref_y)
    check(f"chunked_prefill h{hidden}", torch.cat(outs), torch.cat(refs))
    check("chunked_prefill state", state[0], ref_state)

    # Sanity: chunked kernel output vs one-shot fp32 reference should agree
    # within bf16 state-quantization noise.
    ref_y_full, _ = lfm2_conv_ref(B, C, X, w)
    check("chunked_vs_oneshot", torch.cat(outs), ref_y_full, atol=1e-1)


def test_ragged_batch(hidden=1024, groups=4):
    # Three requests in one batch: T = [5, 1, 9]; request 1 is a decode
    # continuation (step > 0), requests 0/2 are fresh prefills.
    lens = [5, 1, 9]
    max_requests = 3
    w = torch.randn(hidden, CONV_L, dtype=dtype, device=device) * 0.5

    Bs, Cs, Xs = [], [], []
    for T in lens:
        Bs.append(torch.randn(T, hidden, dtype=dtype, device=device))
        Cs.append(torch.randn(T, hidden, dtype=dtype, device=device))
        Xs.append(torch.randn(T, hidden, dtype=dtype, device=device))

    # Request 1 continues from a random pre-existing state.
    prev_state_r1 = torch.randn(hidden, CONV_L - 1, dtype=dtype, device=device)

    state = torch.zeros(max_requests, hidden, CONV_L - 1, dtype=dtype, device=device)
    state[1] = prev_state_r1
    qo_indptr = torch.tensor([0, 5, 6, 15], dtype=torch.int32, device=device)
    step = torch.tensor([0, 42, 0], dtype=torch.int32, device=device)

    bcx = torch.cat(
        [make_grouped_bcx(Bs[i], Cs[i], Xs[i], groups) for i in range(3)]
    )
    out = run_kernel(bcx, w, state, qo_indptr, step, groups)

    offset = 0
    for i, T in enumerate(lens):
        init = prev_state_r1 if i == 1 else None
        ref_y, ref_state = lfm2_conv_ref(Bs[i], Cs[i], Xs[i], w, state=init)
        check(f"ragged req{i}", out[offset:offset+T], ref_y)
        check(f"ragged req{i} state", state[i], ref_state)
        offset += T


if __name__ == "__main__":
    for hidden, groups in [(1024, 4), (2048, 4), (2048, 8), (256, 2), (1024, 1)]:
        test_prefill_fresh(hidden=hidden, groups=groups)
    test_decode_continuation()
    test_chunked_prefill()
    test_ragged_batch()
    print("All LFM2 conv kernel tests passed.")
