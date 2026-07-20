"""Canonical PyTorch references for the LFM2 gated short-conv task.

Semantics follow HuggingFace Lfm2ShortConv (transformers
models/lfm2/modeling_lfm2.py) minus in_proj/out_proj, which MPK runs as
separate linear tasks:

    Bx        = B * x
    conv_out  = causal_depthwise_conv1d(Bx, weight, kernel=L)
    y         = C * conv_out

The conv state is the rolling window of the last L-1 Bx values per request
(oldest first), zero-initialized for a request's first chunk.
"""

import torch


def lfm2_conv_ref(B, C, X, conv_weight, state=None):
    """Reference gated short conv over one contiguous token window.

    B, C, X: (T, H)   split outputs of in_proj
    conv_weight: (H, L)
    state: (H, L-1) previous Bx window, oldest first, or None (fresh request)

    Returns (y, new_state): y (T, H), new_state (H, L-1).
    """
    T, H = B.shape
    L = conv_weight.shape[1]
    Bx = (B.float() * X.float())
    if state is None:
        prev = torch.zeros(L - 1, H, dtype=torch.float32, device=B.device)
    else:
        prev = state.float().t()  # (L-1, H)
    padded = torch.cat([prev, Bx], dim=0)  # (T + L - 1, H)
    conv_out = torch.zeros(T, H, dtype=torch.float32, device=B.device)
    for k in range(L):
        conv_out += conv_weight[:, k].float() * padded[k : k + T]
    y = (C.float() * conv_out).to(torch.bfloat16)
    new_state = padded[-(L - 1):].t().to(torch.bfloat16)  # (H, L-1)
    return y, new_state


def make_grouped_bcx(B, C, X, groups):
    """Pack B/C/X (T, H) into the fused (T, 3H) tensor whose channel groups
    are laid out as [B_g | C_g | x_g] (the layout produced by interleaving the
    in_proj weight rows with shuffle_tensors, num_groups=groups)."""
    T, H = B.shape
    Hg = H // groups
    parts = torch.cat(
        [
            B.view(T, groups, Hg),
            C.view(T, groups, Hg),
            X.view(T, groups, Hg),
        ],
        dim=-1,
    )  # (T, groups, 3*Hg)
    return parts.reshape(T, 3 * H).contiguous()


def shuffle_in_proj_weight(w_in_proj, groups):
    """Reorder HF in_proj.weight (3H, H) rows so that a plain linear produces
    the grouped [B_g | C_g | x_g] layout expected by the conv task."""
    out_dim, hidden = w_in_proj.shape
    H = out_dim // 3
    Hg = H // groups
    w = w_in_proj.view(3, groups, Hg, hidden)
    w = w.permute(1, 0, 2, 3)  # (groups, 3, Hg, hidden)
    return w.reshape(3 * H, hidden).contiguous()
