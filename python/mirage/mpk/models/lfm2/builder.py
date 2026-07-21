"""MPK graph builders for LiquidAI LFM2 / LFM2.5 hybrid models.

LFM2 is a hybrid architecture: every layer has a mixer (either GQA attention
with per-head QK RMSNorm + RoPE, or a gated short convolution) followed by a
SwiGLU feed-forward block:

    residual = x
    h = operator_norm(x)
    h = attention(h)  |  short_conv(h)      # per config.layer_types
    x = x + h
    x = x + feed_forward(ffn_norm(x))       # SwiGLU (dense) or sparse MoE
    ...
    logits = embedding_norm(x) @ embed_tokens.T   # tied lm head

The attention layers reuse the Qwen3 path (QK-norm + RoPE are fused inside
paged_attention_layer). The short-conv mixer uses the TASK_LFM2_CONV kernel
(tasks/ampere/lfm2_conv.cuh) with a per-request conv-state cache.

LFM2.5-8B-A1B (Lfm2MoeForCausalLM) additionally replaces the dense FFN with a
sigmoid-routed MoE (DeepSeek-V3-style expert bias, top-4 of 32 experts,
norm_topk_prob) in all layers except the first ``num_dense_layers``.
"""

import torch

from ..utils import grid_for_rmsnorm_linear_layer
from ..graph_builder import GraphBuilder, MirageModelConfig
from ...persistent_kernel import PersistentKernel
from ...model_registry import register_model_builder
from ....core import bfloat16, int64, int32, float32

from typing import Optional

# grid_dim.y of the conv task: the channel dimension is split into this many
# groups (the in_proj weight rows are interleaved accordingly).
CONV_CHANNEL_GROUPS = 4


@register_model_builder("LFM2", "lfm2", "LiquidAI/LFM2.5-230M")
class Lfm2Builder(GraphBuilder):
    def __init__(self, mpk: PersistentKernel, weights: Optional[dict] = None):
        super().__init__(mpk, weights)
        self.max_num_pages = mpk.max_num_pages
        self.page_size = mpk.page_size
        self.world_size = mpk.world_size
        self.input_tokens = mpk.meta_tensors["input_tokens"]
        self.output_tokens = mpk.meta_tensors["output_tokens"]
        self.tokenizer = None
        self.model_name: str = None
        self.model_path: str = None
        self.shuffled_tensors = {}
        self.rank = mpk.mpi_rank
        self.eos_token_id = 7  # <|im_end|>, overwritten from config
        assert self.world_size == 1, "LFM2 builder currently supports single GPU"

    def _hf_model_class(self):
        from transformers.models.lfm2.modeling_lfm2 import Lfm2ForCausalLM
        return Lfm2ForCausalLM

    def build_from_config(self, model_config: MirageModelConfig):
        raise NotImplementedError(
            "LFM2 builder requires weight_from_model=True (hybrid layer types "
            "and conv state caches are derived from the HF config)")

    def build_from_model(self, model_name: str, model_path: str | None = None):
        from transformers import AutoTokenizer
        model_cls = self._hf_model_class()
        source = model_path if model_path is not None else model_name
        self.model_name = model_name
        self.model = model_cls.from_pretrained(
            source, dtype=torch.bfloat16).to("cuda")
        self.tokenizer = AutoTokenizer.from_pretrained(source)

        config = self.model.config
        self.config = config
        self.eos_token_id = config.eos_token_id
        self.norm_eps = config.norm_eps

        dummy_x = torch.empty(0, dtype=torch.bfloat16, device="cuda")
        positions = torch.arange(32768).unsqueeze(0).to(self.model.device)
        # dense models name the rotary module rotary_emb, MoE models pos_emb
        rotary = getattr(self.model.model, "rotary_emb", None)
        if rotary is None:
            rotary = self.model.model.pos_emb
        self.position_embeddings = rotary(dummy_x, positions)

        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.vocab_size = config.vocab_size
        # lm-head linear + two-stage argmax require the vocab dimension to
        # split evenly across their grids
        import math
        pad_unit = 256 * self.mpk.num_workers // math.gcd(
            256, self.mpk.num_workers)
        self.padded_vocab_size = (
            (self.vocab_size + pad_unit - 1) // pad_unit * pad_unit)
        self.num_q_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_local_q_heads = self.num_q_heads // self.world_size
        self.num_local_kv_heads = self.num_kv_heads // self.world_size
        self.head_dim = getattr(
            config, "head_dim",
            config.hidden_size // config.num_attention_heads)
        self.fused_outdim_1 = (
            self.num_local_q_heads + 2 * self.num_local_kv_heads
        ) * self.head_dim

        self.num_layers = config.num_hidden_layers
        import os as _os
        _trunc = int(_os.environ.get("LFM2_TRUNC_LAYERS", "0"))
        if _trunc > 0:
            # debug aid: build only the first K layers (compare against an
            # identically truncated HF model)
            self.num_layers = min(self.num_layers, _trunc)
        self.layer_types = list(config.layer_types)
        self.conv_l = config.conv_L_cache
        assert not getattr(config, "conv_bias", False), \
            "conv_bias is not supported"

        attn_layers = [i for i, t in enumerate(self.layer_types)
                       if t == "full_attention"]
        conv_layers = [i for i, t in enumerate(self.layer_types)
                       if t != "full_attention"]

        self.k_cache = {}
        self.v_cache = {}
        for i in attn_layers:
            for cache in (self.k_cache, self.v_cache):
                cache[i] = torch.empty(
                    (self.max_num_pages, self.page_size,
                     self.num_local_kv_heads, self.head_dim),
                    dtype=torch.bfloat16, device="cuda")
        # rolling per-request cache of the last conv_l - 1 Bx values
        self.conv_state = {
            i: torch.zeros(
                (self.mpk.max_num_batched_requests, self.hidden_size,
                 self.conv_l - 1),
                dtype=torch.bfloat16, device="cuda")
            for i in conv_layers
        }

        print(f"build_from_model: {model_name}: num_layers={self.num_layers} "
              f"({len(attn_layers)} attention / {len(conv_layers)} conv), "
              f"hidden={self.hidden_size}, intermediate={self.intermediate_size}, "
              f"vocab={self.vocab_size} (padded {self.padded_vocab_size}), "
              f"heads={self.num_q_heads}/{self.num_kv_heads}x{self.head_dim}, "
              f"conv_L={self.conv_l}, eps={self.norm_eps}")

        self.build_from_dict(self.model.state_dict(), True)

    def new_intermediate_tensors(self):
        self.max_num_batched_tokens = self.mpk.max_num_batched_tokens
        mbt = self.max_num_batched_tokens
        new_tensor = self.mpk.new_tensor
        self.y = new_tensor(
            dims=(mbt, self.hidden_size), dtype=bfloat16,
            name="embed_out", io_category="cuda_tensor")
        self.rmsnorm_out = new_tensor(
            dims=(mbt, self.hidden_size), dtype=bfloat16,
            name="rmsnorm_out", io_category="cuda_tensor")
        self.attn_in = new_tensor(
            dims=(mbt, self.fused_outdim_1), dtype=bfloat16,
            name="attn_in", io_category="cuda_tensor")
        self.attn_out = new_tensor(
            dims=(mbt, self.num_local_q_heads * self.head_dim),
            dtype=bfloat16, name="attn_out", io_category="cuda_tensor")
        self.attn_proj_out = new_tensor(
            dims=(mbt, self.hidden_size), dtype=bfloat16,
            name="attn_proj_out", io_category="cuda_tensor")
        self.conv_bcx = new_tensor(
            dims=(mbt, 3 * self.hidden_size), dtype=bfloat16,
            name="conv_bcx", io_category="cuda_tensor")
        self.conv_y = new_tensor(
            dims=(mbt, self.hidden_size), dtype=bfloat16,
            name="conv_y", io_category="cuda_tensor")
        self.mlp_mid = new_tensor(
            dims=(mbt, 2 * self.intermediate_size), dtype=bfloat16,
            name="mlp_mid", io_category="cuda_tensor")
        self.silu_mul_out = new_tensor(
            dims=(mbt, self.intermediate_size), dtype=bfloat16,
            name="silu_mul_out", io_category="cuda_tensor")
        self.mlp_out = new_tensor(
            dims=(mbt, self.hidden_size), dtype=bfloat16,
            name="mlp_out", io_category="cuda_tensor")
        self.argmax_in = new_tensor(
            dims=(mbt, self.padded_vocab_size), dtype=bfloat16,
            name="argmax_in", io_category="cuda_tensor")
        self.argmax_part_value = new_tensor(
            dims=(mbt, self.mpk.num_workers), dtype=bfloat16,
            name="argmax_part_value", io_category="cuda_tensor")
        self.argmax_part_index = new_tensor(
            dims=(mbt, self.mpk.num_workers), dtype=int64,
            name="argmax_part_index", io_category="cuda_tensor")

    def _linear_with_residual_into_x(self, input, weight, output_buffer):
        """out_proj / down_proj matmul with the residual add, following the
        Qwen3 pattern: split-K accumulates into the residual tensor in place
        on Blackwell, other targets use linear_with_residual."""
        if self.use_splitk:
            out = self.x
            self.mpk.splitk_linear_layer(
                input=input,
                weight=weight,
                output=out,
                grid_dim=(self.hidden_size // 128,
                          128 * 128 // self.hidden_size, 1),
                block_dim=(256, 1, 1),
            )
        else:
            out = output_buffer
            self.mpk.linear_with_residual_layer(
                input=input,
                weight=weight,
                residual=self.x,
                output=out,
                grid_dim=(self.hidden_size // 64, 1, 1),
                block_dim=(128, 1, 1),
            )
        self.x = out

    def _rmsnorm(self, weight):
        self.mpk.rmsnorm_layer(
            input=self.x,
            weight=weight,
            output=self.rmsnorm_out,
            grid_dim=(self.mpk.max_num_batched_tokens, 1, 1),
            block_dim=(128, 1, 1),
            eps=self.norm_eps,
        )

    def build_attention_mixer(self, i, prefix, state_dict):
        w_norm = self.mpk.attach_input(
            torch_tensor=state_dict[f"{prefix}operator_norm.weight"],
            name=f"layer_{i}_operator_norm")
        w_q = self.mpk.attach_input(
            torch_tensor=state_dict[f"{prefix}self_attn.q_proj.weight"],
            name=f"layer_{i}_q_proj")
        w_k = self.mpk.attach_input(
            torch_tensor=state_dict[f"{prefix}self_attn.k_proj.weight"],
            name=f"layer_{i}_k_proj")
        w_v = self.mpk.attach_input(
            torch_tensor=state_dict[f"{prefix}self_attn.v_proj.weight"],
            name=f"layer_{i}_v_proj")
        w_qkv = self.mpk.shuffle_tensors(
            inputs=[w_q, w_k, w_v],
            shuffled_dim=0,
            num_groups=self.num_local_kv_heads,
            name=f"layer_{i}_qkv_proj")

        self._rmsnorm(w_norm)
        self.mpk.linear_layer(
            input=self.rmsnorm_out,
            weight=w_qkv,
            output=self.attn_in,
            grid_dim=(grid_for_rmsnorm_linear_layer(w_qkv.dim(0)), 1, 1),
            block_dim=(128, 1, 1),
        )

        # NOTE: the attention kernel applies the per-head QK RMSNorm with a
        # hard-coded eps of 1e-6 while LFM2 uses 1e-5; the difference is far
        # below bf16 resolution for normalized activations.
        w_q_norm = self.mpk.attach_input(
            torch_tensor=state_dict[f"{prefix}self_attn.q_layernorm.weight"],
            name=f"layer_{i}_q_layernorm")
        w_k_norm = self.mpk.attach_input(
            torch_tensor=state_dict[f"{prefix}self_attn.k_layernorm.weight"],
            name=f"layer_{i}_k_layernorm")
        k_cache = self.mpk.attach_input(
            torch_tensor=self.k_cache[i], name=f"layer_{i}_k_cache")
        v_cache = self.mpk.attach_input(
            torch_tensor=self.v_cache[i], name=f"layer_{i}_v_cache")
        self.mpk.paged_attention_layer(
            input=self.attn_in,
            k_cache=k_cache,
            v_cache=v_cache,
            q_norm=w_q_norm,
            k_norm=w_k_norm,
            cos_pos_embed=self.cos_pos_embed,
            sin_pos_embed=self.sin_pos_embed,
            output=self.attn_out,
            grid_dim=(self.mpk.max_num_batched_requests,
                      self.num_local_kv_heads, 1),
            block_dim=(128, 1, 1),
        )

        w_o = self.mpk.attach_input(
            torch_tensor=state_dict[f"{prefix}self_attn.out_proj.weight"],
            name=f"layer_{i}_out_proj")
        self._linear_with_residual_into_x(
            self.attn_out, w_o, self.attn_proj_out)

    def build_conv_mixer(self, i, prefix, state_dict):
        w_norm = self.mpk.attach_input(
            torch_tensor=state_dict[f"{prefix}operator_norm.weight"],
            name=f"layer_{i}_operator_norm")
        # in_proj produces [B | C | x]; interleave its rows so each of the
        # CONV_CHANNEL_GROUPS channel groups is a contiguous [B_g | C_g | x_g]
        # block (what the conv task's grid.y partitioning expects)
        w_in = state_dict[f"{prefix}conv.in_proj.weight"]
        H = self.hidden_size
        w_b = self.mpk.attach_input(
            torch_tensor=w_in[0:H], name=f"layer_{i}_conv_in_proj_B")
        w_c = self.mpk.attach_input(
            torch_tensor=w_in[H:2 * H], name=f"layer_{i}_conv_in_proj_C")
        w_x = self.mpk.attach_input(
            torch_tensor=w_in[2 * H:3 * H], name=f"layer_{i}_conv_in_proj_x")
        w_bcx = self.mpk.shuffle_tensors(
            inputs=[w_b, w_c, w_x],
            shuffled_dim=0,
            num_groups=CONV_CHANNEL_GROUPS,
            name=f"layer_{i}_conv_in_proj")

        self._rmsnorm(w_norm)
        self.mpk.linear_layer(
            input=self.rmsnorm_out,
            weight=w_bcx,
            output=self.conv_bcx,
            grid_dim=(grid_for_rmsnorm_linear_layer(w_bcx.dim(0)), 1, 1),
            block_dim=(128, 1, 1),
        )

        conv_weight = state_dict[f"{prefix}conv.conv.weight"].view(
            self.hidden_size, self.conv_l)
        self.shuffled_tensors[f"layer_{i}_conv_weight"] = conv_weight
        w_conv = self.mpk.attach_input(
            torch_tensor=conv_weight, name=f"layer_{i}_conv_weight")
        conv_state = self.mpk.attach_input(
            torch_tensor=self.conv_state[i], name=f"layer_{i}_conv_state")
        self.mpk.lfm2_conv_layer(
            input=self.conv_bcx,
            conv_weight=w_conv,
            conv_state=conv_state,
            output=self.conv_y,
            grid_dim=(self.mpk.max_num_batched_requests,
                      CONV_CHANNEL_GROUPS, 1),
            block_dim=(128, 1, 1),
        )

        w_o = self.mpk.attach_input(
            torch_tensor=state_dict[f"{prefix}conv.out_proj.weight"],
            name=f"layer_{i}_conv_out_proj")
        self._linear_with_residual_into_x(self.conv_y, w_o, self.attn_proj_out)

    def build_ffn(self, i, prefix, state_dict):
        """Dense SwiGLU FFN: x = x + w2(silu(w1(h)) * w3(h))."""
        w_norm = self.mpk.attach_input(
            torch_tensor=state_dict[f"{prefix}ffn_norm.weight"],
            name=f"layer_{i}_ffn_norm")
        w_gate = self.mpk.attach_input(
            torch_tensor=state_dict[f"{prefix}feed_forward.w1.weight"],
            name=f"layer_{i}_w1")
        w_up = self.mpk.attach_input(
            torch_tensor=state_dict[f"{prefix}feed_forward.w3.weight"],
            name=f"layer_{i}_w3")
        rmsnorm_num_tasks = grid_for_rmsnorm_linear_layer(
            w_gate.dim(0) + w_up.dim(0))
        w_gatedup = self.mpk.shuffle_tensors(
            inputs=[w_gate, w_up],
            shuffled_dim=0,
            num_groups=rmsnorm_num_tasks // 2,
            name=f"layer_{i}_gatedup_proj")
        self._rmsnorm(w_norm)
        self.mpk.linear_layer(
            input=self.rmsnorm_out,
            weight=w_gatedup,
            output=self.mlp_mid,
            grid_dim=(rmsnorm_num_tasks, 1, 1),
            block_dim=(128, 1, 1),
        )
        self.mpk.silu_mul_layer(
            input=self.mlp_mid,
            output=self.silu_mul_out,
            grid_dim=(rmsnorm_num_tasks // 2, 1, 1),
            block_dim=(128, 1, 1),
        )
        w_down = self.mpk.attach_input(
            torch_tensor=state_dict[f"{prefix}feed_forward.w2.weight"],
            name=f"layer_{i}_w2")
        self._linear_with_residual_into_x(
            self.silu_mul_out, w_down, self.mlp_out)

    def build_layers(self, state_dict: dict):
        self.use_splitk = (self.mpk.target_cc == 100)
        for i in range(self.num_layers):
            prefix = f"model.layers.{i}."
            if self.layer_types[i] == "full_attention":
                self.build_attention_mixer(i, prefix, state_dict)
            else:
                self.build_conv_mixer(i, prefix, state_dict)
            self.build_ffn(i, prefix, state_dict)

    def build_from_dict(self, state_dict: dict, with_lm_head: bool):
        if with_lm_head:
            # tied embeddings: LFM2 checkpoints ship no separate lm_head
            lm_head = state_dict.get("lm_head.weight")
            if lm_head is None:
                lm_head = state_dict["model.embed_tokens.weight"]
            if self.padded_vocab_size != self.vocab_size:
                self.lm_head_weight = torch.cat(
                    (lm_head,
                     torch.zeros(
                         (self.padded_vocab_size - self.vocab_size,
                          self.hidden_size),
                         dtype=lm_head.dtype, device="cuda")),
                    0)
            else:
                self.lm_head_weight = lm_head

        self.x = self.mpk.attach_input(
            torch_tensor=self.input_tokens, name="input_token")
        self.cos_pos_embed = self.mpk.attach_input(
            torch_tensor=self.position_embeddings[0][0, :4096, :],
            name="cos_position_embedding")
        self.sin_pos_embed = self.mpk.attach_input(
            torch_tensor=self.position_embeddings[1][0, :4096, :],
            name="sin_position_embedding")

        self.new_intermediate_tensors()

        argmax_out = self.mpk.attach_input(
            torch_tensor=self.output_tokens, name="output_token")

        w_embed = self.mpk.attach_input(
            torch_tensor=state_dict["model.embed_tokens.weight"],
            name="embed_tokens")
        self.mpk.embed_layer(
            input=self.x,
            weight=w_embed,
            output=self.y,
            grid_dim=(1, 1, 1),
            block_dim=(128, 1, 1),
            input_source=1,
        )
        self.x = self.y

        self.build_layers(state_dict)

        w_norm = self.mpk.attach_input(
            torch_tensor=state_dict["model.embedding_norm.weight"],
            name="model_embedding_norm")
        self._rmsnorm(w_norm)
        if with_lm_head:
            w_proj = self.mpk.attach_input(
                torch_tensor=self.lm_head_weight, name="lm_head")
            self.mpk.linear_layer(
                input=self.rmsnorm_out,
                weight=w_proj,
                output=self.argmax_in,
                grid_dim=(grid_for_rmsnorm_linear_layer(w_proj.dim(0)), 1, 1),
                block_dim=(128, 1, 1),
            )
            self.mpk.argmax_partial_layer(
                input=self.argmax_in,
                output=(self.argmax_part_value, self.argmax_part_index),
                grid_dim=(self.mpk.num_workers, 1, 1),
                block_dim=(128, 1, 1),
            )
            self.mpk.argmax_reduce_layer(
                input=(self.argmax_part_value, self.argmax_part_index),
                output=argmax_out,
                grid_dim=(1, 1, 1),
                block_dim=(128, 1, 1),
            )

    def encode(self, text: str):
        return self.tokenizer.encode(text, add_special_tokens=True)

    def decode(self, ids: torch.Tensor):
        return self.tokenizer.decode(ids, skip_special_tokens=True)


@register_model_builder("LFM2-MoE", "lfm2_moe", "LiquidAI/LFM2.5-8B-A1B")
class Lfm2MoeBuilder(Lfm2Builder):
    def _hf_model_class(self):
        from transformers.models.lfm2_moe.modeling_lfm2_moe import (
            Lfm2MoeForCausalLM)
        return Lfm2MoeForCausalLM

    def build_from_model(self, model_name: str, model_path: str | None = None):
        super().build_from_model(model_name, model_path)

    def build_from_dict(self, state_dict: dict, with_lm_head: bool):
        config = self.config
        self.num_dense_layers = config.num_dense_layers
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.moe_intermediate_size = config.moe_intermediate_size
        assert config.use_expert_bias, "expected expert-bias sigmoid routing"
        assert config.norm_topk_prob, (
            "the sigmoid routing kernel always normalizes top-k weights")
        self.routed_scaling_factor = float(config.routed_scaling_factor)
        # topk_sigmoid_task_impl covers 8 rows per warp at 32 experts, so a
        # single-task routing launch handles at most 64 batched tokens
        mbt = self.mpk.max_num_batched_tokens
        assert mbt <= 64, (
            "LFM2.5 MoE routing currently requires max_num_batched_tokens<=64")
        # the bf16 MoE group-GEMM tiles tokens in blocks of MMA_N=16 with
        # floor division, dropping a partial tail tile
        assert mbt % 16 == 0 or (mbt <= 16 and 16 % mbt == 0), (
            "max_num_batched_tokens must be a multiple of 16 (or divide 16)")
        super().build_from_dict(state_dict, with_lm_head)

    def new_intermediate_tensors(self):
        super().new_intermediate_tensors()
        mbt = self.mpk.max_num_batched_tokens
        new_tensor = self.mpk.new_tensor
        self.router_logits = new_tensor(
            dims=(mbt, self.num_experts), dtype=bfloat16,
            name="router_logits", io_category="cuda_tensor")
        self.moe_topk_weights = new_tensor(
            dims=(mbt, self.num_experts_per_tok), dtype=float32,
            name="moe_topk_weights", io_category="cuda_tensor")
        self.moe_routing_indices = new_tensor(
            dims=(self.num_experts, mbt), dtype=int32,
            name="moe_routing_indices", io_category="cuda_tensor")
        self.moe_mask = new_tensor(
            dims=(self.num_experts + 1,), dtype=int32,
            name="moe_mask", io_category="cuda_tensor")
        self.moe_mid = new_tensor(
            dims=(mbt, self.num_experts_per_tok,
                  2 * self.moe_intermediate_size),
            dtype=bfloat16, name="moe_mid", io_category="cuda_tensor")
        self.moe_silu_out = new_tensor(
            dims=(mbt, self.num_experts_per_tok, self.moe_intermediate_size),
            dtype=bfloat16, name="moe_silu_out", io_category="cuda_tensor")
        self.moe_down_out = new_tensor(
            dims=(mbt, self.num_experts_per_tok, self.hidden_size),
            dtype=bfloat16, name="moe_down_out", io_category="cuda_tensor")
        # two combine buffers so consecutive MoE layers never read and write
        # the same tensor (residual comes from the previous combine output)
        self.moe_out = [
            new_tensor(dims=(mbt, self.hidden_size), dtype=bfloat16,
                       name=f"moe_out_{j}", io_category="cuda_tensor")
            for j in range(2)
        ]

    def _stacked_expert_weights(self, prefix, state_dict):
        """Return (w13, w2) stacked expert tensors:
        w13 (E, 2*moe_inter, hidden) rows [w1(gate); w3(up)] per expert,
        w2  (E, hidden, moe_inter)."""
        gate_up_key = f"{prefix}feed_forward.experts.gate_up_proj"
        down_key = f"{prefix}feed_forward.experts.down_proj"
        if gate_up_key in state_dict:
            return (state_dict[gate_up_key].contiguous(),
                    state_dict[down_key].contiguous())
        w13 = torch.stack([
            torch.cat(
                (state_dict[f"{prefix}feed_forward.experts.{e}.w1.weight"],
                 state_dict[f"{prefix}feed_forward.experts.{e}.w3.weight"]),
                dim=0)
            for e in range(self.num_experts)
        ]).contiguous()
        w2 = torch.stack([
            state_dict[f"{prefix}feed_forward.experts.{e}.w2.weight"]
            for e in range(self.num_experts)
        ]).contiguous()
        return w13, w2

    def build_ffn(self, i, prefix, state_dict):
        if i < self.num_dense_layers:
            super().build_ffn(i, prefix, state_dict)
            return

        w_norm = self.mpk.attach_input(
            torch_tensor=state_dict[f"{prefix}ffn_norm.weight"],
            name=f"layer_{i}_ffn_norm")
        self._rmsnorm(w_norm)

        # router: sigmoid scores + expert bias for selection (bias excluded
        # from the returned weights), top-4, normalized; no expert groups
        w_router = self.mpk.attach_input(
            torch_tensor=state_dict[f"{prefix}feed_forward.gate.weight"],
            name=f"layer_{i}_router_gate")
        w_bias = self.mpk.attach_input(
            torch_tensor=state_dict[f"{prefix}feed_forward.expert_bias"],
            name=f"layer_{i}_expert_bias")
        router_grid = max(1, self.num_experts // 8)
        self.mpk.linear_layer(
            input=self.rmsnorm_out,
            weight=w_router,
            output=self.router_logits,
            grid_dim=(router_grid, 1, 1),
            block_dim=(128, 1, 1),
        )
        self.mpk.moe_topk_sigmoid_routing_layer(
            input=self.router_logits,
            bias=w_bias,
            output=(self.moe_topk_weights, self.moe_routing_indices,
                    self.moe_mask),
            grid_dim=(1, 1, 1),
            block_dim=(256, 1, 1),
            num_groups=1,
            topk_group=1,
            routed_scaling_factor=self.routed_scaling_factor,
        )

        w13, w2 = self._stacked_expert_weights(prefix, state_dict)
        self.shuffled_tensors[f"layer_{i}_experts_w13"] = w13
        self.shuffled_tensors[f"layer_{i}_experts_w2"] = w2
        w13_dt = self.mpk.attach_input(
            torch_tensor=w13, name=f"layer_{i}_experts_w13")
        w2_dt = self.mpk.attach_input(
            torch_tensor=w2, name=f"layer_{i}_experts_w2")

        self.mpk.moe_w13_linear_layer(
            input=self.rmsnorm_out,
            weight=w13_dt,
            moe_routing_indices=self.moe_routing_indices,
            moe_mask=self.moe_mask,
            output=self.moe_mid,
            grid_dim=(self.num_experts, 1, 1),
            block_dim=(128, 1, 1),
        )
        self.mpk.moe_silu_mul_layer(
            input=self.moe_mid,
            output=self.moe_silu_out,
            grid_dim=(self.mpk.max_num_batched_tokens,
                      self.num_experts_per_tok, 1),
            block_dim=(128, 1, 1),
        )
        self.mpk.moe_w2_linear_layer(
            input=self.moe_silu_out,
            weight=w2_dt,
            moe_routing_indices=self.moe_routing_indices,
            moe_mask=self.moe_mask,
            output=self.moe_down_out,
            grid_dim=(self.num_experts, 1, 1),
            block_dim=(128, 1, 1),
        )
        # combine: x = residual(x) + sum_k weight_k * expert_out_k
        # (LFM2.5 has no shared expert, so the layer residual rides in the
        # kernel's residual slot and no separate elementwise add is needed)
        moe_out = self.moe_out[i % 2]
        self.mpk.moe_mul_sum_add_layer(
            input=self.moe_down_out,
            weight=self.moe_topk_weights,
            residual=self.x,
            output=moe_out,
            grid_dim=(self.mpk.max_num_batched_tokens, 1, 1),
            block_dim=(128, 1, 1),
        )
        self.x = moe_out
