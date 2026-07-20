/* Copyright 2025 CMU
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once
#include "tasks/common/common_header.cuh"

namespace kernel {

// LFM2 gated short convolution (Lfm2ShortConv without in_proj/out_proj).
//
// The task consumes the output of the fused in_proj linear, laid out per
// channel group g as [B_g | C_g | x_g] (achieved by interleaving the in_proj
// weight rows with shuffle_tensors, num_groups = grid_dim.y):
//   Bx[t, c]       = B[t, c] * x[t, c]
//   conv_out[t, c] = sum_{k=0}^{L-1} w[c, k] * Bx[t - (L-1) + k, c]
//   y[t, c]        = C[t, c] * conv_out[t, c]
// Bx values at positions before the current token window come from the
// per-request conv state cache (last L-1 Bx values), which this task also
// updates in place. A request's first prefill chunk (step == 0) starts from
// zero state, matching HuggingFace's causal left-padding.
//
// One task processes all tokens of one request (prefill: the whole in-window
// token range from qo_indptr; decode: a single token) for CHANNELS channels.
// grid_dim.x indexes requests (task_metadata.request_id), grid_dim.y
// partitions the channel dimension.
template <typename T,
          int CHANNELS,       // channels handled by this task (H / grid_dim.y)
          int CONV_L,         // conv kernel size (conv_L_cache), e.g. 3
          int BCX_STRIDE,     // row stride of the fused BCx tensor (= 3*H)
          int Y_STRIDE,       // row stride of the output tensor (= H)
          int W_STRIDE,       // row stride of the conv weight tensor (= CONV_L)
          int STATE_REQ_STRIDE> // request stride of the state tensor
                                // (= H * (CONV_L - 1))
__device__ __forceinline__ void
    lfm2_conv_task_impl(void const *bcx_ptr,
                        void const *conv_weight_ptr,
                        void *conv_state_ptr,
                        void *output_ptr,
                        int const *qo_indptr_buffer_ptr,
                        int const *step_ptr,
                        int16_t request_id) {
  constexpr int L_STATE = CONV_L - 1;
  int const first_token_pos = qo_indptr_buffer_ptr[request_id];
  int const last_token_pos = qo_indptr_buffer_ptr[request_id + 1];
  int const num_tokens = last_token_pos - first_token_pos;
  if (num_tokens <= 0) {
    return;
  }

  T const *__restrict__ d_bcx =
      static_cast<T const *>(bcx_ptr) + (size_t)first_token_pos * BCX_STRIDE;
  T const *__restrict__ d_weight = static_cast<T const *>(conv_weight_ptr);
  T *__restrict__ d_state = static_cast<T *>(conv_state_ptr) +
                            (size_t)request_id * STATE_REQ_STRIDE;
  T *__restrict__ d_output =
      static_cast<T *>(output_ptr) + (size_t)first_token_pos * Y_STRIDE;

  // step[request_id] counts tokens finished in previous iterations; zero
  // means this is the request's first chunk and the conv state is empty.
  bool const is_first_chunk = (step_ptr[request_id] == 0);

  for (int c = threadIdx.x; c < CHANNELS; c += blockDim.x) {
    float w[CONV_L];
#pragma unroll
    for (int k = 0; k < CONV_L; ++k) {
      w[k] = float(d_weight[c * W_STRIDE + k]);
    }
    // Rolling window of the previous L-1 Bx values, oldest first.
    float prev[L_STATE];
#pragma unroll
    for (int j = 0; j < L_STATE; ++j) {
      prev[j] = is_first_chunk ? 0.0f : float(d_state[c * L_STATE + j]);
    }
    for (int t = 0; t < num_tokens; ++t) {
      float b_val = float(d_bcx[t * BCX_STRIDE + c]);
      float c_val = float(d_bcx[t * BCX_STRIDE + CHANNELS + c]);
      float x_val = float(d_bcx[t * BCX_STRIDE + 2 * CHANNELS + c]);
      float bx = b_val * x_val;
      float acc = w[CONV_L - 1] * bx;
#pragma unroll
      for (int j = 0; j < L_STATE; ++j) {
        acc += w[j] * prev[j];
      }
      d_output[t * Y_STRIDE + c] = T(c_val * acc);
#pragma unroll
      for (int j = 0; j < L_STATE - 1; ++j) {
        prev[j] = prev[j + 1];
      }
      prev[L_STATE - 1] = bx;
    }
#pragma unroll
    for (int j = 0; j < L_STATE; ++j) {
      d_state[c * L_STATE + j] = T(prev[j]);
    }
  }
}

} // namespace kernel
