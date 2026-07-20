// Test wrapper for the LFM2 gated short-conv task kernel.
// Mimics the MPK runtime: blockIdx.x = request (task_metadata.request_id),
// blockIdx.y = channel group (TBGraph dim-1 partition, pre-offset pointers).

#include "tasks/ampere/lfm2_conv.cuh"

#include <torch/extension.h>

using kernel::lfm2_conv_task_impl;
using bfloat16 = type::bfloat16_t;

template <typename T,
          int CHANNELS,
          int CONV_L,
          int BCX_STRIDE,
          int Y_STRIDE,
          int W_STRIDE,
          int STATE_REQ_STRIDE>
__global__ void lfm2_conv_kernel_wrapper(void const *bcx_ptr,
                                         void const *conv_weight_ptr,
                                         void *conv_state_ptr,
                                         void *output_ptr,
                                         int const *qo_indptr_ptr,
                                         int const *step_ptr) {
  // Channel-group offsets mimic the TBGraph dim-1 partitioning that the MPK
  // runtime bakes into the per-task pointers.
  T const *bcx_g =
      static_cast<T const *>(bcx_ptr) + (size_t)blockIdx.y * 3 * CHANNELS;
  T const *w_g = static_cast<T const *>(conv_weight_ptr) +
                 (size_t)blockIdx.y * CHANNELS * W_STRIDE;
  T *state_g = static_cast<T *>(conv_state_ptr) +
               (size_t)blockIdx.y * CHANNELS * (CONV_L - 1);
  T *y_g = static_cast<T *>(output_ptr) + (size_t)blockIdx.y * CHANNELS;
  lfm2_conv_task_impl<T,
                      CHANNELS,
                      CONV_L,
                      BCX_STRIDE,
                      Y_STRIDE,
                      W_STRIDE,
                      STATE_REQ_STRIDE>(bcx_g,
                                        w_g,
                                        state_g,
                                        y_g,
                                        qo_indptr_ptr,
                                        step_ptr,
                                        (int16_t)blockIdx.x);
}

template <int HIDDEN, int GROUPS, int CONV_L>
void launch_lfm2_conv(torch::Tensor bcx,
                      torch::Tensor conv_weight,
                      torch::Tensor conv_state,
                      torch::Tensor output,
                      torch::Tensor qo_indptr,
                      torch::Tensor step,
                      int num_requests) {
  constexpr int CHANNELS = HIDDEN / GROUPS;
  dim3 grid_dim(num_requests, GROUPS, 1);
  dim3 block_dim(256, 1, 1);
  lfm2_conv_kernel_wrapper<bfloat16,
                           CHANNELS,
                           CONV_L,
                           3 * HIDDEN,
                           HIDDEN,
                           CONV_L,
                           HIDDEN *(CONV_L - 1)>
      <<<grid_dim, block_dim>>>(bcx.data_ptr(),
                                conv_weight.data_ptr(),
                                conv_state.data_ptr(),
                                output.data_ptr(),
                                qo_indptr.data_ptr<int>(),
                                step.data_ptr<int>());
  cudaError_t err = cudaDeviceSynchronize();
  TORCH_CHECK(err == cudaSuccess, "CUDA error: ", cudaGetErrorString(err));
}

void lfm2_conv(torch::Tensor bcx,
               torch::Tensor conv_weight,
               torch::Tensor conv_state,
               torch::Tensor output,
               torch::Tensor qo_indptr,
               torch::Tensor step,
               int64_t num_requests,
               int64_t groups) {
  int hidden = output.size(1);
  int conv_l = conv_weight.size(1);
  TORCH_CHECK(bcx.size(1) == 3 * hidden, "bcx must be (tokens, 3*hidden)");
  TORCH_CHECK(conv_state.size(2) == conv_l - 1,
              "conv_state must be (requests, hidden, conv_l - 1)");
  TORCH_CHECK(conv_l == 3, "only conv_l == 3 is instantiated in this wrapper");
  if (hidden == 1024 && groups == 4) {
    launch_lfm2_conv<1024, 4, 3>(
        bcx, conv_weight, conv_state, output, qo_indptr, step, num_requests);
  } else if (hidden == 2048 && groups == 4) {
    launch_lfm2_conv<2048, 4, 3>(
        bcx, conv_weight, conv_state, output, qo_indptr, step, num_requests);
  } else if (hidden == 2048 && groups == 8) {
    launch_lfm2_conv<2048, 8, 3>(
        bcx, conv_weight, conv_state, output, qo_indptr, step, num_requests);
  } else if (hidden == 256 && groups == 2) {
    launch_lfm2_conv<256, 2, 3>(
        bcx, conv_weight, conv_state, output, qo_indptr, step, num_requests);
  } else if (hidden == 1024 && groups == 1) {
    launch_lfm2_conv<1024, 1, 3>(
        bcx, conv_weight, conv_state, output, qo_indptr, step, num_requests);
  } else {
    TORCH_CHECK(false, "Unsupported (hidden, groups): ", hidden, ", ", groups);
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("lfm2_conv", &lfm2_conv, "LFM2 gated short-conv task kernel");
}
