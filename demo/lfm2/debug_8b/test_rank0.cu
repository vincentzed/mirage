#include "persistent_kernel.cuh"
#include <nlohmann/json.hpp>
#include <fstream>
#include <filesystem>
using json = nlohmann::json;
using namespace mirage::runtime;

// Global variable for runtime JSON path (referenced by Python for kernel reuse)
std::string g_task_graph_json_path;

size_t get_event_id(int my_gpu_id, size_t event_pos, bool nvshmem_event) {
  size_t event_id = ((static_cast<size_t>(my_gpu_id) << 32) | event_pos);
  if (nvshmem_event) {
    event_id = event_id | EVENT_NVSHMEM_TAG;
  }
  return event_id;
}

void construct_task_graph(int num_gpus,
                          int my_gpu_id,
                          std::vector<FullTaskDesc> &all_tasks,
                          std::vector<EventDesc> &all_events,
                          std::vector<TaskId> &first_tasks,
                          std::map<std::string, void*> const &all_tensors) {
    std::string json_path = g_task_graph_json_path;
    if (json_path.empty()) {
        // Fall back to __FILE__ based path for backward compatibility
        std::filesystem::path file_path(__FILE__);
        json_path = file_path.parent_path().string()+"/task_graph.json";
    }
    std::ifstream json_file(json_path);
    if (!json_file.is_open()) {
        fprintf(stderr, "ERROR: Failed to open task graph JSON file: %s\n", json_path.c_str());
        abort();
    }
  nlohmann::json json_task_graph;
  json_file >> json_task_graph;
  for (json const &task : json_task_graph["all_tasks"]) {
    FullTaskDesc task_desc(static_cast<TaskType>(task.at("task_type")),
                task.at("variant_id"));
    task_desc.task_metadata.request_id = task.at("request_id").get<int>();
    task_desc.task_metadata.expert_offset = task.at("expert_offset").get<int>();
    task_desc.task_metadata.kv_idx = task.at("kv_idx").get<int>();
    task_desc.task_metadata.merge_task_offset = task.at("merge_task_offset").get<int>();
    task_desc.task_metadata.task_offset = task.at("task_offset").get<int>();
    if (task.at("trigger_event").is_number_integer()) {
      task_desc.trigger_event = task.at("trigger_event").get<unsigned long long int>();
    }
    else {
      assert(false);
    }
    if (task.at("dependent_event").is_number_integer()) {
      task_desc.dependent_event = task.at("dependent_event").get<unsigned long long int>();
    }
    else {
      assert(false);
    }
    task_desc.num_inputs = 0;
    for (json const &tensor : task["inputs"]) {
      TensorDesc input;
      std::string name = tensor.at("base_ptr").get<std::string>();
      assert(all_tensors.find(name) != all_tensors.end());
      off_t offset = tensor.at("offset").get<off_t>();
      input.base_ptr = static_cast<char*>(all_tensors.at(name))+offset;
      assert(tensor.at("dims").size() == tensor.at("strides").size());
      input.num_dims = tensor.at("dims").size();
      input.data_type = tensor.at("data_type").get<int>();
      for (int i = 0; i < input.num_dims; i++) {
        input.dim[i] = tensor["dims"][i].get<int>();
        input.stride[i] = tensor["strides"][i].get<int>();
      }
      task_desc.inputs[task_desc.num_inputs++] = input;
    }
    task_desc.num_outputs = 0;
    for (json const &tensor : task["outputs"]) {
      TensorDesc output;
      std::string name = tensor.at("base_ptr").get<std::string>();
      assert(all_tensors.find(name) != all_tensors.end());
      off_t offset = tensor.at("offset").get<off_t>();
      output.base_ptr = static_cast<char*>(all_tensors.at(name))+offset;
      assert(tensor.at("dims").size() == tensor.at("strides").size());
      output.num_dims = tensor.at("dims").size();
      output.data_type = tensor.at("data_type").get<int>();
      for (int i = 0; i < output.num_dims; i++) {
        output.dim[i] = tensor["dims"][i];
        output.stride[i] = tensor["strides"][i];
      }
      task_desc.outputs[task_desc.num_outputs++] = output;
    }
    #ifdef MPK_ENABLE_TMA
    if (task.at("task_type") > TASK_HOPPER_TASK_BEGIN && task.at("task_type") < TASK_HOPPER_TASK_END) {
      create_tma_desc_by_task(task_desc);
    }
    if (task.at("task_type") > TASK_SM100_TMA_START_TASK && task.at("task_type") < TASK_SM100_TMA_END_TASK) {
      create_tma_desc_by_task(task_desc);
    }
    if (task.at("task_type") == TASK_MLA_DECODE_SM100 || task.at("task_type") == TASK_MLA_MTP_DECODE_SM100 || task.at("task_type") == TASK_MLA_MTP_DECODE_TP2_SM100 || task.at("task_type") == TASK_MLA_MTP_DECODE_TP4_SM100 || task.at("task_type") == TASK_MLA_MTP_DECODE_TP8_SM100 || task.at("task_type") == TASK_MLA_PREFILL_TP8_SM100) {
      create_tma_desc_by_task(task_desc);
    }
    if (task.at("task_type") == TASK_LINEAR_FP8_SM100 || task.at("task_type") == TASK_LINEAR_FP8_WITH_RESIDUAL_SM100) {
      create_tma_desc_by_task(task_desc);
    }
    #endif
    all_tasks.push_back(task_desc);
  }
  for (json const &e : json_task_graph["all_events"]) {
    EventType event_type = static_cast<EventType>(e.at("event_type").get<int>());
    int num_triggers = e.at("num_triggers").get<int>();
    int first_task_id = e.at("first_task_id").get<int>();
    int last_task_id = e.at("last_task_id").get<int>();
    all_events.push_back(EventDesc(event_type, num_triggers, first_task_id, last_task_id));
  }
  for (json const &t : json_task_graph["first_tasks"]) {
    first_tasks.push_back(t.get<int>());
  }
}

static void _init_persistent_kernel(std::vector<FullTaskDesc> &all_tasks,
                                    std::vector<EventDesc> &all_events,
                                  std::vector<TaskId> &first_tasks,
                                  int num_gpus,
                                  int my_gpu_id,
                                  std::map<std::string, void*> const &model_tensors) {
  assert(num_gpus = 1);
  std::map<std::string, void*> all_tensors;
  char *input_token = static_cast<char*>(model_tensors.at("input_token"));
  all_tensors["input_token"] = input_token;
  char *cos_position_embedding = static_cast<char*>(model_tensors.at("cos_position_embedding"));
  all_tensors["cos_position_embedding"] = cos_position_embedding;
  char *sin_position_embedding = static_cast<char*>(model_tensors.at("sin_position_embedding"));
  all_tensors["sin_position_embedding"] = sin_position_embedding;
  void *embed_out;
  CUDA_CHECK(cudaMalloc(&embed_out, 262144));
  all_tensors["embed_out"] = embed_out;
  void *rmsnorm_out;
  CUDA_CHECK(cudaMalloc(&rmsnorm_out, 262144));
  all_tensors["rmsnorm_out"] = rmsnorm_out;
  void *attn_in;
  CUDA_CHECK(cudaMalloc(&attn_in, 393216));
  all_tensors["attn_in"] = attn_in;
  void *attn_out;
  CUDA_CHECK(cudaMalloc(&attn_out, 262144));
  all_tensors["attn_out"] = attn_out;
  void *attn_proj_out;
  CUDA_CHECK(cudaMalloc(&attn_proj_out, 262144));
  all_tensors["attn_proj_out"] = attn_proj_out;
  void *conv_bcx;
  CUDA_CHECK(cudaMalloc(&conv_bcx, 786432));
  all_tensors["conv_bcx"] = conv_bcx;
  void *conv_y;
  CUDA_CHECK(cudaMalloc(&conv_y, 262144));
  all_tensors["conv_y"] = conv_y;
  void *mlp_mid;
  CUDA_CHECK(cudaMalloc(&mlp_mid, 1835008));
  all_tensors["mlp_mid"] = mlp_mid;
  void *silu_mul_out;
  CUDA_CHECK(cudaMalloc(&silu_mul_out, 917504));
  all_tensors["silu_mul_out"] = silu_mul_out;
  void *mlp_out;
  CUDA_CHECK(cudaMalloc(&mlp_out, 262144));
  all_tensors["mlp_out"] = mlp_out;
  void *argmax_in;
  CUDA_CHECK(cudaMalloc(&argmax_in, 16384000));
  all_tensors["argmax_in"] = argmax_in;
  void *argmax_part_value;
  CUDA_CHECK(cudaMalloc(&argmax_part_value, 16384));
  all_tensors["argmax_part_value"] = argmax_part_value;
  void *argmax_part_index;
  CUDA_CHECK(cudaMalloc(&argmax_part_index, 65536));
  all_tensors["argmax_part_index"] = argmax_part_index;
  void *router_logits;
  CUDA_CHECK(cudaMalloc(&router_logits, 4096));
  all_tensors["router_logits"] = router_logits;
  void *moe_topk_weights;
  CUDA_CHECK(cudaMalloc(&moe_topk_weights, 1024));
  all_tensors["moe_topk_weights"] = moe_topk_weights;
  void *moe_routing_indices;
  CUDA_CHECK(cudaMalloc(&moe_routing_indices, 8192));
  all_tensors["moe_routing_indices"] = moe_routing_indices;
  void *moe_mask;
  CUDA_CHECK(cudaMalloc(&moe_mask, 132));
  all_tensors["moe_mask"] = moe_mask;
  void *moe_mid;
  CUDA_CHECK(cudaMalloc(&moe_mid, 1835008));
  all_tensors["moe_mid"] = moe_mid;
  void *moe_silu_out;
  CUDA_CHECK(cudaMalloc(&moe_silu_out, 917504));
  all_tensors["moe_silu_out"] = moe_silu_out;
  void *moe_down_out;
  CUDA_CHECK(cudaMalloc(&moe_down_out, 1048576));
  all_tensors["moe_down_out"] = moe_down_out;
  void *moe_out_0;
  CUDA_CHECK(cudaMalloc(&moe_out_0, 262144));
  all_tensors["moe_out_0"] = moe_out_0;
  void *moe_out_1;
  CUDA_CHECK(cudaMalloc(&moe_out_1, 262144));
  all_tensors["moe_out_1"] = moe_out_1;
  char *output_token = static_cast<char*>(model_tensors.at("output_token"));
  all_tensors["output_token"] = output_token;
  char *embed_tokens = static_cast<char*>(model_tensors.at("embed_tokens"));
  all_tensors["embed_tokens"] = embed_tokens;
  char *layer_0_operator_norm = static_cast<char*>(model_tensors.at("layer_0_operator_norm"));
  all_tensors["layer_0_operator_norm"] = layer_0_operator_norm;
  char *layer_0_conv_in_proj;
  CUDA_CHECK(cudaMalloc(&layer_0_conv_in_proj, 25165824));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_0_conv_in_proj + 0), 6291456, model_tensors.at("layer_0_conv_in_proj_B"), 2097152, 2097152, 4, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_0_conv_in_proj + 2097152), 6291456, model_tensors.at("layer_0_conv_in_proj_C"), 2097152, 2097152, 4, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_0_conv_in_proj + 4194304), 6291456, model_tensors.at("layer_0_conv_in_proj_x"), 2097152, 2097152, 4, cudaMemcpyDeviceToDevice));
  all_tensors["layer_0_conv_in_proj"] = layer_0_conv_in_proj;
  char *layer_0_conv_weight = static_cast<char*>(model_tensors.at("layer_0_conv_weight"));
  all_tensors["layer_0_conv_weight"] = layer_0_conv_weight;
  char *layer_0_conv_state = static_cast<char*>(model_tensors.at("layer_0_conv_state"));
  all_tensors["layer_0_conv_state"] = layer_0_conv_state;
  char *layer_0_conv_out_proj = static_cast<char*>(model_tensors.at("layer_0_conv_out_proj"));
  all_tensors["layer_0_conv_out_proj"] = layer_0_conv_out_proj;
  char *layer_0_ffn_norm = static_cast<char*>(model_tensors.at("layer_0_ffn_norm"));
  all_tensors["layer_0_ffn_norm"] = layer_0_ffn_norm;
  char *layer_0_gatedup_proj;
  CUDA_CHECK(cudaMalloc(&layer_0_gatedup_proj, 58720256));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_0_gatedup_proj + 0), 1835008, model_tensors.at("layer_0_w1"), 917504, 917504, 32, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_0_gatedup_proj + 917504), 1835008, model_tensors.at("layer_0_w3"), 917504, 917504, 32, cudaMemcpyDeviceToDevice));
  all_tensors["layer_0_gatedup_proj"] = layer_0_gatedup_proj;
  char *layer_0_w2 = static_cast<char*>(model_tensors.at("layer_0_w2"));
  all_tensors["layer_0_w2"] = layer_0_w2;
  char *layer_1_operator_norm = static_cast<char*>(model_tensors.at("layer_1_operator_norm"));
  all_tensors["layer_1_operator_norm"] = layer_1_operator_norm;
  char *layer_1_conv_in_proj;
  CUDA_CHECK(cudaMalloc(&layer_1_conv_in_proj, 25165824));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_1_conv_in_proj + 0), 6291456, model_tensors.at("layer_1_conv_in_proj_B"), 2097152, 2097152, 4, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_1_conv_in_proj + 2097152), 6291456, model_tensors.at("layer_1_conv_in_proj_C"), 2097152, 2097152, 4, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_1_conv_in_proj + 4194304), 6291456, model_tensors.at("layer_1_conv_in_proj_x"), 2097152, 2097152, 4, cudaMemcpyDeviceToDevice));
  all_tensors["layer_1_conv_in_proj"] = layer_1_conv_in_proj;
  char *layer_1_conv_weight = static_cast<char*>(model_tensors.at("layer_1_conv_weight"));
  all_tensors["layer_1_conv_weight"] = layer_1_conv_weight;
  char *layer_1_conv_state = static_cast<char*>(model_tensors.at("layer_1_conv_state"));
  all_tensors["layer_1_conv_state"] = layer_1_conv_state;
  char *layer_1_conv_out_proj = static_cast<char*>(model_tensors.at("layer_1_conv_out_proj"));
  all_tensors["layer_1_conv_out_proj"] = layer_1_conv_out_proj;
  char *layer_1_ffn_norm = static_cast<char*>(model_tensors.at("layer_1_ffn_norm"));
  all_tensors["layer_1_ffn_norm"] = layer_1_ffn_norm;
  char *layer_1_gatedup_proj;
  CUDA_CHECK(cudaMalloc(&layer_1_gatedup_proj, 58720256));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_1_gatedup_proj + 0), 1835008, model_tensors.at("layer_1_w1"), 917504, 917504, 32, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_1_gatedup_proj + 917504), 1835008, model_tensors.at("layer_1_w3"), 917504, 917504, 32, cudaMemcpyDeviceToDevice));
  all_tensors["layer_1_gatedup_proj"] = layer_1_gatedup_proj;
  char *layer_1_w2 = static_cast<char*>(model_tensors.at("layer_1_w2"));
  all_tensors["layer_1_w2"] = layer_1_w2;
  char *layer_2_operator_norm = static_cast<char*>(model_tensors.at("layer_2_operator_norm"));
  all_tensors["layer_2_operator_norm"] = layer_2_operator_norm;
  char *layer_2_qkv_proj;
  CUDA_CHECK(cudaMalloc(&layer_2_qkv_proj, 12582912));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_2_qkv_proj + 0), 1572864, model_tensors.at("layer_2_q_proj"), 1048576, 1048576, 8, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_2_qkv_proj + 1048576), 1572864, model_tensors.at("layer_2_k_proj"), 262144, 262144, 8, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_2_qkv_proj + 1310720), 1572864, model_tensors.at("layer_2_v_proj"), 262144, 262144, 8, cudaMemcpyDeviceToDevice));
  all_tensors["layer_2_qkv_proj"] = layer_2_qkv_proj;
  char *layer_2_q_layernorm = static_cast<char*>(model_tensors.at("layer_2_q_layernorm"));
  all_tensors["layer_2_q_layernorm"] = layer_2_q_layernorm;
  char *layer_2_k_layernorm = static_cast<char*>(model_tensors.at("layer_2_k_layernorm"));
  all_tensors["layer_2_k_layernorm"] = layer_2_k_layernorm;
  char *layer_2_k_cache = static_cast<char*>(model_tensors.at("layer_2_k_cache"));
  all_tensors["layer_2_k_cache"] = layer_2_k_cache;
  char *layer_2_v_cache = static_cast<char*>(model_tensors.at("layer_2_v_cache"));
  all_tensors["layer_2_v_cache"] = layer_2_v_cache;
  char *layer_2_out_proj = static_cast<char*>(model_tensors.at("layer_2_out_proj"));
  all_tensors["layer_2_out_proj"] = layer_2_out_proj;
  char *layer_2_ffn_norm = static_cast<char*>(model_tensors.at("layer_2_ffn_norm"));
  all_tensors["layer_2_ffn_norm"] = layer_2_ffn_norm;
  char *layer_2_router_gate = static_cast<char*>(model_tensors.at("layer_2_router_gate"));
  all_tensors["layer_2_router_gate"] = layer_2_router_gate;
  char *layer_2_expert_bias = static_cast<char*>(model_tensors.at("layer_2_expert_bias"));
  all_tensors["layer_2_expert_bias"] = layer_2_expert_bias;
  char *layer_2_experts_w13 = static_cast<char*>(model_tensors.at("layer_2_experts_w13"));
  all_tensors["layer_2_experts_w13"] = layer_2_experts_w13;
  char *layer_2_experts_w2 = static_cast<char*>(model_tensors.at("layer_2_experts_w2"));
  all_tensors["layer_2_experts_w2"] = layer_2_experts_w2;
  char *layer_3_operator_norm = static_cast<char*>(model_tensors.at("layer_3_operator_norm"));
  all_tensors["layer_3_operator_norm"] = layer_3_operator_norm;
  char *layer_3_conv_in_proj;
  CUDA_CHECK(cudaMalloc(&layer_3_conv_in_proj, 25165824));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_3_conv_in_proj + 0), 6291456, model_tensors.at("layer_3_conv_in_proj_B"), 2097152, 2097152, 4, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_3_conv_in_proj + 2097152), 6291456, model_tensors.at("layer_3_conv_in_proj_C"), 2097152, 2097152, 4, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_3_conv_in_proj + 4194304), 6291456, model_tensors.at("layer_3_conv_in_proj_x"), 2097152, 2097152, 4, cudaMemcpyDeviceToDevice));
  all_tensors["layer_3_conv_in_proj"] = layer_3_conv_in_proj;
  char *layer_3_conv_weight = static_cast<char*>(model_tensors.at("layer_3_conv_weight"));
  all_tensors["layer_3_conv_weight"] = layer_3_conv_weight;
  char *layer_3_conv_state = static_cast<char*>(model_tensors.at("layer_3_conv_state"));
  all_tensors["layer_3_conv_state"] = layer_3_conv_state;
  char *layer_3_conv_out_proj = static_cast<char*>(model_tensors.at("layer_3_conv_out_proj"));
  all_tensors["layer_3_conv_out_proj"] = layer_3_conv_out_proj;
  char *layer_3_ffn_norm = static_cast<char*>(model_tensors.at("layer_3_ffn_norm"));
  all_tensors["layer_3_ffn_norm"] = layer_3_ffn_norm;
  char *layer_3_router_gate = static_cast<char*>(model_tensors.at("layer_3_router_gate"));
  all_tensors["layer_3_router_gate"] = layer_3_router_gate;
  char *layer_3_expert_bias = static_cast<char*>(model_tensors.at("layer_3_expert_bias"));
  all_tensors["layer_3_expert_bias"] = layer_3_expert_bias;
  char *layer_3_experts_w13 = static_cast<char*>(model_tensors.at("layer_3_experts_w13"));
  all_tensors["layer_3_experts_w13"] = layer_3_experts_w13;
  char *layer_3_experts_w2 = static_cast<char*>(model_tensors.at("layer_3_experts_w2"));
  all_tensors["layer_3_experts_w2"] = layer_3_experts_w2;
  char *layer_4_operator_norm = static_cast<char*>(model_tensors.at("layer_4_operator_norm"));
  all_tensors["layer_4_operator_norm"] = layer_4_operator_norm;
  char *layer_4_conv_in_proj;
  CUDA_CHECK(cudaMalloc(&layer_4_conv_in_proj, 25165824));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_4_conv_in_proj + 0), 6291456, model_tensors.at("layer_4_conv_in_proj_B"), 2097152, 2097152, 4, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_4_conv_in_proj + 2097152), 6291456, model_tensors.at("layer_4_conv_in_proj_C"), 2097152, 2097152, 4, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_4_conv_in_proj + 4194304), 6291456, model_tensors.at("layer_4_conv_in_proj_x"), 2097152, 2097152, 4, cudaMemcpyDeviceToDevice));
  all_tensors["layer_4_conv_in_proj"] = layer_4_conv_in_proj;
  char *layer_4_conv_weight = static_cast<char*>(model_tensors.at("layer_4_conv_weight"));
  all_tensors["layer_4_conv_weight"] = layer_4_conv_weight;
  char *layer_4_conv_state = static_cast<char*>(model_tensors.at("layer_4_conv_state"));
  all_tensors["layer_4_conv_state"] = layer_4_conv_state;
  char *layer_4_conv_out_proj = static_cast<char*>(model_tensors.at("layer_4_conv_out_proj"));
  all_tensors["layer_4_conv_out_proj"] = layer_4_conv_out_proj;
  char *layer_4_ffn_norm = static_cast<char*>(model_tensors.at("layer_4_ffn_norm"));
  all_tensors["layer_4_ffn_norm"] = layer_4_ffn_norm;
  char *layer_4_router_gate = static_cast<char*>(model_tensors.at("layer_4_router_gate"));
  all_tensors["layer_4_router_gate"] = layer_4_router_gate;
  char *layer_4_expert_bias = static_cast<char*>(model_tensors.at("layer_4_expert_bias"));
  all_tensors["layer_4_expert_bias"] = layer_4_expert_bias;
  char *layer_4_experts_w13 = static_cast<char*>(model_tensors.at("layer_4_experts_w13"));
  all_tensors["layer_4_experts_w13"] = layer_4_experts_w13;
  char *layer_4_experts_w2 = static_cast<char*>(model_tensors.at("layer_4_experts_w2"));
  all_tensors["layer_4_experts_w2"] = layer_4_experts_w2;
  char *layer_5_operator_norm = static_cast<char*>(model_tensors.at("layer_5_operator_norm"));
  all_tensors["layer_5_operator_norm"] = layer_5_operator_norm;
  char *layer_5_conv_in_proj;
  CUDA_CHECK(cudaMalloc(&layer_5_conv_in_proj, 25165824));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_5_conv_in_proj + 0), 6291456, model_tensors.at("layer_5_conv_in_proj_B"), 2097152, 2097152, 4, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_5_conv_in_proj + 2097152), 6291456, model_tensors.at("layer_5_conv_in_proj_C"), 2097152, 2097152, 4, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_5_conv_in_proj + 4194304), 6291456, model_tensors.at("layer_5_conv_in_proj_x"), 2097152, 2097152, 4, cudaMemcpyDeviceToDevice));
  all_tensors["layer_5_conv_in_proj"] = layer_5_conv_in_proj;
  char *layer_5_conv_weight = static_cast<char*>(model_tensors.at("layer_5_conv_weight"));
  all_tensors["layer_5_conv_weight"] = layer_5_conv_weight;
  char *layer_5_conv_state = static_cast<char*>(model_tensors.at("layer_5_conv_state"));
  all_tensors["layer_5_conv_state"] = layer_5_conv_state;
  char *layer_5_conv_out_proj = static_cast<char*>(model_tensors.at("layer_5_conv_out_proj"));
  all_tensors["layer_5_conv_out_proj"] = layer_5_conv_out_proj;
  char *layer_5_ffn_norm = static_cast<char*>(model_tensors.at("layer_5_ffn_norm"));
  all_tensors["layer_5_ffn_norm"] = layer_5_ffn_norm;
  char *layer_5_router_gate = static_cast<char*>(model_tensors.at("layer_5_router_gate"));
  all_tensors["layer_5_router_gate"] = layer_5_router_gate;
  char *layer_5_expert_bias = static_cast<char*>(model_tensors.at("layer_5_expert_bias"));
  all_tensors["layer_5_expert_bias"] = layer_5_expert_bias;
  char *layer_5_experts_w13 = static_cast<char*>(model_tensors.at("layer_5_experts_w13"));
  all_tensors["layer_5_experts_w13"] = layer_5_experts_w13;
  char *layer_5_experts_w2 = static_cast<char*>(model_tensors.at("layer_5_experts_w2"));
  all_tensors["layer_5_experts_w2"] = layer_5_experts_w2;
  char *layer_6_operator_norm = static_cast<char*>(model_tensors.at("layer_6_operator_norm"));
  all_tensors["layer_6_operator_norm"] = layer_6_operator_norm;
  char *layer_6_qkv_proj;
  CUDA_CHECK(cudaMalloc(&layer_6_qkv_proj, 12582912));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_6_qkv_proj + 0), 1572864, model_tensors.at("layer_6_q_proj"), 1048576, 1048576, 8, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_6_qkv_proj + 1048576), 1572864, model_tensors.at("layer_6_k_proj"), 262144, 262144, 8, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_6_qkv_proj + 1310720), 1572864, model_tensors.at("layer_6_v_proj"), 262144, 262144, 8, cudaMemcpyDeviceToDevice));
  all_tensors["layer_6_qkv_proj"] = layer_6_qkv_proj;
  char *layer_6_q_layernorm = static_cast<char*>(model_tensors.at("layer_6_q_layernorm"));
  all_tensors["layer_6_q_layernorm"] = layer_6_q_layernorm;
  char *layer_6_k_layernorm = static_cast<char*>(model_tensors.at("layer_6_k_layernorm"));
  all_tensors["layer_6_k_layernorm"] = layer_6_k_layernorm;
  char *layer_6_k_cache = static_cast<char*>(model_tensors.at("layer_6_k_cache"));
  all_tensors["layer_6_k_cache"] = layer_6_k_cache;
  char *layer_6_v_cache = static_cast<char*>(model_tensors.at("layer_6_v_cache"));
  all_tensors["layer_6_v_cache"] = layer_6_v_cache;
  char *layer_6_out_proj = static_cast<char*>(model_tensors.at("layer_6_out_proj"));
  all_tensors["layer_6_out_proj"] = layer_6_out_proj;
  char *layer_6_ffn_norm = static_cast<char*>(model_tensors.at("layer_6_ffn_norm"));
  all_tensors["layer_6_ffn_norm"] = layer_6_ffn_norm;
  char *layer_6_router_gate = static_cast<char*>(model_tensors.at("layer_6_router_gate"));
  all_tensors["layer_6_router_gate"] = layer_6_router_gate;
  char *layer_6_expert_bias = static_cast<char*>(model_tensors.at("layer_6_expert_bias"));
  all_tensors["layer_6_expert_bias"] = layer_6_expert_bias;
  char *layer_6_experts_w13 = static_cast<char*>(model_tensors.at("layer_6_experts_w13"));
  all_tensors["layer_6_experts_w13"] = layer_6_experts_w13;
  char *layer_6_experts_w2 = static_cast<char*>(model_tensors.at("layer_6_experts_w2"));
  all_tensors["layer_6_experts_w2"] = layer_6_experts_w2;
  char *layer_7_operator_norm = static_cast<char*>(model_tensors.at("layer_7_operator_norm"));
  all_tensors["layer_7_operator_norm"] = layer_7_operator_norm;
  char *layer_7_conv_in_proj;
  CUDA_CHECK(cudaMalloc(&layer_7_conv_in_proj, 25165824));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_7_conv_in_proj + 0), 6291456, model_tensors.at("layer_7_conv_in_proj_B"), 2097152, 2097152, 4, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_7_conv_in_proj + 2097152), 6291456, model_tensors.at("layer_7_conv_in_proj_C"), 2097152, 2097152, 4, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_7_conv_in_proj + 4194304), 6291456, model_tensors.at("layer_7_conv_in_proj_x"), 2097152, 2097152, 4, cudaMemcpyDeviceToDevice));
  all_tensors["layer_7_conv_in_proj"] = layer_7_conv_in_proj;
  char *layer_7_conv_weight = static_cast<char*>(model_tensors.at("layer_7_conv_weight"));
  all_tensors["layer_7_conv_weight"] = layer_7_conv_weight;
  char *layer_7_conv_state = static_cast<char*>(model_tensors.at("layer_7_conv_state"));
  all_tensors["layer_7_conv_state"] = layer_7_conv_state;
  char *layer_7_conv_out_proj = static_cast<char*>(model_tensors.at("layer_7_conv_out_proj"));
  all_tensors["layer_7_conv_out_proj"] = layer_7_conv_out_proj;
  char *layer_7_ffn_norm = static_cast<char*>(model_tensors.at("layer_7_ffn_norm"));
  all_tensors["layer_7_ffn_norm"] = layer_7_ffn_norm;
  char *layer_7_router_gate = static_cast<char*>(model_tensors.at("layer_7_router_gate"));
  all_tensors["layer_7_router_gate"] = layer_7_router_gate;
  char *layer_7_expert_bias = static_cast<char*>(model_tensors.at("layer_7_expert_bias"));
  all_tensors["layer_7_expert_bias"] = layer_7_expert_bias;
  char *layer_7_experts_w13 = static_cast<char*>(model_tensors.at("layer_7_experts_w13"));
  all_tensors["layer_7_experts_w13"] = layer_7_experts_w13;
  char *layer_7_experts_w2 = static_cast<char*>(model_tensors.at("layer_7_experts_w2"));
  all_tensors["layer_7_experts_w2"] = layer_7_experts_w2;
  char *layer_8_operator_norm = static_cast<char*>(model_tensors.at("layer_8_operator_norm"));
  all_tensors["layer_8_operator_norm"] = layer_8_operator_norm;
  char *layer_8_conv_in_proj;
  CUDA_CHECK(cudaMalloc(&layer_8_conv_in_proj, 25165824));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_8_conv_in_proj + 0), 6291456, model_tensors.at("layer_8_conv_in_proj_B"), 2097152, 2097152, 4, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_8_conv_in_proj + 2097152), 6291456, model_tensors.at("layer_8_conv_in_proj_C"), 2097152, 2097152, 4, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_8_conv_in_proj + 4194304), 6291456, model_tensors.at("layer_8_conv_in_proj_x"), 2097152, 2097152, 4, cudaMemcpyDeviceToDevice));
  all_tensors["layer_8_conv_in_proj"] = layer_8_conv_in_proj;
  char *layer_8_conv_weight = static_cast<char*>(model_tensors.at("layer_8_conv_weight"));
  all_tensors["layer_8_conv_weight"] = layer_8_conv_weight;
  char *layer_8_conv_state = static_cast<char*>(model_tensors.at("layer_8_conv_state"));
  all_tensors["layer_8_conv_state"] = layer_8_conv_state;
  char *layer_8_conv_out_proj = static_cast<char*>(model_tensors.at("layer_8_conv_out_proj"));
  all_tensors["layer_8_conv_out_proj"] = layer_8_conv_out_proj;
  char *layer_8_ffn_norm = static_cast<char*>(model_tensors.at("layer_8_ffn_norm"));
  all_tensors["layer_8_ffn_norm"] = layer_8_ffn_norm;
  char *layer_8_router_gate = static_cast<char*>(model_tensors.at("layer_8_router_gate"));
  all_tensors["layer_8_router_gate"] = layer_8_router_gate;
  char *layer_8_expert_bias = static_cast<char*>(model_tensors.at("layer_8_expert_bias"));
  all_tensors["layer_8_expert_bias"] = layer_8_expert_bias;
  char *layer_8_experts_w13 = static_cast<char*>(model_tensors.at("layer_8_experts_w13"));
  all_tensors["layer_8_experts_w13"] = layer_8_experts_w13;
  char *layer_8_experts_w2 = static_cast<char*>(model_tensors.at("layer_8_experts_w2"));
  all_tensors["layer_8_experts_w2"] = layer_8_experts_w2;
  char *layer_9_operator_norm = static_cast<char*>(model_tensors.at("layer_9_operator_norm"));
  all_tensors["layer_9_operator_norm"] = layer_9_operator_norm;
  char *layer_9_conv_in_proj;
  CUDA_CHECK(cudaMalloc(&layer_9_conv_in_proj, 25165824));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_9_conv_in_proj + 0), 6291456, model_tensors.at("layer_9_conv_in_proj_B"), 2097152, 2097152, 4, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_9_conv_in_proj + 2097152), 6291456, model_tensors.at("layer_9_conv_in_proj_C"), 2097152, 2097152, 4, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_9_conv_in_proj + 4194304), 6291456, model_tensors.at("layer_9_conv_in_proj_x"), 2097152, 2097152, 4, cudaMemcpyDeviceToDevice));
  all_tensors["layer_9_conv_in_proj"] = layer_9_conv_in_proj;
  char *layer_9_conv_weight = static_cast<char*>(model_tensors.at("layer_9_conv_weight"));
  all_tensors["layer_9_conv_weight"] = layer_9_conv_weight;
  char *layer_9_conv_state = static_cast<char*>(model_tensors.at("layer_9_conv_state"));
  all_tensors["layer_9_conv_state"] = layer_9_conv_state;
  char *layer_9_conv_out_proj = static_cast<char*>(model_tensors.at("layer_9_conv_out_proj"));
  all_tensors["layer_9_conv_out_proj"] = layer_9_conv_out_proj;
  char *layer_9_ffn_norm = static_cast<char*>(model_tensors.at("layer_9_ffn_norm"));
  all_tensors["layer_9_ffn_norm"] = layer_9_ffn_norm;
  char *layer_9_router_gate = static_cast<char*>(model_tensors.at("layer_9_router_gate"));
  all_tensors["layer_9_router_gate"] = layer_9_router_gate;
  char *layer_9_expert_bias = static_cast<char*>(model_tensors.at("layer_9_expert_bias"));
  all_tensors["layer_9_expert_bias"] = layer_9_expert_bias;
  char *layer_9_experts_w13 = static_cast<char*>(model_tensors.at("layer_9_experts_w13"));
  all_tensors["layer_9_experts_w13"] = layer_9_experts_w13;
  char *layer_9_experts_w2 = static_cast<char*>(model_tensors.at("layer_9_experts_w2"));
  all_tensors["layer_9_experts_w2"] = layer_9_experts_w2;
  char *layer_10_operator_norm = static_cast<char*>(model_tensors.at("layer_10_operator_norm"));
  all_tensors["layer_10_operator_norm"] = layer_10_operator_norm;
  char *layer_10_qkv_proj;
  CUDA_CHECK(cudaMalloc(&layer_10_qkv_proj, 12582912));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_10_qkv_proj + 0), 1572864, model_tensors.at("layer_10_q_proj"), 1048576, 1048576, 8, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_10_qkv_proj + 1048576), 1572864, model_tensors.at("layer_10_k_proj"), 262144, 262144, 8, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_10_qkv_proj + 1310720), 1572864, model_tensors.at("layer_10_v_proj"), 262144, 262144, 8, cudaMemcpyDeviceToDevice));
  all_tensors["layer_10_qkv_proj"] = layer_10_qkv_proj;
  char *layer_10_q_layernorm = static_cast<char*>(model_tensors.at("layer_10_q_layernorm"));
  all_tensors["layer_10_q_layernorm"] = layer_10_q_layernorm;
  char *layer_10_k_layernorm = static_cast<char*>(model_tensors.at("layer_10_k_layernorm"));
  all_tensors["layer_10_k_layernorm"] = layer_10_k_layernorm;
  char *layer_10_k_cache = static_cast<char*>(model_tensors.at("layer_10_k_cache"));
  all_tensors["layer_10_k_cache"] = layer_10_k_cache;
  char *layer_10_v_cache = static_cast<char*>(model_tensors.at("layer_10_v_cache"));
  all_tensors["layer_10_v_cache"] = layer_10_v_cache;
  char *layer_10_out_proj = static_cast<char*>(model_tensors.at("layer_10_out_proj"));
  all_tensors["layer_10_out_proj"] = layer_10_out_proj;
  char *layer_10_ffn_norm = static_cast<char*>(model_tensors.at("layer_10_ffn_norm"));
  all_tensors["layer_10_ffn_norm"] = layer_10_ffn_norm;
  char *layer_10_router_gate = static_cast<char*>(model_tensors.at("layer_10_router_gate"));
  all_tensors["layer_10_router_gate"] = layer_10_router_gate;
  char *layer_10_expert_bias = static_cast<char*>(model_tensors.at("layer_10_expert_bias"));
  all_tensors["layer_10_expert_bias"] = layer_10_expert_bias;
  char *layer_10_experts_w13 = static_cast<char*>(model_tensors.at("layer_10_experts_w13"));
  all_tensors["layer_10_experts_w13"] = layer_10_experts_w13;
  char *layer_10_experts_w2 = static_cast<char*>(model_tensors.at("layer_10_experts_w2"));
  all_tensors["layer_10_experts_w2"] = layer_10_experts_w2;
  char *layer_11_operator_norm = static_cast<char*>(model_tensors.at("layer_11_operator_norm"));
  all_tensors["layer_11_operator_norm"] = layer_11_operator_norm;
  char *layer_11_conv_in_proj;
  CUDA_CHECK(cudaMalloc(&layer_11_conv_in_proj, 25165824));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_11_conv_in_proj + 0), 6291456, model_tensors.at("layer_11_conv_in_proj_B"), 2097152, 2097152, 4, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_11_conv_in_proj + 2097152), 6291456, model_tensors.at("layer_11_conv_in_proj_C"), 2097152, 2097152, 4, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_11_conv_in_proj + 4194304), 6291456, model_tensors.at("layer_11_conv_in_proj_x"), 2097152, 2097152, 4, cudaMemcpyDeviceToDevice));
  all_tensors["layer_11_conv_in_proj"] = layer_11_conv_in_proj;
  char *layer_11_conv_weight = static_cast<char*>(model_tensors.at("layer_11_conv_weight"));
  all_tensors["layer_11_conv_weight"] = layer_11_conv_weight;
  char *layer_11_conv_state = static_cast<char*>(model_tensors.at("layer_11_conv_state"));
  all_tensors["layer_11_conv_state"] = layer_11_conv_state;
  char *layer_11_conv_out_proj = static_cast<char*>(model_tensors.at("layer_11_conv_out_proj"));
  all_tensors["layer_11_conv_out_proj"] = layer_11_conv_out_proj;
  char *layer_11_ffn_norm = static_cast<char*>(model_tensors.at("layer_11_ffn_norm"));
  all_tensors["layer_11_ffn_norm"] = layer_11_ffn_norm;
  char *layer_11_router_gate = static_cast<char*>(model_tensors.at("layer_11_router_gate"));
  all_tensors["layer_11_router_gate"] = layer_11_router_gate;
  char *layer_11_expert_bias = static_cast<char*>(model_tensors.at("layer_11_expert_bias"));
  all_tensors["layer_11_expert_bias"] = layer_11_expert_bias;
  char *layer_11_experts_w13 = static_cast<char*>(model_tensors.at("layer_11_experts_w13"));
  all_tensors["layer_11_experts_w13"] = layer_11_experts_w13;
  char *layer_11_experts_w2 = static_cast<char*>(model_tensors.at("layer_11_experts_w2"));
  all_tensors["layer_11_experts_w2"] = layer_11_experts_w2;
  char *layer_12_operator_norm = static_cast<char*>(model_tensors.at("layer_12_operator_norm"));
  all_tensors["layer_12_operator_norm"] = layer_12_operator_norm;
  char *layer_12_conv_in_proj;
  CUDA_CHECK(cudaMalloc(&layer_12_conv_in_proj, 25165824));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_12_conv_in_proj + 0), 6291456, model_tensors.at("layer_12_conv_in_proj_B"), 2097152, 2097152, 4, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_12_conv_in_proj + 2097152), 6291456, model_tensors.at("layer_12_conv_in_proj_C"), 2097152, 2097152, 4, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_12_conv_in_proj + 4194304), 6291456, model_tensors.at("layer_12_conv_in_proj_x"), 2097152, 2097152, 4, cudaMemcpyDeviceToDevice));
  all_tensors["layer_12_conv_in_proj"] = layer_12_conv_in_proj;
  char *layer_12_conv_weight = static_cast<char*>(model_tensors.at("layer_12_conv_weight"));
  all_tensors["layer_12_conv_weight"] = layer_12_conv_weight;
  char *layer_12_conv_state = static_cast<char*>(model_tensors.at("layer_12_conv_state"));
  all_tensors["layer_12_conv_state"] = layer_12_conv_state;
  char *layer_12_conv_out_proj = static_cast<char*>(model_tensors.at("layer_12_conv_out_proj"));
  all_tensors["layer_12_conv_out_proj"] = layer_12_conv_out_proj;
  char *layer_12_ffn_norm = static_cast<char*>(model_tensors.at("layer_12_ffn_norm"));
  all_tensors["layer_12_ffn_norm"] = layer_12_ffn_norm;
  char *layer_12_router_gate = static_cast<char*>(model_tensors.at("layer_12_router_gate"));
  all_tensors["layer_12_router_gate"] = layer_12_router_gate;
  char *layer_12_expert_bias = static_cast<char*>(model_tensors.at("layer_12_expert_bias"));
  all_tensors["layer_12_expert_bias"] = layer_12_expert_bias;
  char *layer_12_experts_w13 = static_cast<char*>(model_tensors.at("layer_12_experts_w13"));
  all_tensors["layer_12_experts_w13"] = layer_12_experts_w13;
  char *layer_12_experts_w2 = static_cast<char*>(model_tensors.at("layer_12_experts_w2"));
  all_tensors["layer_12_experts_w2"] = layer_12_experts_w2;
  char *layer_13_operator_norm = static_cast<char*>(model_tensors.at("layer_13_operator_norm"));
  all_tensors["layer_13_operator_norm"] = layer_13_operator_norm;
  char *layer_13_conv_in_proj;
  CUDA_CHECK(cudaMalloc(&layer_13_conv_in_proj, 25165824));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_13_conv_in_proj + 0), 6291456, model_tensors.at("layer_13_conv_in_proj_B"), 2097152, 2097152, 4, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_13_conv_in_proj + 2097152), 6291456, model_tensors.at("layer_13_conv_in_proj_C"), 2097152, 2097152, 4, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_13_conv_in_proj + 4194304), 6291456, model_tensors.at("layer_13_conv_in_proj_x"), 2097152, 2097152, 4, cudaMemcpyDeviceToDevice));
  all_tensors["layer_13_conv_in_proj"] = layer_13_conv_in_proj;
  char *layer_13_conv_weight = static_cast<char*>(model_tensors.at("layer_13_conv_weight"));
  all_tensors["layer_13_conv_weight"] = layer_13_conv_weight;
  char *layer_13_conv_state = static_cast<char*>(model_tensors.at("layer_13_conv_state"));
  all_tensors["layer_13_conv_state"] = layer_13_conv_state;
  char *layer_13_conv_out_proj = static_cast<char*>(model_tensors.at("layer_13_conv_out_proj"));
  all_tensors["layer_13_conv_out_proj"] = layer_13_conv_out_proj;
  char *layer_13_ffn_norm = static_cast<char*>(model_tensors.at("layer_13_ffn_norm"));
  all_tensors["layer_13_ffn_norm"] = layer_13_ffn_norm;
  char *layer_13_router_gate = static_cast<char*>(model_tensors.at("layer_13_router_gate"));
  all_tensors["layer_13_router_gate"] = layer_13_router_gate;
  char *layer_13_expert_bias = static_cast<char*>(model_tensors.at("layer_13_expert_bias"));
  all_tensors["layer_13_expert_bias"] = layer_13_expert_bias;
  char *layer_13_experts_w13 = static_cast<char*>(model_tensors.at("layer_13_experts_w13"));
  all_tensors["layer_13_experts_w13"] = layer_13_experts_w13;
  char *layer_13_experts_w2 = static_cast<char*>(model_tensors.at("layer_13_experts_w2"));
  all_tensors["layer_13_experts_w2"] = layer_13_experts_w2;
  char *layer_14_operator_norm = static_cast<char*>(model_tensors.at("layer_14_operator_norm"));
  all_tensors["layer_14_operator_norm"] = layer_14_operator_norm;
  char *layer_14_qkv_proj;
  CUDA_CHECK(cudaMalloc(&layer_14_qkv_proj, 12582912));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_14_qkv_proj + 0), 1572864, model_tensors.at("layer_14_q_proj"), 1048576, 1048576, 8, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_14_qkv_proj + 1048576), 1572864, model_tensors.at("layer_14_k_proj"), 262144, 262144, 8, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_14_qkv_proj + 1310720), 1572864, model_tensors.at("layer_14_v_proj"), 262144, 262144, 8, cudaMemcpyDeviceToDevice));
  all_tensors["layer_14_qkv_proj"] = layer_14_qkv_proj;
  char *layer_14_q_layernorm = static_cast<char*>(model_tensors.at("layer_14_q_layernorm"));
  all_tensors["layer_14_q_layernorm"] = layer_14_q_layernorm;
  char *layer_14_k_layernorm = static_cast<char*>(model_tensors.at("layer_14_k_layernorm"));
  all_tensors["layer_14_k_layernorm"] = layer_14_k_layernorm;
  char *layer_14_k_cache = static_cast<char*>(model_tensors.at("layer_14_k_cache"));
  all_tensors["layer_14_k_cache"] = layer_14_k_cache;
  char *layer_14_v_cache = static_cast<char*>(model_tensors.at("layer_14_v_cache"));
  all_tensors["layer_14_v_cache"] = layer_14_v_cache;
  char *layer_14_out_proj = static_cast<char*>(model_tensors.at("layer_14_out_proj"));
  all_tensors["layer_14_out_proj"] = layer_14_out_proj;
  char *layer_14_ffn_norm = static_cast<char*>(model_tensors.at("layer_14_ffn_norm"));
  all_tensors["layer_14_ffn_norm"] = layer_14_ffn_norm;
  char *layer_14_router_gate = static_cast<char*>(model_tensors.at("layer_14_router_gate"));
  all_tensors["layer_14_router_gate"] = layer_14_router_gate;
  char *layer_14_expert_bias = static_cast<char*>(model_tensors.at("layer_14_expert_bias"));
  all_tensors["layer_14_expert_bias"] = layer_14_expert_bias;
  char *layer_14_experts_w13 = static_cast<char*>(model_tensors.at("layer_14_experts_w13"));
  all_tensors["layer_14_experts_w13"] = layer_14_experts_w13;
  char *layer_14_experts_w2 = static_cast<char*>(model_tensors.at("layer_14_experts_w2"));
  all_tensors["layer_14_experts_w2"] = layer_14_experts_w2;
  char *layer_15_operator_norm = static_cast<char*>(model_tensors.at("layer_15_operator_norm"));
  all_tensors["layer_15_operator_norm"] = layer_15_operator_norm;
  char *layer_15_conv_in_proj;
  CUDA_CHECK(cudaMalloc(&layer_15_conv_in_proj, 25165824));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_15_conv_in_proj + 0), 6291456, model_tensors.at("layer_15_conv_in_proj_B"), 2097152, 2097152, 4, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_15_conv_in_proj + 2097152), 6291456, model_tensors.at("layer_15_conv_in_proj_C"), 2097152, 2097152, 4, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_15_conv_in_proj + 4194304), 6291456, model_tensors.at("layer_15_conv_in_proj_x"), 2097152, 2097152, 4, cudaMemcpyDeviceToDevice));
  all_tensors["layer_15_conv_in_proj"] = layer_15_conv_in_proj;
  char *layer_15_conv_weight = static_cast<char*>(model_tensors.at("layer_15_conv_weight"));
  all_tensors["layer_15_conv_weight"] = layer_15_conv_weight;
  char *layer_15_conv_state = static_cast<char*>(model_tensors.at("layer_15_conv_state"));
  all_tensors["layer_15_conv_state"] = layer_15_conv_state;
  char *layer_15_conv_out_proj = static_cast<char*>(model_tensors.at("layer_15_conv_out_proj"));
  all_tensors["layer_15_conv_out_proj"] = layer_15_conv_out_proj;
  char *layer_15_ffn_norm = static_cast<char*>(model_tensors.at("layer_15_ffn_norm"));
  all_tensors["layer_15_ffn_norm"] = layer_15_ffn_norm;
  char *layer_15_router_gate = static_cast<char*>(model_tensors.at("layer_15_router_gate"));
  all_tensors["layer_15_router_gate"] = layer_15_router_gate;
  char *layer_15_expert_bias = static_cast<char*>(model_tensors.at("layer_15_expert_bias"));
  all_tensors["layer_15_expert_bias"] = layer_15_expert_bias;
  char *layer_15_experts_w13 = static_cast<char*>(model_tensors.at("layer_15_experts_w13"));
  all_tensors["layer_15_experts_w13"] = layer_15_experts_w13;
  char *layer_15_experts_w2 = static_cast<char*>(model_tensors.at("layer_15_experts_w2"));
  all_tensors["layer_15_experts_w2"] = layer_15_experts_w2;
  char *layer_16_operator_norm = static_cast<char*>(model_tensors.at("layer_16_operator_norm"));
  all_tensors["layer_16_operator_norm"] = layer_16_operator_norm;
  char *layer_16_conv_in_proj;
  CUDA_CHECK(cudaMalloc(&layer_16_conv_in_proj, 25165824));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_16_conv_in_proj + 0), 6291456, model_tensors.at("layer_16_conv_in_proj_B"), 2097152, 2097152, 4, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_16_conv_in_proj + 2097152), 6291456, model_tensors.at("layer_16_conv_in_proj_C"), 2097152, 2097152, 4, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_16_conv_in_proj + 4194304), 6291456, model_tensors.at("layer_16_conv_in_proj_x"), 2097152, 2097152, 4, cudaMemcpyDeviceToDevice));
  all_tensors["layer_16_conv_in_proj"] = layer_16_conv_in_proj;
  char *layer_16_conv_weight = static_cast<char*>(model_tensors.at("layer_16_conv_weight"));
  all_tensors["layer_16_conv_weight"] = layer_16_conv_weight;
  char *layer_16_conv_state = static_cast<char*>(model_tensors.at("layer_16_conv_state"));
  all_tensors["layer_16_conv_state"] = layer_16_conv_state;
  char *layer_16_conv_out_proj = static_cast<char*>(model_tensors.at("layer_16_conv_out_proj"));
  all_tensors["layer_16_conv_out_proj"] = layer_16_conv_out_proj;
  char *layer_16_ffn_norm = static_cast<char*>(model_tensors.at("layer_16_ffn_norm"));
  all_tensors["layer_16_ffn_norm"] = layer_16_ffn_norm;
  char *layer_16_router_gate = static_cast<char*>(model_tensors.at("layer_16_router_gate"));
  all_tensors["layer_16_router_gate"] = layer_16_router_gate;
  char *layer_16_expert_bias = static_cast<char*>(model_tensors.at("layer_16_expert_bias"));
  all_tensors["layer_16_expert_bias"] = layer_16_expert_bias;
  char *layer_16_experts_w13 = static_cast<char*>(model_tensors.at("layer_16_experts_w13"));
  all_tensors["layer_16_experts_w13"] = layer_16_experts_w13;
  char *layer_16_experts_w2 = static_cast<char*>(model_tensors.at("layer_16_experts_w2"));
  all_tensors["layer_16_experts_w2"] = layer_16_experts_w2;
  char *layer_17_operator_norm = static_cast<char*>(model_tensors.at("layer_17_operator_norm"));
  all_tensors["layer_17_operator_norm"] = layer_17_operator_norm;
  char *layer_17_conv_in_proj;
  CUDA_CHECK(cudaMalloc(&layer_17_conv_in_proj, 25165824));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_17_conv_in_proj + 0), 6291456, model_tensors.at("layer_17_conv_in_proj_B"), 2097152, 2097152, 4, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_17_conv_in_proj + 2097152), 6291456, model_tensors.at("layer_17_conv_in_proj_C"), 2097152, 2097152, 4, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_17_conv_in_proj + 4194304), 6291456, model_tensors.at("layer_17_conv_in_proj_x"), 2097152, 2097152, 4, cudaMemcpyDeviceToDevice));
  all_tensors["layer_17_conv_in_proj"] = layer_17_conv_in_proj;
  char *layer_17_conv_weight = static_cast<char*>(model_tensors.at("layer_17_conv_weight"));
  all_tensors["layer_17_conv_weight"] = layer_17_conv_weight;
  char *layer_17_conv_state = static_cast<char*>(model_tensors.at("layer_17_conv_state"));
  all_tensors["layer_17_conv_state"] = layer_17_conv_state;
  char *layer_17_conv_out_proj = static_cast<char*>(model_tensors.at("layer_17_conv_out_proj"));
  all_tensors["layer_17_conv_out_proj"] = layer_17_conv_out_proj;
  char *layer_17_ffn_norm = static_cast<char*>(model_tensors.at("layer_17_ffn_norm"));
  all_tensors["layer_17_ffn_norm"] = layer_17_ffn_norm;
  char *layer_17_router_gate = static_cast<char*>(model_tensors.at("layer_17_router_gate"));
  all_tensors["layer_17_router_gate"] = layer_17_router_gate;
  char *layer_17_expert_bias = static_cast<char*>(model_tensors.at("layer_17_expert_bias"));
  all_tensors["layer_17_expert_bias"] = layer_17_expert_bias;
  char *layer_17_experts_w13 = static_cast<char*>(model_tensors.at("layer_17_experts_w13"));
  all_tensors["layer_17_experts_w13"] = layer_17_experts_w13;
  char *layer_17_experts_w2 = static_cast<char*>(model_tensors.at("layer_17_experts_w2"));
  all_tensors["layer_17_experts_w2"] = layer_17_experts_w2;
  char *layer_18_operator_norm = static_cast<char*>(model_tensors.at("layer_18_operator_norm"));
  all_tensors["layer_18_operator_norm"] = layer_18_operator_norm;
  char *layer_18_qkv_proj;
  CUDA_CHECK(cudaMalloc(&layer_18_qkv_proj, 12582912));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_18_qkv_proj + 0), 1572864, model_tensors.at("layer_18_q_proj"), 1048576, 1048576, 8, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_18_qkv_proj + 1048576), 1572864, model_tensors.at("layer_18_k_proj"), 262144, 262144, 8, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_18_qkv_proj + 1310720), 1572864, model_tensors.at("layer_18_v_proj"), 262144, 262144, 8, cudaMemcpyDeviceToDevice));
  all_tensors["layer_18_qkv_proj"] = layer_18_qkv_proj;
  char *layer_18_q_layernorm = static_cast<char*>(model_tensors.at("layer_18_q_layernorm"));
  all_tensors["layer_18_q_layernorm"] = layer_18_q_layernorm;
  char *layer_18_k_layernorm = static_cast<char*>(model_tensors.at("layer_18_k_layernorm"));
  all_tensors["layer_18_k_layernorm"] = layer_18_k_layernorm;
  char *layer_18_k_cache = static_cast<char*>(model_tensors.at("layer_18_k_cache"));
  all_tensors["layer_18_k_cache"] = layer_18_k_cache;
  char *layer_18_v_cache = static_cast<char*>(model_tensors.at("layer_18_v_cache"));
  all_tensors["layer_18_v_cache"] = layer_18_v_cache;
  char *layer_18_out_proj = static_cast<char*>(model_tensors.at("layer_18_out_proj"));
  all_tensors["layer_18_out_proj"] = layer_18_out_proj;
  char *layer_18_ffn_norm = static_cast<char*>(model_tensors.at("layer_18_ffn_norm"));
  all_tensors["layer_18_ffn_norm"] = layer_18_ffn_norm;
  char *layer_18_router_gate = static_cast<char*>(model_tensors.at("layer_18_router_gate"));
  all_tensors["layer_18_router_gate"] = layer_18_router_gate;
  char *layer_18_expert_bias = static_cast<char*>(model_tensors.at("layer_18_expert_bias"));
  all_tensors["layer_18_expert_bias"] = layer_18_expert_bias;
  char *layer_18_experts_w13 = static_cast<char*>(model_tensors.at("layer_18_experts_w13"));
  all_tensors["layer_18_experts_w13"] = layer_18_experts_w13;
  char *layer_18_experts_w2 = static_cast<char*>(model_tensors.at("layer_18_experts_w2"));
  all_tensors["layer_18_experts_w2"] = layer_18_experts_w2;
  char *layer_19_operator_norm = static_cast<char*>(model_tensors.at("layer_19_operator_norm"));
  all_tensors["layer_19_operator_norm"] = layer_19_operator_norm;
  char *layer_19_conv_in_proj;
  CUDA_CHECK(cudaMalloc(&layer_19_conv_in_proj, 25165824));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_19_conv_in_proj + 0), 6291456, model_tensors.at("layer_19_conv_in_proj_B"), 2097152, 2097152, 4, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_19_conv_in_proj + 2097152), 6291456, model_tensors.at("layer_19_conv_in_proj_C"), 2097152, 2097152, 4, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_19_conv_in_proj + 4194304), 6291456, model_tensors.at("layer_19_conv_in_proj_x"), 2097152, 2097152, 4, cudaMemcpyDeviceToDevice));
  all_tensors["layer_19_conv_in_proj"] = layer_19_conv_in_proj;
  char *layer_19_conv_weight = static_cast<char*>(model_tensors.at("layer_19_conv_weight"));
  all_tensors["layer_19_conv_weight"] = layer_19_conv_weight;
  char *layer_19_conv_state = static_cast<char*>(model_tensors.at("layer_19_conv_state"));
  all_tensors["layer_19_conv_state"] = layer_19_conv_state;
  char *layer_19_conv_out_proj = static_cast<char*>(model_tensors.at("layer_19_conv_out_proj"));
  all_tensors["layer_19_conv_out_proj"] = layer_19_conv_out_proj;
  char *layer_19_ffn_norm = static_cast<char*>(model_tensors.at("layer_19_ffn_norm"));
  all_tensors["layer_19_ffn_norm"] = layer_19_ffn_norm;
  char *layer_19_router_gate = static_cast<char*>(model_tensors.at("layer_19_router_gate"));
  all_tensors["layer_19_router_gate"] = layer_19_router_gate;
  char *layer_19_expert_bias = static_cast<char*>(model_tensors.at("layer_19_expert_bias"));
  all_tensors["layer_19_expert_bias"] = layer_19_expert_bias;
  char *layer_19_experts_w13 = static_cast<char*>(model_tensors.at("layer_19_experts_w13"));
  all_tensors["layer_19_experts_w13"] = layer_19_experts_w13;
  char *layer_19_experts_w2 = static_cast<char*>(model_tensors.at("layer_19_experts_w2"));
  all_tensors["layer_19_experts_w2"] = layer_19_experts_w2;
  char *layer_20_operator_norm = static_cast<char*>(model_tensors.at("layer_20_operator_norm"));
  all_tensors["layer_20_operator_norm"] = layer_20_operator_norm;
  char *layer_20_conv_in_proj;
  CUDA_CHECK(cudaMalloc(&layer_20_conv_in_proj, 25165824));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_20_conv_in_proj + 0), 6291456, model_tensors.at("layer_20_conv_in_proj_B"), 2097152, 2097152, 4, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_20_conv_in_proj + 2097152), 6291456, model_tensors.at("layer_20_conv_in_proj_C"), 2097152, 2097152, 4, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_20_conv_in_proj + 4194304), 6291456, model_tensors.at("layer_20_conv_in_proj_x"), 2097152, 2097152, 4, cudaMemcpyDeviceToDevice));
  all_tensors["layer_20_conv_in_proj"] = layer_20_conv_in_proj;
  char *layer_20_conv_weight = static_cast<char*>(model_tensors.at("layer_20_conv_weight"));
  all_tensors["layer_20_conv_weight"] = layer_20_conv_weight;
  char *layer_20_conv_state = static_cast<char*>(model_tensors.at("layer_20_conv_state"));
  all_tensors["layer_20_conv_state"] = layer_20_conv_state;
  char *layer_20_conv_out_proj = static_cast<char*>(model_tensors.at("layer_20_conv_out_proj"));
  all_tensors["layer_20_conv_out_proj"] = layer_20_conv_out_proj;
  char *layer_20_ffn_norm = static_cast<char*>(model_tensors.at("layer_20_ffn_norm"));
  all_tensors["layer_20_ffn_norm"] = layer_20_ffn_norm;
  char *layer_20_router_gate = static_cast<char*>(model_tensors.at("layer_20_router_gate"));
  all_tensors["layer_20_router_gate"] = layer_20_router_gate;
  char *layer_20_expert_bias = static_cast<char*>(model_tensors.at("layer_20_expert_bias"));
  all_tensors["layer_20_expert_bias"] = layer_20_expert_bias;
  char *layer_20_experts_w13 = static_cast<char*>(model_tensors.at("layer_20_experts_w13"));
  all_tensors["layer_20_experts_w13"] = layer_20_experts_w13;
  char *layer_20_experts_w2 = static_cast<char*>(model_tensors.at("layer_20_experts_w2"));
  all_tensors["layer_20_experts_w2"] = layer_20_experts_w2;
  char *layer_21_operator_norm = static_cast<char*>(model_tensors.at("layer_21_operator_norm"));
  all_tensors["layer_21_operator_norm"] = layer_21_operator_norm;
  char *layer_21_qkv_proj;
  CUDA_CHECK(cudaMalloc(&layer_21_qkv_proj, 12582912));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_21_qkv_proj + 0), 1572864, model_tensors.at("layer_21_q_proj"), 1048576, 1048576, 8, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_21_qkv_proj + 1048576), 1572864, model_tensors.at("layer_21_k_proj"), 262144, 262144, 8, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_21_qkv_proj + 1310720), 1572864, model_tensors.at("layer_21_v_proj"), 262144, 262144, 8, cudaMemcpyDeviceToDevice));
  all_tensors["layer_21_qkv_proj"] = layer_21_qkv_proj;
  char *layer_21_q_layernorm = static_cast<char*>(model_tensors.at("layer_21_q_layernorm"));
  all_tensors["layer_21_q_layernorm"] = layer_21_q_layernorm;
  char *layer_21_k_layernorm = static_cast<char*>(model_tensors.at("layer_21_k_layernorm"));
  all_tensors["layer_21_k_layernorm"] = layer_21_k_layernorm;
  char *layer_21_k_cache = static_cast<char*>(model_tensors.at("layer_21_k_cache"));
  all_tensors["layer_21_k_cache"] = layer_21_k_cache;
  char *layer_21_v_cache = static_cast<char*>(model_tensors.at("layer_21_v_cache"));
  all_tensors["layer_21_v_cache"] = layer_21_v_cache;
  char *layer_21_out_proj = static_cast<char*>(model_tensors.at("layer_21_out_proj"));
  all_tensors["layer_21_out_proj"] = layer_21_out_proj;
  char *layer_21_ffn_norm = static_cast<char*>(model_tensors.at("layer_21_ffn_norm"));
  all_tensors["layer_21_ffn_norm"] = layer_21_ffn_norm;
  char *layer_21_router_gate = static_cast<char*>(model_tensors.at("layer_21_router_gate"));
  all_tensors["layer_21_router_gate"] = layer_21_router_gate;
  char *layer_21_expert_bias = static_cast<char*>(model_tensors.at("layer_21_expert_bias"));
  all_tensors["layer_21_expert_bias"] = layer_21_expert_bias;
  char *layer_21_experts_w13 = static_cast<char*>(model_tensors.at("layer_21_experts_w13"));
  all_tensors["layer_21_experts_w13"] = layer_21_experts_w13;
  char *layer_21_experts_w2 = static_cast<char*>(model_tensors.at("layer_21_experts_w2"));
  all_tensors["layer_21_experts_w2"] = layer_21_experts_w2;
  char *layer_22_operator_norm = static_cast<char*>(model_tensors.at("layer_22_operator_norm"));
  all_tensors["layer_22_operator_norm"] = layer_22_operator_norm;
  char *layer_22_conv_in_proj;
  CUDA_CHECK(cudaMalloc(&layer_22_conv_in_proj, 25165824));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_22_conv_in_proj + 0), 6291456, model_tensors.at("layer_22_conv_in_proj_B"), 2097152, 2097152, 4, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_22_conv_in_proj + 2097152), 6291456, model_tensors.at("layer_22_conv_in_proj_C"), 2097152, 2097152, 4, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_22_conv_in_proj + 4194304), 6291456, model_tensors.at("layer_22_conv_in_proj_x"), 2097152, 2097152, 4, cudaMemcpyDeviceToDevice));
  all_tensors["layer_22_conv_in_proj"] = layer_22_conv_in_proj;
  char *layer_22_conv_weight = static_cast<char*>(model_tensors.at("layer_22_conv_weight"));
  all_tensors["layer_22_conv_weight"] = layer_22_conv_weight;
  char *layer_22_conv_state = static_cast<char*>(model_tensors.at("layer_22_conv_state"));
  all_tensors["layer_22_conv_state"] = layer_22_conv_state;
  char *layer_22_conv_out_proj = static_cast<char*>(model_tensors.at("layer_22_conv_out_proj"));
  all_tensors["layer_22_conv_out_proj"] = layer_22_conv_out_proj;
  char *layer_22_ffn_norm = static_cast<char*>(model_tensors.at("layer_22_ffn_norm"));
  all_tensors["layer_22_ffn_norm"] = layer_22_ffn_norm;
  char *layer_22_router_gate = static_cast<char*>(model_tensors.at("layer_22_router_gate"));
  all_tensors["layer_22_router_gate"] = layer_22_router_gate;
  char *layer_22_expert_bias = static_cast<char*>(model_tensors.at("layer_22_expert_bias"));
  all_tensors["layer_22_expert_bias"] = layer_22_expert_bias;
  char *layer_22_experts_w13 = static_cast<char*>(model_tensors.at("layer_22_experts_w13"));
  all_tensors["layer_22_experts_w13"] = layer_22_experts_w13;
  char *layer_22_experts_w2 = static_cast<char*>(model_tensors.at("layer_22_experts_w2"));
  all_tensors["layer_22_experts_w2"] = layer_22_experts_w2;
  char *layer_23_operator_norm = static_cast<char*>(model_tensors.at("layer_23_operator_norm"));
  all_tensors["layer_23_operator_norm"] = layer_23_operator_norm;
  char *layer_23_conv_in_proj;
  CUDA_CHECK(cudaMalloc(&layer_23_conv_in_proj, 25165824));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_23_conv_in_proj + 0), 6291456, model_tensors.at("layer_23_conv_in_proj_B"), 2097152, 2097152, 4, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_23_conv_in_proj + 2097152), 6291456, model_tensors.at("layer_23_conv_in_proj_C"), 2097152, 2097152, 4, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>(layer_23_conv_in_proj + 4194304), 6291456, model_tensors.at("layer_23_conv_in_proj_x"), 2097152, 2097152, 4, cudaMemcpyDeviceToDevice));
  all_tensors["layer_23_conv_in_proj"] = layer_23_conv_in_proj;
  char *layer_23_conv_weight = static_cast<char*>(model_tensors.at("layer_23_conv_weight"));
  all_tensors["layer_23_conv_weight"] = layer_23_conv_weight;
  char *layer_23_conv_state = static_cast<char*>(model_tensors.at("layer_23_conv_state"));
  all_tensors["layer_23_conv_state"] = layer_23_conv_state;
  char *layer_23_conv_out_proj = static_cast<char*>(model_tensors.at("layer_23_conv_out_proj"));
  all_tensors["layer_23_conv_out_proj"] = layer_23_conv_out_proj;
  char *layer_23_ffn_norm = static_cast<char*>(model_tensors.at("layer_23_ffn_norm"));
  all_tensors["layer_23_ffn_norm"] = layer_23_ffn_norm;
  char *layer_23_router_gate = static_cast<char*>(model_tensors.at("layer_23_router_gate"));
  all_tensors["layer_23_router_gate"] = layer_23_router_gate;
  char *layer_23_expert_bias = static_cast<char*>(model_tensors.at("layer_23_expert_bias"));
  all_tensors["layer_23_expert_bias"] = layer_23_expert_bias;
  char *layer_23_experts_w13 = static_cast<char*>(model_tensors.at("layer_23_experts_w13"));
  all_tensors["layer_23_experts_w13"] = layer_23_experts_w13;
  char *layer_23_experts_w2 = static_cast<char*>(model_tensors.at("layer_23_experts_w2"));
  all_tensors["layer_23_experts_w2"] = layer_23_experts_w2;
  char *model_embedding_norm = static_cast<char*>(model_tensors.at("model_embedding_norm"));
  all_tensors["model_embedding_norm"] = model_embedding_norm;
  char *lm_head = static_cast<char*>(model_tensors.at("lm_head"));
  all_tensors["lm_head"] = lm_head;
  all_tensors["nullptr"] = nullptr;
  construct_task_graph(num_gpus, my_gpu_id, all_tasks, all_events, first_tasks, all_tensors);
  cudaDeviceSynchronize();
}

__device__ __forceinline__
void _execute_task(TaskDesc const* task_desc,
                   RuntimeConfig const &runtime_config) {
  if (task_desc->task_type == TASK_EMBEDDING && task_desc->variant_id == 0) {
      kernel::embedding_kernel<bfloat16, 64, 2048, 2048>(
      task_desc->input_ptrs[0],
      task_desc->input_ptrs[1],
      task_desc->output_ptrs[0]);

  }
  else if (task_desc->task_type == TASK_SILU_MUL && task_desc->variant_id == 0) {
      kernel::silu_mul_task_impl<bfloat16, 64, 224, 14336, 7168>(
      task_desc->input_ptrs[0],
      task_desc->output_ptrs[0],
      runtime_config.qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS]);

  }
  else if (task_desc->task_type == TASK_SILU_MUL && task_desc->variant_id == 1) {
      kernel::silu_mul_task_impl<bfloat16, 1, 1792, 3584, 1792>(
      task_desc->input_ptrs[0],
      task_desc->output_ptrs[0],
      1);

  }
  else if (task_desc->task_type == TASK_LFM2_CONV && task_desc->variant_id == 0) {
      kernel::lfm2_conv_task_impl<bfloat16, 512, 3, 6144, 2048, 3, 4096>(
      task_desc->input_ptrs[0],
      task_desc->input_ptrs[1],
      task_desc->input_ptrs[2],
      task_desc->output_ptrs[0],
      runtime_config.qo_indptr_buffer,
      runtime_config.step,
      task_desc->task_metadata.request_id);

  }
  else if (task_desc->task_type == TASK_RMS_NORM_HOPPER && task_desc->variant_id == 0) {
      kernel::rms_norm_hopper_impl<bfloat16, 1, 2048>(
      task_desc->input_ptrs[0],
      task_desc->input_ptrs[1],
      task_desc->output_ptrs[0],
      0.000010f);

  }
  else if (task_desc->task_type == TASK_SPLITK_LINEAR_SM100 && task_desc->variant_id == 0) {
      using TMA_A = kernel::tma::tma_2d<cute::bfloat16_t, 3, 3, 3, 128, 256, 128, 64, 2048, 1, 1, 1, 8192, true>;
  using TMA_B = kernel::tma::tma_2d<cute::bfloat16_t, 3, 3, 3, 64, 256, 16, 64, 2048, 1, 1, 1, 1024, true>;
  using TMA_OUT = kernel::tma::tma_2d<cute::bfloat16_t, 0, 3, 3, 64, 128, 16, 128, 2048, 1, 1, 1, 2048, true>;
    TMA_A tma_a(static_cast<CUtensorMap*>(task_desc->input_tma_desc_ptrs[1][0]));
    TMA_B tma_b(static_cast<CUtensorMap*>(task_desc->input_tma_desc_ptrs[0][0]));
    TMA_OUT tma_out(static_cast<CUtensorMap*>(task_desc->output_tma_desc_ptrs[0][0]));
    cute::Layout layout_Bias = cute::make_layout(cute::make_shape(64, 128), cute::make_stride(2048, cute::Int<1>{}));
    cute::Tensor mBias = cute::make_tensor(cute::make_gmem_ptr(static_cast<cute::bfloat16_t*>(nullptr)), layout_Bias);
    kernel::linear_sm100_mpk_task_impl<cute::bfloat16_t, TMA_A, TMA_B, decltype(mBias), TMA_OUT, 128, 16, 64, 128, 256, true, true, 8, 2, 4>(
        tma_a,
        tma_b,
        mBias,
        tma_out); 

  }
  else if (task_desc->task_type == TASK_SPLITK_LINEAR_SM100 && task_desc->variant_id == 1) {
      using TMA_A = kernel::tma::tma_2d<cute::bfloat16_t, 3, 3, 3, 128, 896, 128, 64, 7168, 1, 1, 1, 8192, true>;
  using TMA_B = kernel::tma::tma_2d<cute::bfloat16_t, 3, 3, 3, 64, 896, 16, 64, 7168, 1, 1, 1, 1024, true>;
  using TMA_OUT = kernel::tma::tma_2d<cute::bfloat16_t, 0, 3, 3, 64, 128, 16, 128, 2048, 1, 1, 1, 2048, true>;
    TMA_A tma_a(static_cast<CUtensorMap*>(task_desc->input_tma_desc_ptrs[1][0]));
    TMA_B tma_b(static_cast<CUtensorMap*>(task_desc->input_tma_desc_ptrs[0][0]));
    TMA_OUT tma_out(static_cast<CUtensorMap*>(task_desc->output_tma_desc_ptrs[0][0]));
    cute::Layout layout_Bias = cute::make_layout(cute::make_shape(64, 128), cute::make_stride(2048, cute::Int<1>{}));
    cute::Tensor mBias = cute::make_tensor(cute::make_gmem_ptr(static_cast<cute::bfloat16_t*>(nullptr)), layout_Bias);
    kernel::linear_sm100_mpk_task_impl<cute::bfloat16_t, TMA_A, TMA_B, decltype(mBias), TMA_OUT, 128, 16, 64, 128, 896, true, true, 8, 2, 4>(
        tma_a,
        tma_b,
        mBias,
        tma_out); 

  }
  else if (task_desc->task_type == TASK_LINEAR_SM100 && task_desc->variant_id == 0) {
      using TMA_A = kernel::tma::tma_2d<cute::bfloat16_t, 3, 3, 3, 64, 2048, 128, 64, 2048, 1, 1, 1, 8192, true>;
  using TMA_B = kernel::tma::tma_2d<cute::bfloat16_t, 3, 3, 3, 64, 2048, 16, 64, 2048, 1, 1, 1, 1024, true>;
  using TMA_OUT = kernel::tma::tma_2d<cute::bfloat16_t, 0, 3, 3, 64, 64, 16, 128, 6144, 1, 1, 1, 2048, true>;
    TMA_A tma_a(static_cast<CUtensorMap*>(task_desc->input_tma_desc_ptrs[1][0]));
    TMA_B tma_b(static_cast<CUtensorMap*>(task_desc->input_tma_desc_ptrs[0][0]));
    TMA_OUT tma_out(static_cast<CUtensorMap*>(task_desc->output_tma_desc_ptrs[0][0]));
    cute::Layout layout_Bias = cute::make_layout(cute::make_shape(64, 64), cute::make_stride(6144, cute::Int<1>{}));
    cute::Tensor mBias = cute::make_tensor(cute::make_gmem_ptr(static_cast<cute::bfloat16_t*>(nullptr)), layout_Bias);
    kernel::linear_sm100_mpk_task_impl<cute::bfloat16_t, TMA_A, TMA_B, decltype(mBias), TMA_OUT, 128, 16, 64, 64, 2048, true, false, 8, 2, 4>(
        tma_a,
        tma_b,
        mBias,
        tma_out); 

  }
  else if (task_desc->task_type == TASK_LINEAR_SM100 && task_desc->variant_id == 1) {
      using TMA_A = kernel::tma::tma_2d<cute::bfloat16_t, 3, 3, 3, 224, 2048, 128, 64, 2048, 1, 1, 1, 8192, true>;
  using TMA_B = kernel::tma::tma_2d<cute::bfloat16_t, 3, 3, 3, 64, 2048, 16, 64, 2048, 1, 1, 1, 1024, true>;
  using TMA_OUT = kernel::tma::tma_2d<cute::bfloat16_t, 0, 3, 3, 64, 224, 16, 128, 14336, 1, 1, 1, 2048, true>;
    TMA_A tma_a(static_cast<CUtensorMap*>(task_desc->input_tma_desc_ptrs[1][0]));
    TMA_B tma_b(static_cast<CUtensorMap*>(task_desc->input_tma_desc_ptrs[0][0]));
    TMA_OUT tma_out(static_cast<CUtensorMap*>(task_desc->output_tma_desc_ptrs[0][0]));
    cute::Layout layout_Bias = cute::make_layout(cute::make_shape(64, 224), cute::make_stride(14336, cute::Int<1>{}));
    cute::Tensor mBias = cute::make_tensor(cute::make_gmem_ptr(static_cast<cute::bfloat16_t*>(nullptr)), layout_Bias);
    kernel::linear_sm100_mpk_task_impl<cute::bfloat16_t, TMA_A, TMA_B, decltype(mBias), TMA_OUT, 128, 16, 64, 224, 2048, true, false, 8, 2, 4>(
        tma_a,
        tma_b,
        mBias,
        tma_out); 

  }
  else if (task_desc->task_type == TASK_LINEAR_SM100 && task_desc->variant_id == 2) {
      using TMA_A = kernel::tma::tma_2d<cute::bfloat16_t, 3, 3, 3, 32, 2048, 128, 64, 2048, 1, 1, 1, 8192, true>;
  using TMA_B = kernel::tma::tma_2d<cute::bfloat16_t, 3, 3, 3, 64, 2048, 16, 64, 2048, 1, 1, 1, 1024, true>;
  using TMA_OUT = kernel::tma::tma_2d<cute::bfloat16_t, 0, 3, 3, 64, 32, 16, 128, 3072, 1, 1, 1, 2048, true>;
    TMA_A tma_a(static_cast<CUtensorMap*>(task_desc->input_tma_desc_ptrs[1][0]));
    TMA_B tma_b(static_cast<CUtensorMap*>(task_desc->input_tma_desc_ptrs[0][0]));
    TMA_OUT tma_out(static_cast<CUtensorMap*>(task_desc->output_tma_desc_ptrs[0][0]));
    cute::Layout layout_Bias = cute::make_layout(cute::make_shape(64, 32), cute::make_stride(3072, cute::Int<1>{}));
    cute::Tensor mBias = cute::make_tensor(cute::make_gmem_ptr(static_cast<cute::bfloat16_t*>(nullptr)), layout_Bias);
    kernel::linear_sm100_mpk_task_impl<cute::bfloat16_t, TMA_A, TMA_B, decltype(mBias), TMA_OUT, 128, 16, 64, 32, 2048, true, false, 8, 2, 4>(
        tma_a,
        tma_b,
        mBias,
        tma_out); 

  }
  else if (task_desc->task_type == TASK_LINEAR_SM100 && task_desc->variant_id == 3) {
      using TMA_A = kernel::tma::tma_2d<cute::bfloat16_t, 3, 3, 3, 8, 2048, 128, 64, 2048, 1, 1, 1, 8192, true>;
  using TMA_B = kernel::tma::tma_2d<cute::bfloat16_t, 3, 3, 3, 64, 2048, 16, 64, 2048, 1, 1, 1, 1024, true>;
  using TMA_OUT = kernel::tma::tma_2d<cute::bfloat16_t, 0, 3, 3, 64, 8, 16, 128, 32, 1, 1, 1, 2048, true>;
    TMA_A tma_a(static_cast<CUtensorMap*>(task_desc->input_tma_desc_ptrs[1][0]));
    TMA_B tma_b(static_cast<CUtensorMap*>(task_desc->input_tma_desc_ptrs[0][0]));
    TMA_OUT tma_out(static_cast<CUtensorMap*>(task_desc->output_tma_desc_ptrs[0][0]));
    cute::Layout layout_Bias = cute::make_layout(cute::make_shape(64, 8), cute::make_stride(32, cute::Int<1>{}));
    cute::Tensor mBias = cute::make_tensor(cute::make_gmem_ptr(static_cast<cute::bfloat16_t*>(nullptr)), layout_Bias);
    kernel::linear_sm100_mpk_task_impl<cute::bfloat16_t, TMA_A, TMA_B, decltype(mBias), TMA_OUT, 128, 16, 64, 8, 2048, true, false, 8, 2, 4>(
        tma_a,
        tma_b,
        mBias,
        tma_out); 

  }
  else if (task_desc->task_type == TASK_LINEAR_SM100 && task_desc->variant_id == 4) {
      using TMA_A = kernel::tma::tma_2d<cute::bfloat16_t, 3, 3, 3, 256, 2048, 128, 64, 2048, 1, 1, 1, 8192, true>;
  using TMA_B = kernel::tma::tma_2d<cute::bfloat16_t, 3, 3, 3, 64, 2048, 16, 64, 2048, 1, 1, 1, 1024, true>;
  using TMA_OUT = kernel::tma::tma_2d<cute::bfloat16_t, 0, 3, 3, 64, 256, 16, 128, 128000, 1, 1, 1, 2048, true>;
    TMA_A tma_a(static_cast<CUtensorMap*>(task_desc->input_tma_desc_ptrs[1][0]));
    TMA_B tma_b(static_cast<CUtensorMap*>(task_desc->input_tma_desc_ptrs[0][0]));
    TMA_OUT tma_out(static_cast<CUtensorMap*>(task_desc->output_tma_desc_ptrs[0][0]));
    cute::Layout layout_Bias = cute::make_layout(cute::make_shape(64, 256), cute::make_stride(128000, cute::Int<1>{}));
    cute::Tensor mBias = cute::make_tensor(cute::make_gmem_ptr(static_cast<cute::bfloat16_t*>(nullptr)), layout_Bias);
    kernel::linear_sm100_mpk_task_impl<cute::bfloat16_t, TMA_A, TMA_B, decltype(mBias), TMA_OUT, 128, 16, 64, 256, 2048, true, false, 8, 2, 4>(
        tma_a,
        tma_b,
        mBias,
        tma_out); 

  }
  else if (task_desc->task_type == TASK_MOE_W13_LINEAR_SM100 && task_desc->variant_id == 0) {
      using TMA_A = kernel::tma::tma_2d<cute::bfloat16_t, 3, 3, 3, 114688, 2048, 128, 64, 2048, 1, 1, 1, 8192, true>;
    TMA_A tma_a(static_cast<CUtensorMap*>(task_desc->input_tma_desc_ptrs[1][0]));
    cute::Layout layout_Bias = cute::make_layout(cute::make_shape(64, 3584, 32), cute::make_stride(3584, cute::Int<1>{}, 229376));
    cute::Tensor mBias = cute::make_tensor(cute::make_gmem_ptr(static_cast<cute::bfloat16_t*>(nullptr)), layout_Bias);
    cute::Layout layout_routing_indices = cute::make_layout(cute::make_shape(32, 64), cute::make_stride(64, cute::Int<1>{}));
    cute::Tensor mRoutingIndices = cute::make_tensor(cute::make_gmem_ptr(static_cast<cute::int32_t*>(task_desc->input_ptrs[2])), layout_routing_indices);
    cute::Layout layout_expert_mask = cute::make_layout(cute::make_shape(33), cute::make_stride(cute::Int<1>{}));
    cute::Tensor mMask = cute::make_tensor(cute::make_gmem_ptr(static_cast<cute::int32_t*>(task_desc->input_ptrs[3])), layout_expert_mask);
    cute::Layout layout_output = cute::make_layout(cute::make_shape(64, 4, 3584), cute::make_stride(14336, 3584, cute::Int<1>{}));
    cute::Tensor mOutput = cute::make_tensor(cute::make_gmem_ptr(static_cast<cute::bfloat16_t*>(task_desc->output_ptrs[0])), layout_output);
    cute::Layout layout_input = cute::make_layout(cute::make_shape(64, 2048), cute::make_stride(2048, cute::Int<1>{}));
    cute::Tensor mInput = cute::make_tensor(cute::make_gmem_ptr(static_cast<cute::bfloat16_t*>(task_desc->input_ptrs[0])), layout_input);
    kernel::moe_linear_sm100_task_impl<cute::bfloat16_t, TMA_A, decltype(mInput), decltype(mBias), decltype(mRoutingIndices), decltype(mMask), decltype(mOutput), 128, 16, 64, 3584, 3584, 2048, 32, 4, 10, true, true, 8, 2, 4>(
        tma_a,
        mInput,
        mBias,
        mRoutingIndices,
        mMask,
        mOutput,
        task_desc->task_metadata.expert_offset);

  }
  else if (task_desc->task_type == TASK_MOE_W2_LINEAR_SM100 && task_desc->variant_id == 0) {
      using TMA_A = kernel::tma::tma_2d<cute::bfloat16_t, 3, 3, 3, 65536, 1792, 128, 64, 1792, 1, 1, 1, 8192, true>;
    TMA_A tma_a(static_cast<CUtensorMap*>(task_desc->input_tma_desc_ptrs[1][0]));
    cute::Layout layout_Bias = cute::make_layout(cute::make_shape(64, 2048, 32), cute::make_stride(2048, cute::Int<1>{}, 131072));
    cute::Tensor mBias = cute::make_tensor(cute::make_gmem_ptr(static_cast<cute::bfloat16_t*>(nullptr)), layout_Bias);
    cute::Layout layout_routing_indices = cute::make_layout(cute::make_shape(32, 64), cute::make_stride(64, cute::Int<1>{}));
    cute::Tensor mRoutingIndices = cute::make_tensor(cute::make_gmem_ptr(static_cast<cute::int32_t*>(task_desc->input_ptrs[2])), layout_routing_indices);
    cute::Layout layout_expert_mask = cute::make_layout(cute::make_shape(33), cute::make_stride(cute::Int<1>{}));
    cute::Tensor mMask = cute::make_tensor(cute::make_gmem_ptr(static_cast<cute::int32_t*>(task_desc->input_ptrs[3])), layout_expert_mask);
    cute::Layout layout_output = cute::make_layout(cute::make_shape(64, 4, 2048), cute::make_stride(8192, 2048, cute::Int<1>{}));
    cute::Tensor mOutput = cute::make_tensor(cute::make_gmem_ptr(static_cast<cute::bfloat16_t*>(task_desc->output_ptrs[0])), layout_output);
    cute::Layout layout_input = cute::make_layout(cute::make_shape(64, 1792, 4), cute::make_stride(7168, cute::Int<1>{}, 1792));
    cute::Tensor mInput = cute::make_tensor(cute::make_gmem_ptr(static_cast<cute::bfloat16_t*>(task_desc->input_ptrs[0])), layout_input);
    kernel::moe_linear_sm100_task_impl<cute::bfloat16_t, TMA_A, decltype(mInput), decltype(mBias), decltype(mRoutingIndices), decltype(mMask), decltype(mOutput), 128, 16, 64, 2048, 2048, 1792, 32, 4, 8, false, true, 8, 2, 4>(
        tma_a,
        mInput,
        mBias,
        mRoutingIndices,
        mMask,
        mOutput,
        task_desc->task_metadata.expert_offset);

  }
  else if (task_desc->task_type == TASK_ATTN_SM100 && task_desc->variant_id == 0) {
      kernel::multitoken_paged_attention_sm100_task_impl<bfloat16, 4, 1, 512, 3072, 2048, 64, 512, 4096, 0, 0, 64>(
      task_desc->input_ptrs[0],
      task_desc->input_ptrs[1],
      task_desc->input_ptrs[2],
      task_desc->output_ptrs[0],
      runtime_config.qo_indptr_buffer,
      runtime_config.paged_kv_indptr_buffer,
      runtime_config.paged_kv_indices_buffer,
      runtime_config.paged_kv_last_page_len_buffer,
      task_desc->task_metadata.request_id,
      true,
      true,
      task_desc->input_ptrs[3],
      task_desc->input_ptrs[4],
      task_desc->input_ptrs[5],
      task_desc->input_ptrs[6],
      1e-6f,
      1e-6f);

  }
  else if (task_desc->task_type == TASK_ARGMAX_REDUCE_SM100 && task_desc->variant_id == 0) {
      kernel::argmax_reduce_sm100_kernel<bfloat16, 64, 1000, 128>(
      task_desc->input_ptrs[0],
      task_desc->input_ptrs[1],
      task_desc->output_ptrs[0],
      runtime_config.qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS]);

  }
  else if (task_desc->task_type == TASK_ARGMAX_PARTIAL_SM100 && task_desc->variant_id == 0) {
      kernel::argmax_partial_sm100_kernel<bfloat16, 64, 1000, 128>(
      task_desc->input_ptrs[0],
      task_desc->output_ptrs[0],
      task_desc->output_ptrs[1],
      runtime_config.qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS]);

  }
  else if (task_desc->task_type == TASK_MOE_MUL_SUM_ADD_SM100 && task_desc->variant_id == 0) {
      kernel::mul_sum_add_sm100_task_impl<cute::bfloat16_t, 1, 2048, 4, 2048>(
      task_desc->input_ptrs[0],
      task_desc->input_ptrs[1],
      task_desc->input_ptrs[2],
      task_desc->output_ptrs[0]);

  }
  else if (task_desc->task_type == TASK_MOE_TOPK_SIGMOID_SM100 && task_desc->variant_id == 0) {
      kernel::topk_sigmoid_task_impl<cute::bfloat16_t, 8, 32, 8, 16, 1, 1, 32, 4>(
      task_desc->input_ptrs[0],
      task_desc->input_ptrs[1],
      nullptr,
      task_desc->output_ptrs[0],
      64,
      task_desc->output_ptrs[1],
      task_desc->output_ptrs[2],
      0,
      32,
      1.000000f);

  }
}

#include <Python.h>
#include <cuda_runtime.h>
#include <string>
#include <vector>

extern std::string g_task_graph_json_path;

static PyObject *init_func(PyObject *self, PyObject *args) {
  PyObject *meta_list, *py_profiler_buffer, *tensor_names_list, *tensor_ptrs_list, *py_json_path;
  std::vector<void*> meta_tensors;
  std::vector<std::string> model_tensor_names;
  std::vector<void*> model_tensor_ptrs;
  int my_mpi_rank, num_workers, num_local_schedulers, num_remote_schedulers, max_seq_length, total_num_requests;
  long long eos_token_id;
  int allocate_nvshmem_teams;
  void *profiler_buffer;

  if (!PyArg_ParseTuple(args, "OOiiiiiiLiOOO", &meta_list, &py_profiler_buffer, &my_mpi_rank, &num_workers, &num_local_schedulers, &num_remote_schedulers, &max_seq_length, &total_num_requests, &eos_token_id, &allocate_nvshmem_teams, &tensor_names_list, &tensor_ptrs_list, &py_json_path)) {
    PyErr_SetString(PyExc_TypeError, "Invalid parameters");
    return NULL;
  }

  if(!PyList_Check(meta_list)) {
    PyErr_SetString(PyExc_TypeError, "arg1 must be a list.");
    return NULL;
  }
  if(!PyList_Check(tensor_names_list)) {
    PyErr_SetString(PyExc_TypeError, "tensor_names must be a list.");
    return NULL;
  }
  if(!PyList_Check(tensor_ptrs_list)) {
    PyErr_SetString(PyExc_TypeError, "tensor_ptrs must be a list.");
    return NULL;
  }

  Py_ssize_t meta_size = PyList_Size(meta_list);
  for(Py_ssize_t i = 0; i < meta_size; i++) {
    PyObject *item = PyList_GetItem(meta_list, i);
    void* tensor = PyLong_AsVoidPtr(item);
    if(!tensor) {
      PyErr_Format(PyExc_TypeError, "Failed to convert item %d (meta) to void pointer", i);
      return NULL;
    }
    meta_tensors.push_back(PyLong_AsVoidPtr(item));
  }
  profiler_buffer = PyLong_AsVoidPtr(py_profiler_buffer);

  Py_ssize_t num_tensors = PyList_Size(tensor_names_list);
  for(Py_ssize_t i = 0; i < num_tensors; i++) {
    PyObject *name_item = PyList_GetItem(tensor_names_list, i);
    PyObject *ptr_item = PyList_GetItem(tensor_ptrs_list, i);
    
    const char *name_str = PyUnicode_AsUTF8(name_item);
    if (!name_str) {
      PyErr_Format(PyExc_TypeError, "Failed to convert tensor name %d to string", i);
      return NULL;
    }
    model_tensor_names.push_back(std::string(name_str));
    
    void *ptr = PyLong_AsVoidPtr(ptr_item);
    model_tensor_ptrs.push_back(ptr);
  }

  if (PyUnicode_Check(py_json_path)) {
    const char *json_path = PyUnicode_AsUTF8(py_json_path);
    if (json_path && strlen(json_path) > 0) {
      g_task_graph_json_path = std::string(json_path);
    }
  }

  init_persistent_kernel(meta_tensors, profiler_buffer, my_mpi_rank, num_workers, num_local_schedulers, num_remote_schedulers, max_seq_length, total_num_requests, eos_token_id, allocate_nvshmem_teams, model_tensor_names, model_tensor_ptrs);

  Py_RETURN_NONE;
}

static PyObject *init_request_func(PyObject *self, PyObject *args) {
  Py_BEGIN_ALLOW_THREADS
  init_request_resources();
  Py_END_ALLOW_THREADS
  Py_RETURN_NONE;
}

static PyObject *launch_func(PyObject *self, PyObject *args) {
  PyObject *py_stream;
  cudaStream_t stream;
  if (!PyArg_ParseTuple(args, "O", &py_stream)) {
    PyErr_SetString(PyExc_TypeError, "Invalid parameters");
    return NULL;
  }
  stream = (cudaStream_t)PyLong_AsVoidPtr(py_stream);
  launch_persistent_kernel(stream);

  Py_RETURN_NONE;
}

static PyObject *finalize_func(PyObject *self, PyObject *args) {
  finalize_persistent_kernel();

  Py_RETURN_NONE;
}

static PyMethodDef ModuleMethods[] = {
  {"init_func", init_func, METH_VARARGS, "initialize persistent kernel"},
  {"init_request_func", init_request_func, METH_VARARGS, "initialize request resources"},
  {"launch_func", launch_func, METH_VARARGS, "launch persistent kernel"},
  {"finalize_func", finalize_func, METH_VARARGS, "finalize persistent kernel"},
  {NULL, NULL, 0, NULL} // sentinel
};

static struct PyModuleDef ModuleDef = {
  PyModuleDef_HEAD_INIT,
  "__mirage_launcher",
  NULL, //documentation
  -1, //size
  ModuleMethods,
  NULL, // m_slots
  NULL, // m_traverse
  NULL, // m_clear
  NULL  // m_free
};

PyMODINIT_FUNC PyInit___mirage_launcher(void) {
  PyObject *m = PyModule_Create(&ModuleDef);
  if(m == NULL) {
    return NULL;
  }
  PyModule_AddFunctions(m, ModuleMethods);
  return m;
}
