#include "multi-device-model-compiler/Kernels/CPU/Ops.h"

#include "oneapi/dnnl/dnnl_graph.hpp"

#include "xxhash.h"

#include <unordered_map>
#include <unordered_set>

using namespace dnnl::graph;
using data_type = logical_tensor::data_type;
using layout_type = logical_tensor::layout_type;
using dim = logical_tensor::dim;
using dims = logical_tensor::dims;

static engine LLMCpuEngine;
static stream LLMCpuStream;

void cpuLLMOpsInit() {
  LLMCpuEngine = getOpsEngine();
  LLMCpuStream = stream(LLMCpuEngine);
}

void cpyLLMOpsDeinit() {}

/// Set any layout according to the connection relationship of partitions
///
/// @param partitions a list of partitions
/// @param id_to_set_any_layout a set of ids of logical tensors with any layout
///     type
void set_any_layout(const std::vector<dnnl::graph::partition> &partitions,
                    std::unordered_set<size_t> &id_to_set_any_layout) {
  // mapping from output tensor id to the all supported flags of
  // supported partitions, we may only need outputs' supported flags
  std::unordered_map<size_t, std::vector<bool>> output_to_flag_map;
  for (const auto &p : partitions) {
    for (const auto &out : p.get_output_ports()) {
      size_t id = out.get_id();
      if (p.is_supported() &&
          output_to_flag_map.find(id) == output_to_flag_map.end()) {
        output_to_flag_map[id] = {};
      }
    }

    for (const auto &in : p.get_input_ports()) {
      size_t id = in.get_id();
      auto iter = output_to_flag_map.find(id);
      if (iter != output_to_flag_map.end()) {
        // collect all of supported flags of this tensor's uses
        // Considering we have such a graph:
        //
        //   partition_A  partition_B
        //        \           |
        //      tensor1    tensor2
        //           \     /     |
        //         partition_C  unsupported partition
        //              |
        //           tensor3
        //              |
        //          framework op
        //
        // so the mapping of partition_A's output will be { true }
        // the mapping of partition_B's output will be { true, false }
        // The mapping of partition_C's output will be { false }
        // Only when all supported flags are true, users can set any
        // layout.
        iter->second.push_back(p.is_supported());
      }
    }
  }

  for (const auto &p : partitions) {
    // no need to set `any` layout if this partition is not supported
    if (!p.is_supported())
      continue;
    for (const auto &in : p.get_input_ports()) {
      size_t id = in.get_id();
      auto iter = output_to_flag_map.find(id);
      // if this input tensor is not an output of another supported
      // partition, just skip
      if (iter == output_to_flag_map.end())
        continue;
      std::vector<bool> flag_vec = iter->second;
      // check if all of uses of this tensor are supported partitions,
      // if not, no need to set ANY layout.
      bool need_set_any = std::all_of(flag_vec.begin(), flag_vec.end(),
                                      [](const bool a) { return a; });
      if (!need_set_any)
        continue;

      /// record the id of logical tensor that will be set to ANY layout
      id_to_set_any_layout.insert(id);
    }
  }
}

struct cpu_deletor {
  cpu_deletor() = default;
  void operator()(void *ptr) {
    if (ptr)
      free(ptr);
  }
};

void allocate_graph_mem(
    std::vector<dnnl::graph::tensor> &tensors,
    const std::vector<dnnl::graph::logical_tensor> &lts,
    std::vector<std::shared_ptr<void>> &data_buffer,
    std::unordered_map<size_t, void *> &io_func,
    std::unordered_map<size_t, dnnl::graph::tensor> &global_outputs_ts_map,
    bool is_input) {
  tensors.reserve(lts.size());
  for (const auto &lt : lts) {
    const auto mem_size = lt.get_mem_size();
    const auto lt_id = lt.get_id();
    if (io_func.count(lt_id)) {
      tensor new_ts{lt, LLMCpuEngine, io_func[lt_id]};
      continue;
    }

    if (is_input) {
      auto pos = global_outputs_ts_map.find(lt_id);
      if (pos != global_outputs_ts_map.end()) {
        tensors.push_back(pos->second);
        continue;
      }
    }

    data_buffer.push_back({});
    data_buffer.back().reset(malloc(mem_size), cpu_deletor{});

    tensor new_ts{lt, LLMCpuEngine, data_buffer.back().get()};
    tensors.push_back(new_ts);

    global_outputs_ts_map[lt_id] = tensors.back();
  }
}

void splitQKV(
    std::vector<dnnl::graph::tensor> &tensors,
    const std::vector<dnnl::graph::logical_tensor> &output_lts,
    std::unordered_map<size_t, dnnl::graph::tensor> &global_outputs_ts_map,
    const logical_tensor &input_lt) {
  const auto id = input_lt.get_id();
  auto input = global_outputs_ts_map[id];
  tensors.reserve(3);

  auto input_data = input.get_data_handle();
  auto output_mem_size = input_lt.get_mem_size() / 3;

  void *q_data = input_data;
  void *k_data = (char *)input_data + output_mem_size;
  void *v_data = (char *)input_data + 2 * output_mem_size;

  tensor q_ts{output_lts[0], LLMCpuEngine, q_data};
  tensor k_ts{output_lts[1], LLMCpuEngine, k_data};
  tensor v_ts{output_lts[2], LLMCpuEngine, v_data};

  tensors.push_back(q_ts);
  global_outputs_ts_map[output_lts[0].get_id()] = tensors.back();
  tensors.push_back(k_ts);
  global_outputs_ts_map[output_lts[1].get_id()] = tensors.back();
  tensors.push_back(v_ts);
  global_outputs_ts_map[output_lts[2].get_id()] = tensors.back();
}

struct partition_combines {
  partition part;
  compiled_partition cp;
  bool support;
};

static std::unordered_map<int64_t, std::vector<partition_combines>>
    DecodingContextLayerMap;

extern "C" MLIR_CPU_OPS_EXPORT void mcpuLLMDecodingContextLayer(
    float *input, float *qkv, float *gemm_weight, float *gemm_bias,
    float *ffn1_weight, float *ffn1_bias, float *ffn2_weight, float *ffn2_bias,
    float *output, int64_t batch, int64_t seq_len, int64_t d_model,
    int64_t feed_forward_dim, int64_t head_num, bool norm_first, bool is_casual,
    bool is_relu) {
  std::array<int64_t, 8> keys{batch,    seq_len,    d_model,   feed_forward_dim,
                              head_num, norm_first, is_casual, is_relu};
  int64_t decodingKey = XXH3_64bits(keys.data(), keys.size() * sizeof(int64_t));

  std::vector<partition_combines> infos;
  if (DecodingContextLayerMap.count(decodingKey)) {
    infos = DecodingContextLayerMap[decodingKey];
  } else {

    int64_t d_head = d_model / head_num;

    dim Batch = batch, Seq = seq_len, Dmodel = d_model, Dhead = d_head,
        Nhead = head_num, Feed = feed_forward_dim, QKV = 3 * d_model;

    dims input_dims{Batch, Seq, Dmodel};
    dims qkv_dims{Dmodel, QKV};
    dims input_qkv_dims{Batch, Seq, QKV};
    dims input_qkv_reshape_dims{Batch, Seq, 3, Nhead, Dhead};
    dims input_qkv_transpose_dims{3, Batch, Nhead, Seq, Dhead};
    dims input_q_dims{Batch, Nhead, Seq, Dhead};
    dims input_k_transpose_dims{Batch, Nhead, Dhead, Seq};
    dims input_qk_dims{Batch, Nhead, Seq, Seq};
    dims input_attn_mlp_dims{Batch, Seq, Nhead, Dhead};
    dims mlp_weight_dims{Dmodel, Dmodel};
    dims mlp_bias_dims{Dmodel};
    dims ffn1_weight_dims{Dmodel, Feed};
    dims ffn1_bias_dims{Feed};
    dims ffn1_output_dims{Batch, Seq, Feed};
    dims ffn2_weight_dims{Feed, Dmodel};
    dims ffn2_bias_dims{Dmodel};

    // input convert
    logical_tensor input_src_desc{0, data_type::f32};
    logical_tensor input_h_src_desc{1, data_type::f16};

    op convert_input(0, op::kind::TypeCast, {input_src_desc},
                     {input_h_src_desc}, "convert_input");

    // input layernorm
    logical_tensor input_layernorm_h_src_desc{2, data_type::f16};
    op layernorm_input(1, op::kind::LayerNorm, {input_h_src_desc},
                       {input_layernorm_h_src_desc}, "layernorm_input");
    layernorm_input.set_attr<bool>(op::attr::keep_stats, false);
    layernorm_input.set_attr<bool>(op::attr::use_affine, false);

    // qkv convert
    logical_tensor qkv_desc{36, data_type::f32};
    logical_tensor qkv_h_desc{37, data_type::f16};

    op convert_qkv(2, op::kind::TypeCast, {qkv_desc}, {qkv_h_desc},
                   "convert_qkv");

    // qkv gemm
    logical_tensor input_qkv_h_desc{3, data_type::f16};

    op gemm_qkv(3, op::kind::MatMul, {input_layernorm_h_src_desc, qkv_h_desc},
                {input_qkv_h_desc}, "gemm_qkv");

    // input qkv reshape
    logical_tensor input_qkv_reshape_h_desc{4, data_type::f16};

    op reshape_qkv(4, op::kind::StaticReshape, {input_qkv_h_desc},
                   {input_qkv_reshape_h_desc}, "reshape_qkv");
    reshape_qkv.set_attr<dims>(op::attr::shape, {Batch, Seq, 3, Nhead, Dhead});
    reshape_qkv.set_attr<bool>(op::attr::special_zero, false);

    // input qkv transpose
    logical_tensor input_qkv_transpose_h_desc{5, data_type::f16};

    op transpose_qkv(5, op::kind::StaticTranspose, {input_qkv_reshape_h_desc},
                     {input_qkv_transpose_h_desc}, "transpose_qkv");
    transpose_qkv.set_attr<dims>(op::attr::order, {2, 0, 3, 1, 4});

    // input qkv split
    logical_tensor input_q_desc{6, data_type::f16};
    logical_tensor input_k_desc{7, data_type::f16};
    logical_tensor input_v_desc{8, data_type::f16};

    op split_qkv(6, op::kind::Wildcard, {input_qkv_transpose_h_desc},
                 {input_q_desc, input_k_desc, input_v_desc}, "split_qkv");

    // input qk gemm
    logical_tensor input_qk_desc{10, data_type::f16};

    op gemm_qk(8, op::kind::MatMul, {input_q_desc, input_k_desc},
               {input_qk_desc}, "gemm_qk");
    gemm_qk.set_attr<bool>(op::attr::transpose_b, true);

    // todo: add mask

    // softmax
    logical_tensor input_softmax_desc{13, data_type::f16};

    op softmax(9, op::kind::SoftMax, {input_qk_desc}, {input_softmax_desc},
               "softmax");
    softmax.set_attr<int64_t>(op::attr::axis, 3);

    // qk*v
    logical_tensor input_attn_desc{14, data_type::f16};

    op gemm_attn(10, op::kind::MatMul, {input_softmax_desc, input_v_desc},
                 {input_attn_desc}, "gemm_attn");

    // attn transpose
    logical_tensor input_attn_transpose_desc{15, data_type::f16};

    op transpose_attn(11, op::kind::StaticTranspose, {input_attn_desc},
                      {input_attn_transpose_desc}, "transpose_attn");
    transpose_attn.set_attr<dims>(op::attr::order, {0, 2, 1, 3});

    // attn reshape
    logical_tensor input_attn_reshape_desc{16, data_type::f16};

    op reshape_attn(12, op::kind::StaticReshape, {input_attn_transpose_desc},
                    {input_attn_reshape_desc}, "reshape_attn");
    reshape_attn.set_attr<dims>(op::attr::shape, {Batch, Seq, Dmodel});
    reshape_attn.set_attr<bool>(op::attr::special_zero, false);

    // mlp weight conevrt
    logical_tensor mlp_weight_desc{17, data_type::f32};
    logical_tensor mlp_weight_h_desc{18, data_type::f16};
    logical_tensor mlp_bias_desc{19, data_type::f32};
    logical_tensor mlp_bias_h_desc{20, data_type::f16};

    op convert_mlp_weight(13, op::kind::TypeCast, {mlp_weight_desc},
                          {mlp_weight_h_desc}, "convert_mlp_weight");
    op convert_mlp_bias(14, op::kind::TypeCast, {mlp_bias_desc},
                        {mlp_bias_h_desc}, "convert_mlp_bias");

    // gemm mlp
    logical_tensor input_mlp_desc{21, data_type::f16};

    op gemm_mlp(15, op::kind::MatMul,
                {input_attn_reshape_desc, mlp_weight_h_desc, mlp_bias_h_desc},
                {input_mlp_desc}, "gemm_mlp");

    // add arg
    logical_tensor input_residual_desc{22, data_type::f16};

    op add_residual(16, op::kind::Add, {input_mlp_desc, input_h_src_desc},
                    {input_residual_desc}, "add_residual");

    // ffn layernorm
    logical_tensor ffn_layernorm_desc{23, data_type::f16};

    op layernorm_ffn(17, op::kind::LayerNorm, {input_residual_desc},
                     {ffn_layernorm_desc}, "layernorm_ffn");
    layernorm_ffn.set_attr<bool>(op::attr::keep_stats, false);
    layernorm_ffn.set_attr<bool>(op::attr::use_affine, false);

    // ffn1 weight convert
    logical_tensor ffn1_weight_desc{24, data_type::f32};
    logical_tensor ffn1_weight_h_desc{25, data_type::f16};
    logical_tensor ffn1_bias_desc{26, data_type::f32};
    logical_tensor ffn1_bias_h_desc{27, data_type::f16};

    op convert_ffn1_weight(18, op::kind::TypeCast, {ffn1_weight_desc},
                           {ffn1_weight_h_desc}, "convert_ffn1_weight");
    op convert_ffn1_bias(19, op::kind::TypeCast, {ffn1_bias_desc},
                         {ffn1_bias_h_desc}, "convert_ffn1_bias");

    // ffn1 gemm
    logical_tensor ffn1_output_desc{28, data_type::f16};

    op gemm_ffn1(20, op::kind::MatMul,
                 {ffn_layernorm_desc, ffn1_weight_h_desc, ffn1_bias_h_desc},
                 {ffn1_output_desc}, "gemm_ffn1");

    // ffn relu
    logical_tensor relu_desc{38, data_type::f16};

    op relu(26, op::kind::ReLU, {ffn1_output_desc}, {relu_desc}, "relu");

    // ffn2 weight conevrt
    logical_tensor ffn2_weight_desc{29, data_type::f32};
    logical_tensor ffn2_weight_h_desc{30, data_type::f16};
    logical_tensor ffn2_bias_desc{31, data_type::f32};
    logical_tensor ffn2_bias_h_desc{32, data_type::f16};

    op convert_ffn2_weight(21, op::kind::TypeCast, {ffn2_weight_desc},
                           {ffn2_weight_h_desc}, "convert_ffn2_weight");
    op convert_ffn2_bias(22, op::kind::TypeCast, {ffn2_bias_desc},
                         {ffn2_bias_h_desc}, "convert_ffn2_bias");

    // ffn2 gemm
    logical_tensor ffn2_output_desc{33, data_type::f16};

    op gemm_ffn2(23, op::kind::MatMul,
                 {relu_desc, ffn2_weight_h_desc, ffn2_bias_h_desc},
                 {ffn2_output_desc}, "gemm_ffn2");

    // add residual 2
    logical_tensor output_h_desc{34, data_type::f16};

    op add_residual2(24, op::kind::Add, {ffn2_output_desc, input_residual_desc},
                     {output_h_desc}, "add_residual2");

    // convert output
    logical_tensor output_desc{35, data_type::f32};

    op convert_output(25, op::kind::TypeCast, {output_h_desc}, {output_desc},
                      "convert_output");

    printf("Before create graph\n");

    // Create graph
    graph g(dnnl::engine::kind::cpu);

    g.add_op(convert_input);
    g.add_op(layernorm_input);
    g.add_op(convert_qkv);
    g.add_op(gemm_qkv);
    g.add_op(reshape_qkv);
    g.add_op(transpose_qkv);
    g.add_op(split_qkv);
    g.add_op(gemm_qk);
    g.add_op(softmax);
    g.add_op(gemm_attn);
    g.add_op(transpose_attn);
    g.add_op(reshape_attn);
    g.add_op(convert_mlp_weight);
    g.add_op(convert_mlp_bias);
    g.add_op(gemm_mlp);
    g.add_op(add_residual);
    g.add_op(layernorm_ffn);
    g.add_op(convert_ffn1_weight);
    g.add_op(convert_ffn1_bias);
    g.add_op(gemm_ffn1);
    g.add_op(relu);
    g.add_op(convert_ffn2_weight);
    g.add_op(convert_ffn2_bias);
    g.add_op(gemm_ffn2);
    g.add_op(add_residual2);
    g.add_op(convert_output);

    g.finalize();

    auto partitions = g.get_partitions();

    std::unordered_map<size_t, logical_tensor> id_to_quired_logical_tensors;
    std::unordered_set<size_t> ids_with_any_layout;
    set_any_layout(partitions, ids_with_any_layout);

    std::unordered_map<size_t, dims> concrete_shapes{
        {0, input_dims},
        {1, input_dims},
        {2, input_dims},
        {36, qkv_dims},
        {37, qkv_dims},
        {3, input_qkv_dims},
        {4, input_qkv_reshape_dims},
        {5, input_qkv_transpose_dims},
        {6, input_q_dims},
        {7, input_q_dims},
        {8, input_q_dims},
        {10, input_qk_dims},
        {13, input_qk_dims},
        {14, input_q_dims},
        {15, input_attn_mlp_dims},
        {16, input_dims},
        {17, mlp_weight_dims},
        {18, mlp_weight_dims},
        {19, mlp_bias_dims},
        {20, mlp_bias_dims},
        {21, input_dims},
        {22, input_dims},
        {23, input_dims},
        {24, ffn1_weight_dims},
        {25, ffn1_weight_dims},
        {26, ffn1_bias_dims},
        {27, ffn1_bias_dims},
        {28, ffn1_output_dims},
        {38, ffn1_output_dims},
        {29, ffn2_weight_dims},
        {30, ffn2_weight_dims},
        {31, ffn2_bias_dims},
        {32, ffn2_bias_dims},
        {33, input_dims},
        {34, input_dims},
        {35, input_dims}};

    for (const auto &part : partitions) {
      if (!part.is_supported()) {
        partition_combines info;
        info.part = part;
        info.support = false;
        infos.emplace_back(info);
        continue;
      }

      std::vector<logical_tensor> inputs = part.get_input_ports();
      std::vector<logical_tensor> outputs = part.get_output_ports();

      for (auto &input : inputs) {
        const auto id = input.get_id();

        if (id_to_quired_logical_tensors.find(id) !=
            id_to_quired_logical_tensors.end()) {
          input = id_to_quired_logical_tensors[id];
        } else {
          input = logical_tensor{id, input.get_data_type(), concrete_shapes[id],
                                 layout_type::strided};
        }
      }

      for (auto &output : outputs) {
        const auto id = output.get_id();
        output = logical_tensor{
            id, output.get_data_type(), DNNL_GRAPH_UNKNOWN_NDIMS,
            ids_with_any_layout.count(id) ? layout_type::any
                                          : layout_type::strided};
      }

      compiled_partition cp = part.compile(inputs, outputs, LLMCpuEngine);

      for (auto &output : outputs) {
        const auto id = output.get_id();
        output = cp.query_logical_tensor(id);
        id_to_quired_logical_tensors[id] = output;
      }

      partition_combines info;
      info.part = part, info.cp = cp, info.support = true;
      infos.emplace_back(info);
    }
  }

  printf("Before exec\n");

  std::unordered_map<size_t, tensor> global_outputs_ts_map;
  std::unordered_map<size_t, void *> io_func{
      {0, input},        {36, qkv},         {17, gemm_weight},
      {19, gemm_bias},   {24, ffn1_weight}, {26, ffn1_bias},
      {29, ffn2_weight}, {31, ffn2_bias},   {35, output}};
  std::vector<std::shared_ptr<void>> data_buffer;

  // execute
  for (auto &info : infos) {
    auto inputs = info.part.get_input_ports();
    auto outputs = info.part.get_output_ports();
    std::vector<tensor> input_ts, output_ts;
    allocate_graph_mem(input_ts, inputs, data_buffer, io_func,
                       global_outputs_ts_map, true);

    if (!info.support) {
      splitQKV(output_ts, outputs, global_outputs_ts_map, inputs[0]);
      continue;
    }

    allocate_graph_mem(output_ts, outputs, data_buffer, io_func,
                       global_outputs_ts_map, false);

    info.cp.execute(LLMCpuStream, input_ts, output_ts);
  }

  LLMCpuStream.wait();
}