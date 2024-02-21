#include "multi-device-model-compiler/Conversion/ConvertONNXToDevice/ConvertONNXToDevice.h"
#include "multi-device-model-compiler/Dialect/Device/IR/Device.h"

#include "src/Dialect/ONNX/ONNXOps.hpp"

#include "mlir/Dialect/Tensor/IR/Tensor.h"

using namespace mlir;

namespace {
class DetectMultiHeadAttentionLayer
    : public OpConversionPattern<ONNXSoftmaxOp> {
public:
  using OpConversionPattern<ONNXSoftmaxOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(ONNXSoftmaxOp softmax, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto qk_output = softmax.getInput();
    auto softmax_output = softmax.getOutput();
    if (!qk_output.hasOneUse() || !softmax_output.hasOneUse() ||
        !qk_output.getType().isa<RankedTensorType>()) {
      return rewriter.notifyMatchFailure(softmax, "Not match");
    }
    auto qk_output_shape = qk_output.getType().cast<RankedTensorType>();
    if (qk_output_shape.getRank() != 4) {
      return rewriter.notifyMatchFailure(softmax, "Not match");
    }
    auto seq_len = qk_output_shape.getDimSize(2),
         batch = qk_output_shape.getDimSize(0),
         n_head = qk_output_shape.getDimSize(1);
    rewriter.eraseOp(softmax);

    ONNXMatMulOp qk_v;
    for (auto user : softmax_output.getUsers()) {
      if (!isa<ONNXMatMulOp>(user)) {
        return rewriter.notifyMatchFailure(softmax, "Not match");
      }
      qk_v = cast<ONNXMatMulOp>(user);
    }
    auto batch_v = qk_v.getB(), attn_out_b_head_seq = qk_v.getResult();
    rewriter.eraseOp(qk_v);

    if (!isa<ONNXAddOp>(qk_output.getDefiningOp())) {
      return rewriter.notifyMatchFailure(softmax, "Not match");
    }
    auto apply_mask = cast<ONNXAddOp>(qk_output.getDefiningOp());
    auto apply_mask_input = apply_mask.getA(), mask = apply_mask.getB();
    rewriter.eraseOp(apply_mask);

    if (!isa<ONNXWhereOp>(mask.getDefiningOp())) {
      return rewriter.notifyMatchFailure(softmax, "Not match");
    }
    auto apply_dropout = cast<ONNXWhereOp>(mask.getDefiningOp());
    auto origin_mask = apply_dropout.getCondition();
    rewriter.eraseOp(apply_dropout);

    if (!isa<ONNXMatMulOp>(apply_mask_input.getDefiningOp())) {
      return rewriter.notifyMatchFailure(softmax, "Not match");
    }
    auto qk_batch_matmul = cast<ONNXMatMulOp>(apply_mask_input.getDefiningOp());
    auto batch_q = qk_batch_matmul.getA(),
         batch_k_transpose = qk_batch_matmul.getB();
    auto batch_q_shape = batch_q.getType().cast<RankedTensorType>();
    auto head_dim = batch_q_shape.getDimSize(3), d_model = n_head * head_dim;
    rewriter.eraseOp(qk_batch_matmul);

    if (!isa<ONNXMulOp>(batch_q.getDefiningOp()) ||
        !isa<ONNXMulOp>(batch_k_transpose.getDefiningOp())) {
      return rewriter.notifyMatchFailure(softmax, "Not match");
    }
    auto q_scale = cast<ONNXMulOp>(batch_q.getDefiningOp()),
         k_scale = cast<ONNXMulOp>(batch_k_transpose.getDefiningOp());
    auto batch_q_before_scale = q_scale.getA(),
         batch_k_transpose_before_scale = k_scale.getA();
    rewriter.eraseOp(q_scale);
    rewriter.eraseOp(k_scale);

    if (!isa<ONNXTransposeOp>(batch_k_transpose_before_scale.getDefiningOp())) {
      return rewriter.notifyMatchFailure(softmax, "Not match");
    }
    auto transepose_batch_k =
        cast<ONNXTransposeOp>(batch_k_transpose_before_scale.getDefiningOp());
    auto batch_k_before_scale = transepose_batch_k.getData();
    rewriter.eraseOp(transepose_batch_k);

    if (!isa<ONNXReshapeOp>(batch_k_before_scale.getDefiningOp()) ||
        !isa<ONNXReshapeOp>(batch_q_before_scale.getDefiningOp()) ||
        !isa<ONNXReshapeOp>(batch_v.getDefiningOp())) {
      return rewriter.notifyMatchFailure(softmax, "Not match");
    }
    auto split_bs_nhead_q =
             cast<ONNXReshapeOp>(batch_q_before_scale.getDefiningOp()),
         split_bs_nhead_k =
             cast<ONNXReshapeOp>(batch_k_before_scale.getDefiningOp()),
         split_bs_nhead_v = cast<ONNXReshapeOp>(batch_v.getDefiningOp());
    auto q_bs_nhead = split_bs_nhead_q.getData(),
         k_bs_nhead = split_bs_nhead_k.getData(),
         v_bs_nhead = split_bs_nhead_v.getData();
    rewriter.eraseOp(split_bs_nhead_q);
    rewriter.eraseOp(split_bs_nhead_k);
    rewriter.eraseOp(split_bs_nhead_v);

    if (!isa<ONNXTransposeOp>(q_bs_nhead.getDefiningOp()) ||
        !isa<ONNXTransposeOp>(k_bs_nhead.getDefiningOp()) ||
        !isa<ONNXTransposeOp>(v_bs_nhead.getDefiningOp())) {
      return rewriter.notifyMatchFailure(softmax, "Not match");
    }
    auto transpose_q_seq = cast<ONNXTransposeOp>(q_bs_nhead.getDefiningOp()),
         transpose_k_seq = cast<ONNXTransposeOp>(k_bs_nhead.getDefiningOp()),
         transpose_v_seq = cast<ONNXTransposeOp>(v_bs_nhead.getDefiningOp());
    auto q_seq_bs_nhead = transpose_q_seq.getData(),
         k_seq_bs_nhead = transpose_k_seq.getData(),
         v_seq_bs_nhead = transpose_v_seq.getData();
    rewriter.eraseOp(transpose_q_seq);
    rewriter.eraseOp(transpose_k_seq);
    rewriter.eraseOp(transpose_v_seq);

    if (!isa<ONNXReshapeOp>(q_seq_bs_nhead.getDefiningOp()) ||
        !isa<ONNXReshapeOp>(k_seq_bs_nhead.getDefiningOp()) ||
        !isa<ONNXReshapeOp>(v_seq_bs_nhead.getDefiningOp())) {
      return rewriter.notifyMatchFailure(softmax, "Not match");
    }
    auto reshape_q_seq = cast<ONNXReshapeOp>(q_seq_bs_nhead.getDefiningOp()),
         reshape_k_seq = cast<ONNXReshapeOp>(k_seq_bs_nhead.getDefiningOp()),
         reshape_v_seq = cast<ONNXReshapeOp>(v_seq_bs_nhead.getDefiningOp());
    auto q_seq_unshape = reshape_q_seq.getData(),
         k_seq_unshape = reshape_k_seq.getData(),
         v_seq_unshape = reshape_v_seq.getData();
    rewriter.eraseOp(reshape_q_seq);
    rewriter.eraseOp(reshape_k_seq);
    rewriter.eraseOp(reshape_v_seq);

    if (!isa<ONNXGatherOp>(q_seq_unshape.getDefiningOp()) ||
        !isa<ONNXGatherOp>(k_seq_unshape.getDefiningOp()) ||
        !isa<ONNXGatherOp>(v_seq_unshape.getDefiningOp())) {
      return rewriter.notifyMatchFailure(softmax, "Not match");
    }
    auto gather_q = cast<ONNXGatherOp>(q_seq_unshape.getDefiningOp()),
         gather_k = cast<ONNXGatherOp>(k_seq_unshape.getDefiningOp()),
         gather_v = cast<ONNXGatherOp>(v_seq_unshape.getDefiningOp());
    auto qkv_squeeze = gather_q.getData();
    rewriter.eraseOp(gather_q);
    rewriter.eraseOp(gather_k);
    rewriter.eraseOp(gather_v);

    if (!isa<ONNXSqueezeOp>(qkv_squeeze.getDefiningOp())) {
      return rewriter.notifyMatchFailure(softmax, "Not match");
    }
    auto squeeze_qkv = cast<ONNXSqueezeOp>(qkv_squeeze.getDefiningOp());
    auto qkv_transpose = squeeze_qkv.getData();
    rewriter.eraseOp(squeeze_qkv);

    if (!isa<ONNXTransposeOp>(qkv_transpose.getDefiningOp())) {
      return rewriter.notifyMatchFailure(softmax, "Not match");
    }
    auto transpose_qkv = cast<ONNXTransposeOp>(qkv_transpose.getDefiningOp());
    auto qkv_unsqueeze = transpose_qkv.getData();
    rewriter.eraseOp(transpose_qkv);

    if (!isa<ONNXUnsqueezeOp>(qkv_unsqueeze.getDefiningOp())) {
      return rewriter.notifyMatchFailure(softmax, "Not match");
    }
    auto unsqueeze_qkv = cast<ONNXUnsqueezeOp>(qkv_unsqueeze.getDefiningOp());
    auto qkv_reshape = unsqueeze_qkv.getData();
    rewriter.eraseOp(unsqueeze_qkv);

    if (!isa<ONNXReshapeOp>(qkv_reshape.getDefiningOp())) {
      return rewriter.notifyMatchFailure(softmax, "Not match");
    }
    auto reshape_qkv = cast<ONNXReshapeOp>(qkv_reshape.getDefiningOp());
    auto qkv_result = reshape_qkv.getData();
    rewriter.eraseOp(reshape_qkv);

    if (!isa<ONNXMatMulOp>(qkv_result.getDefiningOp())) {
      return rewriter.notifyMatchFailure(softmax, "Not match");
    }
    auto matmul_qkv = cast<ONNXMatMulOp>(qkv_result.getDefiningOp());
    auto input_transpose = matmul_qkv.getA(), qkv_weight = matmul_qkv.getB();
    rewriter.eraseOp(matmul_qkv);

    if (!isa<ONNXTransposeOp>(input_transpose.getDefiningOp())) {
      return rewriter.notifyMatchFailure(softmax, "Not match");
    }
    auto transpose_input =
        cast<ONNXTransposeOp>(input_transpose.getDefiningOp());
    auto norm_div = transpose_input.getData();
    rewriter.eraseOp(transpose_input);

    if (!isa<ONNXDivOp>(norm_div.getDefiningOp())) {
      return rewriter.notifyMatchFailure(softmax, "Not match");
    }
    auto div_norm = cast<ONNXDivOp>(norm_div.getDefiningOp());
    auto sigma_sqrt = div_norm.getB(), query_mean = div_norm.getA();
    rewriter.eraseOp(div_norm);

    if (!isa<ONNXSqrtOp>(sigma_sqrt.getDefiningOp())) {
      return rewriter.notifyMatchFailure(softmax, "Not match");
    }
    auto sqrt_sigma = cast<ONNXSqrtOp>(sigma_sqrt.getDefiningOp());
    auto sigma_add = sqrt_sigma.getX();
    rewriter.eraseOp(sqrt_sigma);

    if (!isa<ONNXAddOp>(sigma_add.getDefiningOp())) {
      return rewriter.notifyMatchFailure(softmax, "Not match");
    }
    auto add_sigma = cast<ONNXAddOp>(sigma_add.getDefiningOp());
    auto sigma_reduce = add_sigma.getA();
    rewriter.eraseOp(add_sigma);

    if (!isa<ONNXReduceMeanV13Op>(sigma_reduce.getDefiningOp())) {
      return rewriter.notifyMatchFailure(softmax, "Not match");
    }
    auto reduce_sigma = cast<ONNXReduceMeanV13Op>(sigma_reduce.getDefiningOp());
    auto sigma_mul = reduce_sigma.getData();
    rewriter.eraseOp(reduce_sigma);

    if (!isa<ONNXMulOp>(sigma_mul.getDefiningOp())) {
      return rewriter.notifyMatchFailure(softmax, "Not match");
    }
    auto mul_sigma = cast<ONNXMulOp>(sigma_mul.getDefiningOp());
    rewriter.eraseOp(mul_sigma);

    if (!isa<ONNXSubOp>(query_mean.getDefiningOp())) {
      return rewriter.notifyMatchFailure(softmax, "Not match");
    }
    auto sub_mean = cast<ONNXSubOp>(query_mean.getDefiningOp());
    auto query = sub_mean.getA();
    rewriter.eraseOp(sub_mean);

    if (!attn_out_b_head_seq.hasOneUse()) {
      return rewriter.notifyMatchFailure(softmax, "Not match");
    }
    ONNXTransposeOp transpose_attn_out;
    for (auto user : attn_out_b_head_seq.getUsers()) {
      transpose_attn_out = dyn_cast<ONNXTransposeOp>(user);
    }
    auto attn_out_seq_b_head = transpose_attn_out.getResult();
    rewriter.eraseOp(transpose_attn_out);

    if (!attn_out_seq_b_head.hasOneUse()) {
      return rewriter.notifyMatchFailure(softmax, "Not match");
    }
    ONNXReshapeOp reshape_attn;
    for (auto user : attn_out_seq_b_head.getUsers()) {
      reshape_attn = dyn_cast<ONNXReshapeOp>(user);
    }
    auto attn_out_reshape = reshape_attn.getResult();
    rewriter.eraseOp(reshape_attn);

    if (!attn_out_reshape.hasOneUse()) {
      return rewriter.notifyMatchFailure(softmax, "Not match");
    }
    ONNXGemmOp gemm_attn;
    for (auto user : attn_out_reshape.getUsers()) {
      gemm_attn = dyn_cast<ONNXGemmOp>(user);
    }
    auto attn_gemm = gemm_attn.getY(), attn_gemm_weight = gemm_attn.getB(),
         attn_gemm_bias = gemm_attn.getC();
    rewriter.eraseOp(gemm_attn);

    if (!attn_gemm.hasOneUse()) {
      return rewriter.notifyMatchFailure(softmax, "Not match");
    }
    ONNXReshapeOp reshape_attn_after;
    for (auto user : attn_gemm.getUsers()) {
      reshape_attn_after = dyn_cast<ONNXReshapeOp>(user);
    }
    auto attn_reshape = reshape_attn_after.getResult();
    rewriter.eraseOp(reshape_attn_after);

    if (!attn_reshape.hasOneUse()) {
      return rewriter.notifyMatchFailure(softmax, "Not match");
    }
    ONNXTransposeOp transpose_attn_batch;
    for (auto user : attn_reshape.getUsers()) {
      transpose_attn_batch = dyn_cast<ONNXTransposeOp>(user);
    }
    auto attn_batch_seq_dim = transpose_attn_batch.getResult();
    rewriter.eraseOp(transpose_attn_batch);

    if (!attn_batch_seq_dim.hasOneUse()) {
      return rewriter.notifyMatchFailure(softmax, "Not match");
    }
    ONNXAddOp add_residual_1;
    for (auto user : attn_batch_seq_dim.getUsers()) {
      add_residual_1 = dyn_cast<ONNXAddOp>(user);
    }
    auto residual_1 = add_residual_1.getC();
    rewriter.eraseOp(add_residual_1);

    ONNXReduceMeanV13Op reduce_norm_avg_2;
    for (auto user : residual_1.getUsers()) {
      reduce_norm_avg_2 = dyn_cast<ONNXReduceMeanV13Op>(user);
      if (reduce_norm_avg_2) {
        break;
      }
    }
    auto norm_2_avg = reduce_norm_avg_2.getResult();
    rewriter.eraseOp(reduce_norm_avg_2);

    ONNXSubOp sub_norm_2_avg;
    for (auto user : norm_2_avg.getUsers()) {
      sub_norm_2_avg = dyn_cast<ONNXSubOp>(user);
    }
    auto norm_2_sub = sub_norm_2_avg.getC();
    rewriter.eraseOp(sub_norm_2_avg);

    ONNXMulOp mul_norm_2_sigma;
    for (auto user : norm_2_sub.getUsers()) {
      mul_norm_2_sigma = dyn_cast<ONNXMulOp>(user);
      if (mul_norm_2_sigma) {
        break;
      }
    }
    auto norm_2_sigma = mul_norm_2_sigma.getResult();
    rewriter.eraseOp(mul_norm_2_sigma);

    ONNXReduceMeanV13Op reduce_norm_2_sigma;
    for (auto user : norm_2_sigma.getUsers()) {
      reduce_norm_2_sigma = dyn_cast<ONNXReduceMeanV13Op>(user);
    }
    auto norm_2_reduce = reduce_norm_2_sigma.getResult();
    rewriter.eraseOp(reduce_norm_2_sigma);

    ONNXAddOp add_norm_2_sigma;
    for (auto user : norm_2_reduce.getUsers()) {
      add_norm_2_sigma = dyn_cast<ONNXAddOp>(user);
    }
    auto norm_2_sigma_add = add_norm_2_sigma.getResult();
    rewriter.eraseOp(add_norm_2_sigma);

    ONNXSqrtOp sqrt_norm_2;
    for (auto user : norm_2_sigma_add.getUsers()) {
      sqrt_norm_2 = dyn_cast<ONNXSqrtOp>(user);
    }
    auto norm_2_sigma_sqrt = sqrt_norm_2.getResult();
    rewriter.eraseOp(sqrt_norm_2);

    ONNXDivOp div_norm_2;
    for (auto user : norm_2_sigma_sqrt.getUsers()) {
      div_norm_2 = dyn_cast<ONNXDivOp>(user);
    }
    auto norm_2_output = div_norm_2.getResult();
    rewriter.eraseOp(div_norm_2);

    ONNXMatMulOp matmul_feedforward_1;
    for (auto user : norm_2_output.getUsers()) {
      matmul_feedforward_1 = dyn_cast<ONNXMatMulOp>(user);
    }
    auto feedforward_1_mat = matmul_feedforward_1.getY(),
         feedforward_1_weight = matmul_feedforward_1.getB();
    rewriter.eraseOp(matmul_feedforward_1);

    ONNXAddOp add_feedforward_1;
    for (auto user : feedforward_1_mat.getUsers()) {
      add_feedforward_1 = dyn_cast<ONNXAddOp>(user);
    }
    auto feedforward_1_out = add_feedforward_1.getC(),
         feedforward_1_bias = add_feedforward_1.getB();
    rewriter.eraseOp(add_feedforward_1);

    ONNXReluOp relu_feedforward;
    for (auto user : feedforward_1_out.getUsers()) {
      relu_feedforward = dyn_cast<ONNXReluOp>(user);
    }
    auto feedforward_1_act = relu_feedforward.getResult();
    rewriter.eraseOp(relu_feedforward);

    ONNXMatMulOp matmul_feedforward_2;
    for (auto user : feedforward_1_act.getUsers()) {
      matmul_feedforward_2 = dyn_cast<ONNXMatMulOp>(user);
    }
    auto feedforward_2_mat = matmul_feedforward_2.getY(),
         feedforward_2_weight = matmul_feedforward_2.getB();
    rewriter.eraseOp(matmul_feedforward_2);

    ONNXAddOp add_residual_2;
    for (auto user : feedforward_2_mat.getUsers()) {
      add_residual_2 = dyn_cast<ONNXAddOp>(user);
    }
    auto residual_2 = add_residual_2.getC();
    rewriter.eraseOp(add_residual_2);

    ONNXAddOp add_feedforward_2;
    for (auto user : residual_2.getUsers()) {
      add_feedforward_2 = dyn_cast<ONNXAddOp>(user);
    }
    auto feedforward_2_bias = add_feedforward_2.getB();
    auto feedforward_2_out = add_feedforward_2.getResult();
    rewriter.eraseOp(add_feedforward_2);

    // now take a deep breath, and create
    rewriter.setInsertionPointAfter(add_feedforward_2);
    auto loc = add_feedforward_2.getLoc();
    auto result_shape = feedforward_2_out.getType().cast<RankedTensorType>();
    auto empty = rewriter.create<tensor::EmptyOp>(
        loc, result_shape.getShape(), result_shape.getElementType());
    auto output_tensor = empty.getResult();
    auto multi_head_attention_layer =
        rewriter.create<multi_device::device::MultiHeadAttentionLayer>(
            loc, TypeRange(), /*input*/ query, /*qkv*/ qkv_weight,
            /*attn gemm weight*/ attn_gemm_weight,
            /*attn gemm bias*/ attn_gemm_bias,
            /*feed forward 1 weight*/ feedforward_1_weight,
            /*feed forward 1 bias*/ feedforward_1_bias,
            /*feed forward 2 weight*/ feedforward_2_weight,
            /*feed forward 2 bias*/ feedforward_2_bias,
            /*output tensor*/ output_tensor, ValueRange(),
            /*batch*/ rewriter.getI64IntegerAttr(batch),
            /*seq_len*/ rewriter.getI64IntegerAttr(seq_len),
            /*d_model*/ rewriter.getI64IntegerAttr(d_model),
            /*feed_forward_dim*/ rewriter.getI64IntegerAttr(d_model),
            /*head_num*/ rewriter.getI64IntegerAttr(head_dim),
            /*norm_first*/ rewriter.getBoolAttr(true),
            /*is_casual*/ rewriter.getBoolAttr(true),
            /*act*/ rewriter.getStringAttr("relu"));
    rewriter.replaceAllUsesWith(feedforward_2_out, output_tensor);

    return success();
  }
};
} // namespace

void multi_device::conversion::populateDetectMultiHeadAttentionLayerPattern(
    ConversionTarget &target, RewritePatternSet &patterns,
    TypeConverter &TypeConverter, MLIRContext &ctx) {
  target.addIllegalOp<ONNXSoftmaxOp>();
  patterns.insert<DetectMultiHeadAttentionLayer>(&ctx, 100);
}