#include "multi-device-model-compiler/Dialect/Device/IR/Device.h"
#include "multi-device-model-compiler/Dialect/Device/Transform/Passes.h"

#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/Transforms/BufferUtils.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Operation.h"

using namespace mlir;

namespace {
struct MatmulOpInterface
    : public bufferization::BufferizableOpInterface::ExternalModel<
          MatmulOpInterface, multi_device::device::MatmulOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const bufferization::AnalysisState &state) const {
    if (opOperand.getOperandNumber() < 3) {
      return true;
    }
    return false;
  }
  bool
  bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                          const bufferization::AnalysisState &state) const {
    if (opOperand.getOperandNumber() == 3) {
      return true;
    }
    return false;
  }

  bufferization::AliasingValueList
  getAliasingValues(Operation *op, OpOperand &opOperand,
                    const bufferization::AnalysisState &state) const {
    return {};
  }

  LogicalResult
  bufferize(Operation *op, RewriterBase &rewriter,
            const bufferization::BufferizationOptions &options) const {
    auto matmul = cast<multi_device::device::MatmulOp>(op);
    auto inputMem =
             bufferization::getBuffer(rewriter, matmul.getInput(), options),
         weightMem =
             bufferization::getBuffer(rewriter, matmul.getWeight(), options),
         biasMem =
             bufferization::getBuffer(rewriter, matmul.getBias(), options),
         outputMem =
             bufferization::getBuffer(rewriter, matmul.getOutput(), options);

    if (failed(inputMem) || failed(weightMem) || failed(biasMem) ||
        failed(outputMem)) {
      return failure();
    }

    bufferization::replaceOpWithNewBufferizedOp<multi_device::device::MatmulOp>(
        rewriter, op, TypeRange(), *inputMem, *weightMem, *biasMem, *outputMem,
        ValueRange());
    return success();
  }
};

struct Conv2dOpInterface
    : public bufferization::BufferizableOpInterface::ExternalModel<
          Conv2dOpInterface, multi_device::device::Conv2DOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const bufferization::AnalysisState &state) const {
    if (opOperand.getOperandNumber() < 3 || opOperand.getOperandNumber() == 4) {
      return true;
    }
    return false;
  }
  bool
  bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                          const bufferization::AnalysisState &state) const {
    if (opOperand.getOperandNumber() == 3) {
      return true;
    }
    return false;
  }

  bufferization::AliasingValueList
  getAliasingValues(Operation *op, OpOperand &opOperand,
                    const bufferization::AnalysisState &state) const {
    return {};
  }

  LogicalResult
  bufferize(Operation *op, RewriterBase &rewriter,
            const bufferization::BufferizationOptions &options) const {
    auto conv2d = cast<multi_device::device::Conv2DOp>(op);
    auto inputMem =
             bufferization::getBuffer(rewriter, conv2d.getInput(), options),
         weightMem =
             bufferization::getBuffer(rewriter, conv2d.getWeight(), options),
         biasMem =
             bufferization::getBuffer(rewriter, conv2d.getBias(), options),
         outputMem =
             bufferization::getBuffer(rewriter, conv2d.getOutput(), options);

    if (failed(inputMem) || failed(weightMem) || failed(biasMem) ||
        failed(outputMem)) {
      return failure();
    }

    Value postAddVal;

    if (conv2d.getPostadd()) {
      auto postAddMem =
          bufferization::getBuffer(rewriter, conv2d.getPostadd(), options);
      if (failed(postAddMem)) {
        return failure();
      }
      postAddVal = *postAddMem;
    }

    bufferization::replaceOpWithNewBufferizedOp<multi_device::device::Conv2DOp>(
        rewriter, op, TypeRange(), *inputMem, *weightMem, *biasMem, *outputMem,
        postAddVal, ValueRange(), conv2d.getKernelAttr(), conv2d.getPadAttr(),
        conv2d.getStrideAttr(), conv2d.getDilationAttr(),
        conv2d.getContainReluAttr());

    return success();
  }
};

struct Pool2dOpInterface
    : public bufferization::BufferizableOpInterface::ExternalModel<
          Pool2dOpInterface, multi_device::device::Pool2DOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const bufferization::AnalysisState &state) const {
    if (opOperand.getOperandNumber() == 0) {
      return true;
    }
    return false;
  }
  bool
  bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                          const bufferization::AnalysisState &state) const {
    if (opOperand.getOperandNumber() == 1) {
      return true;
    }
    return false;
  }

  bufferization::AliasingValueList
  getAliasingValues(Operation *op, OpOperand &opOperand,
                    const bufferization::AnalysisState &state) const {
    return {};
  }

  LogicalResult
  bufferize(Operation *op, RewriterBase &rewriter,
            const bufferization::BufferizationOptions &options) const {
    auto pool2d = cast<multi_device::device::Pool2DOp>(op);
    auto inputMem =
             bufferization::getBuffer(rewriter, pool2d.getInput(), options),
         outputMem =
             bufferization::getBuffer(rewriter, pool2d.getOutput(), options);

    if (failed(inputMem) || failed(outputMem)) {
      return failure();
    }

    bufferization::replaceOpWithNewBufferizedOp<multi_device::device::Pool2DOp>(
        rewriter, op, TypeRange(), *inputMem, *outputMem, ValueRange(),
        pool2d.getKernelAttr(), pool2d.getPadAttr(), pool2d.getStrideAttr(),
        pool2d.getMethodAttr());

    return success();
  }
};

struct MultiHeadAttentionLayerInterface
    : public bufferization::BufferizableOpInterface::ExternalModel<
          MultiHeadAttentionLayerInterface,
          multi_device::device::MultiHeadAttentionLayer> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const bufferization::AnalysisState &state) const {
    if (opOperand.getOperandNumber() < 8) {
      return true;
    }
    return false;
  }
  bool
  bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                          const bufferization::AnalysisState &state) const {
    if (opOperand.getOperandNumber() == 8) {
      return true;
    }
    return false;
  }

  bufferization::AliasingValueList
  getAliasingValues(Operation *op, OpOperand &opOperand,
                    const bufferization::AnalysisState &state) const {
    return {};
  }

  LogicalResult
  bufferize(Operation *op, RewriterBase &rewriter,
            const bufferization::BufferizationOptions &options) const {
    auto multiHeadAttentionLayer =
        cast<multi_device::device::MultiHeadAttentionLayer>(op);

    auto inputMem = bufferization::getBuffer(
             rewriter, multiHeadAttentionLayer.getInput(), options),
         qkvMem = bufferization::getBuffer(
             rewriter, multiHeadAttentionLayer.getQKV(), options),
         attnGemmWeightMem = bufferization::getBuffer(
             rewriter, multiHeadAttentionLayer.getAttnGemmWeight(), options),
         attnGemmBiasMem = bufferization::getBuffer(
             rewriter, multiHeadAttentionLayer.getAttnGemmBias(), options),
         ffn1WeightMem = bufferization::getBuffer(
             rewriter, multiHeadAttentionLayer.getFfn1Weight(), options),
         ffn1BiasMem = bufferization::getBuffer(
             rewriter, multiHeadAttentionLayer.getFfn1Bias(), options),
         ffn2WeightMem = bufferization::getBuffer(
             rewriter, multiHeadAttentionLayer.getFfn2Weight(), options),
         ffn2BiasMem = bufferization::getBuffer(
             rewriter, multiHeadAttentionLayer.getFfn2Bias(), options),
         outputMem = bufferization::getBuffer(
             rewriter, multiHeadAttentionLayer.getOutput(), options);

    if (failed(inputMem) || failed(qkvMem) || failed(attnGemmWeightMem) ||
        failed(attnGemmBiasMem) || failed(ffn1WeightMem) ||
        failed(ffn1BiasMem) || failed(ffn2WeightMem) || failed(ffn2BiasMem) ||
        failed(outputMem)) {
      return failure();
    }

    bufferization::replaceOpWithNewBufferizedOp<
        multi_device::device::MultiHeadAttentionLayer>(
        rewriter, op, TypeRange(), /*input*/ *inputMem, /*qkv*/ *qkvMem,
        /*attn gemm weight*/ *attnGemmWeightMem,
        /*attn gemm bias*/ *attnGemmBiasMem,
        /*feed forward 1 weight*/ *ffn1WeightMem,
        /*feed forward 1 bias*/ *ffn1BiasMem,
        /*feed forward 2 weight*/ *ffn2WeightMem,
        /*feed forward 2 bias*/ *ffn2BiasMem, /*output tensor*/ *outputMem,
        ValueRange(), /*batch*/ multiHeadAttentionLayer.getBatchAttr(),
        /*seq_len*/ multiHeadAttentionLayer.getSeqLenAttr(),
        /*d_model*/ multiHeadAttentionLayer.getDModelAttr(),
        /*feed forward dim*/ multiHeadAttentionLayer.getFeedForwardDimAttr(),
        /*head num*/ multiHeadAttentionLayer.getHeadNumAttr(),
        /*norm first*/ multiHeadAttentionLayer.getNormFirstAttr(),
        /*is casual*/ multiHeadAttentionLayer.getIsCasualAttr(),
        /*act*/ multiHeadAttentionLayer.getActAttr());

    return success();
  }
};
} // namespace

void multi_device::device::registerBufferizableOpInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, DeviceDialect *dialect) {
    MatmulOp::attachInterface<MatmulOpInterface>(*ctx);
    Conv2DOp::attachInterface<Conv2dOpInterface>(*ctx);
    Pool2DOp::attachInterface<Pool2dOpInterface>(*ctx);
    MultiHeadAttentionLayer::attachInterface<MultiHeadAttentionLayerInterface>(
        *ctx);
  });
}