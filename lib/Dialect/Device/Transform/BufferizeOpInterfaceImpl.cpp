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
} // namespace

void multi_device::device::registerBufferizableOpInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, DeviceDialect *dialect) {
    MatmulOp::attachInterface<MatmulOpInterface>(*ctx);
    Conv2DOp::attachInterface<Conv2dOpInterface>(*ctx);
    Pool2DOp::attachInterface<Pool2dOpInterface>(*ctx);
  });
}