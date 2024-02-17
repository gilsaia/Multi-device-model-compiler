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
} // namespace

void multi_device::device::registerBufferizableOpInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, DeviceDialect *dialect) {
    MatmulOp::attachInterface<MatmulOpInterface>(*ctx);
  });
}