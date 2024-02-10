#include "multi-device-model-compiler/Dialect/Device/Transform/Passes.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

using namespace mlir;

namespace multi_device {
namespace device {
#define GEN_PASS_DEF_BUFFERIZEOPWITHANALYSIS
#include "multi-device-model-compiler/Dialect/Device/Transform/Passes.h.inc"
} // namespace device
} // namespace multi_device

namespace {
class BufferizeOpWithAnalysisPass final
    : public multi_device::device::impl::BufferizeOpWithAnalysisBase<
          BufferizeOpWithAnalysisPass> {
  using BufferizeOpWithAnalysisBase::BufferizeOpWithAnalysisBase;
  void runOnOperation() override final;
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<bufferization::BufferizationDialect, memref::MemRefDialect,
                    tensor::TensorDialect, linalg::LinalgDialect>();
    linalg::registerBufferizableOpInterfaceExternalModels(registry);
  }
};
} // namespace

void BufferizeOpWithAnalysisPass::runOnOperation() {
  bufferization::BufferizationOptions options =
      bufferization::getPartialBufferizationOptions();
  options.opFilter.allowDialect<linalg::LinalgDialect>();
  if (failed(bufferization::bufferizeOp(getOperation(), options, false))) {
    signalPassFailure();
  }
}