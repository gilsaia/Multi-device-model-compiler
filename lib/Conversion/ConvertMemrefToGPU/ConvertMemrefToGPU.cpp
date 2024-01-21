#include "multi-device-model-compiler/Conversion/ConvertMemrefToGPU/ConvertMemrefToGPU.h"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"

using namespace mlir;
using namespace multi_device;

namespace multi_device {
#define GEN_PASS_DEF_CONVERTMEMREFTOGPU
#include "multi-device-model-compiler/Conversion/Passes.h.inc"

void populateMemrefToGPUPatterns(ConversionTarget &target,
                                 RewritePatternSet &patterns,
                                 TypeConverter &typeConverter,
                                 MLIRContext &ctx) {
  conversion::populateMemrefAllocToGPUConversionPattern(target, patterns,
                                                        typeConverter, ctx);
  conversion::populateMemrefDmaToGPUConversionPattern(target, patterns,
                                                      typeConverter, ctx);
}
} // namespace multi_device

namespace {
class ConvertMemrefToGPUPass
    : public multi_device::impl::ConvertMemrefToGPUBase<
          ConvertMemrefToGPUPass> {
public:
  ConvertMemrefToGPUPass() = default;
  void runOnOperation() override final;

private:
  static bool checkMemrefTypeLegal(BaseMemRefType type) {
    return !type.getMemorySpace();
  }
};
} // namespace

void ConvertMemrefToGPUPass::runOnOperation() {
  MLIRContext &ctx = getContext();
  RewritePatternSet patterns(&ctx);
  ConversionTarget target(ctx);
  TypeConverter keepConverter;

  keepConverter.addConversion(
      [](Type type) -> std::optional<Type> { return type; });

  target.addDynamicallyLegalOp<memref::AllocOp, memref::DeallocOp>(
      [](Operation *op) -> bool {
        if (auto tOp = mlir::dyn_cast<memref::AllocOp>(op)) {
          return ConvertMemrefToGPUPass::checkMemrefTypeLegal(
              tOp.getMemref().getType());
        } else if (auto tOp = mlir::dyn_cast<memref::DeallocOp>(op)) {
          return ConvertMemrefToGPUPass::checkMemrefTypeLegal(
              tOp.getMemref().getType());
        }
        return false;
      });
  target.addIllegalOp<memref::DmaStartOp, memref::DmaWaitOp>();
  target.addLegalDialect<gpu::GPUDialect>();

  populateMemrefToGPUPatterns(target, patterns, keepConverter, ctx);
  if (failed(applyPartialConversion(getOperation(), target,
                                    std::move(patterns)))) {
    signalPassFailure();
  }
}