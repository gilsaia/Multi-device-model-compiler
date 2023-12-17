#include "mlir/IR/Operation.h"

#include "src/Dialect/ONNX/ONNXDialect.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"

#include "multi-device-model-compiler/Pass/ONNX/Passes.h"

namespace multi_device {
#define GEN_PASS_DEF_ELIMINATEENTRYPOINT
#include "multi-device-model-compiler/Pass/ONNX/Passes.h.inc"
} // namespace multi_device

using namespace mlir;

namespace {
struct EliminateEntryPointPass final
    : public multi_device::impl::EliminateEntryPointBase<
          EliminateEntryPointPass> {
  void runOnOperation() override;
};
} // namespace

void EliminateEntryPointPass::runOnOperation() {
  ModuleOp moduleOp = getOperation();
  moduleOp.walk([](ONNXEntryPointOp op) {
    op.erase();
    return WalkResult::skip();
  });
}

std::unique_ptr<OperationPass<ModuleOp>>
multi_device::createEliminateEntryPointPass() {
  return std::make_unique<EliminateEntryPointPass>();
}