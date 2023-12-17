#include "src/Dialect/ONNX/ONNXDialect.hpp"

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

void EliminateEntryPointPass::runOnOperation() {}

std::unique_ptr<Pass> multi_device::createEliminateEntryPointPass() {
  return std::make_unique<EliminateEntryPointPass>();
}