#include "multi-device-model-compiler/Conversion/ConvertTosaToTPU/ConvertTosaToTPU.h"

#include "tpu_mlir/Dialect/Top/IR/TopOps.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"

using namespace mlir;

namespace multi_device {

struct TosaLoweringToTPUPass
    : public PassWrapper<TosaLoweringToTPUPass, OperationPass<func::FuncOp>> {
  StringRef getArgument() const override { return "convert-tosa-to-tpu"; }

  StringRef getDescription() const override {
    return "Lower tosa ops to TOP/TPU dialect.";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<top::TopDialect>();
    registry.insert<tpu::TpuDialect>();
  }

  TosaLoweringToTPUPass() = default;
  TosaLoweringToTPUPass(const TosaLoweringToTPUPass &pass)
      : PassWrapper<TosaLoweringToTPUPass, OperationPass<func::FuncOp>>() {}

  void runOnOperation() override final;
};

void TosaLoweringToTPUPass::runOnOperation() {
  func::FuncOp func = getOperation();
  OpBuilder builder(func.getRegion());
  for (BlockArgument &arg : func.getArguments()) {
    Location loc = builder.getInsertionPoint()->getLoc();
    builder.create<top::InputOp>(loc, arg.getType(), arg);
  }
}

} // namespace multi_device

std::unique_ptr<mlir::Pass> multi_device::createConvertTosaToTPUPass() {
  return std::make_unique<TosaLoweringToTPUPass>();
}
