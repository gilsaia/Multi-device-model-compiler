#include "multi-device-model-compiler/Conversion/ConvertTosaToTPU/ConvertTosaToTPU.h"

#include "tpu_mlir/Dialect/Top/IR/TopOps.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace multi_device {

void populateTosaToTPUConversionPattern(ConversionTarget &target,
                                        RewritePatternSet &patterns,
                                        TypeConverter &TypeConverter,
                                        MLIRContext &ctx) {
  conversion::populateTosaElementWiseToTPUConversionPattern(target, patterns,
                                                            TypeConverter, ctx);
}

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
    auto topInput = builder.create<top::InputOp>(loc, arg.getType(), arg);
    Value res = topInput.getResult();
    func.walk([&](Operation *op) {
      if (op == topInput.getOperation()) {
        return WalkResult::skip();
      }
      for (auto &input : op->getOpOperands()) {
        if (input.get() == arg) {
          input.set(res);
        }
      }
      return WalkResult::advance();
    });
  }
  MLIRContext &context = getContext();
  RewritePatternSet patterns(&context);
  ConversionTarget target(context);

  TypeConverter typeConverter;
  typeConverter.addConversion([](Type type) -> std::optional<Type> {
    if (type.isa<Float32Type, Float16Type, BFloat16Type>() ||
        type.isa<NoneType>()) {
      return type;
    }
    if (type.isa<IntegerType>()) {
      IntegerType intType = type.cast<IntegerType>();
      std::set<unsigned> intWidth{8, 16, 32, 48, 64};
      if (intType.isSignless() &&
          (intWidth.find(intType.getWidth()) != intWidth.end())) {
        return type;
      }
    }
    return std::nullopt;
  });
  typeConverter.addConversion([&](TensorType type) -> std::optional<Type> {
    if (typeConverter.isLegal(type.getElementType())) {
      return type;
    }
    return std::nullopt;
  });

  target.addLegalDialect<func::FuncDialect, top::TopDialect, tpu::TpuDialect>();

  populateTosaToTPUConversionPattern(target, patterns, typeConverter, context);

  if (failed(applyPartialConversion(func, target, std::move(patterns)))) {
    signalPassFailure();
  }
}

} // namespace multi_device

std::unique_ptr<mlir::Pass> multi_device::createConvertTosaToTPUPass() {
  return std::make_unique<TosaLoweringToTPUPass>();
}
