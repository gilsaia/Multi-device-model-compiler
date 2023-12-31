#include "multi-device-model-compiler/Conversion/ConvertTosaToTPU/ConvertTosaToTPU.h"

#include "tpu_mlir/Dialect/Top/IR/TopOps.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Support/TensorFile.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/ErrorHandling.h"

#include <queue>

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
  void replaceFuncInput(func::FuncOp func);
  void addElementName(func::FuncOp func);
  void saveWeights(func::FuncOp func);
};

void TosaLoweringToTPUPass::replaceFuncInput(func::FuncOp func) {
  func.setName("main");
  OpBuilder builder(func.getRegion());
  ArrayAttr input_names = func->getAttr("input_names").dyn_cast<ArrayAttr>();
  if (!input_names || input_names.size() != func.getNumArguments()) {
    llvm_unreachable("func input name not match argument.");
  }
  std::queue<StringAttr> que;
  for (auto &attr : input_names) {
    que.push(attr.dyn_cast<StringAttr>());
  }
  for (BlockArgument &arg : func.getArguments()) {
    Location loc = builder.getInsertionPoint()->getLoc();
    std::vector<NamedAttribute> attrs;
    attrs.emplace_back(builder.getStringAttr("channel_format"),
                       builder.getStringAttr("nchw"));
    attrs.emplace_back(builder.getStringAttr("pixel_format"),
                       builder.getStringAttr("bgr"));
    auto topInput =
        builder.create<top::InputOp>(loc, arg.getType(), arg, attrs);
    topInput->setLoc(NameLoc::get(que.front()));
    que.pop();
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
}

void TosaLoweringToTPUPass::addElementName(func::FuncOp func) {
  llvm::StringSet nameset;
  func.walk([&](tosa::TosaOp op) {
    llvm::StringRef ori_name = op->getName().stripDialect();
    int suffix = 1;
    std::string name = ori_name.str();
    while (nameset.contains(name)) {
      name = ori_name.str() + "_" + std::to_string(suffix);
      ++suffix;
    }
    nameset.insert(name);
    op->setLoc(NameLoc::get(StringAttr::get(op->getContext(), name)));
  });
}

void TosaLoweringToTPUPass::saveWeights(func::FuncOp func) {
  ModuleOp module = dyn_cast<ModuleOp>(func->getParentOp());
  if (!module) {
    llvm_unreachable("Can't find func's parent moduleop.");
  }
  auto weight_file =
      module->getAttr("module.weight_file").dyn_cast<StringAttr>();
  TensorFile file(weight_file.getValue(), false, true);
  file.save();
}

void TosaLoweringToTPUPass::runOnOperation() {
  addElementName(getOperation());
  replaceFuncInput(getOperation());
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
  target.addIllegalDialect<tosa::TosaDialect>();

  populateTosaToTPUConversionPattern(target, patterns, typeConverter, context);

  if (failed(applyPartialConversion(getOperation(), target,
                                    std::move(patterns)))) {
    signalPassFailure();
  }
}

} // namespace multi_device

std::unique_ptr<mlir::Pass> multi_device::createConvertTosaToTPUPass() {
  return std::make_unique<TosaLoweringToTPUPass>();
}
