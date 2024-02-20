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

#define GEN_PASS_DEF_TOSALOWERTOTPU
#include "multi-device-model-compiler/Conversion/Passes.h.inc"

void populateTosaToTPUConversionPattern(ConversionTarget &target,
                                        RewritePatternSet &patterns,
                                        TypeConverter &TypeConverter,
                                        MLIRContext &ctx) {
  conversion::populateTosaElementWiseToTPUConversionPattern(target, patterns,
                                                            TypeConverter, ctx);
  conversion::populateTosaTensorToTPUConversionPattern(target, patterns,
                                                       TypeConverter, ctx);
  conversion::populateTosaMatmulToTPUConversionPattern(target, patterns,
                                                       TypeConverter, ctx);
}

void populateTosaFuseToTPUConversionPattern(ConversionTarget &target,
                                            RewritePatternSet &patterns,
                                            TypeConverter &TypeConverter,
                                            MLIRContext &ctx) {
  conversion::populateTosaFuseElementWiseToTPUConversionPattern(
      target, patterns, TypeConverter, ctx);
}

namespace {
class TosaLoweringToTPUPass
    : public multi_device::impl::TosaLowerToTPUBase<TosaLoweringToTPUPass> {
public:
  TosaLoweringToTPUPass() = default;
  void runOnOperation() override final;

private:
  void replaceFuncInput(func::FuncOp func);
  void addElementName(func::FuncOp func);
  void saveWeights(func::FuncOp func);
};
} // namespace

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
    attrs.emplace_back(
        builder.getNamedAttr("channel_format", builder.getStringAttr("nchw")));
    attrs.emplace_back(
        builder.getNamedAttr("pixel_format", builder.getStringAttr("bgr")));
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
  func.walk([&](tosa::ConstOp op) {
    auto elements = op.getValue();
    if (!elements.isa<DenseIntOrFPElementsAttr>()) {
      llvm_unreachable("Don't has a right weight type.");
    }
    DenseIntOrFPElementsAttr dataAttr =
        elements.cast<DenseIntOrFPElementsAttr>();
    llvm::StringRef weightname =
        op.getLoc().cast<NameLoc>().getName().getValue();
    auto tensorType = op.getType().cast<RankedTensorType>();
    auto elementType = tensorType.getElementType();
    if (elementType.isF32()) {
      LogicalResult res = success();
      if (dataAttr.isSplat()) {
        float val = dataAttr.getValues<float>()[0];
        std::vector<float> vals(dataAttr.getNumElements(), val);
        res = file.addTensor<float>(weightname, &vals, tensorType);
      } else {
        const float *dataPtr =
            reinterpret_cast<const float *>(dataAttr.getRawData().data());
        res = file.addTensor<float>(weightname, dataPtr, tensorType);
      }
      if (failed(res)) {
        llvm_unreachable("File can't add tensor becouse of name used.");
      }
    } else {
      llvm_unreachable("other type not implemented");
    }
  });
  file.save();
}

void TosaLoweringToTPUPass::runOnOperation() {
  addElementName(getOperation());
  replaceFuncInput(getOperation());
  saveWeights(getOperation());
  MLIRContext &context = getContext();
  RewritePatternSet patterns(&context), fusePatterns(&context);
  ConversionTarget target(context), fuseTarget(context);

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
  target.addIllegalOp<tosa::FullyConnectedOp, tosa::AddOp, tosa::ConstOp,
                      tosa::ReshapeOp>();

  populateTosaToTPUConversionPattern(target, patterns, typeConverter, context);

  if (failed(applyPartialConversion(getOperation(), target,
                                    std::move(patterns)))) {
    signalPassFailure();
  }

  fuseTarget
      .addLegalDialect<func::FuncDialect, top::TopDialect, tpu::TpuDialect>();
  fuseTarget.addIllegalOp<tosa::ClampOp>();

  populateTosaFuseToTPUConversionPattern(fuseTarget, fusePatterns,
                                         typeConverter, context);

  if (failed(applyPartialConversion(getOperation(), fuseTarget,
                                    std::move(fusePatterns)))) {
    signalPassFailure();
  }
}

} // namespace multi_device
