#include "multi-device-model-compiler/Conversion/ConvertDeviceToTPU/ConvertDeviceToTPU.h"
#include "multi-device-model-compiler/Dialect/Device/IR/Device.h"

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
#define GEN_PASS_DEF_DEVICELOWERTOTPU
#include "multi-device-model-compiler/Conversion/Passes.h.inc"

void populateDeviceToTPUConversionPattern(ConversionTarget &target,
                                          RewritePatternSet &patterns,
                                          TypeConverter &TypeConverter,
                                          MLIRContext &ctx) {
  conversion::populateDeviceMatmulToTPUConversionPattern(target, patterns,
                                                         TypeConverter, ctx);
  conversion::populateDeviceConv2dToTPUConversionPattern(target, patterns,
                                                         TypeConverter, ctx);
  conversion::populateDevicePool2dToTPUConversionPattern(target, patterns,
                                                         TypeConverter, ctx);
}
} // namespace multi_device

namespace {
class DeviceLoweringToTPUPass
    : public multi_device::impl::DeviceLowerToTPUBase<DeviceLoweringToTPUPass> {
public:
  DeviceLoweringToTPUPass() = default;
  void runOnOperation() override final;

private:
  void addElementName(ModuleOp moduleOp);
};
} // namespace

void DeviceLoweringToTPUPass::addElementName(ModuleOp moduleOp) {
  llvm::StringSet nameset;
  moduleOp.walk([&](Operation *op) {
    if (!(isa<multi_device::device::MatmulOp>(op) ||
          isa<multi_device::device::Conv2DOp>(op) ||
          isa<multi_device::device::Pool2DOp>(op))) {
      return WalkResult::advance();
    }

    llvm::StringRef ori_name = op->getName().stripDialect();
    int suffix = 1;
    std::string name = ori_name.str();
    while (nameset.contains(name)) {
      name = ori_name.str() + "_" + std::to_string(suffix);
      ++suffix;
    }
    nameset.insert(name);
    op->setLoc(NameLoc::get(StringAttr::get(op->getContext(), name)));
    return WalkResult::advance();
  });
}

void DeviceLoweringToTPUPass::runOnOperation() {
  addElementName(getOperation());

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
      std::set<unsigned> intWidth{1, 2, 4, 8, 16, 32, 48, 64};
      if ((intWidth.find(intType.getWidth()) != intWidth.end())) {
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
  target.addIllegalDialect<multi_device::device::DeviceDialect>();

  multi_device::populateDeviceToTPUConversionPattern(target, patterns,
                                                     typeConverter, context);

  if (failed(applyPartialConversion(getOperation(), target,
                                    std::move(patterns)))) {
    signalPassFailure();
  }

  fuseTarget
      .addLegalDialect<func::FuncDialect, top::TopDialect, tpu::TpuDialect>();

  multi_device::conversion::populateFuseClipOpToTPUConversionPattern(
      fuseTarget, fusePatterns, typeConverter, context);

  if (failed(applyPartialConversion(getOperation(), fuseTarget,
                                    std::move(fusePatterns)))) {
    signalPassFailure();
  }
}