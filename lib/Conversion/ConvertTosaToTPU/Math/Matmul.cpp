#include "multi-device-model-compiler/Conversion/ConvertTosaToTPU/ConvertTosaToTPU.h"

#include "tpu_mlir/Dialect/Top/IR/TopOps.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"

#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/TypeUtilities.h"

using namespace mlir;

namespace {
class TosaMatmulLoweringToTPU
    : public OpConversionPattern<tosa::FullyConnectedOp> {
public:
  using OpConversionPattern<tosa::FullyConnectedOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(tosa::FullyConnectedOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto input = op.getInput();
    auto inputDefine = input.getDefiningOp();
    if (mlir::isa<tosa::ReshapeOp>(inputDefine)) {
      input = mlir::cast<tosa::ReshapeOp>(inputDefine).getInput1();
      rewriter.eraseOp(inputDefine);
    }
    auto output = op.getOutput();
    auto outputType = output.getType();
    Operation *outputReshape = nullptr;
    for (auto user : output.getUsers()) {
      if (!mlir::isa<tosa::ReshapeOp>(user)) {
        outputReshape = nullptr;
        outputType = output.getType();
        break;
      }
      outputReshape = user;
      outputType =
          mlir::cast<tosa::ReshapeOp>(outputReshape).getOutput().getType();
    }

    llvm::SmallVector<NamedAttribute> attrs;
    attrs.emplace_back(
        rewriter.getNamedAttr("do_relu", rewriter.getBoolAttr(false)));
    attrs.emplace_back(
        rewriter.getNamedAttr("keep_dims", rewriter.getBoolAttr(true)));
    attrs.emplace_back(
        rewriter.getNamedAttr("hdim_is_batch", rewriter.getBoolAttr(false)));
    attrs.emplace_back(
        rewriter.getNamedAttr("left_transpose", rewriter.getBoolAttr(false)));
    attrs.emplace_back(
        rewriter.getNamedAttr("output_transpose", rewriter.getBoolAttr(false)));
    attrs.emplace_back(
        rewriter.getNamedAttr("right_transpose", rewriter.getBoolAttr(false)));

    llvm::SmallVector<Value> args = {input, op.getWeight(), op.getBias()};
    ValueRange operands(args);

    auto matmul = rewriter.create<tpu::MatMulOp>(op.getLoc(), outputType,
                                                 operands, attrs);

    if (outputReshape) {
      rewriter.eraseOp(op);
      rewriter.replaceOp(outputReshape, matmul);
    } else {
      rewriter.replaceOp(op, matmul);
    }
    return success();
  }
};
} // namespace

void multi_device::conversion::populateTosaMatmulToTPUConversionPattern(
    ConversionTarget &target, RewritePatternSet &patterns,
    TypeConverter &TypeConverter, MLIRContext &ctx) {
  patterns.insert<TosaMatmulLoweringToTPU>(TypeConverter, &ctx, 10);
}