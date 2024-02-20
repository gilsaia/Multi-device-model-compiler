#include "multi-device-model-compiler/Conversion/ConvertTosaToLinalg/ConvertTosaToLinalgSaveTensor.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace {
class TosaFullConnectLoweringToLinalg
    : public OpConversionPattern<tosa::FullyConnectedOp> {
public:
  using OpConversionPattern<tosa::FullyConnectedOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(tosa::FullyConnectedOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto outputResultType = op.getResult().getType().cast<RankedTensorType>();
    auto outputTensor = rewriter.create<tensor::EmptyOp>(
        op.getLoc(), outputResultType.getShape(),
        outputResultType.getElementType());
    auto zero = rewriter.create<arith::ConstantOp>(
        op.getLoc(), rewriter.getZeroAttr(outputResultType.getElementType()));
    auto fill = rewriter.create<linalg::FillOp>(op.getLoc(), zero.getResult(),
                                                outputTensor.getResult());

    auto weightType = op.getWeight().getType().cast<RankedTensorType>();
    llvm::SmallVector<int64_t, 2> weightDims{weightType.getShape()[1],
                                             weightType.getShape()[0]};
    auto transposeWeightType =
        RankedTensorType::get(weightDims, weightType.getElementType());
    auto permVal = rewriter.getI64TensorAttr({1, 0});
    auto perm =
        rewriter.create<tosa::ConstOp>(op.getLoc(), permVal.getType(), permVal);
    auto transpose = rewriter.create<tosa::TransposeOp>(
        op.getLoc(), transposeWeightType, op.getWeight(), perm.getResult());

    auto matmul = rewriter.create<linalg::MatmulOp>(
        op.getLoc(), ValueRange{op.getInput(), transpose.getResult()},
        fill.getResult(0));

    llvm::SmallVector<AffineMap> affineMaps;
    affineMaps.push_back(AffineMap::get(2, 0, {rewriter.getAffineDimExpr(1)}));
    affineMaps.push_back(rewriter.getMultiDimIdentityMap(2));
    affineMaps.push_back(rewriter.getMultiDimIdentityMap(2));
    llvm::SmallVector<utils::IteratorType> iters(2,
                                                 utils::IteratorType::parallel);
    auto linalgOp = rewriter.create<linalg::GenericOp>(
        op.getLoc(), outputResultType,
        ValueRange{op.getBias(), matmul.getResult(0)}, matmul.getResult(0),
        affineMaps, iters,
        [&](OpBuilder &opBuilder, Location loc, ValueRange blockArgs) {
          Value add;
          if (outputResultType.getElementType().isa<FloatType>()) {
            auto addOp = opBuilder.create<arith::AddFOp>(
                loc, outputResultType.getElementType(), blockArgs[0],
                blockArgs[1]);
            add = addOp.getResult();
          } else {
            auto addOp = opBuilder.create<arith::AddIOp>(
                loc, outputResultType.getElementType(), blockArgs[0],
                blockArgs[1]);
            add = addOp.getResult();
          }
          opBuilder.create<linalg::YieldOp>(loc, add);
        });

    rewriter.replaceOp(op, linalgOp);
    return success();
  }
};
} // namespace

void multi_device::conversion::populateTosaFullConnectToLinalgConversionPattern(
    mlir::ConversionTarget &target, mlir::RewritePatternSet &patterns,
    mlir::MLIRContext &ctx) {
  target.addIllegalOp<tosa::FullyConnectedOp>();
  patterns.insert<TosaFullConnectLoweringToLinalg>(&ctx);
}