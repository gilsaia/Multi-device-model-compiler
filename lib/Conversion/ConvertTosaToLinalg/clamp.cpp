#include "multi-device-model-compiler/Conversion/ConvertTosaToLinalg/ConvertTosaToLinalgSaveTensor.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

static Value createLinalgBodyForClamp(OpBuilder &builder, Location loc,
                                      Value input, Type resultType,
                                      tosa::ClampOp op) {
  APFloat minApf = op.getMinFp();
  APFloat maxApf = op.getMaxFp();
  bool needCheckMax = !(maxApf == APFloat::getLargest(maxApf.getSemantics()));
  auto min = builder.create<arith::ConstantOp>(
      loc, resultType, builder.getFloatAttr(resultType, minApf));
  auto MaxResult = builder.create<arith::MaxFOp>(loc, input, min.getResult());
  if (needCheckMax) {
    auto max = builder.create<arith::ConstantOp>(
        loc, resultType, builder.getFloatAttr(resultType, maxApf));
    auto MinResult = builder.create<arith::MinFOp>(loc, MaxResult.getResult(),
                                                   max.getResult());
    return MinResult.getResult();
  }
  return MaxResult.getResult();
}

namespace {
class TosaClampOpLoweringToLinalg : public OpConversionPattern<tosa::ClampOp> {
public:
  using OpConversionPattern<tosa::ClampOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(tosa::ClampOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto resultType = op.getResult().getType().cast<RankedTensorType>();
    auto input = op.getInput();

    auto rank = resultType.getRank();
    llvm::SmallVector<AffineMap> affineMaps;
    affineMaps.push_back(rewriter.getMultiDimIdentityMap(rank));
    affineMaps.push_back(rewriter.getMultiDimIdentityMap(rank));

    llvm::SmallVector<utils::IteratorType> iters(rank,
                                                 utils::IteratorType::parallel);
    auto linalgOp = rewriter.create<linalg::GenericOp>(
        op.getLoc(), resultType, input, input, affineMaps, iters,
        [&](OpBuilder &opBuilder, Location loc, ValueRange blockArgs) {
          Value opResult =
              createLinalgBodyForClamp(opBuilder, loc, blockArgs.front(),
                                       resultType.getElementType(), op);
          opBuilder.create<linalg::YieldOp>(loc, opResult);
        });

    rewriter.replaceOp(op, linalgOp);

    return success();
  }
};
} // namespace

void multi_device::conversion::populateTosaClampToLinalgConversionPattern(
    mlir::ConversionTarget &target, mlir::RewritePatternSet &patterns,
    mlir::MLIRContext &ctx) {
  target.addDynamicallyLegalOp<tosa::ClampOp>([](tosa::ClampOp op) {
    auto input = op.getInput();
    for (auto user : input.getUsers()) {
      if (user != op) {
        return true;
      }
    }
    return false;
  });
  patterns.insert<TosaClampOpLoweringToLinalg>(&ctx);
}