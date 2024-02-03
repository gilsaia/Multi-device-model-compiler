#include "mlir/Dialect/GPU/IR/CompilationInterfaces.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TypeSwitch.h"

#include "multi-device-model-compiler/Dialect/Device/IR/Device.h"

using namespace multi_device;
using namespace multi_device::device;

#include "multi-device-model-compiler/Dialect/Device/IR/DeviceOpsDialect.cpp.inc"

void device::DeviceDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "multi-device-model-compiler/Dialect/Device/IR/DeviceOps.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "multi-device-model-compiler/Dialect/Device/IR/DeviceOpsAttributes.cpp.inc"
      >();
}

namespace {
/// Parses an optional list of async operands with an optional leading
/// keyword.
/// (`async`)? (`[` ssa-id-list `]`)?
///
/// This method is used by the tablegen assembly format for async ops as well.
static ParseResult parseAsyncDependencies(
    OpAsmParser &parser, Type &asyncTokenType,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &asyncDependencies) {
  auto loc = parser.getCurrentLocation();
  if (succeeded(parser.parseOptionalKeyword("async"))) {
    if (parser.getNumResults() == 0)
      return parser.emitError(loc, "needs to be named when marked 'async'");
    asyncTokenType = parser.getBuilder().getType<mlir::gpu::AsyncTokenType>();
  }
  return parser.parseOperandList(asyncDependencies,
                                 OpAsmParser::Delimiter::OptionalSquare);
}

/// Prints optional async dependencies with its leading keyword.
///   (`async`)? (`[` ssa-id-list `]`)?
// Used by the tablegen assembly format for several async ops.
static void printAsyncDependencies(OpAsmPrinter &printer, Operation *op,
                                   Type asyncTokenType,
                                   OperandRange asyncDependencies) {
  if (asyncTokenType)
    printer << "async";
  if (asyncDependencies.empty())
    return;
  if (asyncTokenType)
    printer << ' ';
  printer << '[';
  llvm::interleaveComma(asyncDependencies, printer);
  printer << ']';
}
} // namespace

//===----------------------------------------------------------------------===//
// Device_WaitOp
//===----------------------------------------------------------------------===//

namespace {

struct EraseRedundantDeviceWaitOpPairs : public OpRewritePattern<WaitOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(WaitOp op,
                                PatternRewriter &rewriter) const final {
    auto predicate = [](Value value) {
      auto waitOp = value.getDefiningOp<WaitOp>();
      return waitOp && waitOp->getNumOperands() == 0;
    };
    if (llvm::none_of(op.getAsyncDependencies(), predicate))
      return failure();
    SmallVector<Value> validOperands;
    for (Value operand : op->getOperands()) {
      if (predicate(operand))
        continue;
      validOperands.push_back(operand);
    }
    rewriter.updateRootInPlace(op, [&]() { op->setOperands(validOperands); });
    return success();
  }
};

struct SimplifyDeviceWaitOp : public OpRewritePattern<WaitOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(WaitOp op,
                                PatternRewriter &rewriter) const final {
    // Erase device.wait ops that neither have any async dependencies nor return
    // any async token.
    if (op.getAsyncDependencies().empty() && !op.getAsyncToken()) {
      rewriter.eraseOp(op);
      return success();
    }
    // Replace uses of %t1 = device.wait async [%t0] ops with %t0 and erase the
    // op.
    if (llvm::hasSingleElement(op.getAsyncDependencies()) &&
        op.getAsyncToken()) {
      rewriter.replaceOp(op, op.getAsyncDependencies());
      return success();
    }
    // Erase %t = device.wait async ... ops, where %t has no uses.
    if (op.getAsyncToken() && op.getAsyncToken().use_empty()) {
      rewriter.eraseOp(op);
      return success();
    }
    return failure();
  }
};

} // end anonymous namespace

void WaitOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                         MLIRContext *context) {
  results.add<EraseRedundantDeviceWaitOpPairs, SimplifyDeviceWaitOp>(context);
}

#include "multi-device-model-compiler/Dialect/Device/IR/DeviceOpsEnums.cpp.inc"

using namespace mlir;
#define GET_ATTRDEF_CLASSES
#include "multi-device-model-compiler/Dialect/Device/IR/DeviceOpsAttributes.cpp.inc"

#define GET_OP_CLASSES
#include "multi-device-model-compiler/Dialect/Device/IR/DeviceOps.cpp.inc"