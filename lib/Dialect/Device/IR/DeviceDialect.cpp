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
// Device_MatmulOp
//===----------------------------------------------------------------------===//

LogicalResult MatmulOp::verify() {
  auto inputShape = getInput().getType().getShape(),
       weightShape = getWeight().getType().getShape(),
       biasShape = getBias().getType().getShape(),
       outputShape = getOutput().getType().getShape();
  if (weightShape.size() != 2 || inputShape.back() != weightShape[0]) {
    return failure();
  }
  if (biasShape.back() != weightShape[1]) {
    return failure();
  }
  if (biasShape.size() > 1) {
    for (auto shape : biasShape.drop_back()) {
      if (shape != 1) {
        return failure();
      }
    }
  }
  if (inputShape.size() != outputShape.size() ||
      outputShape.back() != weightShape.back()) {
    return failure();
  }
  for (auto [input, output] :
       llvm::zip(inputShape.drop_back(), outputShape.drop_back())) {
    if (input != output) {
      return failure();
    }
  }
  return success();
}

void MatmulOp::getEffects(
    llvm::SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), getInput());
  effects.emplace_back(MemoryEffects::Read::get(), getWeight());
  effects.emplace_back(MemoryEffects::Read::get(), getBias());
  effects.emplace_back(MemoryEffects::Write::get(), getOutput());
}

void MatmulOp::addAsyncDependency(Value token) {
  this->getAsyncDependenciesMutable().append(token);
}

#include "multi-device-model-compiler/Dialect/Device/IR/DeviceOpsEnums.cpp.inc"

using namespace mlir;
#define GET_ATTRDEF_CLASSES
#include "multi-device-model-compiler/Dialect/Device/IR/DeviceOpsAttributes.cpp.inc"

#define GET_OP_CLASSES
#include "multi-device-model-compiler/Dialect/Device/IR/DeviceOps.cpp.inc"