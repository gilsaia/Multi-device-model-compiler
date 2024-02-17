#include "multi-device-model-compiler/Conversion/ConvertDeviceToLLVM/ConvertDeviceToLLVM.h"
#include "multi-device-model-compiler/Dialect/Device/IR/Device.h"

#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/AsyncToLLVM/AsyncToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/GPUCommon/GPUCommonPass.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/LoweringOptions.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"

using namespace mlir;

namespace multi_device {
#define GEN_PASS_DEF_CONVERTDEVICETOLLVM
#include "multi-device-model-compiler/Conversion/Passes.h.inc"
} // namespace multi_device

namespace {
class ConvertDeviceToLLVMPass final
    : public multi_device::impl::ConvertDeviceToLLVMBase<
          ConvertDeviceToLLVMPass> {
public:
  ConvertDeviceToLLVMPass() = default;
  ConvertDeviceToLLVMPass(
      const multi_device::ConvertDeviceToLLVMOptions &options)
      : ConvertDeviceToLLVMBase(options) {}
  void runOnOperation() override final;
};
} // namespace

void ConvertDeviceToLLVMPass::runOnOperation() {
  LowerToLLVMOptions options(&getContext());
  options.useOpaquePointers = useOpaquePointers;
  options.useBarePtrCallConv = hostBarePtrCallConv;

  LLVMTypeConverter converter(&getContext(), options);
  RewritePatternSet patterns(&getContext());
  LLVMConversionTarget target(getContext());

  ModuleOp moduleOp = getOperation();
  auto device = moduleOp->getAttr("module.device")
                    .cast<multi_device::device::DeviceTypeAttr>()
                    .getValue();

  SymbolTable symbolTable = SymbolTable(getOperation());
  // Preserve GPU modules if they have target attributes.
  target.addDynamicallyLegalOp<gpu::GPUModuleOp>(
      [](gpu::GPUModuleOp module) -> bool {
        return module.getTargetsAttr() != nullptr;
      });
  // Accept as legal LaunchFuncOps if they refer to GPU Modules with targets and
  // the operands have been lowered.
  target.addDynamicallyLegalOp<gpu::LaunchFuncOp>(
      [&](gpu::LaunchFuncOp op) -> bool {
        auto module =
            symbolTable.lookup<gpu::GPUModuleOp>(op.getKernelModuleName());
        return converter.isLegal(op->getOperandTypes()) &&
               converter.isLegal(op->getResultTypes()) &&
               (module && module.getTargetsAttr() &&
                module.getTargetsAttr().size());
      });

  mlir::arith::populateArithToLLVMConversionPatterns(converter, patterns);
  mlir::cf::populateControlFlowToLLVMConversionPatterns(converter, patterns);
  populateVectorToLLVMConversionPatterns(converter, patterns);
  populateFinalizeMemRefToLLVMConversionPatterns(converter, patterns);
  populateFuncToLLVMConversionPatterns(converter, patterns);
  populateAsyncStructuralTypeConversionsAndLegality(converter, patterns,
                                                    target);
  populateGpuToLLVMConversionPatterns(converter, patterns, gpuBinaryAnnotation,
                                      kernelBarePtrCallConv, &symbolTable);
  multi_device::conversion::populateDeviceToLLVMConversionPatterns(
      converter, patterns, &symbolTable, device);

  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}

template <typename OpTy>
class ConvertOpToDeviceRuntimeCallPattern
    : public ConvertOpToLLVMPattern<OpTy> {
public:
  explicit ConvertOpToDeviceRuntimeCallPattern(
      const LLVMTypeConverter &TypeConverter, SymbolTable *cachedModuleTable)
      : ConvertOpToLLVMPattern<OpTy>(TypeConverter),
        cachedModuleTable(cachedModuleTable) {}
  MLIRContext *context = &this->getTypeConverter()->getContext();
  Type llvmVoidType = LLVM::LLVMVoidType::get(context);
  LLVM::LLVMPointerType llvmPointerType =
      this->getTypeConverter()->getPointerType(IntegerType::get(context, 8));
  Type llvmPointerPointerType =
      this->getTypeConverter()->getPointerType(llvmPointerType);
  Type llvmInt8Type = IntegerType::get(context, 8);
  Type llvmInt16Type = IntegerType::get(context, 16);
  Type llvmInt32Type = IntegerType::get(context, 32);
  Type llvmInt64Type = IntegerType::get(context, 64);
  Type llvmFloat32Type = Float32Type::get(context);
  Type llvmInt8PointerType =
      this->getTypeConverter()->getPointerType(llvmInt8Type);
  Type llvmInt64PointerType =
      this->getTypeConverter()->getPointerType(llvmInt64Type);
  Type llvmIntPtrType = IntegerType::get(
      context, this->getTypeConverter()->getPointerBitwidth(0));
  FunctionCallBuilder streamCreateCallBuilder = {
      "mgpuStreamCreate", llvmPointerType /* void *stream */, {}};
  FunctionCallBuilder streamDestroyCallBuilder = {
      "mgpuStreamDestroy", llvmVoidType, {llvmPointerType /* void *stream */}};
  FunctionCallBuilder streamSynchronizeCallBuilder = {
      "mgpuStreamSynchronize",
      llvmVoidType,
      {llvmPointerType /* void *stream */}};
  FunctionCallBuilder streamWaitEventCallBuilder = {
      "mgpuStreamWaitEvent",
      llvmVoidType,
      {llvmPointerType /* void *stream */, llvmPointerType /* void *event */}};
  FunctionCallBuilder eventCreateCallBuilder = {
      "mgpuEventCreate", llvmPointerType /* void *event */, {}};
  FunctionCallBuilder eventCreateWithStreamCallBuilder = {
      "mgpuEventCreateWithStream",
      llvmPointerType /* void *event */,
      {llvmPointerType}};
  FunctionCallBuilder eventDestroyCallBuilder = {
      "mgpuEventDestroy", llvmVoidType, {llvmPointerType /* void *event */}};
  FunctionCallBuilder eventSynchronizeCallBuilder = {
      "mgpuEventSynchronize",
      llvmVoidType,
      {llvmPointerType /* void *event */}};
  FunctionCallBuilder eventRecordCallBuilder = {
      "mgpuEventRecord",
      llvmVoidType,
      {llvmPointerType /* void *event */, llvmPointerType /* void *stream */}};

  /*
  Ops Function Call
  */
  FunctionCallBuilder cpuMatmulCallBuilder = {
      "mcpuMatmul",
      llvmVoidType,
      {llvmPointerType /* input */, llvmPointerType /* weight */,
       llvmPointerType /* bias */, llvmPointerType /* output */,
       llvmInt64Type /* M */, llvmInt64Type /* N */, llvmInt64Type /* K */}};
  FunctionCallBuilder gpuMatmulCallBuilder = {
      "mgpuMatmul",
      llvmVoidType,
      {llvmPointerType /* input */, llvmPointerType /* weight */,
       llvmPointerType /* bias */, llvmPointerType /* output */,
       llvmInt64Type /* M */, llvmInt64Type /* N */, llvmInt64Type /* K */}};

protected:
  SymbolTable *cachedModuleTable;
};

class ConvertWaitOpToDeviceRuntimeCallPattern
    : public ConvertOpToDeviceRuntimeCallPattern<multi_device::device::WaitOp> {
public:
  ConvertWaitOpToDeviceRuntimeCallPattern(
      const LLVMTypeConverter &typeConverter, SymbolTable *cachedModuleTable)
      : ConvertOpToDeviceRuntimeCallPattern<multi_device::device::WaitOp>(
            typeConverter, cachedModuleTable) {}

private:
  LogicalResult
  matchAndRewrite(multi_device::device::WaitOp waitOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

class ConvertWaitAsyncOpToDeviceRuntimeCallPattern
    : public ConvertOpToDeviceRuntimeCallPattern<multi_device::device::WaitOp> {
public:
  ConvertWaitAsyncOpToDeviceRuntimeCallPattern(
      const LLVMTypeConverter &typeConverter, SymbolTable *cachedModuleTable)
      : ConvertOpToDeviceRuntimeCallPattern<multi_device::device::WaitOp>(
            typeConverter, cachedModuleTable) {}

private:
  LogicalResult
  matchAndRewrite(multi_device::device::WaitOp waitOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

class ConvertRecordOpToDeviceRuntimeCallPattern
    : public ConvertOpToDeviceRuntimeCallPattern<
          multi_device::device::RecordOp> {
public:
  ConvertRecordOpToDeviceRuntimeCallPattern(
      const LLVMTypeConverter &typeConverter, SymbolTable *cachedModuleTable)
      : ConvertOpToDeviceRuntimeCallPattern<multi_device::device::RecordOp>(
            typeConverter, cachedModuleTable) {}

private:
  LogicalResult
  matchAndRewrite(multi_device::device::RecordOp recordOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

class ConvertMatmulOpToDeviceRuntimeCallPattern
    : public ConvertOpToDeviceRuntimeCallPattern<
          multi_device::device::MatmulOp> {
public:
  ConvertMatmulOpToDeviceRuntimeCallPattern(
      const LLVMTypeConverter &typeConverter, SymbolTable *cachedModuleTable,
      multi_device::device::DeviceType device)
      : ConvertOpToDeviceRuntimeCallPattern<multi_device::device::MatmulOp>(
            typeConverter, cachedModuleTable),
        device(device) {}
  LogicalResult
  matchAndRewrite(multi_device::device::MatmulOp matmulOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;

private:
  multi_device::device::DeviceType device;
};

static LLVM::CallOp getEventByStream(Value stream) {
  LLVM::CallOp event;
  for (auto *user : stream.getUsers()) {
    auto call = mlir::dyn_cast<LLVM::CallOp>(user);
    if (!call) {
      continue;
    }
    if (call.getCallee()->equals("mgpuEventCreateWithStream")) {
      event = call;
      break;
    }
  }
  return event;
}

LogicalResult ConvertWaitOpToDeviceRuntimeCallPattern::matchAndRewrite(
    multi_device::device::WaitOp waitOp, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  if (waitOp.getAsyncToken()) {
    return rewriter.notifyMatchFailure(waitOp, "Cannot convert async wait.");
  }
  Location loc = waitOp.getLoc();

  for (auto operand : adaptor.getOperands()) {
    auto event = getEventByStream(operand);
    if (!event) {
      return rewriter.notifyMatchFailure(waitOp,
                                         "Can't find event for this stream");
    }
    eventDestroyCallBuilder.create(loc, rewriter, {event.getResult()});
    streamSynchronizeCallBuilder.create(loc, rewriter, {operand});
    streamDestroyCallBuilder.create(loc, rewriter, {operand});
  }

  rewriter.eraseOp(waitOp);
  return success();
}

LogicalResult ConvertWaitAsyncOpToDeviceRuntimeCallPattern::matchAndRewrite(
    multi_device::device::WaitOp waitOp, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  if (!waitOp.getAsyncToken()) {
    return rewriter.notifyMatchFailure(waitOp, "Can only convert async op.");
  }

  Location loc = waitOp.getLoc();

  auto stream = streamCreateCallBuilder.create(loc, rewriter, {});
  eventCreateWithStreamCallBuilder.create(loc, rewriter, {stream.getResult()});

  rewriter.replaceOp(waitOp, {stream});
  return success();
}

LogicalResult ConvertRecordOpToDeviceRuntimeCallPattern::matchAndRewrite(
    multi_device::device::RecordOp recordOp, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  if (adaptor.getSrc() == adaptor.getDst()) {
    // same stream,just erase
    rewriter.eraseOp(recordOp);
    return success();
  }
  auto stream = adaptor.getSrc(), dstStream = adaptor.getDst();
  auto event = getEventByStream(stream);
  if (!event) {
    return rewriter.notifyMatchFailure(recordOp, "Can't find event.");
  }
  auto loc = recordOp.getLoc();
  eventRecordCallBuilder.create(loc, rewriter, {event.getResult(), stream});
  streamWaitEventCallBuilder.create(loc, rewriter,
                                    {dstStream, event.getResult()});

  rewriter.eraseOp(recordOp);
  return success();
}

LogicalResult ConvertMatmulOpToDeviceRuntimeCallPattern::matchAndRewrite(
    multi_device::device::MatmulOp matmulOp, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  int64_t m = 1, n = 1, k = 1;
  auto inputShape = matmulOp.getInput().getType().getShape(),
       weightShape = matmulOp.getWeight().getType().getShape();
  for (auto &shape : inputShape.drop_back()) {
    m *= shape;
  }
  k = inputShape.back(), n = weightShape.back();

  auto loc = matmulOp.getLoc();
  Value mVal = rewriter.create<LLVM::ConstantOp>(loc, llvmInt64Type, m),
        kVal = rewriter.create<LLVM::ConstantOp>(loc, llvmInt64Type, k),
        nVal = rewriter.create<LLVM::ConstantOp>(loc, llvmInt64Type, n);

  SmallVector<Value, 4> opOperandsVec{matmulOp.getInput(), matmulOp.getWeight(),
                                      matmulOp.getBias(), matmulOp.getOutput()},
      operandsVec{adaptor.getInput(), adaptor.getWeight(), adaptor.getBias(),
                  adaptor.getOutput()};

  llvm::SmallVector<Value, 4> arguments = getTypeConverter()->promoteOperands(
      loc, opOperandsVec, operandsVec, rewriter, true);

  if (device == multi_device::device::DeviceType::CPU) {
    cpuMatmulCallBuilder.create(loc, rewriter,
                                {arguments[0], arguments[1], arguments[2],
                                 arguments[3], mVal, nVal, kVal});
  } else if (device == multi_device::device::DeviceType::GPU) {
    gpuMatmulCallBuilder.create(loc, rewriter,
                                {arguments[0], arguments[1], arguments[2],
                                 arguments[3], mVal, nVal, kVal});
  } else {
    return rewriter.notifyMatchFailure(matmulOp, "Wrong device");
  }

  // async control
  if (matmulOp.getAsyncToken()) {
    if (matmulOp.getAsyncDependencies().size() != 1) {
      return rewriter.notifyMatchFailure(matmulOp, "Wrong async dependency");
    }
    rewriter.replaceOp(matmulOp, {matmulOp.getAsyncDependencies()[0]});
  } else {
    rewriter.eraseOp(matmulOp);
  }

  return success();
}

void multi_device::conversion::populateDeviceToLLVMConversionPatterns(
    mlir::LLVMTypeConverter &converter, mlir::RewritePatternSet &patterns,
    mlir::SymbolTable *cachedModuleTable, device::DeviceType device) {
  patterns.add<ConvertWaitOpToDeviceRuntimeCallPattern,
               ConvertWaitAsyncOpToDeviceRuntimeCallPattern,
               ConvertRecordOpToDeviceRuntimeCallPattern>(converter,
                                                          cachedModuleTable);
  patterns.add<ConvertMatmulOpToDeviceRuntimeCallPattern>(
      converter, cachedModuleTable, device);
}