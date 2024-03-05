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
  Type llvmBoolType = IntegerType::get(context, 1);
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
       llvmInt64Type /* M */, llvmInt64Type /* N */, llvmInt64Type /* K */,
       llvmPointerType /* stream */}};
  FunctionCallBuilder cpuConv2dCallBuilder = {
      "mcpuConv2d",
      llvmVoidType,
      {
          llvmPointerType /* input */,    llvmPointerType /* weight */,
          llvmPointerType /* bias */,     llvmPointerType /* output */,
          llvmPointerType /* post add */, llvmInt64Type /* N */,
          llvmInt64Type /* IC */,         llvmInt64Type /* H */,
          llvmInt64Type /* W */,          llvmInt64Type /* OC */,
          llvmInt64Type /* KH */,         llvmInt64Type /* KW */,
          llvmInt64Type /* OH */,         llvmInt64Type /* OW */,
          llvmInt64Type /* PHL */,        llvmInt64Type /* PWL */,
          llvmInt64Type /* PHR */,        llvmInt64Type /* PWR */,
          llvmInt64Type /* SH */,         llvmInt64Type /* SW */,
          llvmInt64Type /* DH */,         llvmInt64Type /* DW */,
          llvmBoolType /* post add */,    llvmBoolType /* contain relu */
      }};
  FunctionCallBuilder gpuConv2dCallBuilder = {
      "mgpuConv2d",
      llvmVoidType,
      {
          llvmPointerType /* input */,
          llvmPointerType /* weight */,
          llvmPointerType /* bias */,
          llvmPointerType /* output */,
          llvmPointerType /* post add */,
          llvmInt64Type /* N */,
          llvmInt64Type /* IC */,
          llvmInt64Type /* H */,
          llvmInt64Type /* W */,
          llvmInt64Type /* OC */,
          llvmInt64Type /* KH */,
          llvmInt64Type /* KW */,
          llvmInt64Type /* OH */,
          llvmInt64Type /* OW */,
          llvmInt64Type /* PHL */,
          llvmInt64Type /* PWL */,
          llvmInt64Type /* PHR */,
          llvmInt64Type /* PWR */,
          llvmInt64Type /* SH */,
          llvmInt64Type /* SW */,
          llvmInt64Type /* DH */,
          llvmInt64Type /* DW */,
          llvmBoolType /* post add */,
          llvmBoolType /* contain relu */,
          llvmPointerType /* stream */
      }};
  FunctionCallBuilder cpuPool2dCallBuilder = {
      "mcpuPool2d",
      llvmVoidType,
      {
          llvmPointerType /* input */, llvmPointerType /* output */,
          llvmInt64Type /* N */, llvmInt64Type /* C */, llvmInt64Type /* H */,
          llvmInt64Type /* W */, llvmInt64Type /* OH */, llvmInt64Type /* OW */,
          llvmInt64Type /* KH */, llvmInt64Type /* KW */,
          llvmInt64Type /* PHL */, llvmInt64Type /* PWL */,
          llvmInt64Type /* PHR */, llvmInt64Type /* PWR */,
          llvmInt64Type /* SH */, llvmInt64Type /* SW */,
          llvmInt64Type /* method,which 0 stand for maxpool,1 stnad for avgpool
                         */
      }};
  FunctionCallBuilder gpuPool2dCallBuilder = {
      "mgpuPool2d",
      llvmVoidType,
      {
          llvmPointerType /* input */, llvmPointerType /* output */,
          llvmInt64Type /* N */, llvmInt64Type /* C */, llvmInt64Type /* H */,
          llvmInt64Type /* W */, llvmInt64Type /* OH */, llvmInt64Type /* OW */,
          llvmInt64Type /* KH */, llvmInt64Type /* KW */,
          llvmInt64Type /* PHL */, llvmInt64Type /* PWL */,
          llvmInt64Type /* PHR */, llvmInt64Type /* PWR */,
          llvmInt64Type /* SH */, llvmInt64Type /* SW */,
          llvmInt64Type /* method */, llvmPointerType /* stream */
      }};
  FunctionCallBuilder cpuMultiHeadAttentionLayerBuild = {
      "mcpuLLMDecodingContextLayer",
      llvmVoidType,
      {
          llvmPointerType /*input*/, llvmPointerType /*qkv*/,
          llvmPointerType /*attn gemm weight*/,
          llvmPointerType /*attn gemm bias*/, llvmPointerType /*ffn1 weight*/,
          llvmPointerType /*ffn1 bias*/, llvmPointerType /*ffn2 weight*/,
          llvmPointerType /*ffn2 bias*/, llvmPointerType /*output*/,
          llvmInt64Type /*batch*/, llvmInt64Type /*seq_len*/,
          llvmInt64Type /*d_model*/, llvmInt64Type /*feed_forward_dim*/,
          llvmInt64Type /*head_num*/, llvmBoolType /*norm first*/,
          llvmBoolType /*is_casual*/, llvmBoolType /*is_relu*/
      }};
  FunctionCallBuilder gpuMultiHeadAttentionLayerBuild = {
      "mgpuLLMDecodingContextLayer",
      llvmVoidType,
      {
          llvmPointerType /*input*/, llvmPointerType /*qkv*/,
          llvmPointerType /*attn gemm weight*/,
          llvmPointerType /*attn gemm bias*/, llvmPointerType /*ffn1 weight*/,
          llvmPointerType /*ffn1 bias*/, llvmPointerType /*ffn2 weight*/,
          llvmPointerType /*ffn2 bias*/, llvmPointerType /*output*/,
          llvmInt64Type /*batch*/, llvmInt64Type /*seq_len*/,
          llvmInt64Type /*d_model*/, llvmInt64Type /*feed_forward_dim*/,
          llvmInt64Type /*head_num*/, llvmBoolType /*norm first*/,
          llvmBoolType /*is_casual*/, llvmBoolType /*is_relu*/,
          llvmPointerType /* stream */
      }};

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

template <typename OpTy>
class ConvertDeviceOpToDeviceRuntimeCallPattern
    : public ConvertOpToDeviceRuntimeCallPattern<OpTy> {
public:
  ConvertDeviceOpToDeviceRuntimeCallPattern(
      const LLVMTypeConverter &typeConverter, SymbolTable *cachedModuleTable,
      multi_device::device::DeviceType device)
      : ConvertOpToDeviceRuntimeCallPattern<OpTy>(typeConverter,
                                                  cachedModuleTable),
        device(device) {}

protected:
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

class ConvertMatmulOpToDeviceRuntimeCallPattern
    : public ConvertDeviceOpToDeviceRuntimeCallPattern<
          multi_device::device::MatmulOp> {
public:
  using ConvertDeviceOpToDeviceRuntimeCallPattern<
      multi_device::device::MatmulOp>::
      ConvertDeviceOpToDeviceRuntimeCallPattern;
  LogicalResult
  matchAndRewrite(multi_device::device::MatmulOp matmulOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

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
                                 arguments[3], mVal, nVal, kVal,
                                 adaptor.getAsyncDependencies()[0]});
  } else {
    return rewriter.notifyMatchFailure(matmulOp, "Wrong device");
  }

  // async control
  if (matmulOp.getAsyncToken()) {
    rewriter.replaceOp(matmulOp, {matmulOp.getAsyncDependencies()[0]});
  } else {
    rewriter.eraseOp(matmulOp);
  }

  return success();
}

class ConvertConv2dOpToDeviceRuntimeCallPattern
    : public ConvertDeviceOpToDeviceRuntimeCallPattern<
          multi_device::device::Conv2DOp> {
public:
  using ConvertDeviceOpToDeviceRuntimeCallPattern<
      multi_device::device::Conv2DOp>::
      ConvertDeviceOpToDeviceRuntimeCallPattern;
  LogicalResult
  matchAndRewrite(multi_device::device::Conv2DOp conv2dOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

LogicalResult ConvertConv2dOpToDeviceRuntimeCallPattern::matchAndRewrite(
    multi_device::device::Conv2DOp conv2dOp, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  int64_t n, c, h, w, oc, kh, kw, oh, ow, phl, pwl, phr, pwr, sh, sw, dh, dw;
  auto inputShape = conv2dOp.getInput().getType().getShape();
  auto outputShape = conv2dOp.getOutput().getType().getShape();
  auto weightShape = conv2dOp.getWeight().getType().getShape();

  n = inputShape[0], c = inputShape[1], h = inputShape[2], w = inputShape[3],
  oc = outputShape[1], kh = weightShape[2], kw = weightShape[3],
  oh = outputShape[2], ow = outputShape[3];

  auto padding = conv2dOp.getPad();
  auto stride = conv2dOp.getStride();
  auto dilation = conv2dOp.getDilation();

  phl = padding[0], pwl = padding[1], phr = padding[2], pwr = padding[3],
  sh = stride[0], sw = stride[1], dh = dilation[0], dw = dilation[1];

  bool postAdd, containRelu;
  if (conv2dOp.getPostadd()) {
    postAdd = true;
  }
  if (conv2dOp.getContainRelu()) {
    containRelu = true;
  }

  auto loc = conv2dOp.getLoc();
  Value nVal = rewriter.create<LLVM::ConstantOp>(loc, llvmInt64Type, n),
        cVal = rewriter.create<LLVM::ConstantOp>(loc, llvmInt64Type, c),
        hVal = rewriter.create<LLVM::ConstantOp>(loc, llvmInt64Type, h),
        wVal = rewriter.create<LLVM::ConstantOp>(loc, llvmInt64Type, w),
        ocVal = rewriter.create<LLVM::ConstantOp>(loc, llvmInt64Type, oc),
        khVal = rewriter.create<LLVM::ConstantOp>(loc, llvmInt64Type, kh),
        kwVal = rewriter.create<LLVM::ConstantOp>(loc, llvmInt64Type, kw),
        ohVal = rewriter.create<LLVM::ConstantOp>(loc, llvmInt64Type, oh),
        owVal = rewriter.create<LLVM::ConstantOp>(loc, llvmInt64Type, ow),
        phlVal = rewriter.create<LLVM::ConstantOp>(loc, llvmInt64Type, phl),
        pwlVal = rewriter.create<LLVM::ConstantOp>(loc, llvmInt64Type, pwl),
        phrVal = rewriter.create<LLVM::ConstantOp>(loc, llvmInt64Type, phr),
        pwrVal = rewriter.create<LLVM::ConstantOp>(loc, llvmInt64Type, pwr),
        shVal = rewriter.create<LLVM::ConstantOp>(loc, llvmInt64Type, sh),
        swVal = rewriter.create<LLVM::ConstantOp>(loc, llvmInt64Type, sw),
        dhVal = rewriter.create<LLVM::ConstantOp>(loc, llvmInt64Type, dh),
        dwVal = rewriter.create<LLVM::ConstantOp>(loc, llvmInt64Type, dw),
        postAddVal =
            rewriter.create<LLVM::ConstantOp>(loc, llvmBoolType, postAdd),
        containReluVal =
            rewriter.create<LLVM::ConstantOp>(loc, llvmBoolType, containRelu);

  SmallVector<Value, 4> opOperandsVec{conv2dOp.getInput(), conv2dOp.getWeight(),
                                      conv2dOp.getBias(), conv2dOp.getOutput()},
      operandsVec{adaptor.getInput(), adaptor.getWeight(), adaptor.getBias(),
                  adaptor.getOutput()};

  SmallVector<Value, 4> arguments = getTypeConverter()->promoteOperands(
      loc, opOperandsVec, operandsVec, rewriter, true);

  Value postAddArgument;
  if (postAdd) {
    auto postargs = getTypeConverter()->promoteOperands(
        loc, {conv2dOp.getPostadd()}, {adaptor.getPostadd()}, rewriter, true);
    postAddArgument = postargs[0];
  } else {
    postAddArgument = rewriter.create<LLVM::NullOp>(loc, llvmPointerType);
  }

  if (device == multi_device::device::DeviceType::CPU) {
    cpuConv2dCallBuilder.create(loc, rewriter, {arguments[0],
                                                arguments[1],
                                                arguments[2],
                                                arguments[3],
                                                postAddArgument,
                                                nVal,
                                                cVal,
                                                hVal,
                                                wVal,
                                                ocVal,
                                                khVal,
                                                kwVal,
                                                ohVal,
                                                owVal,
                                                phlVal,
                                                pwlVal,
                                                phrVal,
                                                pwrVal,
                                                shVal,
                                                swVal,
                                                dhVal,
                                                dwVal,
                                                postAddVal,
                                                containReluVal});
  } else if (device == multi_device::device::DeviceType::GPU) {
    gpuConv2dCallBuilder.create(loc, rewriter,
                                {arguments[0],
                                 arguments[1],
                                 arguments[2],
                                 arguments[3],
                                 postAddArgument,
                                 nVal,
                                 cVal,
                                 hVal,
                                 wVal,
                                 ocVal,
                                 khVal,
                                 kwVal,
                                 ohVal,
                                 owVal,
                                 phlVal,
                                 pwlVal,
                                 phrVal,
                                 pwrVal,
                                 shVal,
                                 swVal,
                                 dhVal,
                                 dwVal,
                                 postAddVal,
                                 containReluVal,
                                 adaptor.getAsyncDependencies()[0]});
  } else {
    return rewriter.notifyMatchFailure(conv2dOp, "Wrong device");
  }

  // async control
  if (conv2dOp.getAsyncToken()) {
    rewriter.replaceOp(conv2dOp, {conv2dOp.getAsyncDependencies()[0]});
  } else {
    rewriter.eraseOp(conv2dOp);
  }

  return success();
}

class ConvertPool2dOpToDeviceRuntimeCallPattern
    : public ConvertDeviceOpToDeviceRuntimeCallPattern<
          multi_device::device::Pool2DOp> {
public:
  using ConvertDeviceOpToDeviceRuntimeCallPattern<
      multi_device::device::Pool2DOp>::
      ConvertDeviceOpToDeviceRuntimeCallPattern;
  LogicalResult
  matchAndRewrite(multi_device::device::Pool2DOp pool2dOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

LogicalResult ConvertPool2dOpToDeviceRuntimeCallPattern::matchAndRewrite(
    multi_device::device::Pool2DOp pool2dOp, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  int64_t n, c, h, w, oh, ow, kh, kw, phl, pwl, phr, pwr, sh, sw, method;
  auto inputShape = pool2dOp.getInput().getType().getShape();
  auto outputShape = pool2dOp.getOutput().getType().getShape();

  n = inputShape[0], c = inputShape[1], h = inputShape[2], w = inputShape[3],
  oh = outputShape[2], ow = outputShape[3];

  auto kernel = pool2dOp.getKernel();
  auto padding = pool2dOp.getPad();
  auto stride = pool2dOp.getStride();
  auto poolMethod = pool2dOp.getMethod();
  kh = kernel[0], kw = kernel[1], phl = padding[0], pwl = padding[1],
  phr = padding[2], pwr = padding[3], sh = stride[0], sw = stride[1];
  if (poolMethod == "max") {
    method = 0;
  } else if (poolMethod == "avg") {
    method = 1;
  } else {
    return rewriter.notifyMatchFailure(pool2dOp, "Wrong method");
  }

  auto loc = pool2dOp.getLoc();
  Value nVal = rewriter.create<LLVM::ConstantOp>(loc, llvmInt64Type, n),
        cVal = rewriter.create<LLVM::ConstantOp>(loc, llvmInt64Type, c),
        hVal = rewriter.create<LLVM::ConstantOp>(loc, llvmInt64Type, h),
        wVal = rewriter.create<LLVM::ConstantOp>(loc, llvmInt64Type, w),
        ohVal = rewriter.create<LLVM::ConstantOp>(loc, llvmInt64Type, oh),
        owVal = rewriter.create<LLVM::ConstantOp>(loc, llvmInt64Type, ow),
        khVal = rewriter.create<LLVM::ConstantOp>(loc, llvmInt64Type, kh),
        kwVal = rewriter.create<LLVM::ConstantOp>(loc, llvmInt64Type, kw),
        phlVal = rewriter.create<LLVM::ConstantOp>(loc, llvmInt64Type, phl),
        pwlVal = rewriter.create<LLVM::ConstantOp>(loc, llvmInt64Type, pwl),
        phrVal = rewriter.create<LLVM::ConstantOp>(loc, llvmInt64Type, phr),
        pwrVal = rewriter.create<LLVM::ConstantOp>(loc, llvmInt64Type, pwr),
        shVal = rewriter.create<LLVM::ConstantOp>(loc, llvmInt64Type, sh),
        swVal = rewriter.create<LLVM::ConstantOp>(loc, llvmInt64Type, sw),
        methodVal =
            rewriter.create<LLVM::ConstantOp>(loc, llvmInt64Type, method);

  SmallVector<Value, 4> opOperandsVec{pool2dOp.getInput(),
                                      pool2dOp.getOutput()},
      operandsVec{adaptor.getInput(), adaptor.getOutput()};

  SmallVector<Value, 4> arguments = getTypeConverter()->promoteOperands(
      loc, opOperandsVec, operandsVec, rewriter, true);

  if (device == multi_device::device::DeviceType::CPU) {
    cpuPool2dCallBuilder.create(loc, rewriter,
                                {arguments[0], arguments[1], nVal, cVal, hVal,
                                 wVal, ohVal, owVal, khVal, kwVal, phlVal,
                                 pwlVal, phrVal, pwrVal, shVal, swVal,
                                 methodVal});
  } else if (device == multi_device::device::DeviceType::GPU) {
    gpuPool2dCallBuilder.create(loc, rewriter,
                                {arguments[0], arguments[1], nVal, cVal, hVal,
                                 wVal, ohVal, owVal, khVal, kwVal, phlVal,
                                 pwlVal, phrVal, pwrVal, shVal, swVal,
                                 methodVal, adaptor.getAsyncDependencies()[0]});
  } else {
    return rewriter.notifyMatchFailure(pool2dOp, "wrong device");
  }

  // async control
  if (pool2dOp.getAsyncToken()) {
    rewriter.replaceOp(pool2dOp, {pool2dOp.getAsyncDependencies()[0]});
  } else {
    rewriter.eraseOp(pool2dOp);
  }

  return success();
}

class ConvertMultiHeadAttentionLayerToDeviceRuntimeCallPattern
    : public ConvertDeviceOpToDeviceRuntimeCallPattern<
          multi_device::device::MultiHeadAttentionLayer> {
public:
  using ConvertDeviceOpToDeviceRuntimeCallPattern<
      multi_device::device::MultiHeadAttentionLayer>::
      ConvertDeviceOpToDeviceRuntimeCallPattern;
  LogicalResult matchAndRewrite(
      multi_device::device::MultiHeadAttentionLayer multiHeadAttentionLayer,
      OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override;
};

LogicalResult
ConvertMultiHeadAttentionLayerToDeviceRuntimeCallPattern::matchAndRewrite(
    multi_device::device::MultiHeadAttentionLayer multiHeadAttentionLayer,
    OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const {
  auto batch = multiHeadAttentionLayer.getBatch(),
       seq_len = multiHeadAttentionLayer.getSeqLen(),
       d_model = multiHeadAttentionLayer.getDModel(),
       feed_forward_dim = multiHeadAttentionLayer.getFeedForwardDim(),
       head_num = multiHeadAttentionLayer.getHeadNum();

  bool norm_first = multiHeadAttentionLayer.getNormFirst(),
       is_casual = multiHeadAttentionLayer.getIsCasual(),
       is_relu = multiHeadAttentionLayer.getAct() == "relu";

  auto loc = multiHeadAttentionLayer.getLoc();
  Value batch_val =
            rewriter.create<LLVM::ConstantOp>(loc, llvmInt64Type, batch),
        seq_len_val =
            rewriter.create<LLVM::ConstantOp>(loc, llvmInt64Type, seq_len),
        d_model_val =
            rewriter.create<LLVM::ConstantOp>(loc, llvmInt64Type, d_model),
        feed_forward_dim_val = rewriter.create<LLVM::ConstantOp>(
            loc, llvmInt64Type, feed_forward_dim),
        head_num_val =
            rewriter.create<LLVM::ConstantOp>(loc, llvmInt64Type, head_num),
        norm_first_val =
            rewriter.create<LLVM::ConstantOp>(loc, llvmBoolType, norm_first),
        is_casual_val =
            rewriter.create<LLVM::ConstantOp>(loc, llvmBoolType, is_casual),
        is_relu_val =
            rewriter.create<LLVM::ConstantOp>(loc, llvmBoolType, is_relu);

  SmallVector<Value> opOperandsVec{multiHeadAttentionLayer.getInput(),
                                   multiHeadAttentionLayer.getQKV(),
                                   multiHeadAttentionLayer.getAttnGemmWeight(),
                                   multiHeadAttentionLayer.getAttnGemmBias(),
                                   multiHeadAttentionLayer.getFfn1Weight(),
                                   multiHeadAttentionLayer.getFfn1Bias(),
                                   multiHeadAttentionLayer.getFfn2Weight(),
                                   multiHeadAttentionLayer.getFfn2Bias(),
                                   multiHeadAttentionLayer.getOutput()},
      operandsVec{adaptor.getInput(),          adaptor.getQKV(),
                  adaptor.getAttnGemmWeight(), adaptor.getAttnGemmBias(),
                  adaptor.getFfn1Weight(),     adaptor.getFfn1Bias(),
                  adaptor.getFfn2Weight(),     adaptor.getFfn2Bias(),
                  adaptor.getOutput()};

  SmallVector<Value> arguments = getTypeConverter()->promoteOperands(
      loc, opOperandsVec, operandsVec, rewriter, true);

  if (device == multi_device::device::DeviceType::GPU) {
    gpuMultiHeadAttentionLayerBuild.create(
        loc, rewriter,
        {arguments[0], arguments[1], arguments[2], arguments[3], arguments[4],
         arguments[5], arguments[6], arguments[7], arguments[8], batch_val,
         seq_len_val, d_model_val, feed_forward_dim_val, head_num_val,
         norm_first_val, is_casual_val, is_relu_val,
         adaptor.getAsyncDependencies()[0]});
  } else if (device == multi_device::device::DeviceType::CPU) {
    cpuMultiHeadAttentionLayerBuild.create(
        loc, rewriter,
        {arguments[0], arguments[1], arguments[2], arguments[3], arguments[4],
         arguments[5], arguments[6], arguments[7], arguments[8], batch_val,
         seq_len_val, d_model_val, feed_forward_dim_val, head_num_val,
         norm_first_val, is_casual_val, is_relu_val});
  } else {
    return rewriter.notifyMatchFailure(multiHeadAttentionLayer, "wrong device");
  }

  // async control
  if (multiHeadAttentionLayer.getAsyncToken()) {
    rewriter.replaceOp(multiHeadAttentionLayer,
                       {multiHeadAttentionLayer.getAsyncDependencies()[0]});
  } else {
    rewriter.eraseOp(multiHeadAttentionLayer);
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
  patterns.add<ConvertMatmulOpToDeviceRuntimeCallPattern,
               ConvertConv2dOpToDeviceRuntimeCallPattern,
               ConvertPool2dOpToDeviceRuntimeCallPattern,
               ConvertMultiHeadAttentionLayerToDeviceRuntimeCallPattern>(
      converter, cachedModuleTable, device);
}