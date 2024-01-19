#include "multi-device-model-compiler/Dialect/Device/IR/Device.h"

#include "mlir/Dialect/GPU/IR/CompilationInterfaces.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace multi_device;
using namespace multi_device::device;

mlir::LogicalResult device::GPUOffloadingAttr::verify(
    ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
    ::llvm::StringRef kernel, Attribute target) {
  if (target) {
    if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(target)) {
      if (intAttr.getInt() < 0) {
        return emitError() << "The object index must be positive.";
      }
    } else if (!(::mlir::isa<mlir::gpu::TargetAttrInterface>(target))) {
      return emitError()
             << "The target attribute must be a GPU Target attribute.";
    }
  }
  return mlir::success();
}

LogicalResult device::GPUOffloadingAttr::embedBinary(
    Operation *binaryOp, llvm::IRBuilderBase &hostBuilder,
    LLVM::ModuleTranslation &hostModuleTranslation) const {
  assert(binaryOp && "The binary operation must be non null.");
  if (!binaryOp) {
    return failure();
  }

  auto op = mlir::dyn_cast<gpu::BinaryOp>(binaryOp);
  if (!op) {
    binaryOp->emitError("Operation must be a GPU binary.");
    return failure();
  }

  llvm::ArrayRef<mlir::Attribute> objects = op.getObjectsAttr().getValue();

  int64_t index = -1;
  if (Attribute target = getTarget()) {
    // If the target attribute is a number it is the index. Otherwise compare
    // the attribute to every target inside the object array to find the index.
    if (auto indexAttr = mlir::dyn_cast<IntegerAttr>(target)) {
      index = indexAttr.getInt();
    } else {
      for (auto [i, attr] : llvm::enumerate(objects)) {
        auto obj = mlir::dyn_cast<gpu::ObjectAttr>(attr);
        if (obj.getTarget() == target) {
          index = i;
        }
      }
    }
  } else {
    // If the target attribute is null then it's selecting the first object in
    // the object array.
    index = 0;
  }

  if (index < 0 || index >= static_cast<int64_t>(objects.size())) {
    op->emitError("The requested target object couldn't be found.");
    return failure();
  }
  auto object = mlir::dyn_cast<gpu::ObjectAttr>(objects[index]);

  llvm::StringRef fileName = getKernel().str() + ".bc";
  std::error_code ec;
  llvm::raw_fd_ostream fs(fileName, ec);
  if (ec) {
    op->emitError("The requested file can't create.");
    return failure();
  }
  fs << object.getObject().getValue();
  fs.close();

  return success();
}

namespace llvm {
namespace {
class MultiDeviceGPULaunch {
public:
  MultiDeviceGPULaunch(Module &module, llvm::IRBuilderBase &builder,
                       mlir::LLVM::ModuleTranslation &moduleTranslation);
  // Get the kernel launch callee.
  FunctionCallee getKernelLaunchFn();

  // Get the module function callee.
  FunctionCallee getModuleFunctionFn();

  // Get the module load callee.
  FunctionCallee getModuleLoadFn();

  FunctionCallee getModuleFileLoadFn();

  // Get the module unload callee.
  FunctionCallee getModuleUnloadFn();

  // Get the stream create callee.
  FunctionCallee getStreamCreateFn();

  // Get the stream destroy callee.
  FunctionCallee getStreamDestroyFn();

  // Get the stream sync callee.
  FunctionCallee getStreamSyncFn();

  // Ger or create the function name global string.
  Value *getOrCreateFunctionName(StringRef moduleName, StringRef kernelName);

  // Create the void* kernel array for passing the arguments.
  Value *createKernelArgArray(mlir::gpu::LaunchFuncOp op);

  // Create the full kernel launch.
  mlir::LogicalResult createKernelLaunch(mlir::gpu::LaunchFuncOp op);

private:
  Module &module;
  IRBuilderBase &builder;
  mlir::LLVM::ModuleTranslation &moduleTranslation;
  Type *i32Ty{};
  Type *voidTy{};
  Type *intPtrTy{};
  PointerType *ptrTy{};
};
} // namespace
} // namespace llvm

LogicalResult device::GPUOffloadingAttr::launchKernel(
    Operation *launchFunc, Operation *binaryOp,
    llvm::IRBuilderBase &hostBuilder,
    LLVM::ModuleTranslation &hostModuleTranslation) const {
  assert(launchFunc && "The launch func operation must be non null.");
  if (!launchFunc)
    return failure();

  auto launchFuncOp = mlir::dyn_cast<gpu::LaunchFuncOp>(launchFunc);
  if (!launchFuncOp) {
    launchFunc->emitError("Operation must be a GPU launch func Op.");
    return failure();
  }

  return llvm::MultiDeviceGPULaunch(*hostModuleTranslation.getLLVMModule(),
                                    hostBuilder, hostModuleTranslation)
      .createKernelLaunch(launchFuncOp);
}

llvm::MultiDeviceGPULaunch::MultiDeviceGPULaunch(
    Module &module, IRBuilderBase &builder,
    mlir::LLVM::ModuleTranslation &moduleTranslation)
    : module(module), builder(builder), moduleTranslation(moduleTranslation) {
  i32Ty = builder.getInt32Ty();
  ptrTy = builder.getPtrTy(0);
  voidTy = builder.getVoidTy();
  intPtrTy = builder.getIntPtrTy(module.getDataLayout());
}

llvm::FunctionCallee llvm::MultiDeviceGPULaunch::getKernelLaunchFn() {
  return module.getOrInsertFunction(
      "mgpuLaunchKernel",
      FunctionType::get(
          voidTy,
          ArrayRef<Type *>({ptrTy, intPtrTy, intPtrTy, intPtrTy, intPtrTy,
                            intPtrTy, intPtrTy, i32Ty, ptrTy, ptrTy, ptrTy}),
          false));
}

llvm::FunctionCallee llvm::MultiDeviceGPULaunch::getModuleFunctionFn() {
  return module.getOrInsertFunction(
      "mgpuModuleGetFunction",
      FunctionType::get(ptrTy, ArrayRef<Type *>({ptrTy, ptrTy}), false));
}

llvm::FunctionCallee llvm::MultiDeviceGPULaunch::getModuleLoadFn() {
  return module.getOrInsertFunction(
      "mgpuModuleLoad",
      FunctionType::get(ptrTy, ArrayRef<Type *>({ptrTy}), false));
}

llvm::FunctionCallee llvm::MultiDeviceGPULaunch::getModuleFileLoadFn() {
  return module.getOrInsertFunction("mgpuModuleFileLoad",
                                    FunctionType::get(ptrTy, false));
}

llvm::FunctionCallee llvm::MultiDeviceGPULaunch::getModuleUnloadFn() {
  return module.getOrInsertFunction(
      "mgpuModuleUnload",
      FunctionType::get(voidTy, ArrayRef<Type *>({ptrTy}), false));
}

llvm::FunctionCallee llvm::MultiDeviceGPULaunch::getStreamCreateFn() {
  return module.getOrInsertFunction("mgpuStreamCreate",
                                    FunctionType::get(ptrTy, false));
}

llvm::FunctionCallee llvm::MultiDeviceGPULaunch::getStreamDestroyFn() {
  return module.getOrInsertFunction(
      "mgpuStreamDestroy",
      FunctionType::get(voidTy, ArrayRef<Type *>({ptrTy}), false));
}

llvm::FunctionCallee llvm::MultiDeviceGPULaunch::getStreamSyncFn() {
  return module.getOrInsertFunction(
      "mgpuStreamSynchronize",
      FunctionType::get(voidTy, ArrayRef<Type *>({ptrTy}), false));
}

// Generates an LLVM IR dialect global that contains the name of the given
// kernel function as a C string, and returns a pointer to its beginning.
llvm::Value *
llvm::MultiDeviceGPULaunch::getOrCreateFunctionName(StringRef moduleName,
                                                    StringRef kernelName) {
  std::string globalName =
      std::string(formatv("{0}_{1}_kernel_name", moduleName, kernelName));

  if (GlobalVariable *gv = module.getGlobalVariable(globalName))
    return gv;

  return builder.CreateGlobalString(kernelName, globalName);
}

// Creates a struct containing all kernel parameters on the stack and returns
// an array of type-erased pointers to the fields of the struct. The array can
// then be passed to the CUDA / ROCm (HIP) kernel launch calls.
// The generated code is essentially as follows:
//
// %struct = alloca(sizeof(struct { Parameters... }))
// %array = alloca(NumParameters * sizeof(void *))
// for (i : [0, NumParameters))
//   %fieldPtr = llvm.getelementptr %struct[0, i]
//   llvm.store parameters[i], %fieldPtr
//   %elementPtr = llvm.getelementptr %array[i]
//   llvm.store %fieldPtr, %elementPtr
// return %array
llvm::Value *
llvm::MultiDeviceGPULaunch::createKernelArgArray(mlir::gpu::LaunchFuncOp op) {
  SmallVector<Value *> args =
      moduleTranslation.lookupValues(op.getKernelOperands());
  SmallVector<Type *> structTypes(args.size(), nullptr);

  for (auto [i, arg] : llvm::enumerate(args))
    structTypes[i] = arg->getType();

  Type *structTy = StructType::create(module.getContext(), structTypes);
  Value *argStruct = builder.CreateAlloca(structTy, 0u);
  Value *argArray = builder.CreateAlloca(
      ptrTy, ConstantInt::get(intPtrTy, structTypes.size()));

  for (auto [i, arg] : enumerate(args)) {
    Value *structMember = builder.CreateStructGEP(structTy, argStruct, i);
    builder.CreateStore(arg, structMember);
    Value *arrayMember = builder.CreateConstGEP1_32(ptrTy, argArray, i);
    builder.CreateStore(structMember, arrayMember);
  }
  return argArray;
}

// Emits LLVM IR to launch a kernel function:
// %0 = call %binarygetter
// %1 = call %moduleLoad(%0)
// %2 = <see generateKernelNameConstant>
// %3 = call %moduleGetFunction(%1, %2)
// %4 = call %streamCreate()
// %5 = <see generateParamsArray>
// call %launchKernel(%3, <launchOp operands 0..5>, 0, %4, %5, nullptr)
// call %streamSynchronize(%4)
// call %streamDestroy(%4)
// call %moduleUnload(%1)
mlir::LogicalResult
llvm::MultiDeviceGPULaunch::createKernelLaunch(mlir::gpu::LaunchFuncOp op) {
  auto llvmValue = [&](mlir::Value value) -> Value * {
    Value *v = moduleTranslation.lookupValue(value);
    assert(v && "Value has not been translated.");
    return v;
  };

  // Get grid dimensions.
  mlir::gpu::KernelDim3 grid = op.getGridSizeOperandValues();
  Value *gx = llvmValue(grid.x), *gy = llvmValue(grid.y),
        *gz = llvmValue(grid.z);

  // Get block dimensions.
  mlir::gpu::KernelDim3 block = op.getBlockSizeOperandValues();
  Value *bx = llvmValue(block.x), *by = llvmValue(block.y),
        *bz = llvmValue(block.z);

  // Get dynamic shared memory size.
  Value *dynamicMemorySize = nullptr;
  if (mlir::Value dynSz = op.getDynamicSharedMemorySize())
    dynamicMemorySize = llvmValue(dynSz);
  else
    dynamicMemorySize = ConstantInt::get(i32Ty, 0);

  // Create the argument array.
  Value *argArray = createKernelArgArray(op);

  // Load the kernel module.
  Value *moduleObject = builder.CreateCall(getModuleFileLoadFn());

  // Load the kernel function.
  StringRef moduleName = op.getKernelModuleName().getValue();
  Value *moduleFunction = builder.CreateCall(
      getModuleFunctionFn(),
      {moduleObject,
       getOrCreateFunctionName(moduleName, op.getKernelName().getValue())});

  // Get the stream to use for execution. If there's no async object then create
  // a stream to make a synchronous kernel launch.
  Value *stream = nullptr;
  bool handleStream = false;
  if (mlir::Value asyncObject = op.getAsyncObject()) {
    stream = llvmValue(asyncObject);
  } else {
    handleStream = true;
    stream = builder.CreateCall(getStreamCreateFn(), {});
  }

  // Create the launch call.
  Value *nullPtr = ConstantPointerNull::get(ptrTy);
  builder.CreateCall(
      getKernelLaunchFn(),
      ArrayRef<Value *>({moduleFunction, gx, gy, gz, bx, by, bz,
                         dynamicMemorySize, stream, argArray, nullPtr}));

  // Sync & destroy the stream, for synchronous launches.
  if (handleStream) {
    builder.CreateCall(getStreamSyncFn(), {stream});
    builder.CreateCall(getStreamDestroyFn(), {stream});
  }

  // Unload the kernel module.
  builder.CreateCall(getModuleUnloadFn(), {moduleObject});

  return success();
}
