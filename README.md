# Multi-device-model-compiler

# Build Project

## Whole Project

1. 下载全部submodule
2. 提前构建LLVM，构建脚本可参考utils/configure-llvm.sh,utils/build-llvm.sh
3. CMake构建本项目，具体选项可参考CMakeLists.txt中选项

## Only Runtime

1. 下载LLVM submodule，commit 91088978d712cd7b33610c59f69d87d5a39e3113
2. 提前构建LLVM，构建脚本同上
3. CMake构建本项目，添加选项 `-DONLY_BUILD_PLAIN_RUNTIME=ON`，仅构建Runtime部分

# Run Example

## Library Generate

在examples目录下存在已经编译好的`add.ll`，`add.mlir`文件，该模型为基本的加法操作，进行两个形状为(3,640,640)的Tensor的加法。ll文件为编译后的llvm ir文件，可通过llvm生态下`llc`工具将其转变为`.o`文件，不直接提供是由于不同机器依赖库不一致，请使用自行构建的llvm工具链完成该操作。

`.o`文件可使用任意编译器将其编译为动态链接库`.so`，之后可直接调用`multi-device-cpu-runner`来进行模型推理

## Data Generate

在examples文件夹下有`data_util.py`文件，可生成指定类型的数据，逻辑简单，有需求可自行添加，Shape一般采取`3_640_640`格式表示对应形状为(3,640,640)

## Run Model

调用构建好的`multi-device-cpu-runner`即可进行模型推理，需要提供动态链接库位置，输入输出形状等。

# Project Structure

doc文件夹下有对Runtime部分实现逻辑的简单解释。

