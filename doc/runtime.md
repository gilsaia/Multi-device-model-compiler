# Runtime实现解释

## 框架架构

Runtime主体实现部分位于`include/multi-device-model-compiler/Runtime`,`lib/Runtime`文件夹下，主要包含几个部分

- TensorDescripter：基本的Tensor描述的包装，其定义格式参考官方文档描述[mlir-target](https://mlir.llvm.org/docs/TargetLLVMIR/#ranked-memref-types)，实现为模板类以应对不同维度，含义可直接参考类型变量名。MLIR中对MemRef最终会固定转变为该形式，并在函数入口对Descripter进行相应的包装解包装操作
- RuntimeUtil：对于生成的动态链接库进行调用，提供相应的包装函数等，使用C中提供的标准库函数来实现对动态链接库的加载与使用，MLIR的生成过程默认会生成固定的函数名，其参数为TensorDescripter，按照先输出后输入的格式，因此对于一个已知的模型，得到其输入与输出的Shape即可推导出具体的函数签名
- ModelInfo：模型相关信息，包装了提供模型的相关信息供调用时使用

## 使用流程

上述Runtime最终被Runner使用，在Runner执行时，根据提供的链接库位置找到编译后的模型动态链接库，而后根据给定的函数名称（默认按照标准命名）和提供的输入输出找到对应的函数入口即调用参数，之后根据给定的输入或随机生成的输入来进行模型推理工作，最终将输出保存。