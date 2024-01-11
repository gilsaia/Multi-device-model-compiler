// RUN: multi-device-opt %s --onnx-to-mlir --mlir-to-tpu

module {
  func.func @main_graph(%arg0: tensor<2x10x10xf32>) -> tensor<2x10x10xf32> attributes {input_names = ["A"], output_names = ["C"]} {
    %0 = onnx.Constant dense<1.000000e+00> : tensor<2x10x10xf32>
    %1 = "onnx.Add"(%arg0, %0) : (tensor<2x10x10xf32>, tensor<2x10x10xf32>) -> tensor<2x10x10xf32>
    return %1 : tensor<2x10x10xf32>
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}