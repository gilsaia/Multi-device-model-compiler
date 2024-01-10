// RUN: multi-device-opt %s --onnx-to-mlir --mlir-to-tpu

module {
  func.func @main_graph(%arg0: tensor<3x640x640xf32>, %arg1: tensor<3x640x640xf32>) -> tensor<3x640x640xf32> attributes {input_names = ["A", "B"], output_names = ["C"]} {
    %0 = "onnx.Add"(%arg0, %arg1) : (tensor<3x640x640xf32>, tensor<3x640x640xf32>) -> tensor<3x640x640xf32>
    return %0 : tensor<3x640x640xf32>
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}