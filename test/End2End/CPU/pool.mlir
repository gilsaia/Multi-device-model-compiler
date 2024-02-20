// RUN: multi-device-opt %s --onnx-to-mlir --mlir-to-cpu --split-input-file

module {
  func.func @main_graph(%arg0: tensor<16x3x256x256xf32>) -> tensor<16x3x64x64xf32> attributes {input_names = ["onnx::MaxPool_0"], output_names = ["1"]} {
    %0 = "onnx.MaxPoolSingleOut"(%arg0) {auto_pad = "NOTSET", ceil_mode = 0 : si64, kernel_shape = [4, 4], onnx_node_name = "/pool/MaxPool", pads = [0, 0, 0, 0], storage_order = 0 : si64, strides = [4, 4]} : (tensor<16x3x256x256xf32>) -> tensor<16x3x64x64xf32>
    onnx.Return %0 : tensor<16x3x64x64xf32>
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
