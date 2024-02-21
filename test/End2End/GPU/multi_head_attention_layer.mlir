module {
  func.func @main_graph(%arg0: tensor<16x512x2048xf32>) -> tensor<16x512x2048xf32> attributes {input_names = ["input.1"], output_names = ["200"]} {
    %0 = onnx.Constant dense<0.000000e+00> : tensor<6144xf32>
    %1 = onnx.Constant dense_resource<__elided__> : tensor<2048x2048xf32>
    %2 = onnx.Constant dense<0.000000e+00> : tensor<2048xf32>
    %3 = onnx.Constant dense_resource<__elided__> : tensor<2048xf32>
    %4 = onnx.Constant dense_resource<__elided__> : tensor<2048xf32>
    %5 = onnx.Constant dense<1.000000e+00> : tensor<2048xf32>
    %6 = onnx.Constant dense_resource<__elided__> : tensor<2048x6144xf32>
    %7 = onnx.Constant dense_resource<__elided__> : tensor<2048x2048xf32>
    %8 = onnx.Constant dense_resource<__elided__> : tensor<2048x2048xf32>
    %9 = "onnx.Identity"(%2) {onnx_node_name = "Identity_0"} : (tensor<2048xf32>) -> tensor<2048xf32>
    %10 = "onnx.Identity"(%5) {onnx_node_name = "Identity_1"} : (tensor<2048xf32>) -> tensor<2048xf32>
    %11 = "onnx.Identity"(%2) {onnx_node_name = "Identity_2"} : (tensor<2048xf32>) -> tensor<2048xf32>
    %12 = "onnx.ReduceMeanV13"(%arg0) {axes = [-1], keepdims = 1 : si64, onnx_node_name = "/encoder/norm1/ReduceMean"} : (tensor<16x512x2048xf32>) -> tensor<16x512x1xf32>
    %13 = "onnx.Sub"(%arg0, %12) {onnx_node_name = "/encoder/norm1/Sub"} : (tensor<16x512x2048xf32>, tensor<16x512x1xf32>) -> tensor<16x512x2048xf32>
    %14 = onnx.Constant dense<2.000000e+00> : tensor<f32>
    %15 = "onnx.Pow"(%13, %14) {onnx_node_name = "/encoder/norm1/Pow"} : (tensor<16x512x2048xf32>, tensor<f32>) -> tensor<16x512x2048xf32>
    %16 = "onnx.ReduceMeanV13"(%15) {axes = [-1], keepdims = 1 : si64, onnx_node_name = "/encoder/norm1/ReduceMean_1"} : (tensor<16x512x2048xf32>) -> tensor<16x512x1xf32>
    %17 = onnx.Constant dense<9.99999974E-6> : tensor<f32>
    %18 = "onnx.Add"(%16, %17) {onnx_node_name = "/encoder/norm1/Add"} : (tensor<16x512x1xf32>, tensor<f32>) -> tensor<16x512x1xf32>
    %19 = "onnx.Sqrt"(%18) {onnx_node_name = "/encoder/norm1/Sqrt"} : (tensor<16x512x1xf32>) -> tensor<16x512x1xf32>
    %20 = "onnx.Div"(%13, %19) {onnx_node_name = "/encoder/norm1/Div"} : (tensor<16x512x2048xf32>, tensor<16x512x1xf32>) -> tensor<16x512x2048xf32>
    %21 = "onnx.Mul"(%20, %5) {onnx_node_name = "/encoder/norm1/Mul"} : (tensor<16x512x2048xf32>, tensor<2048xf32>) -> tensor<16x512x2048xf32>
    %22 = "onnx.Add"(%21, %11) {onnx_node_name = "/encoder/norm1/Add_1"} : (tensor<16x512x2048xf32>, tensor<2048xf32>) -> tensor<16x512x2048xf32>
    %23 = onnx.Constant dense<1> : tensor<i64>
    %24 = onnx.Constant dense<0> : tensor<i64>
    %25 = "onnx.Transpose"(%22) {onnx_node_name = "/encoder/self_attn/Transpose", perm = [1, 0, 2]} : (tensor<16x512x2048xf32>) -> tensor<512x16x2048xf32>
    %26 = onnx.Constant dense<2> : tensor<i64>
    %27 = "onnx.MatMul"(%25, %6) {onnx_node_name = "/encoder/self_attn/MatMul"} : (tensor<512x16x2048xf32>, tensor<2048x6144xf32>) -> tensor<512x16x6144xf32>
    %28 = "onnx.Add"(%0, %27) {onnx_node_name = "/encoder/self_attn/Add"} : (tensor<6144xf32>, tensor<512x16x6144xf32>) -> tensor<512x16x6144xf32>
    %29 = onnx.Constant dense<[3, 2048]> : tensor<2xi64>
    %30 = onnx.Constant dense<2> : tensor<1xi64>
    %31 = onnx.Constant dense<3> : tensor<1xi64>
    %32 = "onnx.Mod"(%30, %31) {fmod = 0 : si64, onnx_node_name = "/encoder/self_attn/Mod"} : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi64>
    %33 = "onnx.Shape"(%28) {onnx_node_name = "/encoder/self_attn/Shape", start = 0 : si64} : (tensor<512x16x6144xf32>) -> tensor<3xi64>
    %34 = onnx.Constant dense<0> : tensor<1xi64>
    %35 = onnx.Constant dense<1> : tensor<1xi64>
    %36 = "onnx.Reshape"(%32, %35) {allowzero = 0 : si64, onnx_node_name = "/encoder/self_attn/Reshape"} : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi64>
    %37 = "onnx.NoValue"() {value} : () -> none
    %38 = "onnx.NoValue"() {value} : () -> none
    %39 = "onnx.Slice"(%33, %34, %36, %37, %38) {onnx_node_name = "/encoder/self_attn/Slice"} : (tensor<3xi64>, tensor<1xi64>, tensor<1xi64>, none, none) -> tensor<*xi64>
    %40 = onnx.Constant dense<1> : tensor<1xi64>
    %41 = "onnx.Add"(%32, %40) {onnx_node_name = "/encoder/self_attn/Add_1"} : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi64>
    %42 = onnx.Constant dense<1> : tensor<1xi64>
    %43 = "onnx.Reshape"(%41, %42) {allowzero = 0 : si64, onnx_node_name = "/encoder/self_attn/Reshape_1"} : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi64>
    %44 = onnx.Constant dense<9223372036854775807> : tensor<1xi64>
    %45 = "onnx.NoValue"() {value} : () -> none
    %46 = "onnx.NoValue"() {value} : () -> none
    %47 = "onnx.Slice"(%33, %43, %44, %45, %46) {onnx_node_name = "/encoder/self_attn/Slice_1"} : (tensor<3xi64>, tensor<1xi64>, tensor<1xi64>, none, none) -> tensor<*xi64>
    %48 = "onnx.Concat"(%39, %29, %47) {axis = 0 : si64, onnx_node_name = "/encoder/self_attn/Concat"} : (tensor<*xi64>, tensor<2xi64>, tensor<*xi64>) -> tensor<*xi64>
    %49 = "onnx.Reshape"(%28, %48) {allowzero = 0 : si64, onnx_node_name = "/encoder/self_attn/Reshape_2"} : (tensor<512x16x6144xf32>, tensor<*xi64>) -> tensor<*xf32>
    %50 = onnx.Constant dense<0> : tensor<1xi64>
    %51 = "onnx.Unsqueeze"(%49, %50) {onnx_node_name = "/encoder/self_attn/Unsqueeze"} : (tensor<*xf32>, tensor<1xi64>) -> tensor<*xf32>
    %52 = "onnx.Transpose"(%51) {onnx_node_name = "/encoder/self_attn/Transpose_1", perm = [3, 1, 2, 0, 4]} : (tensor<*xf32>) -> tensor<*xf32>
    %53 = onnx.Constant dense<3> : tensor<1xi64>
    %54 = "onnx.Squeeze"(%52, %53) {onnx_node_name = "/encoder/self_attn/Squeeze"} : (tensor<*xf32>, tensor<1xi64>) -> tensor<*xf32>
    %55 = "onnx.Gather"(%54, %24) {axis = 0 : si64, onnx_node_name = "/encoder/self_attn/Gather"} : (tensor<*xf32>, tensor<i64>) -> tensor<*xf32>
    %56 = "onnx.Gather"(%54, %23) {axis = 0 : si64, onnx_node_name = "/encoder/self_attn/Gather_1"} : (tensor<*xf32>, tensor<i64>) -> tensor<*xf32>
    %57 = "onnx.Gather"(%54, %26) {axis = 0 : si64, onnx_node_name = "/encoder/self_attn/Gather_2"} : (tensor<*xf32>, tensor<i64>) -> tensor<*xf32>
    %58 = onnx.Constant dense<[512, 256, 128]> : tensor<3xi64>
    %59 = "onnx.Reshape"(%55, %58) {allowzero = 0 : si64, onnx_node_name = "/encoder/self_attn/Reshape_3"} : (tensor<*xf32>, tensor<3xi64>) -> tensor<512x256x128xf32>
    %60 = "onnx.Transpose"(%59) {onnx_node_name = "/encoder/self_attn/Transpose_2", perm = [1, 0, 2]} : (tensor<512x256x128xf32>) -> tensor<256x512x128xf32>
    %61 = onnx.Constant dense<[512, 256, 128]> : tensor<3xi64>
    %62 = "onnx.Reshape"(%56, %61) {allowzero = 0 : si64, onnx_node_name = "/encoder/self_attn/Reshape_4"} : (tensor<*xf32>, tensor<3xi64>) -> tensor<512x256x128xf32>
    %63 = "onnx.Transpose"(%62) {onnx_node_name = "/encoder/self_attn/Transpose_3", perm = [1, 0, 2]} : (tensor<512x256x128xf32>) -> tensor<256x512x128xf32>
    %64 = onnx.Constant dense<[512, 256, 128]> : tensor<3xi64>
    %65 = "onnx.Reshape"(%57, %64) {allowzero = 0 : si64, onnx_node_name = "/encoder/self_attn/Reshape_5"} : (tensor<*xf32>, tensor<3xi64>) -> tensor<512x256x128xf32>
    %66 = "onnx.Transpose"(%65) {onnx_node_name = "/encoder/self_attn/Transpose_4", perm = [1, 0, 2]} : (tensor<512x256x128xf32>) -> tensor<256x512x128xf32>
    %67 = onnx.Constant dense<[16, 16, 512, 128]> : tensor<4xi64>
    %68 = "onnx.Reshape"(%60, %67) {allowzero = 0 : si64, onnx_node_name = "/encoder/self_attn/Reshape_6"} : (tensor<256x512x128xf32>, tensor<4xi64>) -> tensor<16x16x512x128xf32>
    %69 = onnx.Constant dense<[16, 16, 512, 128]> : tensor<4xi64>
    %70 = onnx.Constant dense<[16, 16, 512, 128]> : tensor<4xi64>
    %71 = "onnx.Reshape"(%63, %69) {allowzero = 0 : si64, onnx_node_name = "/encoder/self_attn/Reshape_7"} : (tensor<256x512x128xf32>, tensor<4xi64>) -> tensor<16x16x512x128xf32>
    %72 = "onnx.Reshape"(%66, %70) {allowzero = 0 : si64, onnx_node_name = "/encoder/self_attn/Reshape_8"} : (tensor<256x512x128xf32>, tensor<4xi64>) -> tensor<16x16x512x128xf32>
    %73 = "onnx.Shape"(%68) {onnx_node_name = "/encoder/self_attn/Shape_1", start = 0 : si64} : (tensor<16x16x512x128xf32>) -> tensor<4xi64>
    %74 = onnx.Constant dense<-1> : tensor<1xi64>
    %75 = onnx.Constant dense<9223372036854775807> : tensor<1xi64>
    %76 = "onnx.NoValue"() {value} : () -> none
    %77 = "onnx.NoValue"() {value} : () -> none
    %78 = "onnx.Slice"(%73, %74, %75, %76, %77) {onnx_node_name = "/encoder/self_attn/Slice_2"} : (tensor<4xi64>, tensor<1xi64>, tensor<1xi64>, none, none) -> tensor<1xi64>
    %79 = "onnx.Cast"(%78) {onnx_node_name = "/encoder/self_attn/Cast", saturate = 1 : si64, to = f32} : (tensor<1xi64>) -> tensor<1xf32>
    %80 = "onnx.Sqrt"(%79) {onnx_node_name = "/encoder/self_attn/Sqrt"} : (tensor<1xf32>) -> tensor<1xf32>
    %81 = onnx.Constant dense<1.000000e+00> : tensor<1xf32>
    %82 = "onnx.Div"(%81, %80) {onnx_node_name = "/encoder/self_attn/Div"} : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
    %83 = "onnx.Shape"(%68) {onnx_node_name = "/encoder/self_attn/Shape_2", start = 0 : si64} : (tensor<16x16x512x128xf32>) -> tensor<4xi64>
    %84 = "onnx.Shape"(%71) {onnx_node_name = "/encoder/self_attn/Shape_3", start = 0 : si64} : (tensor<16x16x512x128xf32>) -> tensor<4xi64>
    %85 = onnx.Constant dense<-1> : tensor<1xi64>
    %86 = onnx.Constant dense<-2> : tensor<1xi64>
    %87 = "onnx.NoValue"() {value} : () -> none
    %88 = "onnx.NoValue"() {value} : () -> none
    %89 = "onnx.Slice"(%83, %86, %85, %87, %88) {onnx_node_name = "/encoder/self_attn/Slice_3"} : (tensor<4xi64>, tensor<1xi64>, tensor<1xi64>, none, none) -> tensor<1xi64>
    %90 = "onnx.NoValue"() {value} : () -> none
    %91 = "onnx.NoValue"() {value} : () -> none
    %92 = "onnx.Slice"(%84, %86, %85, %90, %91) {onnx_node_name = "/encoder/self_attn/Slice_4"} : (tensor<4xi64>, tensor<1xi64>, tensor<1xi64>, none, none) -> tensor<1xi64>
    %93 = "onnx.Concat"(%89, %92) {axis = 0 : si64, onnx_node_name = "/encoder/self_attn/Concat_1"} : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
    %94 = onnx.Constant dense<1.000000e+00> : tensor<1xf32>
    %95 = "onnx.Expand"(%94, %93) {onnx_node_name = "/encoder/self_attn/Expand"} : (tensor<1xf32>, tensor<2xi64>) -> tensor<?x?xf32>
    %96 = "onnx.NoValue"() {value} : () -> none
    %97 = "onnx.Trilu"(%95, %96) {onnx_node_name = "/encoder/self_attn/Trilu", upper = 0 : si64} : (tensor<?x?xf32>, none) -> tensor<?x?xf32>
    %98 = onnx.Constant dense<0.000000e+00> : tensor<1xf32>
    %99 = onnx.Constant dense<0xFF800000> : tensor<1xf32>
    %100 = onnx.Constant dense<0.000000e+00> : tensor<1xf32>
    %101 = "onnx.Equal"(%97, %100) {onnx_node_name = "/encoder/self_attn/Equal"} : (tensor<?x?xf32>, tensor<1xf32>) -> tensor<?x?xi1>
    %102 = "onnx.Where"(%101, %99, %98) {onnx_node_name = "/encoder/self_attn/Where"} : (tensor<?x?xi1>, tensor<1xf32>, tensor<1xf32>) -> tensor<?x?xf32>
    %103 = "onnx.Transpose"(%71) {onnx_node_name = "/encoder/self_attn/Transpose_5", perm = [0, 1, 3, 2]} : (tensor<16x16x512x128xf32>) -> tensor<16x16x128x512xf32>
    %104 = "onnx.Sqrt"(%82) {onnx_node_name = "/encoder/self_attn/Sqrt_1"} : (tensor<1xf32>) -> tensor<1xf32>
    %105 = "onnx.Mul"(%68, %104) {onnx_node_name = "/encoder/self_attn/Mul"} : (tensor<16x16x512x128xf32>, tensor<1xf32>) -> tensor<16x16x512x128xf32>
    %106 = "onnx.Sqrt"(%82) {onnx_node_name = "/encoder/self_attn/Sqrt_2"} : (tensor<1xf32>) -> tensor<1xf32>
    %107 = "onnx.Mul"(%103, %106) {onnx_node_name = "/encoder/self_attn/Mul_1"} : (tensor<16x16x128x512xf32>, tensor<1xf32>) -> tensor<16x16x128x512xf32>
    %108 = "onnx.MatMul"(%105, %107) {onnx_node_name = "/encoder/self_attn/MatMul_1"} : (tensor<16x16x512x128xf32>, tensor<16x16x128x512xf32>) -> tensor<16x16x512x512xf32>
    %109 = "onnx.Add"(%108, %102) {onnx_node_name = "/encoder/self_attn/Add_2"} : (tensor<16x16x512x512xf32>, tensor<?x?xf32>) -> tensor<16x16x512x512xf32>
    %110 = "onnx.Softmax"(%109) {axis = -1 : si64, onnx_node_name = "/encoder/self_attn/Softmax"} : (tensor<16x16x512x512xf32>) -> tensor<16x16x512x512xf32>
    %111 = "onnx.MatMul"(%110, %72) {onnx_node_name = "/encoder/self_attn/MatMul_2"} : (tensor<16x16x512x512xf32>, tensor<16x16x512x128xf32>) -> tensor<16x16x512x128xf32>
    %112 = "onnx.Transpose"(%111) {onnx_node_name = "/encoder/self_attn/Transpose_6", perm = [2, 0, 1, 3]} : (tensor<16x16x512x128xf32>) -> tensor<512x16x16x128xf32>
    %113 = onnx.Constant dense<[8192, 2048]> : tensor<2xi64>
    %114 = "onnx.Reshape"(%112, %113) {allowzero = 0 : si64, onnx_node_name = "/encoder/self_attn/Reshape_9"} : (tensor<512x16x16x128xf32>, tensor<2xi64>) -> tensor<8192x2048xf32>
    %115 = "onnx.Gemm"(%114, %1, %2) {alpha = 1.000000e+00 : f32, beta = 1.000000e+00 : f32, onnx_node_name = "/encoder/self_attn/Gemm", transA = 0 : si64, transB = 1 : si64} : (tensor<8192x2048xf32>, tensor<2048x2048xf32>, tensor<2048xf32>) -> tensor<8192x2048xf32>
    %116 = onnx.Constant dense<[512, 16, 2048]> : tensor<3xi64>
    %117 = "onnx.Reshape"(%115, %116) {allowzero = 0 : si64, onnx_node_name = "/encoder/self_attn/Reshape_10"} : (tensor<8192x2048xf32>, tensor<3xi64>) -> tensor<512x16x2048xf32>
    %118 = "onnx.Transpose"(%117) {onnx_node_name = "/encoder/self_attn/Transpose_7", perm = [1, 0, 2]} : (tensor<512x16x2048xf32>) -> tensor<16x512x2048xf32>
    %119 = "onnx.Add"(%arg0, %118) {onnx_node_name = "/encoder/Add"} : (tensor<16x512x2048xf32>, tensor<16x512x2048xf32>) -> tensor<16x512x2048xf32>
    %120 = "onnx.ReduceMeanV13"(%119) {axes = [-1], keepdims = 1 : si64, onnx_node_name = "/encoder/norm2/ReduceMean"} : (tensor<16x512x2048xf32>) -> tensor<16x512x1xf32>
    %121 = "onnx.Sub"(%119, %120) {onnx_node_name = "/encoder/norm2/Sub"} : (tensor<16x512x2048xf32>, tensor<16x512x1xf32>) -> tensor<16x512x2048xf32>
    %122 = onnx.Constant dense<2.000000e+00> : tensor<f32>
    %123 = "onnx.Pow"(%121, %122) {onnx_node_name = "/encoder/norm2/Pow"} : (tensor<16x512x2048xf32>, tensor<f32>) -> tensor<16x512x2048xf32>
    %124 = "onnx.ReduceMeanV13"(%123) {axes = [-1], keepdims = 1 : si64, onnx_node_name = "/encoder/norm2/ReduceMean_1"} : (tensor<16x512x2048xf32>) -> tensor<16x512x1xf32>
    %125 = onnx.Constant dense<9.99999974E-6> : tensor<f32>
    %126 = "onnx.Add"(%124, %125) {onnx_node_name = "/encoder/norm2/Add"} : (tensor<16x512x1xf32>, tensor<f32>) -> tensor<16x512x1xf32>
    %127 = "onnx.Sqrt"(%126) {onnx_node_name = "/encoder/norm2/Sqrt"} : (tensor<16x512x1xf32>) -> tensor<16x512x1xf32>
    %128 = "onnx.Div"(%121, %127) {onnx_node_name = "/encoder/norm2/Div"} : (tensor<16x512x2048xf32>, tensor<16x512x1xf32>) -> tensor<16x512x2048xf32>
    %129 = "onnx.Mul"(%128, %10) {onnx_node_name = "/encoder/norm2/Mul"} : (tensor<16x512x2048xf32>, tensor<2048xf32>) -> tensor<16x512x2048xf32>
    %130 = "onnx.Add"(%129, %9) {onnx_node_name = "/encoder/norm2/Add_1"} : (tensor<16x512x2048xf32>, tensor<2048xf32>) -> tensor<16x512x2048xf32>
    %131 = "onnx.MatMul"(%130, %7) {onnx_node_name = "/encoder/linear1/MatMul"} : (tensor<16x512x2048xf32>, tensor<2048x2048xf32>) -> tensor<16x512x2048xf32>
    %132 = "onnx.Add"(%3, %131) {onnx_node_name = "/encoder/linear1/Add"} : (tensor<2048xf32>, tensor<16x512x2048xf32>) -> tensor<16x512x2048xf32>
    %133 = "onnx.Relu"(%132) {onnx_node_name = "/encoder/Relu"} : (tensor<16x512x2048xf32>) -> tensor<16x512x2048xf32>
    %134 = "onnx.MatMul"(%133, %8) {onnx_node_name = "/encoder/linear2/MatMul"} : (tensor<16x512x2048xf32>, tensor<2048x2048xf32>) -> tensor<16x512x2048xf32>
    %135 = "onnx.Add"(%4, %134) {onnx_node_name = "/encoder/linear2/Add"} : (tensor<2048xf32>, tensor<16x512x2048xf32>) -> tensor<16x512x2048xf32>
    %136 = "onnx.Add"(%119, %135) {onnx_node_name = "/encoder/Add_1"} : (tensor<16x512x2048xf32>, tensor<16x512x2048xf32>) -> tensor<16x512x2048xf32>
    onnx.Return %136 : tensor<16x512x2048xf32>
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
