; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

declare ptr @malloc(i64)

declare void @free(ptr)

define { ptr, ptr, i64, [3 x i64], [3 x i64] } @main_graph(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6, i64 %7, i64 %8, ptr %9, ptr %10, i64 %11, i64 %12, i64 %13, i64 %14, i64 %15, i64 %16, i64 %17) {
  %19 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1228800) to i64), i64 64))
  %20 = ptrtoint ptr %19 to i64
  %21 = add i64 %20, 63
  %22 = urem i64 %21, 64
  %23 = sub i64 %21, %22
  %24 = inttoptr i64 %23 to ptr
  %25 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } undef, ptr %19, 0
  %26 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %25, ptr %24, 1
  %27 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %26, i64 0, 2
  %28 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %27, i64 3, 3, 0
  %29 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %28, i64 640, 3, 1
  %30 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %29, i64 640, 3, 2
  %31 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %30, i64 409600, 4, 0
  %32 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %31, i64 640, 4, 1
  %33 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %32, i64 1, 4, 2
  br label %34

34:                                               ; preds = %65, %18
  %35 = phi i64 [ %66, %65 ], [ 0, %18 ]
  %36 = icmp slt i64 %35, 3
  br i1 %36, label %37, label %67

37:                                               ; preds = %34
  br label %38

38:                                               ; preds = %63, %37
  %39 = phi i64 [ %64, %63 ], [ 0, %37 ]
  %40 = icmp slt i64 %39, 640
  br i1 %40, label %41, label %65

41:                                               ; preds = %38
  br label %42

42:                                               ; preds = %45, %41
  %43 = phi i64 [ %62, %45 ], [ 0, %41 ]
  %44 = icmp slt i64 %43, 40
  br i1 %44, label %45, label %63

45:                                               ; preds = %42
  %46 = mul i64 %43, 16
  %47 = sub i64 640, %46
  %48 = trunc i64 %47 to i32
  %49 = insertelement <16 x i32> undef, i32 %48, i32 0
  %50 = shufflevector <16 x i32> %49, <16 x i32> undef, <16 x i32> zeroinitializer
  %51 = icmp sgt <16 x i32> %50, <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %52 = mul i64 %35, 409600
  %53 = mul i64 %39, 640
  %54 = add i64 %52, %53
  %55 = add i64 %54, %46
  %56 = getelementptr float, ptr %1, i64 %55
  %57 = call <16 x float> @llvm.masked.load.v16f32.p0(ptr %56, i32 4, <16 x i1> %51, <16 x float> zeroinitializer)
  %58 = getelementptr float, ptr %10, i64 %55
  %59 = call <16 x float> @llvm.masked.load.v16f32.p0(ptr %58, i32 4, <16 x i1> %51, <16 x float> zeroinitializer)
  %60 = fadd <16 x float> %57, %59
  %61 = getelementptr float, ptr %24, i64 %55
  call void @llvm.masked.store.v16f32.p0(<16 x float> %60, ptr %61, i32 4, <16 x i1> %51)
  %62 = add i64 %43, 1
  br label %42

63:                                               ; preds = %42
  %64 = add i64 %39, 1
  br label %38

65:                                               ; preds = %38
  %66 = add i64 %35, 1
  br label %34

67:                                               ; preds = %34
  ret { ptr, ptr, i64, [3 x i64], [3 x i64] } %33
}

define void @_mlir_ciface_main_graph(ptr %0, ptr %1, ptr %2) {
  %4 = load { ptr, ptr, i64, [3 x i64], [3 x i64] }, ptr %1, align 8
  %5 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %4, 0
  %6 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %4, 1
  %7 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %4, 2
  %8 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %4, 3, 0
  %9 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %4, 3, 1
  %10 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %4, 3, 2
  %11 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %4, 4, 0
  %12 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %4, 4, 1
  %13 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %4, 4, 2
  %14 = load { ptr, ptr, i64, [3 x i64], [3 x i64] }, ptr %2, align 8
  %15 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %14, 0
  %16 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %14, 1
  %17 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %14, 2
  %18 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %14, 3, 0
  %19 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %14, 3, 1
  %20 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %14, 3, 2
  %21 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %14, 4, 0
  %22 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %14, 4, 1
  %23 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %14, 4, 2
  %24 = call { ptr, ptr, i64, [3 x i64], [3 x i64] } @main_graph(ptr %5, ptr %6, i64 %7, i64 %8, i64 %9, i64 %10, i64 %11, i64 %12, i64 %13, ptr %15, ptr %16, i64 %17, i64 %18, i64 %19, i64 %20, i64 %21, i64 %22, i64 %23)
  store { ptr, ptr, i64, [3 x i64], [3 x i64] } %24, ptr %0, align 8
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: read)
declare <16 x float> @llvm.masked.load.v16f32.p0(ptr nocapture, i32 immarg, <16 x i1>, <16 x float>) #0

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: write)
declare void @llvm.masked.store.v16f32.p0(<16 x float>, ptr nocapture, i32 immarg, <16 x i1>) #1

attributes #0 = { nocallback nofree nosync nounwind willreturn memory(argmem: read) }
attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: write) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
