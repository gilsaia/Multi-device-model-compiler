#include "mlir/IR/DialectRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "Builder/FrontendDialectTransformer.hpp"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Model-Converter", registry));
}