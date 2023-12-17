#include "mlir/InitAllTranslations.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"

using namespace mlir;

int main(int argc, char **argv) {
  registerToCppTranslation();
  registerToLLVMIRTranslation();
  return failed(mlirTranslateMain(argc, argv, "Multi-device Translation Tool"));
}