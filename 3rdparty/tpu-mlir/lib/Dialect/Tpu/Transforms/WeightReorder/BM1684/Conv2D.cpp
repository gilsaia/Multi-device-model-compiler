//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "../WeightReorder.h"
#include "tpu_mlir/Support/MathUtils.h"

using namespace bm1684;

void conv_weight_transform(int ic, int oc, int kh, int kw,
                           const void *weight_orig, const void *weight_trans,
                           int type_bytes) {
  int trans_offset;
  for (int oc_idx = 0; oc_idx < oc; oc_idx++) {
    for (int ic_idx = 0; ic_idx < ic; ic_idx++) {
      for (int k_idx = 0; k_idx < kh * kw; k_idx++) {
        int orig_offset = ic_idx * kh * kw + k_idx + oc_idx * kh * kw * ic;
        switch (type_bytes) {
        case 4:
          trans_offset = ic_idx + k_idx * align_up(ic, 2) +
                         oc_idx * kh * kw * align_up(ic, 2);
          *((float *)weight_trans + trans_offset) =
              *((float *)weight_orig + orig_offset);
          break;
        case 1:
          trans_offset = ic_idx + k_idx * align_up(ic, 4) +
                         oc_idx * kh * kw * align_up(ic, 4);
          *((char *)weight_trans + trans_offset) =
              *((char *)weight_orig + orig_offset);
          break;
        case 2:
          trans_offset = ic_idx + k_idx * ic + oc_idx * kh * kw * ic;
          *((short *)weight_trans + trans_offset) =
              *((short *)weight_orig + orig_offset);
          break;
        default:
          llvm_unreachable("wrong conv weight data type");
        }
      }
    }
  }
}

template <>
LogicalResult WeightReorder<tpu::Conv2DOp, int8_t>::matchAndRewrite(
    tpu::Conv2DOp op, PatternRewriter &rewriter) const {
  if (!module::getStorageType(op.getFilter()).isInteger(8))
    return failure();
  if (module::isWeight(op.getFilter()) == false) {
    return failure();
  }
  auto attr = op.parseParam();
  auto type_bytes = 1;
  auto filterOp = cast<top::WeightOp>(op.getFilter().getDefiningOp());
  auto filter_int8 = filterOp.read<int8_t>();
  auto filter_type = module::getElementType(op.getFilter());
  if ((attr.dh > 15 || attr.dw > 15) && attr.groups == 1) {
    int64_t factor_dh = 2, factor_dw = 2;
    int64_t new_kh = attr.kh, new_kw = attr.kw;
    int64_t new_dh = attr.dh, new_dw = attr.dw;
    if (attr.dh > 15) {
      while (factor_dh <= 15 &&
             (attr.dh % factor_dh || attr.dh / factor_dh > 15))
        factor_dh++;
      if (factor_dh > 15) {
        llvm_unreachable("Un-supported dh by conv layer");
        op.dump();
        return failure();
      }
      new_dh /= factor_dh;
      new_kh = (attr.kh - 1) * factor_dh + 1;
    }
    if (attr.dw > 15) {
      while (factor_dw <= 15 &&
             (attr.dw % factor_dw || attr.dw / factor_dw > 15))
        factor_dw++;
      if (factor_dw > 15) {
        llvm_unreachable("Un-supported dh by conv layer");
        op.dump();
        return failure();
      }
      new_dw /= factor_dw;
      new_kw = (attr.kw - 1) * factor_dw + 1;
    }
    auto input_shape = module::getShape(op->getOperand(0));
    auto filter_op = op.getFilter().getDefiningOp<top::WeightOp>();
    auto filter_type = module::getStorageType(op.getFilter());
    auto filter_i8 = filter_op.read<int8_t>();
    std::vector<int64_t> new_shape = {1, attr.oc, new_kh * new_kw,
                                      align_up(attr.ic, 4l)};
    auto filter_new = std::make_shared<std::vector<int8_t>>(
        attr.oc * new_kh * new_kw * align_up(attr.ic, 4l));
    for (int ioc = 0; ioc < attr.oc; ioc++) {
      for (int iic = 0; iic < attr.ic; iic++) {
        for (int ikh = 0; ikh < attr.kh; ikh++) {
          for (int ikw = 0; ikw < attr.kw; ikw++) {
            int offset =
                ((ioc * input_shape[1] + iic) * attr.kh + ikh) * attr.kw + ikw;
            int new_offset = iic + ikw * factor_dw * align_up(attr.ic, 4) +
                             ikh * factor_dh * new_kw * align_up(attr.ic, 4) +
                             ioc * new_kh * new_kw * align_up(attr.ic, 4);
            filter_new->at(new_offset) = filter_i8->at(offset);
          }
        }
      }
    }
    filter_i8 = filter_new;
    auto new_type = RankedTensorType::get(new_shape, filter_type);
    auto new_filter = top::WeightOp::create(op.getFilter().getDefiningOp(),
                                            "reorderd", *filter_new, new_type);
    op->setOperand(1, new_filter);
    op.setDilationsAttr(rewriter.getI64ArrayAttr({new_dh, new_dw}));
    op.setKernelShapeAttr(rewriter.getI64ArrayAttr({new_kh, new_kw}));
  } else if (attr.is_dw == false) {
    std::vector<int64_t> new_shape = {1, attr.oc, attr.kh * attr.kw,
                                      align_up(attr.ic / attr.groups, 4l)};
    int new_count =
        align_up(attr.ic / attr.groups, 4l) * attr.oc * attr.kh * attr.kw;
    auto filter_new = std::make_shared<std::vector<int8_t>>(new_count, 0);
    conv_weight_transform(attr.ic / attr.groups, attr.oc, attr.kh, attr.kw,
                          filter_int8->data(), filter_new->data(), type_bytes);
    auto new_type = RankedTensorType::get(new_shape, filter_type);
    auto new_filter = top::WeightOp::create(op.getFilter().getDefiningOp(),
                                            "reorderd", *filter_new, new_type);
    op->setOperand(1, new_filter);
  } else {
    int64_t filter_shape[4];
    filter_shape[0] = 1;
    filter_shape[1] = attr.oc;
    filter_shape[2] = attr.ic / attr.groups;
    filter_shape[3] = attr.kh * attr.kw;
    auto new_type = RankedTensorType::get(filter_shape, filter_type);
    op.getFilter().setType(new_type);
  }
  // bias op
  if (attr.has_bias) {
    auto bias_type = module::getElementType(op.getBias());
    int64_t bias_shape[4] = {1, attr.oc, 1, 1};
    auto new_type = RankedTensorType::get(bias_shape, bias_type);
    op.getBias().setType(new_type);
  }
  return success();
}

template <>
LogicalResult WeightReorder<tpu::Conv2DOp, Float32Type>::matchAndRewrite(
    tpu::Conv2DOp op, PatternRewriter &rewriter) const {
  if (!module::getStorageType(op.getFilter()).isF32())
    return failure();
  if (module::isWeight(op.getFilter()) == false) {
    return failure();
  }
  auto attr = op.parseParam();
  auto type_bytes = 4;
  auto out_type = module::getStorageType(op.getOutput());
  // filter reorder
  auto filterOp = op.getFilter().getDefiningOp<top::WeightOp>();
  auto weight_data = filterOp.read_as_byte();
  if (attr.is_dw == false) {
    std::vector<int64_t> new_shape = {1, attr.oc, attr.kh * attr.kw,
                                      align_up(attr.ic / attr.groups, 2l)};
    int new_count =
        align_up(attr.ic / attr.groups, 2l) * attr.oc * attr.kh * attr.kw;
    auto filter_new = std::make_shared<std::vector<float>>(new_count, 0);
    conv_weight_transform(attr.ic / attr.groups, attr.oc, attr.kh, attr.kw,
                          weight_data->data(), filter_new->data(), type_bytes);
    auto new_type = RankedTensorType::get(new_shape, out_type);
    auto new_filter = top::WeightOp::create(op.getFilter().getDefiningOp(),
                                            "reorderd", *filter_new, new_type);
    op->setOperand(1, new_filter);
  } else {
    int64_t filter_shape[4];
    filter_shape[0] = 1;
    filter_shape[1] = attr.oc;
    filter_shape[2] = attr.ic / attr.groups;
    filter_shape[3] = attr.kh * attr.kw;
    auto new_type = RankedTensorType::get(filter_shape, out_type);
    op.getFilter().setType(new_type);
  }

  // bias op
  if (attr.has_bias) {
    int64_t bias_shape[4] = {1, attr.oc, 1, 1};
    auto new_type = RankedTensorType::get(bias_shape, out_type);
    op.getBias().setType(new_type);
  }
  return success();
}
