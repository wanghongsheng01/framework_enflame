/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
// #ifdef WITH_ENFLAME

#include "oneflow/core/framework/framework.h"
#include "oneflow/core/device/dtu_util.h"
#include "oneflow/core/kernel/new_kernel_util.h"

namespace oneflow {

namespace {

template<typename T>
class NormalizationInferenceKernel final : public user_op::OpKernel {
 public:
  NormalizationInferenceKernel() = default;
  ~NormalizationInferenceKernel() override = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const bool training = ctx->Attr<bool>("training");
    CHECK(!training);
    topsContext_t context = ctx->device_ctx()->enflame_handle();
    const auto* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    auto* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    const auto* gamma = ctx->Tensor4ArgNameAndIndex("gamma", 0);
    const auto* beta = ctx->Tensor4ArgNameAndIndex("beta", 0);
    auto* moving_mean = ctx->Tensor4ArgNameAndIndex("moving_mean", 0);
    auto* moving_variance = ctx->Tensor4ArgNameAndIndex("moving_variance", 0);
    const auto axis = ctx->Attr<int32_t>("axis");
    const auto epsilon = ctx->Attr<float>("epsilon");

    const DataType data_type = x->data_type();
    CHECK_EQ(x->shape(), y->shape());
    CHECK_EQ(y->data_type(), data_type);
    CHECK_GE(axis, 0);
    CHECK_LT(axis, x->shape().NumAxes());

    const auto& in_shape = ctx->Tensor4ArgNameAndIndex("x", 0)->shape();
    size_t ndim = in_shape.NumAxes();
    int dim_vec[4];
    size_t i = 0;
    for (; i < ndim; ++i) { dim_vec[i] = in_shape.At(i); }
    for (; i < 4; ++i) { dim_vec[i] = 1; }

    const auto& out_shape = ctx->Tensor4ArgNameAndIndex("y", 0)->shape();
    size_t ndim_y = out_shape.NumAxes();
    int dim_vec_y[4];
    i = 0;
    for (; i < ndim_y; ++i) { dim_vec_y[i] = out_shape.At(i); }
    for (; i < 4; ++i) { dim_vec_y[i] = 1; }

    // xDesc
    T type;
    topsTensorDescriptor_t xDesc;
    OF_ENFLAME_CHECK(topsCreateTensorDescriptor(&xDesc));
    OF_ENFLAME_CHECK(topsSetTensorDescriptor(xDesc, TOPS_TENSOR_NHWC, oneflowDataType2TopsDType<T>(type), dim_vec[0], dim_vec[3], dim_vec[1], dim_vec[2]));

    // yDesc
    topsTensorDescriptor_t yDesc;
    OF_ENFLAME_CHECK(topsCreateTensorDescriptor(&yDesc));
    OF_ENFLAME_CHECK(topsSetTensorDescriptor(yDesc, TOPS_TENSOR_NHWC, oneflowDataType2TopsDType<T>(type), dim_vec_y[0], dim_vec_y[3], dim_vec_y[1], dim_vec_y[2]));

    // gamma, beta, mean, variance
    auto *gamma_p = gamma->dptr();
    auto *beta_p = beta->dptr();
    auto *mean_p = moving_mean->dptr();
    auto *variance_p = moving_variance->dptr();

    topsMemory_t out_mem;
    OF_ENFLAME_CHECK(topsBatchNormalizationForwardInference(context, 
                                            xDesc, static_cast<topsMemory_t>(const_cast<void*>(x->dptr())), 
                                            yDesc, &out_mem, 
                                            static_cast<topsMemory_t>(const_cast<void*>(static_cast<const void*>(gamma_p))),
                                            static_cast<topsMemory_t>(const_cast<void*>(static_cast<const void*>(beta_p))),
                                            static_cast<topsMemory_t>(const_cast<void*>(static_cast<const void*>(mean_p))),
                                            static_cast<topsMemory_t>(const_cast<void*>(static_cast<const void*>(variance_p))),
                                            epsilon));

    OF_ENFLAME_CHECK(topsMemcpy(context, out_mem, static_cast<topsMemory_t>(y->mut_dptr())));
    OF_ENFLAME_CHECK(topsFree(context, out_mem));
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_BN_INFERENCE_KERNEL(dtype)                                                     \
  REGISTER_USER_KERNEL("normalization")                                                         \
      .SetCreateFn<NormalizationInferenceKernel<dtype>>()                                       \
      .SetIsMatchedHob((user_op::HobDeviceTag() == "enflame")                                   \
                       & (user_op::HobDataType("y", 0) == GetDataType<dtype>::value)            \
                       & (user_op::HobAttr<bool>("training") == false))                         \
      .SetInplaceProposalFn([](const user_op::InferContext& ctx,                                \
                               user_op::AddInplaceArgPair AddInplaceArgPairFn) -> Maybe<void> { \
        if (ctx.user_op_conf().has_input("_add_to_output", 0)) {                                \
          OF_RETURN_IF_ERROR(AddInplaceArgPairFn("y", 0, "_add_to_output", 0, true));           \
        }                                                                                       \
        return Maybe<void>::Ok();                                                               \
      });

REGISTER_BN_INFERENCE_KERNEL(float)
REGISTER_BN_INFERENCE_KERNEL(double)

#undef REGISTER_BN_INFERENCE_KERNEL

}  // namespace
}  // namespace oneflow

// #endif  // WITH_ENFLAME