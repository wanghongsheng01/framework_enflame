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
#ifdef WITH_ENFLAME
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/new_kernel_util.h"

namespace oneflow {

namespace {

template<DeviceType device_type, typename T>
class ReluKernelDtu final : public user_op::OpKernel {
 public:
  ReluKernelDtu() = default;
  ~ReluKernelDtu() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {

    std::cout << "-------------------------relu forward-------------------\n";/////////////Debug code///////

    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("in", 0);
    const auto& in_shape = ctx->Tensor4ArgNameAndIndex("in", 0)->shape();
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("out", 0);
    
    topsContext_t context = ctx->device_ctx()->enflame_handle();

    size_t ndim = in_shape.NumAxes();
    int dim_vec[ndim];
    for (size_t i = 0; i < ndim; ++i) { dim_vec[i] = in_shape.At(i); }

    topsTensorDescriptor_t tDesc;
    OF_ENFLAME_CHECK(topsCreateTensorDescriptor(&tDesc));
    OF_ENFLAME_CHECK(topsSetTensorDescriptor(tDesc, TOPS_TENSOR_NCHW, topsDataType::TOPS_DATA_FLOAT, dim_vec[0], dim_vec[1], dim_vec[2], dim_vec[3]));

    topsActivationDescriptor_t actDesc;
    OF_ENFLAME_CHECK(topsCreateActivationDescriptor(&actDesc));
    OF_ENFLAME_CHECK(topsSetActivationDescriptor(actDesc, TOPS_ACTIVATION_RELU));

    const void* in = x->dptr();
    // void* out = y->dptr();
    topsMemory_t out_mem;
    OF_ENFLAME_CHECK(topsActivationForward(context, actDesc, tDesc, static_cast<topsMemory_t>(const_cast<void*>(in)), tDesc, (&out_mem)));
    OF_ENFLAME_CHECK(topsMemcpy(context, out_mem, static_cast<topsMemory_t>(y->mut_dptr())));
    OF_ENFLAME_CHECK(topsFree(context, out_mem));
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_RELU_KERNEL(device, dtype)                                                     \
  REGISTER_USER_KERNEL("relu")                                                                  \
      .SetCreateFn<ReluKernelDtu<device, dtype>>()                                              \
      .SetIsMatchedHob((user_op::HobDeviceTag() == device)                                      \
                       & (user_op::HobDataType("out", 0) == GetDataType<dtype>::value))         \
      .SetInplaceProposalFn([](const user_op::InferContext&,                                    \
                               user_op::AddInplaceArgPair AddInplaceArgPairFn) -> Maybe<void> { \
        OF_RETURN_IF_ERROR(AddInplaceArgPairFn("out", 0, "in", 0, true));                       \
        return Maybe<void>::Ok();                                                               \
      });

REGISTER_RELU_KERNEL(DeviceType::kENFLAME, float)


template<DeviceType device_type, typename T>
class ReluGradKernelDtu final : public user_op::OpKernel {
 public:
  ReluGradKernelDtu() = default;
  ~ReluGradKernelDtu() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {

    std::cout << "-------------------------relu backward-------------------\n";/////////////Debug code///////

    const user_op::Tensor* y_blob = ctx->Tensor4ArgNameAndIndex("y", 0);
    const auto& y_shape = ctx->Tensor4ArgNameAndIndex("y", 0)->shape();
    const user_op::Tensor* dy_blob = ctx->Tensor4ArgNameAndIndex("dy", 0);
    user_op::Tensor* dx_blob = ctx->Tensor4ArgNameAndIndex("dx", 0);
    
    topsContext_t context = ctx->device_ctx()->enflame_handle();

    size_t ndim = y_shape.NumAxes();
    int dim_vec[ndim];
    for (size_t i = 0; i < ndim; ++i) { dim_vec[i] = y_shape.At(i); }

    topsTensorDescriptor_t tDesc;
    OF_ENFLAME_CHECK(topsCreateTensorDescriptor(&tDesc));
    OF_ENFLAME_CHECK(topsSetTensorDescriptor(tDesc, TOPS_TENSOR_NCHW, topsDataType::TOPS_DATA_FLOAT, dim_vec[0], dim_vec[1], dim_vec[2], dim_vec[3]));

    topsActivationDescriptor_t actDesc;
    OF_ENFLAME_CHECK(topsCreateActivationDescriptor(&actDesc));
    OF_ENFLAME_CHECK(topsSetActivationDescriptor(actDesc, TOPS_ACTIVATION_RELU));

    topsMemory_t out_mem;
    OF_ENFLAME_CHECK(topsActivationBackward(context, actDesc, 
                           tDesc, static_cast<topsMemory_t>(const_cast<void*>(y_blob->dptr())),
                           tDesc, static_cast<topsMemory_t>(const_cast<void*>(y_blob->dptr())),
                           tDesc, static_cast<topsMemory_t>(const_cast<void*>(dy_blob->dptr())),
                           tDesc, &out_mem));
    OF_ENFLAME_CHECK(topsMemcpy(context, out_mem, static_cast<topsMemory_t>(dx_blob->mut_dptr())));
    OF_ENFLAME_CHECK(topsFree(context, out_mem));
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_RELU_GRAD_KERNEL(device, dtype)                                                \
  REGISTER_USER_KERNEL("relu_grad")                                                             \
      .SetCreateFn<ReluGradKernelDtu<device, dtype>>()                                          \
      .SetIsMatchedHob((user_op::HobDeviceTag() == device)                                      \
                       & (user_op::HobDataType("dx", 0) == GetDataType<dtype>::value))          \
      .SetInplaceProposalFn([](const user_op::InferContext&,                                    \
                               user_op::AddInplaceArgPair AddInplaceArgPairFn) -> Maybe<void> { \
        OF_RETURN_IF_ERROR(AddInplaceArgPairFn("dx", 0, "dy", 0, true));                        \
        return Maybe<void>::Ok();                                                               \
      });

REGISTER_RELU_GRAD_KERNEL(DeviceType::kENFLAME, float)

}  // namespace

}  // namespace oneflow

#endif  // WITH_ENFLAME