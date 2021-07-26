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
class SoftmaxKernelDtu final : public user_op::OpKernel {
 public:
  SoftmaxKernelDtu() = default;
  ~SoftmaxKernelDtu() override = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    topsContext_t context = ctx->device_ctx()->enflame_handle();
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    const int64_t num_classes = in->shape().At(in->shape().NumAxes() - 1);
    const int64_t num_instances = in->shape().Count(0, in->shape().NumAxes() - 1);

    // softmaxDesc
    topsSoftmaxDescriptor_t softmaxDesc;
    OF_ENFLAME_CHECK(topsCreateSoftmaxDescriptor(&softmaxDesc));
    OF_ENFLAME_CHECK(topsSetSoftmaxDescriptor(softmaxDesc, TOPS_SOFTMAX_ACCURATE, TOPS_SOFTMAX_MODE_INSTANCE));

    // xDesc
    T type;
    topsTensorDescriptor_t xDesc;
    OF_ENFLAME_CHECK(topsCreateTensorDescriptor(&xDesc));
    OF_ENFLAME_CHECK(topsSetTensorDescriptor(xDesc, TOPS_TENSOR_NHWC, oneflowDataType2TopsDType<T>(type), num_instances, num_classes, 1, 1));

    topsMemory_t out_mem;
    OF_ENFLAME_CHECK(topsSoftmaxForward(context, softmaxDesc, xDesc, static_cast<topsMemory_t>(const_cast<void*>(in->dptr())), xDesc, &out_mem));
    OF_ENFLAME_CHECK(topsMemcpy(context, out_mem, static_cast<topsMemory_t>(out->mut_dptr())));
    OF_ENFLAME_CHECK(topsFree(context, out_mem));
    OF_ENFLAME_CHECK(topsDestroyTensorDescriptor(xDesc));
    OF_ENFLAME_CHECK(topsDestroySoftmaxDescriptor(softmaxDesc));
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_SOFTMAX_KERNEL_DTU(device, dtype)                                      \
  REGISTER_USER_KERNEL("softmax")                                                       \
      .SetCreateFn<SoftmaxKernelDtu<device, dtype>>()                                   \
      .SetIsMatchedHob((user_op::HobDeviceTag() == device)                              \
                       & (user_op::HobDataType("out", 0) == GetDataType<dtype>::value));

REGISTER_SOFTMAX_KERNEL_DTU(DeviceType::kENFLAME, float)


}  // namespace

}  // namespace oneflow

#endif  // WITH_ENFLAME