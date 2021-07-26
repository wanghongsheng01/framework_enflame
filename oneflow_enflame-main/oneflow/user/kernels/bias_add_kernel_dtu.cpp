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
#include "oneflow/core/device/dtu_util.h"

namespace oneflow {

namespace {

template<DeviceType device_type, typename T>
class BiasAddKernelDtu final : public user_op::OpKernel {
 public:
  BiasAddKernelDtu() = default;
  ~BiasAddKernelDtu() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    topsContext_t context = ctx->device_ctx()->enflame_handle();
    const auto* a_tensor = ctx->Tensor4ArgNameAndIndex("a", 0);
    const auto* b_tensor = ctx->Tensor4ArgNameAndIndex("b", 0);
    auto* out_tensor = ctx->Tensor4ArgNameAndIndex("out", 0);
    const int32_t bias_add_axis = ctx->Attr<int32_t>("axis");
    const size_t num_axis = a_tensor->shape().NumAxes();
    int dim_vec_a[num_axis];
    for(int i = 0; i < num_axis; ++i) {
      dim_vec_a[i] = a_tensor->shape().At(i);
    }
    int dim_vec_b[1];
    dim_vec_b[0] = b_tensor->shape().At(0);

    // bDesc
    T type;
    topsTensorNdDescriptor_t bDesc;
    OF_ENFLAME_CHECK(topsCreateTensorNdDescriptor(&bDesc));
    OF_ENFLAME_CHECK(topsSetTensorNdDescriptor(bDesc, oneflowDataType2TopsDType<T>(type), 1, dim_vec_b));

    // broadcastDims[]
    int broadcastDims[] = { bias_add_axis };

    // aDesc
    topsTensorNdDescriptor_t aDesc;
    OF_ENFLAME_CHECK(topsCreateTensorNdDescriptor(&aDesc));
    OF_ENFLAME_CHECK(topsSetTensorNdDescriptor(aDesc, oneflowDataType2TopsDType<T>(type), num_axis, dim_vec_a));

    // broadcast
    topsMemory_t b_broad;
    OF_ENFLAME_CHECK(topsBroadcast(context, bDesc, static_cast<topsMemory_t>(const_cast<void*>(b_tensor->dptr())), broadcastDims, aDesc, &b_broad));

    // elem_add
    topsMemory_t out_mem;
    OF_ENFLAME_CHECK(topsBinaryOp(context, TOPS_BINARY_OP_ADD, 
                                  aDesc, static_cast<topsMemory_t>(const_cast<void*>(a_tensor->dptr())), 
                                  aDesc, b_broad, 
                                  aDesc, &out_mem));
    OF_ENFLAME_CHECK(topsMemcpy(context, out_mem, static_cast<topsMemory_t>(out_tensor->mut_dptr())));
    OF_ENFLAME_CHECK(topsFree(context, out_mem));
    OF_ENFLAME_CHECK(topsFree(context, b_broad));
    OF_ENFLAME_CHECK(topsDestroyTensorNdDescriptor(aDesc));
    OF_ENFLAME_CHECK(topsDestroyTensorNdDescriptor(bDesc));
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_BIAS_ADD_USER_KERNEL_DTU(op_device_type, dtype)                                \
  REGISTER_USER_KERNEL("bias_add")                                                              \
      .SetCreateFn<BiasAddKernelDtu<DeviceType::k##op_device_type, dtype>>()                    \
      .SetIsMatchedHob((user_op::HobDeviceTag() == DeviceType::k##op_device_type)               \
                       & (user_op::HobDataType("out", 0) == GetDataType<dtype>::value))         \
      .SetInplaceProposalFn([](const user_op::InferContext&,                                    \
                               user_op::AddInplaceArgPair AddInplaceArgPairFn) -> Maybe<void> { \
        OF_RETURN_IF_ERROR(AddInplaceArgPairFn("out", 0, "a", 0, true));                        \
        return Maybe<void>::Ok();                                                               \
      });

REGISTER_BIAS_ADD_USER_KERNEL_DTU(ENFLAME, float)

}  // namespace

}  // namespace oneflow

#endif  // WITH_ENFLAME