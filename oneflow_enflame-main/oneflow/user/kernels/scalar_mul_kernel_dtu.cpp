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

template<DeviceType device_type, typename T>
class ScalarMulUserKernelDtu final : public user_op::OpKernel {
 public:
  ScalarMulUserKernelDtu() = default;
  ~ScalarMulUserKernelDtu() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    topsContext_t context = ctx->device_ctx()->enflame_handle();

    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    const void* in_ptr = in->dptr();

    T scalar_operand = static_cast<T>(0);
    if (ctx->Attr<bool>("has_int_operand")) {
      scalar_operand = static_cast<T>(ctx->Attr<int64_t>("int_operand"));
    } else if (ctx->Attr<bool>("has_float_operand")) {
      scalar_operand = static_cast<T>(ctx->Attr<double>("float_operand"));
    } else {
      UNIMPLEMENTED();
    }
    topsMemory_t scalar_mem = nullptr;
    topsMalloc(context, &scalar_mem, sizeof(scalar_operand));
    topsMemcpyHostToDevice(context, &scalar_operand, scalar_mem);

    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    void* out_ptr = out->mut_dptr();
    
    const auto& in_shape = ctx->Tensor4ArgNameAndIndex("in", 0)->shape();
    size_t ndim = in_shape.NumAxes();
    int dim_vec[ndim];
    for (size_t i = 0; i < ndim; ++i) {
      dim_vec[i] = in_shape.At(i);
    }

    topsTensorNdDescriptor_t tDesc;
    OF_ENFLAME_CHECK(topsCreateTensorNdDescriptor(&tDesc));
    topsSetTensorNdDescriptor(tDesc, TOPS_DATA_FLOAT, ndim, dim_vec);

    // scalarDesc
    int scalarDescDim[0];
    topsTensorNdDescriptor_t scalarDesc;
    OF_ENFLAME_CHECK(topsCreateTensorNdDescriptor(&scalarDesc));
    topsSetTensorNdDescriptor(scalarDesc, TOPS_DATA_FLOAT, 0, scalarDescDim);

    // run broadcast
    topsMemory_t broadcast_mem;
    topsBroadcast(context, scalarDesc, scalar_mem, scalarDescDim, tDesc, &broadcast_mem);

    // run elementwise binary
    topsMemory_t mul_mem = nullptr;
    topsBinaryOp(context, TOPS_BINARY_OP_MUL, 
                                  tDesc, static_cast<topsMemory_t>(const_cast<void*>(in_ptr)),
                                  tDesc, broadcast_mem, tDesc, &mul_mem);

    topsMemcpy(context, mul_mem, static_cast<topsMemory_t>(out_ptr));
    topsFree(context, scalar_mem);
    topsFree(context, broadcast_mem);
    topsFree(context, mul_mem);
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_KERNEL(kernel_device_type, dtype)                                              \
  REGISTER_USER_KERNEL("scalar_mul")                                                            \
      .SetCreateFn<ScalarMulUserKernelDtu<DeviceType::k##kernel_device_type, dtype>>()          \
      .SetIsMatchedHob((user_op::HobDeviceTag() == DeviceType::k##kernel_device_type)           \
                       & (user_op::HobDataType("out", 0) == GetDataType<dtype>::value))         \
      .SetInplaceProposalFn([](const user_op::InferContext&,                                    \
                               user_op::AddInplaceArgPair AddInplaceArgPairFn) -> Maybe<void> { \
        OF_RETURN_IF_ERROR(AddInplaceArgPairFn("out", 0, "in", 0, true));                       \
        return Maybe<void>::Ok();                                                               \
      });

REGISTER_KERNEL(ENFLAME, float)

}  // namespace oneflow

#endif  // WITH_ENFLAME
