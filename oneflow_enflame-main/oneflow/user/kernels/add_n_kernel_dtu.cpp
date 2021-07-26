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
#include "oneflow/core/device/dtu_util.h"
#include <stdio.h>

namespace oneflow {

typedef struct Add_ {
  topsTensorNdDescriptor_t input_desc = nullptr;
  topsTensorNdDescriptor_t output_desc = nullptr;
} Add;

template<DeviceType device_type, TopsDataType T>
class DtuAddNKernel : public user_op::OpKernel {
 public:
  DtuAddNKernel() = default;
  ~DtuAddNKernel() = default;

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    size_t in_num = ctx->inputs().size();
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    topsContext_t context = ctx->device_ctx()->enflame_handle();
    topsMemory_t out_mem;

    const auto& first_input_shape = ctx->Tensor4ArgNameAndIndex("in", 0)->shape();
    size_t ndim = first_input_shape.NumAxes();
    int dim_vec[ndim];
    for (size_t i = 0; i < ndim; ++i) { dim_vec[i] = first_input_shape.At(i); }

    Add add;
    topsCreateTensorNdDescriptor(&add.input_desc);
    topsSetTensorNdDescriptor(add.input_desc, T, ndim, dim_vec);
    topsCreateTensorNdDescriptor(&add.output_desc);
    topsSetTensorNdDescriptor(add.output_desc, T, ndim, dim_vec);

    const auto* first_input_tensor = ctx->Tensor4ArgNameAndIndex("in", 0);
    CHECK_EQ(first_input_shape, first_input_tensor->shape());
    const void* first_in_dptr = first_input_tensor->dptr();

    const auto* second_input_tensor = ctx->Tensor4ArgNameAndIndex("in", 1);
    CHECK_EQ(first_input_shape, second_input_tensor->shape());
    const void* second_in_dptr = second_input_tensor->dptr();
    
    OF_ENFLAME_CHECK(topsBinaryOp(context, TOPS_BINARY_OP_ADD, 
                                  add.input_desc, static_cast<topsMemory_t>(const_cast<void*>(first_in_dptr)), 
                                  add.input_desc, static_cast<topsMemory_t>(const_cast<void*>(second_in_dptr)), 
                                  add.output_desc, (&out_mem)));

    OF_ENFLAME_CHECK(topsMemcpy(context, out_mem, static_cast<topsMemory_t>(out->mut_dptr())));
    OF_ENFLAME_CHECK(topsFree(context, out_mem));

    for (size_t i = 2; i < in_num; ++i) {
      const auto* input_i_tensor = ctx->Tensor4ArgNameAndIndex("in", i);
      CHECK_EQ(first_input_shape, input_i_tensor->shape());
      const void* in_i_dptr = const_cast<void*>(input_i_tensor->dptr());
      OF_ENFLAME_CHECK(topsBinaryOp(context, TOPS_BINARY_OP_ADD, 
                                    add.input_desc, static_cast<topsMemory_t>(const_cast<void*>(in_i_dptr)), 
                                    add.input_desc, static_cast<topsMemory_t>(out->mut_dptr()), 
                                    add.output_desc, &out_mem));
     OF_ENFLAME_CHECK(topsMemcpy(context, out_mem, static_cast<topsMemory_t>(out->mut_dptr())));
     OF_ENFLAME_CHECK(topsFree(context, out_mem));
    }

    topsDestroyTensorNdDescriptor(add.input_desc);
    topsDestroyTensorNdDescriptor(add.output_desc);

  }
};

#define REGISTER_ADDN_ENFLAME_KERNEL(dtype)                                \
  REGISTER_USER_KERNEL("add_n")                                            \
      .SetCreateFn<DtuAddNKernel<DeviceType::kENFLAME, dtype>>()           \
      .SetIsMatchedHob((user_op::HobDeviceTag() == DeviceType::kENFLAME));

REGISTER_ADDN_ENFLAME_KERNEL(topsDataType::TOPS_DATA_FLOAT)

}  // namespace oneflow

#endif  // WITH_ENFLAME
