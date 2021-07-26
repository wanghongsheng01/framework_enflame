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
#include "oneflow/core/framework/config_def.h"
#include "oneflow/core/job/job_desc.h"

namespace oneflow {

template<topsDataType T>
class MatmulDtuFloatingKernel final : public user_op::OpKernel {
 public:
  MatmulDtuFloatingKernel() = default;
  ~MatmulDtuFloatingKernel() = default;

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    topsContext_t context = ctx->device_ctx()->enflame_handle();

    const user_op::Tensor* a = ctx->Tensor4ArgNameAndIndex("a", 0);
    const user_op::Tensor* b = ctx->Tensor4ArgNameAndIndex("b", 0);
    const int nDims_a = a->shape().NumAxes();
    const int nDims_b = b->shape().NumAxes();
    int contractDim_a = ctx->Attr<bool>("transpose_a") ? (nDims_a - 2) : (nDims_a - 1);
    int contractDim_b = ctx->Attr<bool>("transpose_b") ? (nDims_a - 1) : (nDims_a - 2);
    user_op::Tensor* c = ctx->Tensor4ArgNameAndIndex("out", 0);

    // a
    int dimA[nDims_a];
    for (size_t i = 0; i < nDims_a; i++)
    {
      dimA[i] = a->shape().At(i);
    }
    const void *a_dptr = a->dptr();

    // b
    int dimB[nDims_b];
    for (size_t i = 0; i < nDims_b; i++)
    {
      dimB[i] = b->shape().At(i);
    }
    const void *b_dptr = b->dptr();

    size_t batch_dim_num = nDims_a - 2;
    int (*batch_dimension_map)[2] = nullptr;
    const size_t numBatch = batch_dim_num;
    int batch_dim_map[numBatch][2];

    if (2 == nDims_a) {
      batch_dim_num = 0;
      batch_dimension_map = nullptr;
    } else {
      batch_dim_num = nDims_a - 2;
      for (size_t i = 0; i < numBatch; i++)
      {
        batch_dim_map[i][0] = i;
        batch_dim_map[i][1] = i;
      }
      batch_dimension_map = batch_dim_map;
    }

    // c
    const int nDims_c = c->shape().NumAxes();
    int dimC[nDims_c];
    for (size_t i = 0; i < nDims_c; i++)
    {
      dimC[i] = c->shape().At(i);
    }
    void *c_dptr = c->mut_dptr();

    // aDesc
    topsTensorNdDescriptor_t aDesc;
    OF_ENFLAME_CHECK(topsCreateTensorNdDescriptor(&aDesc));
    OF_ENFLAME_CHECK(topsSetTensorNdDescriptor(aDesc, T, a->shape().NumAxes(), dimA));

    // bDesc
    topsTensorNdDescriptor_t bDesc;
    OF_ENFLAME_CHECK(topsCreateTensorNdDescriptor(&bDesc));
    OF_ENFLAME_CHECK(topsSetTensorNdDescriptor(bDesc, T, b->shape().NumAxes(), dimB));

    // dotDesc
    topsDotDescriptor_t dotDesc;
    OF_ENFLAME_CHECK(topsCreateDotDescriptor(&dotDesc));
    OF_ENFLAME_CHECK(topsSetDotDescriptor(dotDesc, contractDim_a, contractDim_b, batch_dim_num, batch_dimension_map));

    // cDesc
    topsTensorNdDescriptor_t cDesc;
    OF_ENFLAME_CHECK(topsCreateTensorNdDescriptor(&cDesc));
    OF_ENFLAME_CHECK(topsSetTensorNdDescriptor(cDesc, T, a->shape().NumAxes(), dimC));

    topsMemory_t out_mem;
    OF_ENFLAME_CHECK(topsDot(context, aDesc, static_cast<topsMemory_t>(const_cast<void*>(a_dptr)),
                                      bDesc, static_cast<topsMemory_t>(const_cast<void*>(b_dptr)),
                                      dotDesc, 
                                      cDesc, &out_mem));
    OF_ENFLAME_CHECK(topsMemcpy(context, out_mem, static_cast<topsMemory_t>(c_dptr)));
  
    OF_ENFLAME_CHECK(topsFree(context, out_mem));
    OF_ENFLAME_CHECK(topsDestroyTensorNdDescriptor(cDesc));
    OF_ENFLAME_CHECK(topsDestroyDotDescriptor(dotDesc));
    OF_ENFLAME_CHECK(topsDestroyTensorNdDescriptor(bDesc));
    OF_ENFLAME_CHECK(topsDestroyTensorNdDescriptor(aDesc));
  }
};

#define REGISTER_MATMUL_KERNEL(name, tops_dtype, dtype)                                         \
  REGISTER_USER_KERNEL(name)                                                                    \
      .SetCreateFn<MatmulDtuFloatingKernel<tops_dtype>>()                                       \
      .SetIsMatchedHob((user_op::HobDeviceTag() == "enflame")                                   \
                       & (user_op::HobDataType("a", 0) == GetDataType<dtype>::value))           \
      .SetInplaceProposalFn([](const user_op::InferContext& ctx,                                \
                               user_op::AddInplaceArgPair AddInplaceArgPairFn) -> Maybe<void> { \
        if (ctx.user_op_conf().has_input("_add_to_output", 0)) {                                \
          OF_RETURN_IF_ERROR(AddInplaceArgPairFn("out", 0, "_add_to_output", 0, true));         \
        }                                                                                       \
        return Maybe<void>::Ok();                                                               \
      });

REGISTER_MATMUL_KERNEL("matmul", topsDataType::TOPS_DATA_FLOAT, float)
REGISTER_MATMUL_KERNEL("batch_matmul", topsDataType::TOPS_DATA_FLOAT, float)

}  // namespace oneflow


#endif // WITH_ENFLAME