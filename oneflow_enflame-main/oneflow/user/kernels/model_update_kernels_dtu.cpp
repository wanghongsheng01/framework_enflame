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
//#ifdef WITH_ENFLAME

#include "oneflow/core/framework/framework.h"
#include "dtu_kernel_utils.h"

namespace oneflow {

namespace {

template<class T>
void Update(DeviceCtx* ctx, int64_t n, double scale,
            float l1, float l2, float weight_decay, float lr,
            void* model_diff, void* model, topsDataType dtype)
{  
  // const auto x = model[i];
  // const auto d = model_diff[i];
  // const float model_diff_t = d * scale + l1 * sgn(x) + l2 * x;
  // const float next_model = x - lr * (model_diff_t + weight_decay * x);
  // model[i] = (1 - lr*weight_decay) * x - lr*l1 * sgn(x) - lr*scale * d;
  //
  // final: scalar1 * model - scalar2 * sgn(model) - scalar3 * model_diff

  float scalar1 = 1.0 - lr * weight_decay;
  float scalar2 = lr * l1;
  double scalar3 = lr * scale;

  topsContext_t context = ctx->enflame_handle();
  auto modelMem = static_cast<topsMemory_t>(model);
  auto modelDiffMem = static_cast<topsMemory_t>(model_diff);
  // tensor desc
  int dims[] = {n};
  topsTensorNdDescriptor_t tensorDesc;
  topsCreateTensorNdDescriptor(&tensorDesc);
  topsSetTensorNdDescriptor(tensorDesc, dtype, 1, dims);

  topsMemory_t mem1 = scalarMulTensor<T>(context, dtype, scalar1, tensorDesc, modelMem);
  
  topsMemory_t result = nullptr;
  topsUnaryOp(context, TOPS_UNARY_OP_SIGN, tensorDesc, modelMem, tensorDesc, &result);
  topsMemory_t mem2 = scalarMulTensor<T>(context, dtype, scalar2, tensorDesc, result);
  topsFree(context, result);
  result = nullptr;

  topsBinaryOp(context, TOPS_BINARY_OP_SUB, 
        tensorDesc, mem1, tensorDesc, mem2, tensorDesc, &result);
  topsFree(context, mem1);
  mem1 = nullptr;
  topsFree(context, mem2);
  mem2 = nullptr;

  topsMemory_t mem3 = scalarMulTensor<T>(context, dtype, scalar3, tensorDesc, modelDiffMem);
  topsMemory_t newModelMem = nullptr;
  topsBinaryOp(context, TOPS_BINARY_OP_SUB, 
        tensorDesc, result, tensorDesc, mem3, tensorDesc, &newModelMem);
  topsFree(context, result);
  result = nullptr;
  topsFree(context, mem3);
  mem3 = nullptr;

  topsMemcpy(context, newModelMem, modelMem);
  topsFree(context, newModelMem);
  topsDestroyTensorNdDescriptor(tensorDesc);
}

template<typename T, typename G>
class SGDUpdateKernel final : public user_op::OpKernel 
{
 public:
  SGDUpdateKernel() = default;
  ~SGDUpdateKernel() override = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override 
  {
    std::cout << "-------------------------SGDUpdateKernel-------------------------\n";

    if (ctx->user_op_conf().has_input("skip_if", 0)) 
    {
      const user_op::Tensor* skip_if = ctx->Tensor4ArgNameAndIndex("skip_if", 0);
      CHECK_EQ(skip_if->shape().elem_cnt(), 1);
      const auto skip_if_ptr = skip_if->dptr<int64_t>();
      if (skip_if_ptr != nullptr && *skip_if_ptr != 0)
        return;
    }    

    auto scale = ctx->Attr<double>("scale");
    const auto l1 = ctx->Attr<float>("l1");
    const auto l2 = ctx->Attr<float>("l2");
    const auto weight_decay = ctx->Attr<float>("weight_decay");
    
    const user_op::Tensor* learning_rate = ctx->Tensor4ArgNameAndIndex("learning_rate", 0);
    user_op::Tensor* model_diff = ctx->Tensor4ArgNameAndIndex("model_diff", 0);
    user_op::Tensor* model = ctx->Tensor4ArgNameAndIndex("model", 0);

    if (ctx->user_op_conf().has_input("scale_by_tensor", 0)) {
      const user_op::Tensor* scale_by_tensor = ctx->Tensor4ArgNameAndIndex("scale_by_tensor", 0);
      CHECK_EQ(scale_by_tensor->data_type(), model->data_type());
      CHECK_EQ(scale_by_tensor->shape().elem_cnt(), 1);
      const auto scale_by_ptr = scale_by_tensor->dptr<T>();
      if (scale_by_ptr != nullptr) 
        scale *= *scale_by_ptr;
    }

    auto dtype = oneflowDataType2TopsDType(model->data_type());
    Update<T>(ctx->device_ctx(), model->shape().elem_cnt(), scale, l1, l2, weight_decay,
        *(learning_rate->dptr<float>()), model_diff->mut_dptr(), model->mut_dptr(), dtype);
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return true; }
};

#define REGISTER_SGD_UPDATE_KERNEL(dtype, gtype)                                 \
  REGISTER_USER_KERNEL("sgd_update")                                             \
      .SetCreateFn<SGDUpdateKernel<dtype, gtype>>()                              \
      .SetIsMatchedHob((user_op::HobDeviceTag() == DeviceType::kENFLAME)         \
                       & (user_op::HobDataType("model", 0) == GetDataType<dtype>::value) \
                       & (user_op::HobDataType("model_diff", 0) == GetDataType<gtype>::value));

REGISTER_SGD_UPDATE_KERNEL(float, float);
REGISTER_SGD_UPDATE_KERNEL(double, double);

}  // namespace

}  // namespace oneflow

//#endif // WITH_ENFLAME