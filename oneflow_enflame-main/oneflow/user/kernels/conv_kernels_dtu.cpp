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
#include "oneflow/user/ops/nn_util.h"
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

namespace {

template<topsDataType T>
class ConvDtuKernel final : public user_op::OpKernel {
 public:
  ConvDtuKernel() = default;
  ~ConvDtuKernel() = default;

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    const user_op::Tensor* weight = ctx->Tensor4ArgNameAndIndex("weight", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    topsContext_t context = ctx->device_ctx()->enflame_handle();

    // in
    const auto& in_shape = ctx->Tensor4ArgNameAndIndex("in", 0)->shape();
    size_t ndim = in_shape.NumAxes();
    int dim_vec_in[ndim];
    const void *in_dptr = in->dptr();

    // weight
    const auto& weight_shape = ctx->Tensor4ArgNameAndIndex("weight", 0)->shape();
    size_t ndim_w = weight_shape.NumAxes();
    int dim_vec_weight[ndim_w];
    const void *weight_dptr = weight->dptr();

    // out
    const auto& out_shape = ctx->Tensor4ArgNameAndIndex("out", 0)->shape();
    size_t ndim_o = out_shape.NumAxes();
    int dim_vec_out[ndim_o];
    void *out_dptr = out->mut_dptr();

    const auto& data_format = ctx->Attr<std::string>("data_format");
    topsTensorFormat tensor_format = (data_format == "channels_first") ? topsTensorFormat::TOPS_TENSOR_NCHW : topsTensorFormat::TOPS_TENSOR_NHWC;

    if (data_format == "channels_first") {
      for (size_t i = 0; i < ndim; ++i) { 
        dim_vec_in[i] = in_shape.At(i);
        dim_vec_weight[i] = weight_shape.At(i);
        dim_vec_out[i] = out_shape.At(i); 
      }
    } else {
      // in vec
      dim_vec_in[0] = in_shape.At(0);
      dim_vec_in[1] = in_shape.At(3);
      dim_vec_in[2] = in_shape.At(1);
      dim_vec_in[3] = in_shape.At(2);

      // weight
      dim_vec_weight[0] = weight_shape.At(0);
      dim_vec_weight[1] = weight_shape.At(3);
      dim_vec_weight[2] = weight_shape.At(1);
      dim_vec_weight[3] = weight_shape.At(2);

      // out vec
      dim_vec_out[0] = out_shape.At(0);
      dim_vec_out[1] = out_shape.At(3);
      dim_vec_out[2] = out_shape.At(1);
      dim_vec_out[3] = out_shape.At(2);
    }
    
    const auto& padding_before = ctx->Attr<std::vector<int32_t>>("padding_before");
    const auto& strides = ctx->Attr<std::vector<int32_t>>("strides");
    const auto& dilation_rate = ctx->Attr<std::vector<int32_t>>("dilation_rate");

    // difine Tensor descriptor xDesc
    topsTensorDescriptor_t xDesc;
    OF_ENFLAME_CHECK(topsCreateTensorDescriptor(&xDesc));
    OF_ENFLAME_CHECK(topsSetTensorDescriptor(xDesc, tensor_format, T, dim_vec_in[0], dim_vec_in[1], dim_vec_in[2], dim_vec_in[3]));

    // difine Filter descriptor wDesc
    topsFilterDescriptor_t wDesc;
    OF_ENFLAME_CHECK(topsCreateFilterDescriptor(&wDesc));
    OF_ENFLAME_CHECK(topsSetFilterDescriptor(wDesc, tensor_format, T, dim_vec_weight[0], dim_vec_weight[1], dim_vec_weight[2], dim_vec_weight[3]));

    // difine Tensor descriptor yDesc
    topsTensorDescriptor_t yDesc;
    OF_ENFLAME_CHECK(topsCreateTensorDescriptor(&yDesc));
    OF_ENFLAME_CHECK(topsSetTensorDescriptor(yDesc, tensor_format, T, dim_vec_out[0], dim_vec_out[1], dim_vec_out[2], dim_vec_out[3]));

    // difine conv descriptor convDesc
    topsConvolutionDescriptor_t convDesc;
    OF_ENFLAME_CHECK(topsCreateConvolutionDescriptor(&convDesc));
    OF_ENFLAME_CHECK(topsSetConvolutionDescriptor(convDesc, padding_before.at(0), padding_before.at(0), 
                                                            padding_before.at(1), padding_before.at(1), 
                                                            strides.at(0), strides.at(1), 
                                                            1, 1, 
                                                            dilation_rate.at(0), dilation_rate.at(1)));

    // conv forward operation
    topsMemory_t mem3;
    OF_ENFLAME_CHECK(topsConvolutionForward(context, xDesc, static_cast<topsMemory_t>(const_cast<void*>(in_dptr)), 
                                                     wDesc, static_cast<topsMemory_t>(const_cast<void*>(weight_dptr)), 
                                                     convDesc, yDesc, (&mem3)));
    OF_ENFLAME_CHECK(topsMemcpy(context, mem3, static_cast<topsMemory_t>(out_dptr)));
    OF_ENFLAME_CHECK(topsFree(context, mem3));
    OF_ENFLAME_CHECK(topsDestroyConvolutionDescriptor(convDesc));
    OF_ENFLAME_CHECK(topsDestroyTensorDescriptor(xDesc));
    OF_ENFLAME_CHECK(topsDestroyFilterDescriptor(wDesc));
    OF_ENFLAME_CHECK(topsDestroyTensorDescriptor(yDesc));
  }
};

#define REGISTER_CONV_KERNEL(op_name, tops_dtype, dtype)                                  \
  REGISTER_USER_KERNEL(#op_name)                                                          \
      .SetCreateFn<ConvDtuKernel<tops_dtype>>()                                           \
      .SetIsMatchedHob((user_op::HobDeviceTag() == "enflame")                             \
                       & (user_op::HobAttr<int32_t>("groups") == 1)                       \
                       & (user_op::HobDataType("in", 0) == GetDataType<dtype>::value))

REGISTER_CONV_KERNEL(conv2d, topsDataType::TOPS_DATA_FLOAT, float);

template<topsDataType T>
class ConvDataGradDtuKernel final : public user_op::OpKernel {
 public:
  ConvDataGradDtuKernel() = default;
  ~ConvDataGradDtuKernel() = default;

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    topsContext_t context = ctx->device_ctx()->enflame_handle();
    const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    const user_op::Tensor* weight = ctx->Tensor4ArgNameAndIndex("weight", 0);
    user_op::Tensor* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);

    // dy
    const auto& dy_shape = dy->shape();
    size_t ndim = dy_shape.NumAxes();
    int dim_vec_dy[ndim];
    const void *dy_dptr = dy->dptr();

    // weight
    const auto& weight_shape = weight->shape();
    size_t ndim_w = weight_shape.NumAxes();
    int dim_vec_weight[ndim_w];
    const void *weight_dptr = weight->dptr();

    // dx
    const auto& dx_shape = dx->shape();
    size_t ndim_dx = dx_shape.NumAxes();
    int dim_vec_dx[ndim_dx];
    void *dx_dptr = dx->mut_dptr();

    const auto& data_format = ctx->Attr<std::string>("data_format");
    topsTensorFormat tensor_format = (data_format == "channels_first") ? topsTensorFormat::TOPS_TENSOR_NCHW : topsTensorFormat::TOPS_TENSOR_NHWC;

    if (data_format == "channels_first") {
      for (size_t i = 0; i < ndim; ++i) {
        dim_vec_dy[i] = dy_shape.At(i);
        dim_vec_weight[i] = weight_shape.At(i);
        dim_vec_dx[i] = dx_shape.At(i);
      }
    } else {
      // dy
      dim_vec_dy[0] = dy_shape.At(0);
      dim_vec_dy[1] = dy_shape.At(3);
      dim_vec_dy[2] = dy_shape.At(1);
      dim_vec_dy[3] = dy_shape.At(2);

      // weight
      dim_vec_weight[0] = weight_shape.At(0);
      dim_vec_weight[1] = weight_shape.At(3);
      dim_vec_weight[2] = weight_shape.At(1);
      dim_vec_weight[3] = weight_shape.At(2);

      // dx
      dim_vec_dx[0] = dx_shape.At(0);
      dim_vec_dx[1] = dx_shape.At(3);
      dim_vec_dx[2] = dx_shape.At(1);
      dim_vec_dx[3] = dx_shape.At(2);
    }

    // dyDesc
    topsTensorDescriptor_t dyDesc;
    OF_ENFLAME_CHECK(topsCreateTensorDescriptor(&dyDesc));
    OF_ENFLAME_CHECK(topsSetTensorDescriptor(dyDesc, tensor_format, T, dim_vec_dy[0], dim_vec_dy[1], dim_vec_dy[2], dim_vec_dy[3]));
    
    // wDesc
    topsFilterDescriptor_t wDesc;
    OF_ENFLAME_CHECK(topsCreateFilterDescriptor(&wDesc));
    OF_ENFLAME_CHECK(topsSetFilterDescriptor(wDesc, tensor_format, T, dim_vec_weight[0], dim_vec_weight[1], dim_vec_weight[2], dim_vec_weight[3]));

    // difine conv descriptor convDesc
    topsConvolutionDescriptor_t convDesc;
    OF_ENFLAME_CHECK(topsCreateConvolutionDescriptor(&convDesc));
    const auto& padding_before = ctx->Attr<std::vector<int32_t>>("padding_before");
    const auto& strides = ctx->Attr<std::vector<int32_t>>("strides");
    const auto& dilation_rate = ctx->Attr<std::vector<int32_t>>("dilation_rate");
    OF_ENFLAME_CHECK(topsSetConvolutionDescriptor(convDesc, padding_before.at(0), padding_before.at(0),
                                                            padding_before.at(1), padding_before.at(1),
                                                            strides.at(0), strides.at(1),
                                                            1, 1,
                                                            dilation_rate.at(0), dilation_rate.at(1)));

    // dxDesc
    topsTensorDescriptor_t dxDesc;
    OF_ENFLAME_CHECK(topsCreateTensorDescriptor(&dxDesc));
    OF_ENFLAME_CHECK(topsSetTensorDescriptor(dxDesc, tensor_format, T, dim_vec_dx[0], dim_vec_dx[1], dim_vec_dx[2], dim_vec_dx[3]));

    // conv backward operation
    topsMemory_t out_mem;
    OF_ENFLAME_CHECK(topsConvolutionBackwardData(context, dxDesc, static_cast<topsMemory_t>(const_cast<void*>(dy_dptr)), 
                                                 wDesc, static_cast<topsMemory_t>(const_cast<void*>(weight_dptr)), 
                                                 convDesc, dxDesc, &out_mem));
    OF_ENFLAME_CHECK(topsMemcpy(context, out_mem, static_cast<topsMemory_t>(dx_dptr)));
    OF_ENFLAME_CHECK(topsFree(context, out_mem));
    OF_ENFLAME_CHECK(topsDestroyTensorDescriptor(dxDesc));
    OF_ENFLAME_CHECK(topsDestroyConvolutionDescriptor(convDesc));
    OF_ENFLAME_CHECK(topsDestroyFilterDescriptor(wDesc));
    OF_ENFLAME_CHECK(topsDestroyTensorDescriptor(dyDesc));
  }
};

#define REGISTER_CONV_DATA_GRAD_KERNEL(op_name, tops_dtype, dtype)                        \
  REGISTER_USER_KERNEL(#op_name)                                                          \
      .SetCreateFn<ConvDataGradDtuKernel<tops_dtype>>()                                   \
      .SetIsMatchedHob((user_op::HobDeviceTag() == "enflame")                             \
                       & (user_op::HobAttr<int32_t>("groups") == 1)                       \
                       & (user_op::HobDataType("dy", 0) == GetDataType<dtype>::value))

REGISTER_CONV_DATA_GRAD_KERNEL(conv_data_grad, topsDataType::TOPS_DATA_FLOAT, float);


template<topsDataType T>
class ConvFilterGradDtuKernel final : public user_op::OpKernel {
 public:
  ConvFilterGradDtuKernel() = default;
  ~ConvFilterGradDtuKernel() = default;

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    topsContext_t context = ctx->device_ctx()->enflame_handle();

    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    user_op::Tensor* filter_diff = ctx->Tensor4ArgNameAndIndex("filter_diff", 0);

    const auto& data_format = ctx->Attr<std::string>("data_format");
    topsTensorFormat tensor_format = (data_format == "channels_first") ? topsTensorFormat::TOPS_TENSOR_NCHW : topsTensorFormat::TOPS_TENSOR_NHWC;

    // x
    const auto& x_shape = x->shape();
    size_t ndim_x = x_shape.NumAxes();
    int dim_vec_x[ndim_x];
    const void *x_dptr = x->dptr();

    // dy
    const auto& dy_shape = dy->shape();
    size_t ndim_dy = dy_shape.NumAxes();
    int dim_vec_dy[ndim_dy];
    const void *dy_dptr = dy->dptr();

    // filter_diff
    const auto& filter_diff_shape = filter_diff->shape();
    size_t ndim_filter_diff = filter_diff_shape.NumAxes();
    int dim_vec_filter_diff[ndim_filter_diff];
    void *filter_diff_dptr = filter_diff->mut_dptr();

    if (data_format == "channels_first") {
      for (size_t i = 0; i < ndim_x; ++i) {
        dim_vec_x[i] = x_shape.At(i);
        dim_vec_dy[i] = dy_shape.At(i);
        dim_vec_filter_diff[i] = filter_diff_shape.At(i);
      }
    } else {
      // x
      dim_vec_x[0] = x_shape.At(0);
      dim_vec_x[1] = x_shape.At(3);
      dim_vec_x[2] = x_shape.At(1);
      dim_vec_x[3] = x_shape.At(2);

      // dy
      dim_vec_dy[0] = dy_shape.At(0);
      dim_vec_dy[1] = dy_shape.At(3);
      dim_vec_dy[2] = dy_shape.At(1);
      dim_vec_dy[3] = dy_shape.At(2);

      // filter_diff
      dim_vec_filter_diff[0] = filter_diff_shape.At(0);
      dim_vec_filter_diff[1] = filter_diff_shape.At(3);
      dim_vec_filter_diff[2] = filter_diff_shape.At(1);
      dim_vec_filter_diff[3] = filter_diff_shape.At(2);
    }

    // xDesc
    topsTensorDescriptor_t xDesc;
    OF_ENFLAME_CHECK(topsCreateTensorDescriptor(&xDesc));
    OF_ENFLAME_CHECK(topsSetTensorDescriptor(xDesc, tensor_format, T, dim_vec_x[0], dim_vec_x[1], dim_vec_x[2], dim_vec_x[3]));

    // dyDesc
    topsTensorDescriptor_t dyDesc;
    OF_ENFLAME_CHECK(topsCreateTensorDescriptor(&dyDesc));
    OF_ENFLAME_CHECK(topsSetTensorDescriptor(dyDesc, tensor_format, T, dim_vec_dy[0], dim_vec_dy[1], dim_vec_dy[2], dim_vec_dy[3]));

    // convDesc
    topsConvolutionDescriptor_t convDesc;
    OF_ENFLAME_CHECK(topsCreateConvolutionDescriptor(&convDesc));
    const auto& padding_before = ctx->Attr<std::vector<int32_t>>("padding_before");
    const auto& strides = ctx->Attr<std::vector<int32_t>>("strides");
    const auto& dilation_rate = ctx->Attr<std::vector<int32_t>>("dilation_rate");
    OF_ENFLAME_CHECK(topsSetConvolutionDescriptor(convDesc, padding_before.at(0), padding_before.at(0), 
                                                            padding_before.at(1), padding_before.at(1), 
                                                            1, 1, 
                                                            1, 1, 
                                                            strides.at(0), strides.at(1))); // strides in forward is dilation of dy in backward

    // dwDesc (filter_diff)
    topsFilterDescriptor_t dwDesc;
    OF_ENFLAME_CHECK(topsCreateFilterDescriptor(&dwDesc));
    OF_ENFLAME_CHECK(topsSetFilterDescriptor(dwDesc, tensor_format, T, dim_vec_filter_diff[0], dim_vec_filter_diff[1], dim_vec_filter_diff[2], dim_vec_filter_diff[3]));

    topsMemory_t out_mem;
    OF_ENFLAME_CHECK(topsConvolutionBackwardFilter(context, xDesc, static_cast<topsMemory_t>(const_cast<void*>(x_dptr)),
                                                            dyDesc, static_cast<topsMemory_t>(const_cast<void*>(dy_dptr)),
                                                            convDesc, dwDesc, &out_mem));
    OF_ENFLAME_CHECK(topsMemcpy(context, out_mem, static_cast<topsMemory_t>(filter_diff_dptr)));
    OF_ENFLAME_CHECK(topsFree(context, out_mem));
    OF_ENFLAME_CHECK(topsDestroyFilterDescriptor(dwDesc));
    OF_ENFLAME_CHECK(topsDestroyConvolutionDescriptor(convDesc));
    OF_ENFLAME_CHECK(topsDestroyTensorDescriptor(dyDesc));
    OF_ENFLAME_CHECK(topsDestroyTensorDescriptor(xDesc));
  }
};

#define REGISTER_CONV_FILTER_GRAD_KERNEL(op_name, tops_dtype, dtype)                        \
  REGISTER_USER_KERNEL(#op_name)                                                            \
    .SetCreateFn<ConvFilterGradDtuKernel<tops_dtype>>()                                     \
      .SetIsMatchedHob((user_op::HobDeviceTag() == "enflame")                               \
                       & (user_op::HobAttr<int32_t>("groups") == 1)                         \
                       & (user_op::HobDataType("dy", 0) == GetDataType<dtype>::value))

REGISTER_CONV_FILTER_GRAD_KERNEL(conv_filter_grad, topsDataType::TOPS_DATA_FLOAT, float);

template<topsDataType T>
class ConvBiasGradDtuKernel final : public user_op::OpKernel {
 public:
  ConvBiasGradDtuKernel() = default;
  ~ConvBiasGradDtuKernel() = default;

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    topsContext_t context = ctx->device_ctx()->enflame_handle();

    const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    user_op::Tensor* bias_diff = ctx->Tensor4ArgNameAndIndex("bias_diff", 0);

    const auto& data_format = ctx->Attr<std::string>("data_format");
    topsTensorFormat tensor_format = (data_format == "channels_first") ? topsTensorFormat::TOPS_TENSOR_NCHW : topsTensorFormat::TOPS_TENSOR_NHWC;

    // dy
    const auto& dy_shape = dy->shape();
    size_t ndim_dy = dy_shape.NumAxes();
    int dim_vec_dy[ndim_dy];
    const void *dy_dptr = dy->dptr();

    // bias_diff
    void *out = bias_diff->mut_dptr();

    for (size_t i = 0; i < ndim_dy; ++i) {
        dim_vec_dy[i] = dy_shape.At(i);
    }

    int dim_reduce[2];
    int dim_out[ndim_dy - 2];
    if (data_format == "channels_first") {
      dim_reduce[0] = dy_shape.At(ndim_dy - 1);
      dim_reduce[1] = dy_shape.At(ndim_dy - 2);
      dim_out[0] = dy_shape.At(0);
      dim_out[1] = dy_shape.At(1);
    } else {
      dim_reduce[0] = dy_shape.At(2);
      dim_reduce[1] = dy_shape.At(1);
      dim_out[0] = dy_shape.At(0);
      dim_out[1] = dy_shape.At(ndim_dy - 1);
    }

    // xDesc
    topsTensorNdDescriptor_t xDesc;
    OF_ENFLAME_CHECK(topsCreateTensorNdDescriptor(&xDesc));
    OF_ENFLAME_CHECK(topsSetTensorNdDescriptor(xDesc, T, ndim_dy, dim_vec_dy));

    // reduceDesc
    topsReduceDescriptor_t reduceDesc;
    OF_ENFLAME_CHECK(topsCreateReduceDescriptor(&reduceDesc));
    OF_ENFLAME_CHECK(topsSetReduceDescriptor(reduceDesc, TOPS_REDUCE_ADD, 2, dim_reduce));

    // initValue
    float initValue = 0.0;

    // yDesc
    topsTensorNdDescriptor_t yDesc;
    OF_ENFLAME_CHECK(topsCreateTensorNdDescriptor(&yDesc));
    topsSetTensorNdDescriptor(yDesc, T, ndim_dy - 2, dim_out);

    topsMemory_t out_mem;
    OF_ENFLAME_CHECK(topsReduceTensor(context, xDesc, static_cast<topsMemory_t>(const_cast<void*>(dy_dptr)), reduceDesc, &initValue, yDesc, &out_mem));
    OF_ENFLAME_CHECK(topsMemcpy(context, out_mem, static_cast<topsMemory_t>(out)));

    OF_ENFLAME_CHECK(topsFree(context, out_mem));
    OF_ENFLAME_CHECK(topsDestroyTensorNdDescriptor(yDesc));
    OF_ENFLAME_CHECK(topsDestroyReduceDescriptor(reduceDesc));
    OF_ENFLAME_CHECK(topsDestroyTensorNdDescriptor(xDesc));

  }
};

#define REGISTER_CONV_BIAS_GRAD_KERNEL(op_name, tops_dtype, dtype)                        \
  REGISTER_USER_KERNEL(#op_name)                                                            \
    .SetCreateFn<ConvBiasGradDtuKernel<tops_dtype>>()                                     \
      .SetIsMatchedHob((user_op::HobDeviceTag() == "enflame")                               \
                       & (user_op::HobDataType("dy", 0) == GetDataType<dtype>::value))

REGISTER_CONV_BIAS_GRAD_KERNEL(conv_bias_grad, topsDataType::TOPS_DATA_FLOAT, float);

}

}  // namespace oneflow

#endif  // WITH_ENFLAME