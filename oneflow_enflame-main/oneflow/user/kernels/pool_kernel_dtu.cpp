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

#include <tuple>
#include "oneflow/core/framework/framework.h"


namespace oneflow {

namespace {

  typedef std::tuple<int, int, int, int> pad_type;
  pad_type PaddingCompute(const std::vector<int32_t>& pool_size, const std::vector<int32_t>& strides, const std::string& padding, int dim_vec_x[]) {
      int window_height = pool_size[0];
      int window_width = pool_size[0];
      if (pool_size.size() > 1) { window_width = pool_size[1]; }

      int vertical_stride = strides[0];
      int horizon_stride = strides[0];
      if (strides.size() > 1) { horizon_stride = strides[1]; }

      int pad_t = 0;
      int pad_b = 0;
      int pad_l = 0;
      int pad_r = 0;

      int input_height = dim_vec_x[2];
      int output_height = (input_height + (vertical_stride - 1)) / vertical_stride;
      int total_vertical_padding =
          (output_height - 1) * vertical_stride + window_height - input_height;

      int input_width = dim_vec_x[3];
      int output_width = (input_width + (horizon_stride - 1)) / horizon_stride;
      int total_horizon_padding = (output_width - 1) * horizon_stride + window_width - input_width;

      int pad_b_t_small = total_vertical_padding / 2;
      int pad_b_t_big = total_vertical_padding - pad_b_t_small;
      int pad_l_r_small = total_horizon_padding / 2;
      int pad_l_r_big = total_horizon_padding - pad_l_r_small;

      if (padding == "same_lower") {
        pad_b = pad_b_t_small;
        pad_t = pad_b_t_big;
        pad_r = pad_l_r_small;
        pad_l = pad_l_r_big;
      } else if (padding == "same_upper") {
        pad_b = pad_b_t_big;
        pad_t = pad_b_t_small;
        pad_r = pad_l_r_big;
        pad_l = pad_l_r_small;
      } else if (padding == "valid") {
        pad_t = 0;
        pad_b = 0;
        pad_l = 0;
        pad_r = 0;
      } else {
        UNIMPLEMENTED();
      }

      return std::make_tuple(pad_t, pad_b, pad_l, pad_r); 
  }

  template<topsDataType T>
  void PoolForward(user_op::KernelComputeContext* ctx, topsPoolingMode mode) {
      topsContext_t context = ctx->device_ctx()->enflame_handle();

      const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
      user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);

      // x
      const auto& x_shape = ctx->Tensor4ArgNameAndIndex("x", 0)->shape();
      size_t ndim = x_shape.NumAxes();
      int dim_vec_x[ndim];
      const void *x_dptr = x->dptr();

      // y
      const auto& y_shape = ctx->Tensor4ArgNameAndIndex("y", 0)->shape();
      size_t ndim_y = y_shape.NumAxes();
      int dim_vec_y[ndim_y];
      void *y_dptr = y->mut_dptr();
      
      const auto& data_format = ctx->Attr<std::string>("data_format");
      topsTensorFormat tensor_format = (data_format == "channels_first") ? topsTensorFormat::TOPS_TENSOR_NCHW : topsTensorFormat::TOPS_TENSOR_NHWC;
      if (data_format == "channels_first") {
        for (size_t i = 0; i < ndim; ++i) {
          dim_vec_x[i] = x_shape.At(i);
          dim_vec_y[i] = y_shape.At(i);
        }
      } else {
        // in vec
        dim_vec_x[0] = x_shape.At(0);
        dim_vec_x[1] = x_shape.At(3);
        dim_vec_x[2] = x_shape.At(1);
        dim_vec_x[3] = x_shape.At(2);

        // out vec
        dim_vec_y[0] = y_shape.At(0);
        dim_vec_y[1] = y_shape.At(3);
        dim_vec_y[2] = y_shape.At(1);
        dim_vec_y[3] = y_shape.At(2);
      }

      const std::vector<int32_t>& pool_size = ctx->Attr<std::vector<int32_t>>("pool_size");
      const std::vector<int32_t>& strides = ctx->Attr<std::vector<int32_t>>("strides");
      const std::string& padding = ctx->Attr<std::string>("padding");
      int pad_t, pad_b, pad_l, pad_r;
      std::tie(pad_t, pad_b, pad_l, pad_r) = PaddingCompute(pool_size, strides, padding, dim_vec_x);

      // xDesc
      topsTensorDescriptor_t xDesc;
      OF_ENFLAME_CHECK(topsCreateTensorDescriptor(&xDesc));
      OF_ENFLAME_CHECK(topsSetTensorDescriptor(xDesc, tensor_format, T, dim_vec_x[0], dim_vec_x[1], dim_vec_x[2], dim_vec_x[3]));

      // poolingDesc
      topsPoolingDescriptor_t poolingDesc;
      OF_ENFLAME_CHECK(topsCreatePoolingDescriptor(&poolingDesc));
      OF_ENFLAME_CHECK(topsSetPoolingDescriptor(poolingDesc, mode, pool_size.at(0),pool_size.at(1), 
                                                pad_t, pad_l,
                                                strides.at(0), strides.at(1)));

      // yDesc
      topsTensorDescriptor_t yDesc;
      OF_ENFLAME_CHECK(topsCreateTensorDescriptor(&yDesc));
      OF_ENFLAME_CHECK(topsSetTensorDescriptor(yDesc, tensor_format, T, dim_vec_y[0], dim_vec_y[1], dim_vec_y[2], dim_vec_y[3]));

      topsMemory_t out_mem;
      OF_ENFLAME_CHECK(topsPoolingForward(context, xDesc, static_cast<topsMemory_t>(const_cast<void*>(x_dptr)), poolingDesc, yDesc, &out_mem));
      
      OF_ENFLAME_CHECK(topsMemcpy(context, out_mem, static_cast<topsMemory_t>(y_dptr)));
      OF_ENFLAME_CHECK(topsFree(context, out_mem));
      OF_ENFLAME_CHECK(topsDestroyTensorDescriptor(yDesc));
      OF_ENFLAME_CHECK(topsDestroyPoolingDescriptor(poolingDesc));
      OF_ENFLAME_CHECK(topsDestroyTensorDescriptor(xDesc));
  }

  template<topsDataType T>
  void PoolBackward(user_op::KernelComputeContext* ctx, topsPoolingMode mode) {
      topsContext_t context = ctx->device_ctx()->enflame_handle();

      const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
      const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
      const user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
      user_op::Tensor* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);

      // x
      const auto& x_shape = ctx->Tensor4ArgNameAndIndex("x", 0)->shape();
      size_t ndim = x_shape.NumAxes();
      int dim_vec_x[ndim];
      const void *x_dptr = x->dptr();

      // y
      const auto& y_shape = ctx->Tensor4ArgNameAndIndex("y", 0)->shape();
      size_t ndim_y = y_shape.NumAxes();
      int dim_vec_y[ndim_y];
      const void *y_dptr = y->dptr();

      // dy
      const auto& dy_shape = ctx->Tensor4ArgNameAndIndex("dy", 0)->shape();
      size_t ndim_dy = dy_shape.NumAxes();
      int dim_vec_dy[ndim_dy];
      const void *dy_dptr = dy->dptr();

      // dx
      const auto& dx_shape = ctx->Tensor4ArgNameAndIndex("dx", 0)->shape();
      size_t ndim_dx = dx_shape.NumAxes();
      int dim_vec_dx[ndim_dx];
      void *dx_dptr = dx->mut_dptr();

      const auto& data_format = ctx->Attr<std::string>("data_format");
      topsTensorFormat tensor_format = (data_format == "channels_first") ? topsTensorFormat::TOPS_TENSOR_NCHW : topsTensorFormat::TOPS_TENSOR_NHWC;
      if (data_format == "channels_first") {
        for (size_t i = 0; i < ndim; ++i) {
          dim_vec_x[i] = x_shape.At(i);
          dim_vec_y[i] = y_shape.At(i);
          dim_vec_dy[i] = dy_shape.At(i);
          dim_vec_dx[i] = dx_shape.At(i);
        }
      } else {
        // x vec
        dim_vec_x[0] = x_shape.At(0);
        dim_vec_x[1] = x_shape.At(3);
        dim_vec_x[2] = x_shape.At(1);
        dim_vec_x[3] = x_shape.At(2);

        // y vec
        dim_vec_y[0] = y_shape.At(0);
        dim_vec_y[1] = y_shape.At(3);
        dim_vec_y[2] = y_shape.At(1);
        dim_vec_y[3] = y_shape.At(2);

        // dy vec
        dim_vec_dy[0] = dy_shape.At(0);
        dim_vec_dy[1] = dy_shape.At(3);
        dim_vec_dy[2] = dy_shape.At(1);
        dim_vec_dy[3] = dy_shape.At(2);

        // dx vec
        dim_vec_dx[0] = dx_shape.At(0);
        dim_vec_dx[1] = dx_shape.At(3);
        dim_vec_dx[2] = dx_shape.At(1);
        dim_vec_dx[3] = dx_shape.At(2);
      }

      const std::vector<int32_t>& pool_size = ctx->Attr<std::vector<int32_t>>("pool_size");
      const std::vector<int32_t>& strides = ctx->Attr<std::vector<int32_t>>("strides");
      const std::string& padding = ctx->Attr<std::string>("padding");
      int pad_t, pad_b, pad_l, pad_r;
      std::tie(pad_t, pad_b, pad_l, pad_r) = PaddingCompute(pool_size, strides, padding, dim_vec_x);

      // xDesc
      topsTensorDescriptor_t xDesc;
      OF_ENFLAME_CHECK(topsCreateTensorDescriptor(&xDesc));
      OF_ENFLAME_CHECK(topsSetTensorDescriptor(xDesc, tensor_format, T, dim_vec_x[0], dim_vec_x[1], dim_vec_x[2], dim_vec_x[3]));

      // yDesc
      topsTensorDescriptor_t yDesc;
      OF_ENFLAME_CHECK(topsCreateTensorDescriptor(&yDesc));
      OF_ENFLAME_CHECK(topsSetTensorDescriptor(yDesc, tensor_format, T, dim_vec_y[0], dim_vec_y[1], dim_vec_y[2], dim_vec_y[3]));

      // dyDesc
      topsTensorDescriptor_t dyDesc;
      OF_ENFLAME_CHECK(topsCreateTensorDescriptor(&dyDesc));
      OF_ENFLAME_CHECK(topsSetTensorDescriptor(dyDesc, tensor_format, T, dim_vec_dy[0], dim_vec_dy[1], dim_vec_dy[2], dim_vec_dy[3]));

      // poolingDesc
      topsPoolingDescriptor_t poolingDesc;
      OF_ENFLAME_CHECK(topsCreatePoolingDescriptor(&poolingDesc));
      OF_ENFLAME_CHECK(topsSetPoolingDescriptor(poolingDesc, mode, pool_size[0], pool_size[1], pad_t, pad_l, strides.at(0), strides.at(1)));

      // dxDesc
      topsTensorDescriptor_t dxDesc;
      OF_ENFLAME_CHECK(topsCreateTensorDescriptor(&dxDesc));
      OF_ENFLAME_CHECK(topsSetTensorDescriptor(dxDesc, tensor_format, T, dim_vec_dx[0], dim_vec_dx[1], dim_vec_dx[2], dim_vec_dx[3]));

      topsMemory_t out_mem;
      OF_ENFLAME_CHECK(topsPoolingBackward(context, yDesc, static_cast<topsMemory_t>(const_cast<void*>(y_dptr)),
                                                    dyDesc, static_cast<topsMemory_t>(const_cast<void*>(dy_dptr)),
                                                    xDesc, static_cast<topsMemory_t>(const_cast<void*>(x_dptr)),
                                                    poolingDesc,
                                                    dxDesc,
                                                    &out_mem));
      OF_ENFLAME_CHECK(topsMemcpy(context, out_mem, static_cast<topsMemory_t>(dx_dptr)));

      OF_ENFLAME_CHECK(topsFree(context, out_mem));
      OF_ENFLAME_CHECK(topsDestroyTensorDescriptor(dxDesc));
      OF_ENFLAME_CHECK(topsDestroyPoolingDescriptor(poolingDesc));
      OF_ENFLAME_CHECK(topsDestroyTensorDescriptor(dyDesc));
      OF_ENFLAME_CHECK(topsDestroyTensorDescriptor(yDesc));
      OF_ENFLAME_CHECK(topsDestroyTensorDescriptor(xDesc));
  }
  
}  // namespace

template<topsDataType T>
class MaxPool2DDtuKernel : public user_op::OpKernel {
 public:
  MaxPool2DDtuKernel() = default;
  ~MaxPool2DDtuKernel() = default;

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    PoolForward<T>(ctx, TOPS_POOLING_MAX);
  }
};

#define REGISTER_MAX_POOL_DTU_KERNEL(name, tops_dtype, dtype)                          \
  REGISTER_USER_KERNEL(name)                                                           \
      .SetCreateFn<MaxPool2DDtuKernel<tops_dtype>>()                                   \
      .SetIsMatchedHob((user_op::HobDeviceTag() == "enflame")                          \
                       & (user_op::HobDataType("x", 0) == GetDataType<dtype>::value));

REGISTER_MAX_POOL_DTU_KERNEL("max_pool_2d", topsDataType::TOPS_DATA_FLOAT, float)


template<topsDataType T>
class AvePool2DDtuKernel : public user_op::OpKernel {
 public:
  AvePool2DDtuKernel() = default;
  ~AvePool2DDtuKernel() = default;

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
      PoolForward<T>(ctx, TOPS_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING);
  }
};

#define REGISTER_AVE_POOL_DTU_KERNEL(name, tops_dtype, dtype)                          \
  REGISTER_USER_KERNEL(name)                                                           \
      .SetCreateFn<AvePool2DDtuKernel<tops_dtype>>()                                   \
      .SetIsMatchedHob((user_op::HobDeviceTag() == "enflame")                          \
                       & (user_op::HobDataType("x", 0) == GetDataType<dtype>::value));

REGISTER_AVE_POOL_DTU_KERNEL("avg_pool_2d", topsDataType::TOPS_DATA_FLOAT, float)


template<topsDataType T>
class MaxPool2DGradDtuKernel : public user_op::OpKernel {
 public:
  MaxPool2DGradDtuKernel() = default;
  ~MaxPool2DGradDtuKernel() = default;

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
      PoolBackward<T>(ctx, TOPS_POOLING_MAX);
  }
};

#define REGISTER_MAX_POOL_DTU_KERNEL(name, tops_dtype, dtype)                          \
  REGISTER_USER_KERNEL(name)                                                           \
      .SetCreateFn<MaxPool2DGradDtuKernel<tops_dtype>>()                               \
      .SetIsMatchedHob((user_op::HobDeviceTag() == "enflame")                          \
                       & (user_op::HobDataType("x", 0) == GetDataType<dtype>::value));

REGISTER_MAX_POOL_DTU_KERNEL("max_pool_2d_grad", topsDataType::TOPS_DATA_FLOAT, float)


template<topsDataType T>
class AvgPool2DGradDtuKernel : public user_op::OpKernel {
 public:
  AvgPool2DGradDtuKernel() = default;
  ~AvgPool2DGradDtuKernel() = default;

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
      PoolBackward<T>(ctx, TOPS_POOLING_MAX);
  }
};

#define REGISTER_MAX_POOL_DTU_KERNEL(name, tops_dtype, dtype)                          \
  REGISTER_USER_KERNEL(name)                                                           \
      .SetCreateFn<AvgPool2DGradDtuKernel<tops_dtype>>()                               \
      .SetIsMatchedHob((user_op::HobDeviceTag() == "enflame")                          \
                       & (user_op::HobDataType("x", 0) == GetDataType<dtype>::value));

REGISTER_MAX_POOL_DTU_KERNEL("avg_pool_2d_grad", topsDataType::TOPS_DATA_FLOAT, float)

}  // namespace oneflow

#endif // WITH_ENFLAME