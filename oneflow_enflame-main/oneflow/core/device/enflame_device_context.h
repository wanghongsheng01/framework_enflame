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
#ifndef ONEFLOW_CORE_DEVICE_ENFLAME_DEVICE_CONTEXT_H_
#define ONEFLOW_CORE_DEVICE_ENFLAME_DEVICE_CONTEXT_H_

#include "oneflow/core/kernel/kernel_context.h"
#include "oneflow/core/device/device_context.h"
#include "oneflow/core/device/enflame_context_handle.h"

namespace oneflow {

#ifdef WITH_ENFLAME

class EnflameDeviceCtx : public DeviceCtx {
 public:
  OF_DISALLOW_COPY_AND_MOVE(EnflameDeviceCtx);
  EnflameDeviceCtx() = delete;
  ~EnflameDeviceCtx() override = default;

  explicit EnflameDeviceCtx(EnflameContextHandle* stream_handler)
      : enflame_handler_(stream_handler) {}

  const topsStream_t& enflame_stream() const override {
    return *(enflame_handler_->enflame_stream());
  }

  const topsContext_t& enflame_handle() const override {
    return *(enflame_handler_->enflame_device_handle());
  }

  void SyncDevice() override { OF_ENFLAME_CHECK(topsStreamSynchronize(enflame_stream())); }

  void AddCallBack(std::function<void()> callback) const override {
    enflame_handler_->AddCallBack(callback);
  }

 protected:
  EnflameContextHandle* enflame_handler_;
};

#endif  // WITH_ENFLAME

}  // namespace oneflow

#endif  // ONEFLOW_CORE_DEVICE_ENFLAME_DEVICE_CONTEXT_H_