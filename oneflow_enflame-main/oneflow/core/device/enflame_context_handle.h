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
#ifndef ONEFLOW_CORE_DEVICE_ENFLAME_CONTEXT_HANDLE_H_
#define ONEFLOW_CORE_DEVICE_ENFLAME_CONTEXT_HANDLE_H_

#ifdef WITH_ENFLAME

#include "oneflow/core/common/channel.h"
#include "oneflow/core/device/dtu_util.h"

namespace oneflow {

struct EnflameCBEvent {
  std::function<void()> callback;
  // topsEvent_t event;
};

class EnflameContextHandle final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(EnflameContextHandle);
  EnflameContextHandle() = delete;
  EnflameContextHandle(Channel<EnflameCBEvent>* cb_event_chan)
      : cb_event_chan_(cb_event_chan) {}

  topsContext_t* enflame_device_handle();
  const topsStream_t* enflame_stream();

  void AddCallBack(std::function<void()> callback);

  ~EnflameContextHandle();

 private:
  Channel<EnflameCBEvent>* cb_event_chan_;
  std::unique_ptr<topsStream_t> enflame_stream_;
};


}  // namespace oneflow

#endif  // WITH_ENFLAME

#endif  // ONEFLOW_CORE_DEVICE_ENFLAME_CONTEXT_HANDLE_H_