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

#include "oneflow/core/device/enflame_context_handle.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/job/global_for.h"
#include "oneflow/core/job/resource_desc.h"

namespace oneflow{

topsContext_t* EnflameContextHandle::enflame_device_handle() {
  return Global<DTUCurrentDeviceContext>::Get()->DTUGetCtx();
}

const topsStream_t* EnflameContextHandle::enflame_stream() {
  if (!enflame_stream_) {
    enflame_stream_.reset(new topsStream_t);
    OF_ENFLAME_CHECK(topsStreamCreate(*enflame_device_handle(), enflame_stream_.get()));
  }
  return enflame_stream_.get();
}

EnflameContextHandle::~EnflameContextHandle() {
  if (enflame_stream_) { OF_ENFLAME_CHECK(topsStreamDestroy(*enflame_stream_)); }
}

void EnflameContextHandle::AddCallBack(std::function<void()> callback) {
  EnflameCBEvent cb_event;
  cb_event.callback = std::move(callback);
  // OF_ENFLAME_CHECK(topsEventCreate(*enflame_device_handle(), &cb_event.event));
  cb_event_chan_->Send(cb_event);
}

}  // namespace oneflow

#endif  // WITH_ENFLAME