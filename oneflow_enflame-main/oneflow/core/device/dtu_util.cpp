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
#include "oneflow/core/device/dtu_util.h"

namespace oneflow {

DTUCurrentDeviceContext::DTUCurrentDeviceContext() : dtu_ctx_(nullptr) {
  int tmp_arr[1] = {0};
  memcpy(clusters_,tmp_arr,sizeof(tmp_arr));

  if (nullptr == dtu_ctx_) {
    dtu_ctx_ = new topsContext_t;
    OF_ENFLAME_CHECK(topsDeviceCreate(dtu_ctx_, chip_, nbClusters_, clusters_));
  }
}

DTUCurrentDeviceContext::~DTUCurrentDeviceContext() {
  DTUDestroyCtx();
}

// DTUCurrentDeviceContext& DTUCurrentDeviceContext::Get() {
//   static DTUCurrentDeviceContext guard;
//   return guard;
// }

topsContext_t* DTUCurrentDeviceContext::DTUGetCtx() {
  // if (nullptr == dtu_ctx_) {
  //   dtu_ctx_.reset(new topsContext_t);
  //   OF_ENFLAME_CHECK(topsDeviceCreate(dtu_ctx_.get(), chip_, nbClusters_, clusters_));
  // }

  // return dtu_ctx_.get();

  return dtu_ctx_;
}

// void DTUCurrentDeviceContext::DTUDestroyCtx() {
//   if (nullptr != dtu_ctx_) {
//      OF_ENFLAME_CHECK(topsDeviceDestroy(*dtu_ctx_)); 
//   }
// }

void DTUCurrentDeviceContext::DTUDestroyCtx() {
  if (nullptr != dtu_ctx_) {
     OF_ENFLAME_CHECK(topsDeviceDestroy(*dtu_ctx_)); 
     delete dtu_ctx_;
  }
}

}  // namespace oneflow
#endif  // WITH_ENFLAME