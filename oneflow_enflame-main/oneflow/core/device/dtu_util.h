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
#ifndef ONEFLOW_CORE_DEVICE_DTU_UTIL_H_
#define ONEFLOW_CORE_DEVICE_DTU_UTIL_H_

#ifdef WITH_ENFLAME

#include<tops.h>
#include"oneflow/core/common/util.h"

namespace oneflow {

using TopsDataType = topsDataType;

#define OF_ENFLAME_CHECK(condition)                                                                           \
  for (topsStatus_t _of_enflame_check_status = (condition); _of_enflame_check_status != TOPS_STATUS_SUCCESS;) \
  LOG(FATAL) << "Check failed: " #condition " : invalid input parameters."                                    \
             << " (" << _of_enflame_check_status << ") "

template<class T>
inline topsDataType oneflowDataType2TopsDType(T type)
{
    if(typeid(type) == typeid(float)) {
        return TOPS_DATA_FLOAT;
    } else if(typeid(type) == typeid(double)) {
        return TOPS_DATA_DOUBLE;
    } else if(typeid(type) == typeid(int8_t)) {
        return TOPS_DATA_INT8;
    } else if(typeid(type) == typeid(int32_t)) {
        return TOPS_DATA_INT32;
    } else {
        assert(false);
    }
}

// void DTUDestroyCtx(topsContext_t *ctx_ptr) {
//   if (ctx_ptr) { OF_ENFLAME_CHECK(topsDeviceDestroy(*ctx_ptr)); }
// }


// class ContextDeleter {  // a deleter class with context
// public:
//     ContextDeleter() {}
//     template <class T>
//     void operator()(T* p) {
//         std::cout << "[deleted #" << "]\n";
//         topsDeviceDestroy(*p);
//     }
// };


class DTUCurrentDeviceContext final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DTUCurrentDeviceContext)
  DTUCurrentDeviceContext();
  ~DTUCurrentDeviceContext();

  // static DTUCurrentDeviceContext& Get();
  topsContext_t* DTUGetCtx();
  void DTUDestroyCtx();

 private:
  // std::unique_ptr<topsContext_t> dtu_ctx_;
  topsContext_t *dtu_ctx_;
  const int chip_ = 0;
  const int nbClusters_ = 1;
  int clusters_[1];
};

}  // namespace oneflow

#endif  // WITH_ENFLAME

#endif  // ONEFLOW_CORE_DEVICE_MLU_UTIL_H_