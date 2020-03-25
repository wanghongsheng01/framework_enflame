#ifndef ONEFLOW_CORE_VM_INFER_STREAM_TYPE_H_
#define ONEFLOW_CORE_VM_INFER_STREAM_TYPE_H_

#include <glog/logging.h>
#include "oneflow/core/common/object_msg.h"
#include "oneflow/core/vm/stream_type.h"
#include "oneflow/core/device/device_context.h"

namespace oneflow {

class Resource;

namespace vm {

class Stream;
class InstrChain;
class InstructionStatusBuffer;

struct InferStreamTypeUtil final {
  static void InitInstructionStatus(const Stream& stream, InstructionStatusBuffer* status_buffer);
  static void DeleteInstructionStatus(const Stream& stream, InstructionStatusBuffer* status_buffer);
  static bool QueryInstructionStatusDone(const Stream& stream,
                                         const InstructionStatusBuffer& status_buffer);
  static void Infer(InstrChain* instr_chain);
};

template<typename T>
class InferStreamType final : public StreamType {
 public:
  InferStreamType() = default;
  ~InferStreamType() = default;

  void InitDeviceCtx(std::unique_ptr<DeviceCtx>* device_ctx, Stream* stream) const override {}

  void InitInstructionStatus(const Stream& stream,
                             InstructionStatusBuffer* status_buffer) const override {
    return InferStreamTypeUtil::InitInstructionStatus(stream, status_buffer);
  }
  void DeleteInstructionStatus(const Stream& stream,
                               InstructionStatusBuffer* status_buffer) const override {
    return InferStreamTypeUtil::DeleteInstructionStatus(stream, status_buffer);
  }
  bool QueryInstructionStatusDone(const Stream& stream,
                                  const InstructionStatusBuffer& status_buffer) const override {
    return InferStreamTypeUtil::QueryInstructionStatusDone(stream, status_buffer);
  }
  void Infer(InstrChain* instr_chain) const override { InferStreamTypeUtil::Infer(instr_chain); }
  void Compute(InstrChain* instr_chain) const override { LOG(FATAL) << "UNIMPLEMENTED"; }

  ObjectMsgPtr<StreamDesc> MakeRemoteStreamDesc(const Resource& resource,
                                                int64_t this_machine_id) const override {
    return T().MakeRemoteStreamDesc(resource, this_machine_id);
  }
  ObjectMsgPtr<StreamDesc> MakeLocalStreamDesc(const Resource& resource) const override {
    return T().MakeLocalStreamDesc(resource);
  }
};

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_INFER_STREAM_TYPE_H_
