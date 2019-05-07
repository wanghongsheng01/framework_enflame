#include "oneflow/core/job_completer/add_parallel_cast_facade_op.h"
#include "oneflow/core/graph/op_graph.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/common/protobuf.h"

namespace oneflow {

void AddParallelCastFacadeOp(const OpGraph& op_graph, Job* job) {
  JobBuilder job_builder(job);
  using BlobParallel = std::pair<ParallelDesc, SbpParallel>;
  using BlobConsumer = std::pair<const OpNode*, std::string>;
  HashMap<LogicalBlobId, const OpNode*> lbi2producer;
  HashMap<LogicalBlobId, HashMap<BlobParallel, std::vector<BlobConsumer>>> lib2consumers;
  op_graph.ForEachNode([&](const OpNode* node) {
    for (const std::string& ibn : node->op().input_bns()) {
      const LogicalBlobId& lbi = node->op().BnInOp2Lbi(ibn);
      const OpNode* producer = node->ProducerOpNode4Lbi(lbi);
      if (producer->parallel_desc().parallel_num() != node->parallel_desc().parallel_num()
          || producer->SbpParallel4Lbi(lbi) != node->SbpParallel4BnInOp(ibn)) {
        lbi2producer.emplace(lbi, producer);
        lib2consumers[lbi][{node->parallel_desc(), node->SbpParallel4BnInOp(ibn)}].push_back(
            {node, ibn});
      }
    }
  });
  for (const auto& lib7consumer : lib2consumers) {
    const LogicalBlobId& lbi = lib7consumer.first;
    const OpNode* producer = lbi2producer.at(lbi);
    const SbpParallel& src_sbp_parallel = producer->SbpParallel4Lbi(lbi);
    const Shape& logical_blob_shape = producer->LogicalBlobDesc4Lbi(lbi).shape();
    for (const auto& blob_parallel7consumers : lib7consumer.second) {
      const ParallelDesc& dst_parallel_desc = blob_parallel7consumers.first.first;
      const SbpParallel& dst_sbp_parallel = blob_parallel7consumers.first.second;
      OperatorConf facade_op_conf{};
      facade_op_conf.set_name("System-Boxing-ParallelCastFacade-" + NewUniqueId());
      ParallelCastFacadeOpConf* cast_conf = facade_op_conf.mutable_parallel_cast_facade_conf();
      cast_conf->set_in(GenLogicalBlobName(lbi));
      cast_conf->set_out("out");
      *cast_conf->mutable_in_sbp_parallel() = src_sbp_parallel;
      *cast_conf->mutable_out_sbp_parallel() = dst_sbp_parallel;
      logical_blob_shape.ToProto(cast_conf->mutable_logical_blob_shape());
      job_builder.AddOps(dst_parallel_desc.parallel_conf(), {facade_op_conf});
      LogicalBlobId casted_lbi;
      casted_lbi.set_op_name(facade_op_conf.name());
      casted_lbi.set_blob_name(cast_conf->out());
      for (const auto& consumer7ibn : blob_parallel7consumers.second) {
        const OpNode* consumer = consumer7ibn.first;
        const std::string& ibn = consumer7ibn.second;
        OperatorConf consumer_op_conf = consumer->op().op_conf();
        PbMessage* consumer_op_type_conf =
            MutableMessageInPbMessage(&consumer_op_conf, consumer_op_conf.op_type_case());
        SetBnValInOpTypeConf(consumer_op_type_conf, ibn, GenLogicalBlobName(lbi),
                             GenLogicalBlobName(casted_lbi));
        job_builder.MutOps({consumer_op_conf});
      }
    }
  }
}

}  // namespace oneflow
