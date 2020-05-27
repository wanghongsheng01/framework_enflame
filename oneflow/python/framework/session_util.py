from __future__ import absolute_import

import threading
from oneflow.core.job.job_set_pb2 import ConfigProto
import oneflow.core.job.job_pb2 as job_util
import oneflow.core.job.job_set_pb2 as job_set_util
import oneflow.python.framework.session_context as session_ctx
from oneflow.python.framework.session_context import SessionStatus
import oneflow.python.framework.compiler as compiler
import oneflow.python.framework.c_api_util as c_api_util
import oneflow.python.framework.c_api_util as c_api_util
import oneflow.python.framework.config_util as config_util
import oneflow.python.framework.env_util as env_util
import oneflow.python.framework.push_util as push_util
import oneflow.python.framework.job_instance as job_instance_util
import oneflow.python.framework.hob as hob
import oneflow.python.lib.core.enable_if as enable_if
from oneflow.python.framework.pull_util import FutureRemoteBlobs
from oneflow.python.oneflow_export import oneflow_export
from oneflow.python.framework.function_desc import FunctionDesc
from contextlib import contextmanager

import oneflow

class Session(object):
    def __init__(self):
        self.job_name2function_desc_ = {}
        self.status_ = SessionStatus.OPEN
        self.cond_var_ = threading.Condition()
        self.running_job_cnt_ = 0
        self.inter_user_job_info_ = None
        self.uuid2watch_handler_ = {}
        self.config_proto_ = _GetDefaultConfigProto()
        self.placement_scope_stack_ = []
        self.is_mirrored_strategy_enabled_stack_ = []
        self.function_flag_name2default_val_ = {}
        self.job_name2var_name2var_blob_ = {}
        self.job_name2name_scope_stack_ = {}
        self.eager_global_function_desc_stack_ = []
        self._UpdateFunctionFlagName2DefaultVal()

    @property
    def status(self): return self.status_

    @property
    def is_running(self): return self.status_ is SessionStatus.RUNNING

    @property
    def config_proto(self): return self.config_proto_

    @property
    def uuid2watch_handler(self): return self.uuid2watch_handler_

    @property
    def placement_scope_stack(self): return self.placement_scope_stack_

    @property
    def is_mirrored_strategy_enabled_stack(self): return self.is_mirrored_strategy_enabled_stack_

    @property
    def function_flag_name2default_val(self): return self.function_flag_name2default_val_

    @property
    def inter_user_job_info(self): return self.inter_user_job_info_

    @property
    def job_name2var_name2var_blob(self): return self.job_name2var_name2var_blob_

    @property
    def job_name2name_scope_stack(self): return self.job_name2name_scope_stack_

    def GetLazyFunctionDesc(self, job_name):
        if job_name in self.job_name2function_desc_: return self.job_name2function_desc_[job_name]
        return None

    def AnyGlobalFunctionDefined(self): return len(self.job_name2function_desc_) > 0

    def GetJobConfigProto(self, job_name):
      return self.job_name2function_desc_[job_name].job_config_proto

    def GetFunctionDesc(self, job_name):
        return self.job_name2function_desc_[job_name]

    def _UpdateFunctionFlagName2DefaultVal(self):
        items = c_api_util.GetFunctionConfigDef().flag_name2flag_def.items()
        self.function_flag_name2default_val_ = {k : v.default_val for k, v in items}

    def TryInit(self):
        if self.status_ is SessionStatus.OPEN: self.Init()
        return self

    def Init(self):
        assert self.status_ is SessionStatus.OPEN
        self.status_ = SessionStatus.RUNNING
        if not c_api_util.IsEnvInited(): oneflow.env.init()
        _TryCompleteConfigProto(self.config_proto_)
        c_api_util.InitGlobalSession(self.config_proto_)
        if not c_api_util.EagerExecutionEnabled():
            for job_name, func_desc in self.job_name2function_desc_.items():
                compiler.Compile(func_desc, self.config_proto_)
            assert len(self.job_name2function_desc_.items()) > 0
            c_api_util.StartGlobalSession()
            self.inter_user_job_info_ = c_api_util.GetInterUserJobInfo()
        return self

    def TryClose(self):
        if self.status_ is SessionStatus.RUNNING: self.Close()

    def Close(self):
        assert self.status_ is SessionStatus.RUNNING
        self.Sync()
        c_api_util.StopGlobalSession()
        c_api_util.DestroyGlobalSession()
        self.status_ = SessionStatus.CLOSED

    def AddJob(self, function_desc):
        assert self.status_ is SessionStatus.OPEN
        assert isinstance(function_desc, FunctionDesc)
        self.job_name2function_desc_[function_desc.job_func.__name__] = function_desc

    def Sync(self):
        assert self.status_ is SessionStatus.RUNNING
        self.cond_var_.acquire()
        while self.running_job_cnt_ > 0:
            self.cond_var_.wait()
        assert self.running_job_cnt_ == 0
        self.cond_var_.release()

    def LazyRun(self, job_func, *arg):
        assert self.status_ is SessionStatus.RUNNING
        remote_blobs = self.LaunchUserJob(job_func, *arg)
        if remote_blobs is None: return
        return FutureRemoteBlobs(self).SetResult(remote_blobs).Inited()

    def EagerRun(self, function_desc, job_func, *arg):
        with self._EagerGlobalFunctionDescScope(function_desc):
            return TODO()

    def LaunchUserJob(self, job_func, *arg):
        assert self.status_ is SessionStatus.RUNNING
        job_name = job_func.__name__
        push_util.AsyncPush(self, job_func, *arg)
        self.LaunchJob(job_instance_util.MakeUserJobInstance(job_name))
        return job_func.__oneflow_output_remote_blobs__

    def LaunchJob(self, job_instance):
        assert self.status_ is SessionStatus.RUNNING
        self._IncRunningJobCnt()
        job_instance.AddPostFinishCallback(lambda _: self._DecRunningJobCnt())
        c_api_util.LaunchJob(job_instance)

    def AsyncPush(self, op_name, push_data_cb):
        assert self.status_ is SessionStatus.RUNNING
        push_job_name = self.inter_user_job_info.input_or_var_op_name2push_job_name[op_name]
        self.LaunchJob(job_instance_util.MakePushJobInstance(push_job_name, op_name, push_data_cb))

    def AsyncPull(self, op_name, pull_data_cb):
        assert self.status_ is SessionStatus.RUNNING
        pull_job_name = self.inter_user_job_info.output_or_var_op_name2pull_job_name[op_name]
        self.LaunchJob(job_instance_util.MakePullJobInstance(pull_job_name, op_name, pull_data_cb))

    def HasAnyCallbackAfterFunctionReturn(self):
        return len(self.uuid2watch_handler) > 0

    def StashVariableBlob4Job(self, job_name, var_name, var_blob):
        if job_name not in self.job_name2var_name2var_blob:
            self.job_name2var_name2var_blob[job_name] = dict()
        assert var_name not in self.job_name2var_name2var_blob[job_name]
        self.job_name2var_name2var_blob[job_name][var_name] = var_blob

    def TryGetVariableBlobOfJobFromStash(self, job_name, var_name):
        if job_name not in self.job_name2var_name2var_blob:
            return None

        var_name2var_blob = self.job_name2var_name2var_blob[job_name]
        if var_name not in var_name2var_blob:
            return None

        return var_name2var_blob[var_name]

    def CurrentEagerGlobalFunctionDesc(self):
        if len(self.eager_global_function_desc_stack_) == 0: return None
        return self.eager_global_function_desc_stack_[0]

    @contextmanager
    def _EagerGlobalFunctionDescScope(self, function_desc):
        self.eager_global_function_desc_stack_.insert(0, function_desc)
        try:
            yield
        finally:
            self.eager_global_function_desc_stack_.pop(0)

    def _IncRunningJobCnt(self):
        assert self.status_ is SessionStatus.RUNNING
        self.cond_var_.acquire()
        self.running_job_cnt_ += 1
        self.cond_var_.release()

    def _DecRunningJobCnt(self):
        self.cond_var_.acquire()
        self.running_job_cnt_ -= 1
        self.cond_var_.notify()
        self.cond_var_.release()

@enable_if.condition(hob.in_normal_mode & ~hob.any_global_function_defined)
def enable_eager_execution(val = True):
    return c_api_util.EnableEagerExecution(val)

@oneflow_export("enable_eager_execution")
def api_enable_eager_execution(val = True):
    return enable_if.unique(enable_eager_execution)(val)

@oneflow_export("eager_execution_enabled")
def api_eager_execution_enabled():
    return c_api_util.EagerExecutionEnabled()

@oneflow_export("clear_default_session")
def clear_default_session():
    session_ctx.TryCloseDefaultSession()
    session_ctx.OpenDefaultSession(Session())
    c_api_util.EnableEagerExecution(False)

@oneflow_export("sync_default_session")
def sync_default_session():
    session_ctx.GetDefaultSession().Sync()

def _TryCompleteConfigProto(config_proto):
    if config_proto.resource.machine_num == 0:
        config_proto.resource.machine_num = len(env_util.default_env_proto.machine)

def _GetDefaultConfigProto():
    config_proto = job_set_util.ConfigProto()
    config_proto.resource.machine_num = 0
    config_proto.resource.gpu_device_num = 1
    config_proto.io_conf.data_fs_conf.localfs_conf.SetInParent()
    config_proto.io_conf.snapshot_fs_conf.localfs_conf.SetInParent()
    return config_proto

session_ctx.OpenDefaultSession(Session())
