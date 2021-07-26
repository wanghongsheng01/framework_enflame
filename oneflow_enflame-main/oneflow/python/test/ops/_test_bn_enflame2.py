"""
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
"""
from collections import OrderedDict

import numpy as np
import oneflow as flow
from oneflow.python.test.ops.test_util import (
    Args,
    GenArgDict,
    type_name_to_flow_type,
    type_name_to_np_type,
)
import oneflow.typing as oft
import unittest

def RunOneflowLayerBn(device_type, x, data_type, flow_args):
    flow.clear_default_session()
    dtype = type_name_to_flow_type[data_type]
    np_dtype = type_name_to_np_type[data_type]
    x = x.astype(np_dtype)

    flow.config.enable_legacy_model_io(True)

    @flow.global_function()
    def FlowJob(of_x: oft.Numpy.Placeholder(x.shape, dtype=dtype)):
        with flow.scope.placement(device_type, "0:0"):
            y = flow.layers.batch_normalization(
                of_x, *flow_args, trainable=False, training=False,
            )

            return y

    checkpoint = flow.train.CheckPoint()
    checkpoint.init()

    y = FlowJob(x).get().numpy()
    return y


def CompareBnWithCPU(
    test_case,
    device_type,
    input_shape,
    data_type,
    op_args=None,
    input_minval=-10,
    input_maxval=10,
    y_rtol=1e-2,
    y_atol=1e-2,
):
    # assert device_type in ["enflame"]
    # tf bn doesn't support double
    assert data_type in ["float32"]
    if op_args is None:
        flow_args, tf_args = [], []
    else:
        flow_args, tf_args = op_args.flow_args, op_args.tf_args

    x = np.random.uniform(low=input_minval, high=input_maxval, size=input_shape)

    msg = (
        "device_type={}, input_shape={}, data_type={}, op_args={}, input_minval={}, input_maxval={}, y_rtol={}, "
        "y_atol={}".format(
            device_type,
            input_shape,
            data_type,
            op_args,
            input_minval,
            input_maxval,
            y_rtol,
            y_atol,
        )
    )

    _y = RunOneflowLayerBn("cpu", x, data_type, flow_args)
    of_y = RunOneflowLayerBn(device_type, x, data_type, flow_args)
    print(of_y)
    test_case.assertTrue(np.allclose(of_y, _y, rtol=y_rtol, atol=y_atol), msg)


@flow.unittest.skip_unless_1n1d()
class TestBatchNormalization(flow.unittest.TestCase):
    def test_layer_batchnorm_inference(test_case):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["enflame"]
        arg_dict["data_type"] = ["float32"]
        arg_dict["input_shape"] = [(1, 3, 2, 2)]
        arg_dict["op_args"] = [
            Args([-1, 0.95, 0.0001]),
            Args([]),
        ]
        for arg in GenArgDict(arg_dict):
            CompareBnWithCPU(test_case, **arg)


if __name__ == "__main__":
    unittest.main()