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
import unittest

import numpy as np
import oneflow as flow
import oneflow.typing as tp

# flow.config.enable_debug_mode(True)
flow.config.enable_legacy_model_io(True)


def flow_relu(x):
    def make_job(input_shape, dtype=flow.float32):
        config = flow.function_config()
        config.default_placement_scope(flow.scope.placement("enflame", "0:0"))

        @flow.global_function(type="predict", function_config=config)
        def relu_job(x: tp.Numpy.Placeholder(input_shape)) -> tp.Numpy:
            return flow.math.relu(x)

        return relu_job

    relu_fakedev_job = make_job(x.shape, dtype=flow.float32)
    y = relu_fakedev_job(x)
    return y

# def flow_relu(x):
#     def make_job(input_shape, dtype=flow.float32):
#         config = flow.function_config()
#         config.default_placement_scope(flow.scope.placement("enflame", "0:0"))
#         # config.default_data_type(flow.float32)

#         @flow.global_function(type="train", function_config=config)
#         def relu_job(x: tp.Numpy.Placeholder(input_shape)) -> tp.Numpy:
#           with flow.scope.placement("enflame", "0:0"):
#             v = flow.get_variable(
#                 shape=input_shape,
#                 dtype=flow.float32,
#                 initializer=flow.zeros_initializer(),
#                 name="x_var",
#             )
#             x_var = x + v

#           of_relu_out = flow.math.relu(x_var)
          
#           with flow.scope.placement("enflame", "0:0"):
#             flow.optimizer.SGD(
#               flow.optimizer.PiecewiseConstantScheduler([], [1e-3]), momentum=0
#             ).minimize(of_relu_out)

#           return of_relu_out

#         return relu_job

#     relu_enflame_job = make_job(x.shape, dtype=flow.float32)
#     y = relu_enflame_job(x)
#     return y


def np_relu(x):
    return np.where(x > 0, x, 0)


def _compare_with_np(test_case, input_shape):
    x = np.random.random(input_shape).astype(np.float32)
    x = (x - 0.5) * 10
    np_res = np_relu(x)
    # print(np_res)

    flow_res = flow_relu(x)
    print(flow_res)
    test_case.assertTrue(np.array_equal(np_res, flow_res))


@flow.unittest.skip_unless_1n1d()
class TestRelu(flow.unittest.TestCase):
    def test_random_value(test_case):
        _compare_with_np(test_case, (1, 224, 224, 3))


if __name__ == "__main__":
    unittest.main()



# ======================================================================================================