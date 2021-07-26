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
import oneflow as flow
import oneflow._oneflow_internal


class Function:
    def __init__(self, func_name, handle):
        self.func_name = func_name
        self.handle = handle

    def __call__(self, *args, **kwargs):
        args = list(args)
        return self.handle(*args, **kwargs)


def RegisterFunctionalApis():
    import inspect
    import oneflow.F
    import oneflow.experimental.F

    for s in dir(oneflow._oneflow_internal.F):
        f = getattr(oneflow._oneflow_internal.F, s)
        if inspect.isbuiltin(f):
            func_name = s
            if s in _function_name_aliases:
                func_name = _function_name_aliases[s]
                setattr(oneflow.F, func_name, Function(func_name, f))
                setattr(oneflow.experimental.F, func_name, Function(func_name, f))
            setattr(oneflow.F, s, Function(func_name, f))
            setattr(oneflow.experimental.F, s, Function(func_name, f))

    del inspect


_function_name_aliases = {
    "add_scalar": "scalar_add",
}
