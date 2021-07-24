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
from collections import OrderedDict

import numpy as np

import oneflow as flow

from test_util import GenArgList


# TODO: auto test


def _test_adaptive_avgpool1d_forward(test_case, device):
    input = flow.Tensor(
        np.array(
            [
                [
                    [
                        0.05580734834074974,
                        -0.6875145435333252,
                        -1.654430866241455,
                        -0.6225992441177368,
                        0.10183599591255188,
                        0.05019790679216385,
                        -1.2537643909454346,
                        0.14907236397266388,
                    ]
                ]
            ]
        ),
        dtype=flow.float32,
        device=flow.device(device),
    )
    m = flow.nn.AdaptiveAvgPool1d(4)
    m.to(device)
    of_out_1 = m(input)
    of_out_2 = flow.adaptive_avg_pool1d(input, 4)
    np_out = np.array(
        [
            [
                [
                    -0.31585359573364258,
                    -1.13851499557495117,
                    0.07601694762706757,
                    -0.55234599113464355,
                ]
            ]
        ]
    )
    test_case.assertTrue(np.allclose(of_out_1.numpy(), np_out, 1e-5, 1e-5))
    test_case.assertTrue(np.allclose(of_out_2.numpy(), np_out, 1e-5, 1e-5))


def _test_adaptive_avgpool1d_backward(test_case, device):
    input = flow.Tensor(
        np.array(
            [
                [
                    [
                        0.05580734834074974,
                        -0.6875145435333252,
                        -1.654430866241455,
                        -0.6225992441177368,
                        0.10183599591255188,
                        0.05019790679216385,
                        -1.2537643909454346,
                        0.14907236397266388,
                    ]
                ]
            ]
        ),
        dtype=flow.float32,
        device=flow.device(device),
        requires_grad=True,
    )
    m = flow.nn.AdaptiveAvgPool1d(4)
    of_out = m(input)
    of_out = of_out.sum()
    of_out.backward()
    np_grad = np.array([[[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]]])
    test_case.assertTrue(np.allclose(input.grad.numpy(), np_grad, 1e-5, 1e-5))


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
def _test_adaptive_avgpool2d_forward(test_case, device):
    input = flow.Tensor(
        np.array(
            [
                [
                    [
                        [
                            0.10039155930280685,
                            0.04879157617688179,
                            -1.0515470504760742,
                            0.9466001987457275,
                        ],
                        [
                            0.45375481247901917,
                            0.23611211776733398,
                            1.343685269355774,
                            0.3979687988758087,
                        ],
                        [
                            0.05580734834074974,
                            -0.6875145435333252,
                            -1.654430866241455,
                            -0.6225992441177368,
                        ],
                        [
                            0.10183599591255188,
                            0.05019790679216385,
                            -1.2537643909454346,
                            0.14907236397266388,
                        ],
                    ]
                ]
            ]
        ),
        dtype=flow.float32,
        device=flow.device(device),
    )
    m = flow.nn.AdaptiveAvgPool2d((2, 2))
    m.to(device)
    of_out_1 = m(input)
    of_out_2 = flow.adaptive_avg_pool2d(input, (2, 2))
    np_out = np.array(
        [
            [
                [
                    [0.20976251363754272, 0.4091767966747284],
                    [-0.1199183315038681, -0.8454304933547974],
                ]
            ]
        ]
    )
    test_case.assertTrue(np.allclose(of_out_1.numpy(), np_out, 1e-5, 1e-5))
    test_case.assertTrue(np.allclose(of_out_2.numpy(), np_out, 1e-5, 1e-5))


def _test_adaptive_avgpool2d_backward(test_case, device):
    input = flow.Tensor(
        np.array(
            [
                [
                    [
                        [
                            0.10039155930280685,
                            0.04879157617688179,
                            -1.0515470504760742,
                            0.9466001987457275,
                        ],
                        [
                            0.45375481247901917,
                            0.23611211776733398,
                            1.343685269355774,
                            0.3979687988758087,
                        ],
                        [
                            0.05580734834074974,
                            -0.6875145435333252,
                            -1.654430866241455,
                            -0.6225992441177368,
                        ],
                        [
                            0.10183599591255188,
                            0.05019790679216385,
                            -1.2537643909454346,
                            0.14907236397266388,
                        ],
                    ]
                ]
            ]
        ),
        dtype=flow.float32,
        device=flow.device(device),
        requires_grad=True,
    )
    m = flow.nn.AdaptiveAvgPool2d((2, 2))
    of_out = m(input)
    of_out = of_out.sum()
    of_out.backward()
    np_grad = np.array(
        [
            [
                [
                    [0.25, 0.25, 0.25, 0.25],
                    [0.25, 0.25, 0.25, 0.25],
                    [0.25, 0.25, 0.25, 0.25],
                    [0.25, 0.25, 0.25, 0.25],
                ]
            ]
        ]
    )
    test_case.assertTrue(np.allclose(input.grad.numpy(), np_grad, 1e-5, 1e-5))


def _test_adaptive_avgpool2d_hw_forward(test_case, device):
    input = flow.Tensor(
        np.array(
            [
                [
                    [
                        [0.28242185711860657, -0.7742040753364563, -0.5439430475234985],
                        [-0.1706847995519638, 0.0430854931473732, 0.34247592091560364],
                        [-1.036131501197815, -1.033642292022705, 0.3455536365509033],
                    ]
                ]
            ]
        ),
        dtype=flow.float32,
        device=flow.device(device),
    )
    m = flow.nn.AdaptiveAvgPool2d((1, 2))
    m.to(device)
    of_out = m(input)
    np_out = np.array([[[[-0.4481925666332245, -0.27011242508888245]]]])
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-5, 1e-5))


def _test_adaptive_avgpool2d_hw_backward(test_case, device):
    input = flow.Tensor(
        np.array(
            [
                [
                    [
                        [0.28242185711860657, -0.7742040753364563, -0.5439430475234985],
                        [-0.1706847995519638, 0.0430854931473732, 0.34247592091560364],
                        [-1.036131501197815, -1.033642292022705, 0.3455536365509033],
                    ]
                ]
            ]
        ),
        dtype=flow.float32,
        device=flow.device(device),
        requires_grad=True,
    )
    m = flow.nn.AdaptiveAvgPool2d((1, 2))
    of_out = m(input)
    of_out = of_out.sum()
    of_out.backward()
    np_grad = np.array(
        [
            [
                [
                    [0.1666666716337204, 0.3333333432674408, 0.1666666716337204],
                    [0.1666666716337204, 0.3333333432674408, 0.1666666716337204],
                    [0.1666666716337204, 0.3333333432674408, 0.1666666716337204],
                ]
            ]
        ]
    )
    test_case.assertTrue(np.allclose(input.grad.numpy(), np_grad, 1e-5, 1e-5))


def _test_adaptive_avgpool3d_forward(test_case, device):
    input = flow.Tensor(
        np.array(
            [
                [
                    [
                        [
                            [
                                -1.07757179960088489,
                                -0.78045388903658375,
                                -1.26275387521194427,
                                0.99935071451204771,
                            ],
                            [
                                2.02225324891575164,
                                1.10345137769946500,
                                -0.43773247548795780,
                                1.89049181058751703,
                            ],
                            [
                                -0.55938618990646538,
                                -0.49495202415265188,
                                -0.18536721363519787,
                                -0.60989698667757719,
                            ],
                            [
                                -1.65362152601718160,
                                -1.03925835404367861,
                                0.36867765976139671,
                                -0.53568828349518050,
                            ],
                        ],
                        [
                            [
                                -1.26179006644499525,
                                -1.43909210916315322,
                                0.20654399652431357,
                                0.81864721019067133,
                            ],
                            [
                                -0.30333788634000142,
                                -0.81732697640762930,
                                -0.37675150976256139,
                                -0.11021655039337777,
                            ],
                            [
                                -0.22977043608192885,
                                1.27171963666499055,
                                -0.47908512978782908,
                                -1.44953694047278558,
                            ],
                            [
                                -1.28020932869777826,
                                -0.11184514806663474,
                                1.70221670872109843,
                                -1.73548372877253554,
                            ],
                        ],
                        [
                            [
                                2.47064979917736061,
                                -0.65497026319732976,
                                -0.93181070795716758,
                                1.46529042716824276,
                            ],
                            [
                                1.14198642343413970,
                                1.38990908108600797,
                                0.96578419005255678,
                                -0.85631142649766190,
                            ],
                            [
                                0.19515087084250754,
                                -0.37808457398571094,
                                0.29386253984961830,
                                0.92799305103533269,
                            ],
                            [
                                -0.93741182779940069,
                                0.33418317304524309,
                                -0.27925427653038332,
                                0.38029090707066726,
                            ],
                        ],
                        [
                            [
                                0.59186866597360410,
                                -0.78706310899389020,
                                -0.95343448742453918,
                                0.31341612954718795,
                            ],
                            [
                                0.75090294441452277,
                                -0.92992883985623231,
                                -0.73430540527824761,
                                -0.88064815906966942,
                            ],
                            [
                                -0.47078530163539850,
                                0.12253641652645629,
                                0.50880220398328457,
                                0.52039178932756203,
                            ],
                            [
                                -0.08613006511636320,
                                0.30291348404866386,
                                -0.62685658736801231,
                                -0.27469204305759976,
                            ],
                        ],
                    ]
                ]
            ]
        ),
        dtype=flow.float32,
        device=flow.device(device),
    )
    m = flow.nn.AdaptiveAvgPool3d((2, 2, 2))
    m.to(device)
    of_out_1 = m(input)
    of_out_2 = flow.adaptive_avg_pool3d(input, (2, 2, 2))
    np_out = np.array(
        [
            [
                [
                    [
                        [-0.31923351254725391, 0.21594741511983859],
                        [-0.51216542128766618, -0.36552048929482639],
                    ],
                    [
                        [0.49666933775477279, -0.20150242993241230],
                        [-0.11470347800925032, 0.18131719803880864],
                    ],
                ]
            ]
        ]
    )
    test_case.assertTrue(np.allclose(of_out_1.numpy(), np_out, 1e-5, 1e-5))
    test_case.assertTrue(np.allclose(of_out_2.numpy(), np_out, 1e-5, 1e-5))


def _test_adaptive_avgpool3d_backward(test_case, device):
    input = flow.Tensor(
        np.array(
            [
                [
                    [
                        [
                            [
                                -1.07757179960088489,
                                -0.78045388903658375,
                                -1.26275387521194427,
                                0.99935071451204771,
                            ],
                            [
                                2.02225324891575164,
                                1.10345137769946500,
                                -0.43773247548795780,
                                1.89049181058751703,
                            ],
                            [
                                -0.55938618990646538,
                                -0.49495202415265188,
                                -0.18536721363519787,
                                -0.60989698667757719,
                            ],
                            [
                                -1.65362152601718160,
                                -1.03925835404367861,
                                0.36867765976139671,
                                -0.53568828349518050,
                            ],
                        ],
                        [
                            [
                                -1.26179006644499525,
                                -1.43909210916315322,
                                0.20654399652431357,
                                0.81864721019067133,
                            ],
                            [
                                -0.30333788634000142,
                                -0.81732697640762930,
                                -0.37675150976256139,
                                -0.11021655039337777,
                            ],
                            [
                                -0.22977043608192885,
                                1.27171963666499055,
                                -0.47908512978782908,
                                -1.44953694047278558,
                            ],
                            [
                                -1.28020932869777826,
                                -0.11184514806663474,
                                1.70221670872109843,
                                -1.73548372877253554,
                            ],
                        ],
                        [
                            [
                                2.47064979917736061,
                                -0.65497026319732976,
                                -0.93181070795716758,
                                1.46529042716824276,
                            ],
                            [
                                1.14198642343413970,
                                1.38990908108600797,
                                0.96578419005255678,
                                -0.85631142649766190,
                            ],
                            [
                                0.19515087084250754,
                                -0.37808457398571094,
                                0.29386253984961830,
                                0.92799305103533269,
                            ],
                            [
                                -0.93741182779940069,
                                0.33418317304524309,
                                -0.27925427653038332,
                                0.38029090707066726,
                            ],
                        ],
                        [
                            [
                                0.59186866597360410,
                                -0.78706310899389020,
                                -0.95343448742453918,
                                0.31341612954718795,
                            ],
                            [
                                0.75090294441452277,
                                -0.92992883985623231,
                                -0.73430540527824761,
                                -0.88064815906966942,
                            ],
                            [
                                -0.47078530163539850,
                                0.12253641652645629,
                                0.50880220398328457,
                                0.52039178932756203,
                            ],
                            [
                                -0.08613006511636320,
                                0.30291348404866386,
                                -0.62685658736801231,
                                -0.27469204305759976,
                            ],
                        ],
                    ]
                ]
            ]
        ),
        dtype=flow.float32,
        device=flow.device(device),
        requires_grad=True,
    )
    m = flow.nn.AdaptiveAvgPool3d((2, 2, 2))
    of_out = m(input)
    of_out = of_out.sum()
    of_out.backward()
    np_grad = np.array(
        [
            [
                [
                    [
                        [0.125, 0.125, 0.125, 0.125],
                        [0.125, 0.125, 0.125, 0.125],
                        [0.125, 0.125, 0.125, 0.125],
                        [0.125, 0.125, 0.125, 0.125],
                    ],
                    [
                        [0.125, 0.125, 0.125, 0.125],
                        [0.125, 0.125, 0.125, 0.125],
                        [0.125, 0.125, 0.125, 0.125],
                        [0.125, 0.125, 0.125, 0.125],
                    ],
                    [
                        [0.125, 0.125, 0.125, 0.125],
                        [0.125, 0.125, 0.125, 0.125],
                        [0.125, 0.125, 0.125, 0.125],
                        [0.125, 0.125, 0.125, 0.125],
                    ],
                    [
                        [0.125, 0.125, 0.125, 0.125],
                        [0.125, 0.125, 0.125, 0.125],
                        [0.125, 0.125, 0.125, 0.125],
                        [0.125, 0.125, 0.125, 0.125],
                    ],
                ]
            ]
        ]
    )

    test_case.assertTrue(np.allclose(input.grad.numpy(), np_grad, 1e-5, 1e-5))


def _test_adaptive_avgpool3d_dhw_forward(test_case, device):
    input = flow.Tensor(
        np.array(
            [
                [
                    [
                        [
                            [
                                -1.07757179960088489,
                                -0.78045388903658375,
                                -1.26275387521194427,
                                0.99935071451204771,
                            ],
                            [
                                2.02225324891575164,
                                1.10345137769946500,
                                -0.43773247548795780,
                                1.89049181058751703,
                            ],
                            [
                                -0.55938618990646538,
                                -0.49495202415265188,
                                -0.18536721363519787,
                                -0.60989698667757719,
                            ],
                            [
                                -1.65362152601718160,
                                -1.03925835404367861,
                                0.36867765976139671,
                                -0.53568828349518050,
                            ],
                        ],
                        [
                            [
                                -1.26179006644499525,
                                -1.43909210916315322,
                                0.20654399652431357,
                                0.81864721019067133,
                            ],
                            [
                                -0.30333788634000142,
                                -0.81732697640762930,
                                -0.37675150976256139,
                                -0.11021655039337777,
                            ],
                            [
                                -0.22977043608192885,
                                1.27171963666499055,
                                -0.47908512978782908,
                                -1.44953694047278558,
                            ],
                            [
                                -1.28020932869777826,
                                -0.11184514806663474,
                                1.70221670872109843,
                                -1.73548372877253554,
                            ],
                        ],
                        [
                            [
                                2.47064979917736061,
                                -0.65497026319732976,
                                -0.93181070795716758,
                                1.46529042716824276,
                            ],
                            [
                                1.14198642343413970,
                                1.38990908108600797,
                                0.96578419005255678,
                                -0.85631142649766190,
                            ],
                            [
                                0.19515087084250754,
                                -0.37808457398571094,
                                0.29386253984961830,
                                0.92799305103533269,
                            ],
                            [
                                -0.93741182779940069,
                                0.33418317304524309,
                                -0.27925427653038332,
                                0.38029090707066726,
                            ],
                        ],
                        [
                            [
                                0.59186866597360410,
                                -0.78706310899389020,
                                -0.95343448742453918,
                                0.31341612954718795,
                            ],
                            [
                                0.75090294441452277,
                                -0.92992883985623231,
                                -0.73430540527824761,
                                -0.88064815906966942,
                            ],
                            [
                                -0.47078530163539850,
                                0.12253641652645629,
                                0.50880220398328457,
                                0.52039178932756203,
                            ],
                            [
                                -0.08613006511636320,
                                0.30291348404866386,
                                -0.62685658736801231,
                                -0.27469204305759976,
                            ],
                        ],
                    ]
                ]
            ]
        ),
        dtype=flow.float32,
        device=flow.device(device),
    )
    m = flow.nn.AdaptiveAvgPool3d((1, 2, 3))
    m.to(device)
    of_out = m(input)
    np_out = np.array(
        [
            [
                [
                    [0.08871791260375947, -0.40249593765093078, 0.00722249259371315],
                    [-0.31343444964845824, 0.08188803218941582, -0.09210164562800888],
                ]
            ]
        ]
    )
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-5, 1e-5))


def _test_adaptive_avgpool3d_dhw_backward(test_case, device):
    input = flow.Tensor(
        np.array(
            [
                [
                    [
                        [
                            [
                                -1.07757179960088489,
                                -0.78045388903658375,
                                -1.26275387521194427,
                                0.99935071451204771,
                            ],
                            [
                                2.02225324891575164,
                                1.10345137769946500,
                                -0.43773247548795780,
                                1.89049181058751703,
                            ],
                            [
                                -0.55938618990646538,
                                -0.49495202415265188,
                                -0.18536721363519787,
                                -0.60989698667757719,
                            ],
                            [
                                -1.65362152601718160,
                                -1.03925835404367861,
                                0.36867765976139671,
                                -0.53568828349518050,
                            ],
                        ],
                        [
                            [
                                -1.26179006644499525,
                                -1.43909210916315322,
                                0.20654399652431357,
                                0.81864721019067133,
                            ],
                            [
                                -0.30333788634000142,
                                -0.81732697640762930,
                                -0.37675150976256139,
                                -0.11021655039337777,
                            ],
                            [
                                -0.22977043608192885,
                                1.27171963666499055,
                                -0.47908512978782908,
                                -1.44953694047278558,
                            ],
                            [
                                -1.28020932869777826,
                                -0.11184514806663474,
                                1.70221670872109843,
                                -1.73548372877253554,
                            ],
                        ],
                        [
                            [
                                2.47064979917736061,
                                -0.65497026319732976,
                                -0.93181070795716758,
                                1.46529042716824276,
                            ],
                            [
                                1.14198642343413970,
                                1.38990908108600797,
                                0.96578419005255678,
                                -0.85631142649766190,
                            ],
                            [
                                0.19515087084250754,
                                -0.37808457398571094,
                                0.29386253984961830,
                                0.92799305103533269,
                            ],
                            [
                                -0.93741182779940069,
                                0.33418317304524309,
                                -0.27925427653038332,
                                0.38029090707066726,
                            ],
                        ],
                        [
                            [
                                0.59186866597360410,
                                -0.78706310899389020,
                                -0.95343448742453918,
                                0.31341612954718795,
                            ],
                            [
                                0.75090294441452277,
                                -0.92992883985623231,
                                -0.73430540527824761,
                                -0.88064815906966942,
                            ],
                            [
                                -0.47078530163539850,
                                0.12253641652645629,
                                0.50880220398328457,
                                0.52039178932756203,
                            ],
                            [
                                -0.08613006511636320,
                                0.30291348404866386,
                                -0.62685658736801231,
                                -0.27469204305759976,
                            ],
                        ],
                    ]
                ]
            ]
        ),
        dtype=flow.float32,
        device=flow.device(device),
        requires_grad=True,
    )
    m = flow.nn.AdaptiveAvgPool3d((1, 2, 3))
    of_out = m(input)
    of_out = of_out.sum()
    of_out.backward()
    np_grad = np.array(
        [
            [
                [
                    [
                        [0.0625, 0.125, 0.125, 0.0625],
                        [0.0625, 0.125, 0.125, 0.0625],
                        [0.0625, 0.125, 0.125, 0.0625],
                        [0.0625, 0.125, 0.125, 0.0625],
                    ],
                    [
                        [0.0625, 0.125, 0.125, 0.0625],
                        [0.0625, 0.125, 0.125, 0.0625],
                        [0.0625, 0.125, 0.125, 0.0625],
                        [0.0625, 0.125, 0.125, 0.0625],
                    ],
                    [
                        [0.0625, 0.125, 0.125, 0.0625],
                        [0.0625, 0.125, 0.125, 0.0625],
                        [0.0625, 0.125, 0.125, 0.0625],
                        [0.0625, 0.125, 0.125, 0.0625],
                    ],
                    [
                        [0.0625, 0.125, 0.125, 0.0625],
                        [0.0625, 0.125, 0.125, 0.0625],
                        [0.0625, 0.125, 0.125, 0.0625],
                        [0.0625, 0.125, 0.125, 0.0625],
                    ],
                ]
            ]
        ]
    )
    test_case.assertTrue(np.allclose(input.grad.numpy(), np_grad, 1e-5, 1e-5))


@flow.unittest.skip_unless_1n1d()
class TestAdaptiveAvgPool(flow.unittest.TestCase):
    def test_adaptive_avgpool1d(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            _test_adaptive_avgpool1d_forward,
            _test_adaptive_avgpool1d_backward,
        ]
        arg_dict["device"] = [
            "cpu",
            "cuda",
        ]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])

    def test_adaptive_avgpool2d(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            _test_adaptive_avgpool2d_forward,
            _test_adaptive_avgpool2d_backward,
            _test_adaptive_avgpool2d_hw_forward,
            _test_adaptive_avgpool2d_hw_backward,
        ]
        arg_dict["device"] = [
            "cpu",
            "cuda",
        ]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])

    def test_adaptive_avgpool3d(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            _test_adaptive_avgpool3d_forward,
            _test_adaptive_avgpool3d_backward,
            _test_adaptive_avgpool3d_dhw_forward,
            _test_adaptive_avgpool3d_dhw_backward,
        ]
        arg_dict["device"] = [
            "cpu",
            "cuda",
        ]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()
