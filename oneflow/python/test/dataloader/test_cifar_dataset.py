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
import os
import unittest

import oneflow as flow
import oneflow.experimental.nn as nn
import oneflow.python.utils.vision.transforms as transforms
import oneflow.experimental.optim as optim


classes = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(flow.F.relu(self.conv1(x)))
        x = self.pool(flow.F.relu(self.conv2(x)))
        x = flow.flatten(x, 1)  # flatten all dimensions except batch
        x = flow.F.relu(self.fc1(x))
        x = flow.F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def test(test_case):
    device = flow.device("cuda")
    net = Net()
    net.to(device)

    optimizer = optim.SGD(net.parameters(), lr=0.002, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    criterion.to(device)

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    train_epoch = 1
    batch_size = 4
    data_dir = os.getenv("ONEFLOW_TEST_CACHE_DIR") + "/data-test/cifar10"

    trainset = flow.utils.vision.datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=transform,
        source_url="https://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/cifar/cifar-10-python.tar.gz",
    )
    trainloader = flow.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    final_loss = 0
    for epoch in range(1, train_epoch + 1):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 1):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(dtype=flow.float32, device=device)
            labels = labels.to(dtype=flow.int64, device=device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.numpy()
            if i % 2000 == 0:  # print every 2000 mini-batches
                final_loss = running_loss / 2000
                print("epoch: %d  step: %5d  loss: %.3f " % (epoch, i, final_loss))
                running_loss = 0.0

    print("final loss : ", final_loss)
    test_case.assertLess(final_loss, 1.50)


@flow.unittest.skip_unless_1n1d()
class TestCifarDataset(flow.unittest.TestCase):
    def test_cifar_dataset(test_case):
        test(test_case)


if __name__ == "__main__":
    unittest.main()
    # 1 epoch training log
    # epoch: 1  step:  2000  loss: 2.107
    # epoch: 1  step:  4000  loss: 1.838
    # epoch: 1  step:  6000  loss: 1.644
    # epoch: 1  step:  8000  loss: 1.535
    # epoch: 1  step: 10000  loss: 1.528
    # epoch: 1  step: 12000  loss: 1.476