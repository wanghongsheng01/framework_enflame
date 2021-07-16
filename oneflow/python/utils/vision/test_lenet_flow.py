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
import time

import oneflow.experimental as flow
import oneflow.experimental.nn as nn

import oneflow.python.utils.data as data
import oneflow.python.utils.vision.datasets as datasets
import oneflow.python.utils.vision.transforms as transforms
import oneflow.experimental.optim as optim


# reference: http://tangshusen.me/Dive-into-DL-PyTorch/#/chapter05_CNN/5.5_lenet
flow.enable_eager_execution()

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5),  # in_channels, out_channels, kernel_size
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # kernel_size, stride
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc = nn.Sequential(
            nn.Linear(16 * 4 * 4, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10),
        )

    def forward(self, img):
        feature = self.conv(img)
        feature = feature.reshape(shape=[img.shape[0], -1])
        output = self.fc(feature)
        return output


device = flow.device("cuda")
net = LeNet()
net.to(device)


def load_data_fashion_mnist(batch_size, resize=None, root="./data-test/fashion-mnist"):
    """Download the Fashion-MNIST dataset and then load into memory."""
    root = os.path.expanduser(root)
    trans = []
    if resize:
        trans.append(transforms.Resize(resize))
    trans.append(transforms.ToTensor())
    transform = transforms.Compose(trans)

    mnist_train = datasets.FashionMNIST(
        root=root, train=True, transform=transform, download=True
    )
    mnist_test = datasets.FashionMNIST(
        root=root, train=False, transform=transform, download=True
    )
    num_workers = 0

    train_iter = data.DataLoader(
        mnist_train, batch_size, shuffle=True, num_workers=num_workers
    )
    test_iter = data.DataLoader(
        mnist_test, batch_size, shuffle=False, num_workers=num_workers
    )
    return train_iter, test_iter


batch_size = 256
train_iter, test_iter = load_data_fashion_mnist(batch_size=batch_size, resize=None)
loss = nn.CrossEntropyLoss()
loss.to(device)

def evaluate_accuracy(data_iter, net, device=None):
    if device is None and isinstance(net, nn.Module):
        device = list(net.parameters())[0].device
    acc_sum, n = 0.0, 0
    with flow.no_grad():
        for X, y in data_iter:
            X = flow.tensor(X.numpy())
            y = flow.tensor(y.numpy())
            X = X.to(device=device)
            y = y.to(device=device)
            if isinstance(net, nn.Module):
                net.eval()  #  evaluating mode
                acc_sum += (
                    (net(X).argmax(dim=1).numpy() == y.numpy()).sum()
                )
                net.train()  # turn to training mode
            else:
                if "is_training" in net.__code__.co_varnames:
                    # set is_training = False
                    acc_sum += (
                        (net(X, is_training=False).argmax(dim=1).numpy() == y.numpy())
                        .float()
                        .sum()
                    )
                else:
                    acc_sum += (net(X).argmax(dim=1).numpy() == y.numpy()).float().sum()
            n += y.shape[0]
    return acc_sum / n


def train(net, train_iter, test_iter, batch_size, optimizer, loss, device, num_epochs):
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
        for X, y in train_iter:
            X = X.to(device=device)
            y = y.to(device=device)
            X.requires_grad = True
            # forward
            y_hat = net(X)
            l = loss(y_hat, y).sum()
            # backward
            l.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_l_sum += l.numpy()
            train_acc_sum += (y_hat.argmax(dim=1).numpy() == y.numpy()).sum()
            n += y.shape[0]
            batch_count += 1

        test_acc = evaluate_accuracy(test_iter, net)
        print(
            "epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec"
            % (
                epoch + 1,
                train_l_sum / batch_count,
                train_acc_sum / n,
                test_acc,
                time.time() - start,
            )
        )


lr, num_epochs = 0.01, 10
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
train(net, train_iter, test_iter, batch_size, optimizer, loss, device, num_epochs)