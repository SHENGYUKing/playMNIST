# -*- coding: utf-8 -*-

# Copyright 2020. SHENGYUKing.
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS
# OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

# The MNIST database of handwritten digits, available from http://yann.lecun.com/exdb/mnist/, has a training set of
# 60,000 examples, and a test set of 10,000 examples. It is a subset of a larger set available from NIST. The digits
# have been size-normalized and centered in a fixed-size image.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as da
import torch.optim as op
import torchvision as tv
from PIL import Image
import model as mod


# # set GPU
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# inherit original data.Dataset
class CustomDataset(da.Dataset):
    """CustomDataset
    Custom-defined dataset can be used more flexibly.
    Args:
        data (numpy.array): the data in the dataset.
        target (numpy.array): the label in the dataset.
        train (bool, optional): If True, creat dataset as training set,
            otherwise as test set.
        transform (callable, optional): A function/transform that takes in
            an PIL image and returns a transformed version.
            E.g, ``transforms.RandomCrop``
        target_transform (callback, optional): A function/transform that
            takes in the target and transforms it.
    """

    def __init__(self, data, target, train=True, transform=None, target_transform=None):
        """
        Required !!!
            TODO
                1. Initialize file path or list of file names.
                2. Something others custom want to do at first.
        """
        super(CustomDataset, self).__init__()
        self.data = data
        self.target = target
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' + '\n'
                               + 'Please make your dataset at first.')
        if self.train:
            self.train_data = self.data
            self.train_labels = self.target
        else:
            self.test_data = self.data
            self.test_labels = self.target

    def __getitem__(self, index):
        """
        Required !!!
            TODO
                1. Read one data (not a serious of data) from file.
                2. Preprocess the data.
                3. Return a data pair (data and label).
        """
        if self.train:
            img_in, target_in = self.train_data[index], self.train_labels[index]
        else:
            img_in, target_in = self.test_data[index], self.test_labels[index]

        img = Image.fromarray(img_in.reshape(28, 28).astype('uint8'), mode='L')
        target = target_in

        if self.transform is not None:
            img_out = self.transform(img)
        else:
            img_out = img

        if self.target_transform is not None:
            target_out = self.target_transform(target)
        else:
            target_out = target
        return img_out, target_out

    def __len__(self):
        """
        Required !!!
            TODO
                1. Return the total scale of the dataset
        """
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def _check_exists(self):
        return (self.data.all() and self.target.all()) is not None

    def targets(self):
        if self.train:
            return self.train_labels
        else:
            return self.test_labels


class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()

        # conv_out = (conv_input - kernel_size) / stride + 1
        # Input: 1 * (28 + 2 * 2) * (28 + 2 * 2)
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=(5, 5), stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        )
        # Output: 6 * 28 * 28 (before pooling), 6 * 14 * 14 (after pooling)

        # Input: 6 * 14 * 14
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=(5, 5)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        )
        # Output: 16 * 10 * 10 (before pooling), 16 * 5 * 5 (after pooling)

        # Input: 16 * 5 * 5
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 120, kernel_size=(5, 5)),
            nn.ReLU()
        )
        # Output: 120 * 1 * 1

        # Input: 120 * 1
        self.fc1 = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU()
        )
        # Output: 84 * 1 (96 printable ASCII chars can be exactly distinguished by 7*12 bitmap)

        # Input: 84 * 1
        self.fc2 = nn.Sequential(
            nn.Linear(84, 10),
            nn.LogSoftmax()
        )
        # Output: 10 * 1

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # x should reshape to 1-D for full-connection layer nn.Linear()
        x = x.view(-1, 120)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()

        """
            Alex.Conv1(3, 96, k=11, s=4, p=0)
            Alex.MaxPool1(k=3, s=2)
        """
        # Input: 1 * (28+2*1) * (28+2*1)
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), stride=1, padding=1),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.ReLU()
        )
        # Output: 32 * 28 * 28 (before pooling), 32 * 14 * 14 (after pooling)

        """
            Alex.Conv2(96, 256, k=5,s=1,p=2)
            Alex.MaxPool2(k=3, s=2)
        """
        # Input: 32 * 14 * 14
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=1, padding=1),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.ReLU()
        )
        # OutPut: 64 * 14 * 14 (before pooling), 64 * 7 * 7 (after pooling)

        """
            Alex.Conv3(256, 384, k=3, s=1, p=1)
            Alex.Conv4(384, 384, k=3, s=1, p=1)
            Alex.Conv5(256, 256, k=3, s=1, p=1)
            Alex.MaxPool3(k=3, s=2)
        """
        # Input: 64 * 7 * 7
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=1, padding=1)
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=1, padding=1),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.ReLU()
        )
        # Output: 256 * 3 * 3

        """
            Alex.FC1(256 * 6 * 6, 4096)
            Alex.FC2(4096, 4096)
            Alex.FC3(4096, 1000)
        """
        # Input: 256 * 3 * 3
        self.fc1 = nn.Sequential(
            nn.Linear(256 * 3 * 3, 1024),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU()
        )
        self.fc3 = nn.Linear(512, 10)
        # Output: 10 * 1

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        # Alex.x = x.view(-1, 256 * 6 * 6)
        x = x.view(-1, 256 * 3 * 3)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


# use for ResNet18/34, conv by 3x3 3x3 kernel
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, out_planes, stride=1):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_planes)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * out_planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_planes),
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


# use for ResNet50/101/152, conv by 1x1 3x3 1x1 kernel
class BarrBlock(nn.Module):
    expansion = 4

    def __init__(self, in_planes, out_planes, stride=1):
        super(BarrBlock, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_planes, out_planes, kernel_size=3, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(out_planes, self.expansion * out_planes, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.expansion * out_planes)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * out_planes, kernel_size=1, bias=False),
                nn.BatchNorm2d(self.expansion * out_planes)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[0], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[0], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[0], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks -1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, out_planes, stride))
            self.in_planes = out_planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50():
    return ResNet(BarrBlock, [3, 4, 6, 3])


def ResNet101():
    return ResNet(BarrBlock, [3, 4, 23, 3])


def ResNet152():
    return ResNet(BarrBlock, [3, 8, 36, 3])


# set Hyper-parameters
INPUT_SIZE = 784
NUM_CLASSES = 10
NUM_EPOCHS = 5
BATCH_SIZE = 20
LEARNING_RATE = 0.001

# load dataset
# custom dataset
train_imgs, train_labs = mod.load_mnist('./mnist/')
train_dataset = CustomDataset(train_imgs, train_labs,
                              train=True,
                              transform=tv.transforms.ToTensor(),
                              target_transform=None)
test_imgs, test_labs = mod.load_mnist('./mnist/', 't10k')
test_dataset = CustomDataset(test_imgs, test_labs,
                             train=False,
                             transform=tv.transforms.ToTensor(),
                             target_transform=None)

# data loader
train_loader = da.DataLoader(dataset=train_dataset,
                             batch_size=BATCH_SIZE,
                             shuffle=True)
test_loader = da.DataLoader(dataset=test_dataset,
                            batch_size=BATCH_SIZE,
                            shuffle=False)

# set basic model
# model = nn.Linear(INPUT_SIZE, NUM_CLASSES)
# model = LeNet5()
# model = AlexNet()
model = ResNet18()

criterion = nn.CrossEntropyLoss()
optimizer = op.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)

# train the model
total_step = len(train_loader)
for epoch in range(NUM_EPOCHS):
    for i, (imgs, labs) in enumerate(train_loader):
        # reshape images to (batch_size, input_size) (use with LR model)
        # imgs = imgs.reshape(-1, 28 * 28)
        labs = labs.long()
        optimizer.zero_grad()

        # forward pass
        outputs = model(imgs)
        loss = criterion(outputs, labs)

        # backward and optimize
        loss.backward()
        optimizer.step()

        # record the process and compute the loss
        if (i + 1) % 100 == 0:
            print("Epoch: [{}/{}], Step: [{}/{}], Loss: {:.6f}"
                  .format(epoch + 1, NUM_EPOCHS, i + 1, total_step, loss.item()))


# test the model
# for memory efficiency without computing gradients
with torch.no_grad():
    correct = 0
    total = 0
    for imgs, labs in test_loader:
        # imgs = imgs.reshape(-1, 28 * 28)
        labs = labs.long()

        outputs = model(imgs)
        _, predicted = torch.max(outputs.data, 1)
        total += labs.size(0)
        correct += (predicted == labs).sum()

    print("Accuracy of the model on the test set (with 10,000 images): {:.4f}%"
          .format(100 * correct / total))

# save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')
