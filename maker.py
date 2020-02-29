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

import os
import torch
import torch.nn as nn
import torch.utils.data as da
import torch.optim as op
from PIL import Image
import model as mod
import torchvision as tv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gsp


if not os.path.exists('./img_gan'):
    os.mkdir('./img_gan')
    os.mkdir('./img_gan/real')
    os.mkdir('./img_gan/fake')

if not os.path.exists('./model'):
    os.mkdir('./model')

if not os.path.exists('./mnist_fake'):
    os.mkdir('./mnist_fake')


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


# GANs discriminator
class Discriminator(nn.Module):
    def __init__(self, num_classes):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.LeakyReLU(0.2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.LeakyReLU(0.2)
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# GANs generator
class Generator(nn.Module):
    def __init__(self, input_size, num_feature):
        super(Generator, self).__init__()

        self.fc = nn.Linear(input_size, num_feature)
        self.br = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.ReLU()
        )
        self.downsample1 = nn.Sequential(
            nn.Conv2d(1, 50, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(50),
            nn.ReLU()
        )
        self.downsample2 = nn.Sequential(
            nn.Conv2d(50, 25, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(25),
            nn.ReLU()
        )
        self.downsample3 = nn.Sequential(
            nn.Conv2d(25, 1, kernel_size=2, stride=2),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 1, 56, 56)
        x = self.br(x)
        x = self.downsample1(x)
        x = self.downsample2(x)
        x = self.downsample3(x)
        return x


def to_img(x):
    out = 0.5 * (x + 1)
    out = out.clamp(0, 1)
    out = out.view(-1, 1, 28, 28)
    return out


def train():
    """
    If there is no Discriminator or Generator, should train D_model and G_model at first
    """
    # set Hyper-parameters for GAN
    BATCH_SIZE = 20
    NUM_EPOCHS = 10
    Z_DIMENSION = 100
    NUM_CLASSES = 10

    # load dataset
    train_imgs, train_labs = mod.load_mnist('./mnist/')
    train_dataset = CustomDataset(train_imgs, train_labs,
                                  train=True,
                                  transform=tv.transforms.ToTensor(),
                                  target_transform=None)
    train_loader = da.DataLoader(dataset=train_dataset,
                                 batch_size=BATCH_SIZE,
                                 shuffle=True)

    # set Module
    D = Discriminator(NUM_CLASSES)
    G = Generator(Z_DIMENSION + NUM_CLASSES, 1 * 56 * 56)
    criterion = nn.BCELoss()
    D_optimizer = op.Adam(D.parameters(), lr=2e-4)
    G_optimizer = op.Adam(G.parameters(), lr=2e-4)

    # training
    total_step = len(train_loader)
    for epoch in range(NUM_EPOCHS):
        for i, (imgs, labs) in enumerate(train_loader):
            num_img = imgs.size(0)
            labs_onehot = np.zeros((num_img, NUM_CLASSES))
            labs_onehot[np.arange(num_img), labs.numpy()] = 1

            # train discriminator
            real_img = imgs
            real_lab = torch.from_numpy(labs_onehot).float()
            fake_lab = torch.zeros(num_img, NUM_CLASSES)

            real_out = D(real_img)
            D_loss_real = criterion(real_out, real_lab)
            real_scores = real_out

            Z = torch.randn(num_img, Z_DIMENSION + NUM_CLASSES)
            fake_img = G(Z)
            fake_out = D(fake_img)
            D_loss_fake = criterion(fake_out, fake_lab)
            fake_scores = fake_out

            D_loss = D_loss_real + D_loss_fake
            D_optimizer.zero_grad()
            D_loss.backward()
            D_optimizer.step()

            # train generator
            Z = torch.randn(num_img, Z_DIMENSION)
            Z = np.concatenate((Z.numpy(), labs_onehot), axis=1)
            Z = torch.from_numpy(Z).float()
            fake_img = G(Z)
            output = D(fake_img)
            G_loss = criterion(output, real_lab)

            G_optimizer.zero_grad()
            G_loss.backward()
            G_optimizer.step()

            """假的数据跟标签对应不上，而且假数据一致性出奇的高都TM是3.。。。"""
            # GAN jepg to mnist_fake on CPU
            if (i + 1) % 1000 == 0:
                real_images = to_img(real_img.data)
                tv.utils.save_image(real_images, './img_gan/real/real_imgs_E{}_{}.png'.format(epoch + 1, i + 1))
                fake_images = to_img(fake_img.data)
                tv.utils.save_image(fake_images, './img_gan/fake/fake_imgs_E{}_{}.png'.format(epoch + 1, i + 1))

            if (i + 1) % 100 == 0:
                print("Epoch: [{}/{}], Step: [{}/{}], D_Loss: {:.6f}, G_Loss: {:.6f}"
                      .format(epoch + 1, NUM_EPOCHS, i + 1, total_step, D_loss.item(), G_loss.item()))

        torch.save(D.state_dict(), r'./model/cgan_discriminator_E{}.pth'.format(epoch + 1))
        torch.save(G.state_dict(), r'./model/cgan_generator_E{}.pth'.format(epoch + 1))


def show(images, NUM_IMGS):
    images = images.detach().numpy()
    images = 255 * (0.5 * images + 0.5)
    images = images.astype(np.uint8)
    plt.figure(figsize=(4, 4))
    width = images.shape[2]
    gs = gsp.GridSpec(1, NUM_IMGS, wspace=0, hspace=0)
    for i, img in enumerate(images):
        ax = plt.subplot(gs[i])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(img.reshape(width, width), cmap=plt.cm.gray)
        plt.axis('off')
        plt.tight_layout()
    plt.tight_layout()


def to_array(x, y):
    if type(x) == torch.Tensor:
        x = x.detach().numpy().reshape(-1, 28 * 28)
    else:
        x = x.reshape(-1, 28 * 28)

    if type(y) == torch.Tensor:
        y = y.detach().numpy().reshape(-1, 1)
    else:
        y = y.reshape(-1, 1)

    array = np.concatenate((y, x), axis=1)
    return array


def make():
    """
    Use trained Discriminator or Generator to make fake dataset
    """
    Z_DIMENSION = 100
    NUM_CLASSES = 10
    NUM_IMGS = 10
    D = Discriminator(NUM_CLASSES)
    G = Generator(Z_DIMENSION + NUM_CLASSES, 1 * 56 * 56)
    D.load_state_dict(torch.load(r'./model/cgan_discriminator_1E10.pth'))
    G.load_state_dict(torch.load(r'./model/cgan_generator_1E10.pth'))

    fake_dataset = np.zeros((1, 28 * 28 + 1))
    classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    for i in range(NUM_CLASSES):
        z = torch.randn((NUM_IMGS, 100))
        x = np.zeros((NUM_IMGS, NUM_CLASSES))
        x[:, i] = 1
        z = np.concatenate((z.numpy(), x), 1)
        z = torch.from_numpy(z).float()
        fake_img = G(z)
        output = D(fake_img)
        lab = np.dot(np.ones(NUM_IMGS), classes[i])
        fake_dataset = np.concatenate((fake_dataset, to_array(fake_img, lab)), axis=0)
        show(fake_img, NUM_IMGS)
        plt.savefig('./mnist_fake/fake_imgs_bymake_{}.png'.format(i+1), bbox_inches='tight')
        print("Classes: [{}/{}], Now_Class: {}".format(i + 1, NUM_CLASSES, classes[i]))

    np.savetxt('./mnist_fake/fake_mnist.csv', np.delete(fake_dataset, 0, 0), delimiter=',')


if __name__ == '__main__':
    make()
