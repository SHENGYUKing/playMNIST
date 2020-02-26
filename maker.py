# -*- coding: utf-8 -*-

import os
import torch
import torch.nn as nn
import torch.utils.data as da
import torch.optim as op
from PIL import Image
import model as mod
import torchvision as tv

if not os.path.exists('./img_gan'):
    os.mkdir('./img_gan')
    os.mkdir('./img_gan/real')
    os.mkdir('./img_gan/fake')


def to_img(x):
    out = 0.5 * (x + 1)
    out = out.clamp(0, 1)
    out = out.view(-1, 1, 28, 28)
    return out


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
    def __init__(self):
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
            nn.Linear(1024, 1),
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


# set Hyper-parameters for GAN
BATCH_SIZE = 20
NUM_EPOCHS = 5
Z_DIMENSION = 100

# Image processing
img_transform = tv.transforms.Compose([
    tv.transforms.ToTensor(),
    tv.transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

# load dataset
train_imgs, train_labs = mod.load_mnist('./mnist/')
train_dataset = CustomDataset(train_imgs, train_labs,
                              train=True,
                              transform=img_transform,
                              target_transform=None)
train_loader = da.DataLoader(dataset=train_dataset,
                             batch_size=BATCH_SIZE,
                             shuffle=True)

# set Module
D = Discriminator()
G = Generator(Z_DIMENSION, 4 * 28 * 28)
criterion = nn.BCELoss()
D_optimizer = op.Adam(D.parameters(), lr=1e-3)
G_optimizer = op.Adam(G.parameters(), lr=1e-3)

# training
total_step = len(train_loader)
for epoch in range(NUM_EPOCHS):
    for i, (imgs, _) in enumerate(train_loader):
        num_img = imgs.size(0)

        # train discriminator
        real_img = imgs
        real_lab = torch.ones(num_img)
        fake_lab = torch.zeros(num_img)

        real_out = D(real_img)
        D_loss_real = criterion(real_out, real_lab)
        real_scores = real_out

        Z = torch.randn(num_img, Z_DIMENSION)
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
        fake_img = G(Z)
        output = D(fake_img)
        G_loss = criterion(output, real_lab)

        G_optimizer.zero_grad()
        G_loss.backward()
        G_optimizer.step()

        if (i + 1) % 100 == 0:
            print("Epoch [{}/{}], Step: [{}/{}], D_Loss: {:.6f}, G_Loss: {:.6f}"
                  .format(epoch + 1, NUM_EPOCHS, i + 1, total_step, D_loss.item(), G_loss.item()))
    if epoch == 0:
        real_images = to_img(real_img.data)
        tv.utils.save_image(real_images, './img_gan/real/real_imgs.png')

    fake_images = to_img(fake_img.data)
    tv.utils.save_image(fake_images, './img_gan/fake/fake_imgs_{}.png'.format(epoch + 1))

torch.save(G.state_dict(), 'gan_generator.pth')
torch.save(D.state_dict(), 'gan_discriminator.pth')
