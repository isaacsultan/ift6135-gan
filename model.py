import torch
import torch.nn as nn
import torch.nn.functional as F


cuda_available = torch.cuda.is_available()

class Generator(nn.Module):

    def __init__(self, z_dim=128):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.model_name = 'DCGAN'

        self.deconv1 = nn.ConvTranspose2d(
            in_channels=100, out_channels=z_dim*8,
            kernel_size=(4, 4), bias=False
        )
        self.bn1 = nn.BatchNorm2d(z_dim * 4)

        self.deconv2 = nn.ConvTranspose2d(
            in_channels=z_dim * 4, out_channels=z_dim * 2,
            kernel_size=(4, 4), stride=2, padding=1, bias=False,
        )
        self.bn2 = nn.BatchNorm2d(z_dim * 2)

        self.deconv3 = nn.ConvTranspose2d(
            in_channels=z_dim * 2, out_channels=z_dim,
            kernel_size=(4, 4), stride=2, padding=1, bias=False,
        )
        self.bn3 = nn.BatchNorm2d(z_dim)

        self.deconv4 = nn.ConvTranspose2d(
            in_channels=z_dim, out_channels=1,
            kernel_size=(4, 4), stride=2, padding=3, bias=False,
        )

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, x):
        x = F.relu(self.bn1(self.deconv1(x)))
        x = F.relu(self.bn2(self.deconv2(x)))
        x = F.relu(self.bn3(self.deconv3(x)))
        return F.tanh(self.deconv4(x))


class Discriminator(nn.Module):

    def __init__(self, z_dim=128):
        super(Discriminator, self).__init__()
        self.z_dim = z_dim
        self.model_name = 'DCGAN'


        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=z_dim,
            kernel_size=(4, 4), stride=2, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(z_dim)

        self.conv2 = nn.Conv2d(
            in_channels=z_dim, out_channels=z_dim * 2,
            kernel_size=(4, 4), stride=2, padding=1, bias=False,
        )
        self.bn2 = nn.BatchNorm2d(z_dim * 2)

        self.conv3 = nn.Conv2d(
            in_channels=z_dim * 2, out_channels=z_dim * 4,
            kernel_size=(4, 4), stride=2, padding=1, bias=False,
        )
        self.bn3 = nn.BatchNorm2d(z_dim * 4)

        self.conv4 = nn.Conv2d(
            in_channels=z_dim * 4, out_channels=1,
            kernel_size=(4, 4), stride=2, padding=1, bias=False,
        )

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.2)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2)
        return F.sigmoid(self.conv4(x)).squeeze()

