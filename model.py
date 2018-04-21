import torch
import torch.nn as nn
import torch.nn.functional as F

cuda_available = torch.cuda.is_available()


class Generator(nn.Module):

    def __init__(self, z_dim=128, model_name='DCGAN'):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.model_name = model_name

        self.deconv1 = nn.ConvTranspose2d(
            in_channels=100, out_channels=z_dim * 8,
            kernel_size=(4, 4), stride=1, padding=0, bias=False
        )
        self.bn1 = nn.BatchNorm2d(z_dim * 8)

        self.deconv2 = nn.ConvTranspose2d(
            in_channels=z_dim * 8, out_channels=z_dim * 4,
            kernel_size=(4, 4), stride=2, padding=1, bias=False,
        )
        self.bn2 = nn.BatchNorm2d(z_dim * 4)

        self.deconv3 = nn.ConvTranspose2d(
            in_channels=z_dim * 4, out_channels=z_dim * 2,
            kernel_size=(4, 4), stride=2, padding=1, bias=False,
        )
        self.bn3 = nn.BatchNorm2d(z_dim * 2)

        self.deconv4 = nn.ConvTranspose2d(
            in_channels=z_dim * 2, out_channels=z_dim,
            kernel_size=(4, 4), stride=2, padding=1, bias=False,
        )

        self.bn4 = nn.BatchNorm2d(z_dim)

        self.deconv5 = nn.ConvTranspose2d(
            in_channels=z_dim, out_channels=3,
            kernel_size=(4, 4), stride=2, padding=1, bias=False,
        )

        # Initialise weights to N(0, 0.02)
        for m in self._modules:
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x = F.relu(self.bn1(self.deconv1(x)))
        x = F.relu(self.bn2(self.deconv2(x)))
        x = F.relu(self.bn3(self.deconv3(x)))
        x = F.relu(self.bn4(self.deconv4(x)))
        x = F.tanh(self.deconv5(x))
        return x


class Discriminator(nn.Module):

    def __init__(self, z_dim=128, model_name='DCGAN'):
        super(Discriminator, self).__init__()
        self.z_dim = z_dim
        self.model_name = model_name

        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=z_dim,
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
            in_channels=z_dim * 4, out_channels=z_dim * 8,
            kernel_size=(4, 4), stride=2, padding=1, bias=False,
        )

        self.bn4 = nn.BatchNorm2d(z_dim * 8)

        self.conv5 = nn.Conv2d(
            in_channels=z_dim * 8, out_channels=1,
            kernel_size=(4, 4), stride=1, padding=0, bias=False,
        )


        self.output_layer = nn.Sequential()
        self.output_layer.add_module('out', self.conv5)
        self.output_layer.add_module('act', nn.Sigmoid())

        # Initialise weights to N(0, 0.02)
        for m in self._modules:
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.2)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.bn4(self.conv4(x)), 0.2)
	
        if self.model_name == 'LSGAN':
            x = self.conv5(x).squeeze()
        else:
            x = self.output_layer(x).squeeze()
        return x
        
