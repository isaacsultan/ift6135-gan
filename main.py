import matplotlib as mpl
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

import inception_score
import utility
from model import Generator, Discriminator

mpl.use('Agg')

z_dim = 128
cuda_available = torch.cuda.is_available()


def build_model(model_type):
    generator = Generator(model_name=model_type)
    discriminator = Discriminator()
    if cuda_available:
        generator = generator.cuda()
        discriminator = discriminator.cuda()

    loss = nn.BCELoss()
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=2e-4)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=2e-4)

    return generator, discriminator, loss, optimizer_g, optimizer_d


def train(trainloader, generator, discriminator, loss, optimizer_g, optimizer_d):
    ctr = 0
    minibatch_disc_losses = []
    minibatch_gen_losses = []

    fixed_noise = Variable(torch.FloatTensor(8 * 8, 128, 1, 1).normal_(0, 1), volatile=True)

    if cuda_available:
        fixed_noise.cuda()

    for epoch in range(50):
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            ctr += 1
            if cuda_available:
                inputs, targets = inputs.cuda(), targets.cuda()

            inputs, targets = Variable(inputs), Variable(targets)

            zeros = Variable(torch.zeros(inputs.size(0)))
            ones = Variable(torch.ones(inputs.size(0)))

            if cuda_available:
                zeros, ones = zeros.cuda(), ones.cuda()

            # UPDATE DISCRIMINATOR

            # Sample z ~ N(0, 1)
            minibatch_noise = Variable(torch.randn((128, 100)).view(-1, 100, 1, 1))

            if cuda_available:
                minibatch_noise = minibatch_noise.cuda()

            # Zero gradients for the discriminator
            optimizer_d.zero_grad()

            # Train with real examples
            d_real = discriminator(inputs)

            if discriminator.model_name == 'DCGAN':
                d_real_loss = loss(d_real, ones)  # Train discriminator to recognize real examples
            else:
                d_real_loss = 0.5 * torch.mean((d_real - ones) ** 2)
            d_real_loss.backward()

            # Train with fake examples from the generator
            fake = generator(minibatch_noise).detach()  # Detach to prevent backpropping through the generator
            d_fake = discriminator(fake)

            d_fake_loss = loss(d_fake, zeros)  # Train discriminator to recognize generator samples
            d_fake_loss.backward()
            minibatch_disc_losses.append(d_real_loss.data[0] + d_fake_loss.data[0])

            # # the discriminator
            optimizer_d.step()

            ### UPDATE GENERATOR

            # Zero gradients for the generator
            optimizer_g.zero_grad()

            # Sample z ~ N(0, 1)
            minibatch_noise = Variable(torch.randn((128, 100)).view(-1, 100, 1, 1))

            if cuda_available:
                minibatch_noise = minibatch_noise.cuda()

            d_fake = discriminator(generator(minibatch_noise))
            if generator.model_name == 'DCGAN':
                g_loss = loss(d_fake, ones)  # Train generator to fool the discriminator into thinking these are real.
            else:
                g_loss = 0.5 * torch.mean((d_fake - ones) ** 2)
            g_loss.backward()

            # Update the generator
            optimizer_g.step()

            minibatch_gen_losses.append(g_loss.data[0])

        print('Generator loss : %.3f' % (np.mean(minibatch_gen_losses)))
        print('Discriminator loss : %.3f' % (np.mean(minibatch_disc_losses)))

        utility.plot_result(generator, fixed_noise, 64, epoch, 'logs')

    utility.save_losses(minibatch_disc_losses, minibatch_gen_losses, generator.model_name)
    utility.save(discriminator, generator)


def main():
    for model_type in ['DCGAN', 'LSGAN']:
        generator, discriminator, loss, optimizer_g, optimizer_d = build_model(model_type)
        trainloader = utility.trainloader()
        train(trainloader, generator, discriminator, loss, optimizer_g, optimizer_d)

        inc_score = inception_score.calculate(utility.trainloader_helper(), cuda=cuda_available, batch_size=32, resize=True, splits=10)
        print('Inception score: {}'.format(inc_score))


if __name__ == '__main__':
    main()
