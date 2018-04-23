import matplotlib as mpl
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

import inception_score
import utility
from mmd_score import linear_mmd2
from model import Generator, Discriminator

mpl.use('Agg')

z_dim = 100
batch_size = 128
cuda_available = torch.cuda.is_available()
if cuda_available:
    print("Cuda is available!")


def build_model(model_type):
    generator = Generator(model_name=model_type, batch_size=128)
    discriminator = Discriminator()
    if cuda_available:
        generator = generator.cuda()
        discriminator = discriminator.cuda()

    loss = nn.BCELoss()
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=2e-4)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=2e-4)

    return generator, discriminator, loss, optimizer_g, optimizer_d


def sample_noise(batch_size, z_dim):
    # Sample z ~ N(0, 1)            
    minibatch_noise = Variable(torch.randn((batch_size, z_dim)).view(-1, z_dim, 1, 1))
    if cuda_available:
        minibatch_noise = minibatch_noise.cuda()
    return minibatch_noise


eval_loader = iter(utility.trainloader(1000))


def eval_mmd(generator, z_dim):
    inputs, targets = next(eval_loader)
    if cuda_available:
        inputs, targets = inputs.cuda(), targets.cuda()

    inputs, targets = Variable(inputs), Variable(targets)

    minibatch_noise = sample_noise(1000, z_dim)
    fake = generator(minibatch_noise)
    return linear_mmd2(inputs, fake).data.cpu().numpy()[0]


def train(trainloader, generator, discriminator, loss, optimizer_g, optimizer_d):
    ctr = 0
    minibatch_disc_losses = []
    minibatch_gen_losses = []

    fixed_noise = Variable(torch.FloatTensor(8 * 8, z_dim, 1, 1).normal_(0, 1), volatile=True)

    if cuda_available:
        print("CUDA is available!")
        fixed_noise.cuda()

    print("Epoch, Inception Score, MMD Score", file=open("logs/eval.log", "a"))

    for epoch in range(50):
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            ctr += 1
            if cuda_available:
                inputs, targets = inputs.cuda(), targets.cuda()

            inputs, targets = Variable(inputs), Variable(targets)

            zeros = Variable(torch.zeros(batch_size))
            ones = Variable(torch.ones(batch_size))

            if cuda_available:
                zeros, ones = zeros.cuda(), ones.cuda()

            # print("Updating discriminator...")
            minibatch_noise = sample_noise(batch_size, z_dim)

            # Zero gradients for the discriminator
            optimizer_d.zero_grad()

            # Train with real examples
            d_real = discriminator(inputs)

            if discriminator.model_name == 'DCGAN':
                d_real_loss = loss(d_real, ones)  # Train discriminator to recognize real examples
            else:
                d_real_loss = 0.5 * torch.mean((d_real - ones) ** 2)

            # print("Applying gradients to discriminator...")
            d_real_loss.backward()

            # print("Train with fake examples from the generator")
            fake = generator(minibatch_noise).detach()  # Detach to prevent backpropping through the generator

            d_fake = discriminator(fake)

            d_fake_loss = loss(d_fake, zeros)  # Train discriminator to recognize generator samples
            d_fake_loss.backward()
            minibatch_disc_losses.append(d_real_loss.data[0] + d_fake_loss.data[0])

            # # the discriminator
            optimizer_d.step()

            # print("Updating the generator...")
            optimizer_g.zero_grad()

            # print("Sample z ~ N(0, 1)")
            minibatch_noise = sample_noise(batch_size, z_dim)

            d_fake = discriminator(generator(minibatch_noise))
            if generator.model_name == 'DCGAN':
                g_loss = loss(d_fake, ones)  # Train generator to fool the discriminator into thinking these are real.
            else:
                g_loss = 0.5 * torch.mean((d_fake - ones) ** 2)
            g_loss.backward()

            # print("Applying gradients to generator...")
            optimizer_g.step()

            minibatch_gen_losses.append(g_loss.data[0])
            if ctr % 10 == 0:
                print("Iteration {} of epoch {}".format(ctr, epoch))

        print('Generator loss : %.3f' % (np.mean(minibatch_gen_losses)))
        print('Discriminator loss : %.3f' % (np.mean(minibatch_disc_losses)))

        inc_score = inception_score.evaluate(generator, z_dim, cuda=cuda_available)
        mmd_score = eval_mmd(generator, z_dim)
        print('MMD score      : {}'.format(mmd_score))
        print('Inception score: {}'.format(inc_score))
        print("{}, {}, {}".format(epoch, inc_score, mmd_score), file=open("logs/eval.log", "a"))

        utility.plot_result(generator, fixed_noise, epoch)
        loss_name = "{0}_epoch{1}".format(generator.model_name, epoch)
        utility.save_losses(minibatch_disc_losses, minibatch_gen_losses, loss_name)
        utility.save(discriminator, generator, epoch)


def main():
    for model_type in ['DCGAN']:
        generator, discriminator, loss, optimizer_g, optimizer_d = build_model(model_type)
        trainloader = utility.trainloader(batch_size)
        print("Loaded training data")
        train(trainloader, generator, discriminator, loss, optimizer_g, optimizer_d)

        # inc_score = inception_score.calculate(utility.trainloader_helper(batch_size), cuda=cuda_available, batch_size=32, resize=True, splits=10)
        # print('Inception score: {}'.format(inc_score))


if __name__ == '__main__':
    main()
