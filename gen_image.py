import os
import pickle as pk
import numpy as np
import torch
from model import Generator
import torch.nn as nn
from torch.autograd import Variable
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

cuda_available = torch.cuda.is_available()

def build_model(model_type):
    generator = Generator(model_name=model_type, z_dim=128)
    # import ipdb; ipdb.set_trace()
    generator.load_state_dict(torch.load('DCGAN_49_G.pkl'))
    if cuda_available:
        generator = generator.cuda()
    return generator

def gen_image(G, num_epoch, save_dir, fig_size=(8, 8), interpolate=False, inception_score=False):
    interpolate = True
    torch.manual_seed(1234)
    if interpolate:
        noise = torch.FloatTensor(64, 100, 1, 1).normal_(0, 1)
        noise2 = (noise * 0 + 1)
        for i in range(64):
            noise2[i] = (1 - i / 63.0) * noise[0] + (i / 63.0) * noise[8]
        # import ipdb; ipdb.set_trace()
        fixed_noise = Variable(noise2, volatile=True)
    else:
        if inception_score:
            noise = torch.FloatTensor(640, 100, 1, 1).normal_(0, 1)
            fixed_noise = Variable(noise, volatile=True)
        else:
            noise = torch.FloatTensor(64, 100, 1, 1).normal_(0, 1)
            fixed_noise = Variable(noise, volatile=True)
    G.eval()
    fixed_noise = fixed_noise.cuda()
    generate_images = G(fixed_noise)
    if inception_score:
        np.save('samples_for_score', generate_images.data.cpu().numpy())
        print("images saved for inception score calculations")
        os._exit(0)
    new_images = torch.ones(64,3,64,64)
    x0 = generate_images[0]
    x1 = generate_images[1]
    alphas = torch.linspace(0, 10, 64)
    for i in range(64): 
        new_images[i] = alphas[i]*x0.data + (1-alphas[i])*x1.data   
    n_rows = n_cols = 8
    fig, axes = plt.subplots(n_rows, n_cols, figsize=fig_size)
    for ax, img in zip(axes.flatten(), new_images):
        ax.axis('off')
        ax.set_adjustable('box-forced')
        img = (((img - img.min()) * 255) / (img.max() - img.min())).numpy().transpose(1, 2, 0).astype(
            np.uint8)
        ax.imshow(img, cmap=None, aspect='equal')
    plt.subplots_adjust(wspace=0, hspace=0)
    title = 'Epoch {0}'.format(num_epoch)
    fig.text(0.5, 0.04, title, ha='center')

    filename = "{0}_x_sweep.png".format(G.model_name, 1)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    plt.savefig(os.path.join(save_dir, filename))
    plt.close()

generator = build_model("DCGAN")

gen_image(generator, 0, "./logs")

