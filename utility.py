import os
import pickle as pk

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.misc import imresize
from torchvision import transforms, datasets



def trainloader():
    dset = trainloader_helper()
    train_loader = torch.utils.data.DataLoader(dset, batch_size=128, shuffle=True)
    return train_loader


def trainloader_helper():
    if not os.path.isdir('data/resized_celebA/'):
        _preprocess_celeb()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    data_dir = 'data/resized_celebA'
    dset = datasets.ImageFolder(data_dir, transform)
    return dset


def _preprocess_celeb():
    root = 'data/img_align_celebA/'
    save_root = 'data/resized_celebA/'
    resize_size = 64

    if not os.path.isdir(save_root):
        os.mkdir(save_root)
    if not os.path.isdir(save_root + 'celebA'):
        os.mkdir(save_root + 'celebA')
    img_list = os.listdir(root)

    # ten_percent = len(img_list) // 10

    for i in range(len(img_list)):
        img = plt.imread(root + img_list[i])
        img = imresize(img, (resize_size, resize_size))
        plt.imsave(fname=save_root + 'celebA/' + img_list[i], arr=img)

        if (i % 1000) == 0:
            print('%d images complete' % i)


def save(discriminator, generator):
    if not os.path.exists('logs'):
        os.makedirs('logs')

    torch.save(generator.state_dict(), os.path.join('logs', generator.model_name + '_G.pkl'))
    torch.save(discriminator.state_dict(), os.path.join('logs', discriminator.model_name + '_D.pkl'))


def save_losses(minibatch_disc_losses, minibatch_gen_losses, model_name):
    if not os.path.exists('logs'):
        os.makedirs('logs')

    fname1 = model_name + '_minibatch_disc_losses.pk'
    fname2 = model_name + '_minibatch_gen_losses.pk'

    with open(os.path.join('logs', fname1)) as f:
        pk.dump(minibatch_disc_losses, f)
    with open(os.path.join('logs', fname2)) as f:
        pk.dump(minibatch_gen_losses, f)


def plot_result(G, fixed_noise, num_epoch, save_dir, fig_size=(8, 8)):
    G.eval()
    generate_images = G(fixed_noise)
    G.train()

    n_rows = n_cols = 8
    fig, axes = plt.subplots(n_rows, n_cols, figsize=fig_size)

    for ax, img in zip(axes.flatten(), generate_images):
        ax.axis('off')
        ax.set_adjustable('box-forced')
        img = (((img - img.min()) * 255) / (img.max() - img.min())).cpu().data.numpy().transpose(1, 2, 0).astype(
            np.uint8)
        ax.imshow(img, cmap=None, aspect='equal')
    plt.subplots_adjust(wspace=0, hspace=0)
    title = 'Epoch {0}'.format(num_epoch)
    fig.text(0.5, 0.04, title, ha='center')

    filename = G.model_name + '_epoch_' + num_epoch + 'png'
    plt.savefig(os.path.join(save_dir, filename))
    plt.close()
