import os
import pickle as pk

import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.misc import imresize
from torchvision import transforms, datasets

log_dir = 'logs'


def trainloader():
    dset = trainloader_helper()
    train_loader = torch.utils.data.DataLoader(dset, batch_size=128, shuffle=True, num_workers=6)
    return train_loader


def trainloader_helper():
    data_dir = '/data/milatmp1/considib/img_align_celebA/celebA/'
    resized_dir = '/data/milatmp1/considib/resized_celebA/'
    if not os.path.isdir(resized_dir):
        _preprocess_celeb(data_dir, resized_dir)

    transform = transforms.Compose([
        # transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    data_dir = '/data/milatmp1/considib/resized_celebA'
    dset = datasets.ImageFolder(data_dir, transform)
    print("Loaded {} images for training".format(len(dset)))
    return dset


def _preprocess_celeb(root, save_root):
    resize_size = 64

    if not os.path.isdir(save_root):
        os.mkdir(save_root)
    if not os.path.isdir(save_root + 'celebA'):
        os.mkdir(save_root + 'celebA')
    img_list = os.listdir(root)

    # ten_percent = len(img_list) // 10
    print(len(img_list))

    for i in range(len(img_list)):
        img = plt.imread(root + img_list[i])
        img = imresize(img, (resize_size, resize_size))
        plt.imsave(fname=save_root + 'celebA/' + img_list[i], arr=img)

        if (i % 1000) == 0:
            print('%d images complete' % i)


def save(discriminator, generator, epoch=0):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    gen_model_name   = "{0}_{1}_G.pkl".format(generator.model_name, epoch)
    disc_model_name  = "{0}_{1}_D.pkl".format(discriminator.model_name, epoch)
    torch.save(generator.state_dict(), os.path.join(log_dir, gen_model_name))
    torch.save(discriminator.state_dict(), os.path.join(log_dir, disc_model_name))


def save_losses(minibatch_disc_losses, minibatch_gen_losses, model_name):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    fname1 = model_name + '_minibatch_disc_losses.pk'
    fname2 = model_name + '_minibatch_gen_losses.pk'

    with open(os.path.join(log_dir, fname1), "wb") as f:
        pk.dump(minibatch_disc_losses, f)
    with open(os.path.join(log_dir, fname2), "wb") as f:
        pk.dump(minibatch_gen_losses, f)


def plot_result(G, fixed_noise, num_epoch, fig_size=(8, 8)):
    G.eval()
    fixed_noise = fixed_noise.cuda()
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

    filename = "{0}_epoch_{1}.png".format(G.model_name, num_epoch)
    if not os.path.exists(log_dir):
       os.mkdir(log_dir)
    plt.savefig(os.path.join(log_dir, filename))
    plt.close()
