import torch
import torchvision
from torchvision import transforms, datasets
import os
import matplotlib.pyplot as plt
from scipy.misc import imresize
import numpy as np
import pickle as pk


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
    data_dir = 'data/resized_celebA/'
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


def save_losses(minibatch_disc_losses, minibatch_gen_losses):
    if not os.path.exists('logs'):
        os.makedirs('logs')

    with open(os.path.join('logs', minibatch_disc_losses + '.pk')) as f:
        pk.dump(minibatch_disc_losses, f)
    with open(os.path.join('logs', minibatch_gen_losses + '.pk')) as f:
        pk.dump(minibatch_gen_losses, f)
