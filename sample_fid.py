import os
import glob
import os
import torch
import tqdm
from tqdm.asyncio import trange, tqdm
from sampler import pc_sampler2, ode_sampler
from torchlevy import LevyStable
from fid_score import fid_score
from models.ncsnpp import NCSNpp
from util import *

levy = LevyStable()
import torchvision.utils as tvu
from Diffusion import VPSDE

import glob
import cv2
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST, CIFAR10, CelebA, CIFAR100
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize
from model import *


def testimg(path, png_name='test_tile'):
    path = path + "/*"
    # path = "./levy-only/*"
    file_list = glob.glob(path)

    images = []

    for file in file_list:
        img = cv2.imread(file)
        images.append(img)

    tile_size = 8

    # 바둑판 타일형 만들기
    def img_tile(img_list):
        return cv2.vconcat([cv2.hconcat(img) for img in img_list])

    imgs = []

    for i in range(0, len(images), tile_size):
        assert tile_size**2 == len(images), "invalid tile size"
        imgs.append(images[i:i+tile_size])

    img_tile_res = img_tile(imgs)
    png_name = "./" + png_name + ".png"
    cv2.imwrite(png_name, img_tile_res)

def sample_fid(config, path=None, image_folder=None):
    datasets = config.datsets.dataset
    sampler = config.sample_fid.sampler
    num_steps = config.sample_fid.num_steps
    if not os.path.isdir(image_folder):
        os.mkdir(image_folder)
    img_id = len(os.listdir(image_folder))
    print(f"starting from image {img_id}")
    n_rounds = (total_n_samples - img_id) // batch_size

    sde = VPSDE(alpha=config.Diffusion.alpha, beta_min=config.Diffusion.beta_min, beta_max=config.Diffusion.beta_max)
    if conditional == None:
        num_classes=None
    score_model =get_model(datset)
    score_model = score_model.to(device)
    if path:
        ckpt = torch.load(path, map_location=device)
        score_model.load_state_dict(ckpt, strict=True)
        score_model.eval()

    j=img_id
    with torch.no_grad():
        for _ in trange(n_rounds, desc="Generating image samples for FID evaluation."):
            n = batch_size

            x_shape = (n, channels, image_size, image_size)
            x = levy.sample(sde.alpha, 0, size=x_shape).to(device)

            if sampler =="ode_sampler":
             x = ode_sampler(score_model,
                sde,
                sde.alpha,
                batch_size,
                num_steps=config.sampling.num_steps,
                device=config.sampling.device,
                clamp=config.sampling.clamp,
                initial_clamp=config.sampling.initial_clamp,
                datasets=datasets, y=sampling.fix_class)
            if sampler =="pc_sampler2":
                x = pc_sampler2(score_model,
                                sde,
                                sde.alpha,
                                batch_size,
                                num_steps=config.sampling.num_steps,
                                device=config.sampling.device,
                                clamp=config.sampling.clamp,
                                initial_clamp=config.sampling.initial_clamp,
                                datasets=datasets, y= sampling.fix_class)

            x = (x+1)/2
            x = x.clamp(0.0, 1.0)

            for i in range(n):
                sam = x[i]

                plt.axis('off')
                if datasets == 'MNIST':
                    fig = plt.figure(figsize=(1,1))
                    fig.patch.set_visible(False)
                    ax = fig.add_subplot(111)
                    ax.set_axis_off()
                    ax.imshow(sam.permute(1, 2, 0).cpu(), vmin=0., vmax=1., cmap='gray')
                else:
                    fig = plt.figure()
                    fig.patch.set_visible(False)
                    ax = fig.add_subplot(111)
                    ax.set_axis_off()
                    ax.imshow(sam.permute(1, 2, 0).cpu(), vmin=0., vmax=1.)
                    
                   
                name = str(f'{j}') + '.png'
                name = os.path.join(image_folder, name)

                plt.savefig(name)
                #plt.savefig(name, dpi=500)
                plt.cla()
                plt.clf()
                plt.close()
                j= j+1




def dataloader2png(data_loader, com_folder,datasets):
    j=0
    for x,y in tqdm(data_loader):
        x = x.to('cuda')
        n = len(x)
        for i in range(n):
            sam = x[i]
            plt.axis('off')
            if datasets == 'MNIST':
                fig = plt.figure(figsize=(1, 1))
                fig.patch.set_visible(False)
                ax = fig.add_subplot(111)
                ax.set_axis_off()
                ax.imshow(sam.permute(1, 2, 0).cpu(), vmin=0., vmax=1., cmap='gray')
            else:
                fig = plt.figure()
                fig.patch.set_visible(False)
                ax = fig.add_subplot(111)
                ax.set_axis_off()
                ax.imshow(sam.permute(1, 2, 0).cpu(), vmin=0., vmax=1.)
            name = str(f'{j}') + '.png'

            name = os.path.join(com_folder, name)
            plt.savefig(name)
            plt.cla()
            plt.clf()
            j = j + 1
def fix_class_dataloader2png(data_loader, com_folder,datasets,fix_class=1):
    j=0
    for x,y in tqdm(data_loader):
        x = x.to('cuda')
        n = len(x)
        if y == fix_class:

         for i in range(n):
            sam = x[i]
            plt.axis('off')
            if datasets == 'MNIST':
                fig = plt.figure(figsize=(1, 1))
                fig.patch.set_visible(False)
                ax = fig.add_subplot(111)
                ax.set_axis_off()
                ax.imshow(sam.permute(1, 2, 0).cpu(), vmin=0., vmax=1., cmap='gray')
            else:
                fig = plt.figure()
                fig.patch.set_visible(False)
                ax = fig.add_subplot(111)
                ax.set_axis_off()
                ax.imshow(sam.permute(1, 2, 0).cpu(), vmin=0., vmax=1.)
            name = str(f'{j}') + '.png'

            name = os.path.join(com_folder, name)
            plt.savefig(name)
            plt.cla()
            plt.clf()
            j = j + 1
def cifar102png(path='/scratch/private/eunbiyoon/sub_Levy/cifar10_1'):
    transform = transforms.Compose([transforms.Resize((32, 32)),
                                      transforms.ToTensor()
                                      ])
    dataset = CIFAR10('/home/eunbiyoon/comb_Levy_motion', train=False, transform=transform, download=True)
    data_loader = DataLoader(dataset, batch_size=1,
                             shuffle=True, generator=torch.Generator(device='cuda'))
    j=0
    for x,y in tqdm(data_loader):
        x = x.to('cuda')
        n = len(x)
        if y ==1:

         for i in range(n):
            sam = x[i]
            fig = plt.figure()
            fig.patch.set_visible(False)
            ax = fig.add_subplot(111)
            ax.set_axis_off()
            ax.imshow(sam.permute(1, 2, 0).cpu(), vmin=0., vmax=1.)
            
            name = str(f'{j}') + '.png'
            name = os.path.join(path, name)
            plt.savefig(name)
            # plt.savefig(name, dpi=500)
            plt.cla()
            plt.clf()
            j = j + 1

def mnist2png(path="/scratch/private/eunbiyoon/sub_Levy/mnist_1"):
    transform = transforms.Compose([transforms.Resize((28, 28)),
                                      transforms.ToTensor()
                                      ])
    dataset = MNIST('/scratch/private/eunbiyoon/data', train=False, transform=transform, download=True)
    data_loader = DataLoader(dataset, batch_size=1,
                             shuffle=True, generator=torch.Generator(device='cuda'))
    j=0
    for x,y in tqdm(data_loader):
        x = x.to('cuda')
        n = len(x)
        if y==1:
         for i in range(n):
            sam = x[i]
            fig = plt.figure(figsize=(1, 1))
            fig.patch.set_visible(False)
            ax = fig.add_subplot(111)
            ax.set_axis_off()
            ax.imshow(sam.permute(1, 2, 0).cpu(), vmin=0., vmax=1., cmap='gray')
            name = str(f'{j}') + '.png'
            name = os.path.join(path, name)
            plt.savefig(name)
            # plt.savefig(name, dpi=500)
            plt.cla()
            plt.clf()
            j = j + 1

