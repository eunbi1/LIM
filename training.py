import os
import glob
from util import *
from sampling import *
from torch.optim import Adam
from sample_fid import sample_fid
from torch.utils.data import DataLoader
import random
from sample_fid import dataloader2png, fix_class_dataloader2png
from torchvision.datasets import MNIST, CIFAR10
import tqdm
import os
from fid_score import fid_score

import matplotlib.pyplot as plt
import time
from model import *
from models.ncsnpp import NCSNpp
from losses import *

from Diffusion import *
import numpy as np
import torch

import torchvision.transforms as transforms
from ema import EMAHelper
from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize

from transformers import get_scheduler
from torchvision import datasets, transforms



from torchlevy import LevyStable

levy = LevyStable()

image_size = 28
channels = 1
batch_size = 128


def train(config):
    lr = config.optim.lr
    batch_size = config.training.batch_size
    n_epochs = config.training.n_epochs
    num_steps = config.training.num_steps
    path = config.training.path
    device = config.training.device
    training_clamp = config.training.training_clamp
    initial_epoch = config.training.initial_epoch
    sample_probs = config.training.sample_probs
    conditional = config.training.conditional
    fix_class = config.training.fix_class
    fid_mode = config.training.fid_mode
    sampling_mode = config.sampling.sampler
    sampling_step = config.sampling.sampling_steps
    store_path = config.training.store_path
    alpha = config.Diffusion.alpha

    sde = VPSDE(config)
    score_model = get_model(config)
    score_model.to(device)
    data_loader, validation_loader = get_datasets(config)

    if path is not None:
        ckpt = torch.load(path, map_location=device)
        score_model.load_state_dict(ckpt, strict=False)

    ema_helper = EMAHelper(mu=config.optim.mu)
    num_training_steps = n_epochs * len(data_loader)
    ema_helper.register(score_model)
    optimizer = torch.optim.AdamW(score_model.parameters(), lr=lr)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )
    score_model.train()

    counter = 0
    L = []
    L_val = []

    j = 0
    for epoch in range(n_epochs):
        counter += 1
        avg_loss = 0.
        num_items = 0
        for x, y in data_loader:
            x = 2 * x - 1

            n = x.size(0)
            x = x.to(device)
            y= y.to(device)
            t = torch.rand(x.shape[0]).to(device)*(sde.T-0.00001)+0.00001
            if alpha==2:
                e_L = torch.randn(size=(x.shape))*np.sqrt(2)
                e_L = e_L.to(device)
            else:
                e_L = levy.sample(alpha, 0, size=(x.shape), is_isotropic=True, clamp=training_clamp).to(device)

            if conditional == False:
                y = None
            if np.random.random() < 0.2:
                y = None

            loss = loss_fn(score_model, sde, x, t,y, e_L=e_L)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(score_model.parameters(), config.optim.grad_clip)
            optimizer.step()
            lr_scheduler.step()
            ema_helper.update(score_model)


            print(f'{epoch} th epoch {j} th step loss: {loss}')
            j += 1
            optimizer.step()
            avg_loss += loss.item() * x.shape[0]
            num_items += x.shape[0]

        else:
            with torch.no_grad():
                counter += 1
                val_avg_loss = 0.
                val_num_items = 0
                for x, y in validation_loader:
                    x= 2*x-1
                    n = x.size(0)
                    x = x.to(device)
                    t = torch.rand(x.shape[0]).to(device)*(sde.T-0.00001)+0.00001
                    if conditional == False:
                        y = None
                    if np.random.random() < 0.2:
                        y = None

                    if alpha == 2:
                        e_L = levy.sample(alpha, 0, size=(x.shape), is_isotropic=False).to(device)
                    else:
                        e_L = levy.sample(alpha, 0, size=(x.shape), is_isotropic=True, clamp=training_clamp).to(device)

                    val_loss = loss_fn(score_model, sde, x, t,y, e_L)
                    val_avg_loss += val_loss.item() * x.shape[0]
                    val_num_items += x.shape[0]
        L.append(avg_loss / num_items)
        L_val.append(val_avg_loss / val_num_items)

        t1 = time.time()
        if epoch%config.training.ckpt_store == 0 :


            ckpt_name = str(datasets) + str(
            f'batch{batch_size}ch{ch}ch_mult{ch_mult}num_res{num_res_blocks}dropout{dropout}') + str(
            f'clamp{training_clamp}') + str(f'_epoch{epoch+initial_epoch}_') + str(f'{alpha}_{beta_min}_{beta_max}.pth')
            dir_path = str(datasets) + str(f'batch{batch_size}lr{lr}ch{ch}ch_mult{ch_mult}num_res{num_res_blocks}dropout{dropout}') + str(
            f'clamp{training_clamp}') + str(f'{alpha}_{beta_min}_{beta_max}')
            if conditional==True:
                dir_path = 'conditional'+dir_path
            if imbalanced==True:
                dir_path = str(sample_probs)+dir_path
            dir_path = os.path.join('/scratch/private/eunbiyoon/sub_Levy', dir_path)
            if not os.path.isdir(dir_path):
                os.mkdir(dir_path)

            X = range(initial_epoch,len(L)+initial_epoch)
            plt.plot(X, L, 'r', label='training loss')
            plt.plot(X, L_val, 'b', label='validation loss')
            plt.legend()
            name_ = os.path.join(dir_path,  'loss.png')
            plt.savefig(name_)
            plt.cla()

            dir_path2 = os.path.join(dir_path, 'ckpt')
            if not os.path.isdir(dir_path2):
                os.mkdir(dir_path2)
            if conditional==False:
                fix_class = None

            ckpt_name = os.path.join(dir_path2, ckpt_name)
            torch.save(score_model.state_dict(), ckpt_name)
            name= str(epoch+initial_epoch)+mode
        if epoch % config.training.sampling_store == 0:
            sample(config, path=ckpt_name, dir_path=dir_path, mode='train')
        # name = 'ode'+ str(epoch + initial_epoch) + mode
        # sample(alpha=sde.alpha, path=ckpt_name,
        #        beta_min=beta_min, beta_max=beta_max, sampler='ode_sampler', batch_size=64, num_steps=20,
        #        LM_steps=50,
        #        Predictor=True, Corrector=False, trajectory=False, clamp=2.3, initial_clamp=training_clamp,
        #        clamp_mode="constant",
        #        datasets=datasets, name=name,
        #        dir_path=dir_path, ch=ch, ch_mult=ch_mult, num_res_blocks=num_res_blocks, resolution=resolution )
    dir_path = str(datasets) + str(
        f'batch{batch_size}lr{lr}ch{ch}ch_mult{ch_mult}num_res{num_res_blocks}dropout{dropout}') + str(
        f'clamp{training_clamp}') + str(f'{alpha}_{beta_min}_{beta_max}')
    if conditional == True:
        dir_path = 'conditional' + dir_path

    dir_path = os.path.join('/scratch/private/eunbiyoon/sub_Levy', dir_path)
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)
        
    name = str(sampling_step)+ sampling_mode+'_'+fid_mode+'_'+datasets+'_'+str(alpha)
    name2 = fid_mode +datasets
    if conditional == False:
        fix_class= None
    if fix_class==0:
        name = '0' + name
        name2 = '0' + name2

    elif fix_class:
        name = str(fix_class)+name
        name2 = str(fix_class)+name2
    image_folder =os.path.join(dir_path  , name)
    com_folder = os.path.join(store_path  , name2)

    if n_epochs==0:
        ckpt_name = path
    if not os.path.isdir(image_folder):
        os.mkdir(image_folder)
    if not os.path.isdir(com_folder):
        os.mkdir(com_folder)
        if fid_mode == "train":
            if fix_class == None:
             validation_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=500,shuffle=False)
             dataloader2png(validation_loader,com_folder, datasets)
            else:
                validation_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1, shuffle=False)
                fix_class_dataloader2png(validation_loader, com_folder, datasets, fix_class)

        else :
            validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=500, shuffle=False)
            dataloader2png(validation_loader, com_folder, datasets)
    plt.axis('off')
    sample_fid(config, com_folder, image_folder)
    fid = fid_score(com_folder, image_folder)
    print(f'alpha:{alpha} fid mode:{fid_mode},sampling mode {sampling_mode} step{sampling_step} FID:{fid}')
