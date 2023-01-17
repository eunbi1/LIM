import os
from sampling import *
from torch.optim import Adam
import random
import tqdm
import os
import matplotlib.pyplot as plt
from models.ncsnpp import NCSNpp
import time
from model import *
from Diffusion import *
import numpy as np
from sampler import ddim_score_update2
import torch
import torchvision.transforms as transforms
from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize

from transformers import  AdamW, get_scheduler
from torchvision import datasets, transforms
from torchlevy import LevyStable

levy = LevyStable()
def image_grid(x):
  size = 32
  channels = 3
  img = x.reshape(-1, size, size, channels)
  w = int(np.sqrt(img.shape[0]))
  img = img.reshape((w, w, size, size, channels)).transpose((0, 2, 1, 3, 4)).reshape((w * size, w * size, channels))
  return img

def show_samples(x,name):
  x = x.permute(0, 2, 3, 1).detach().cpu().numpy()
  img = image_grid(x)
  plt.figure(figsize=(8,8))
  plt.axis('off')
  plt.imshow(img)
  plt.savefig(name)
  plt.show()



def impainted_noise(sde, data, noise, mask,t):

    sigma = sde.marginal_std(t)
    alpha =sde.alpha
    x_coeff = sde.diffusion_coeff(t)

    if alpha == 2:
        e_L = torch.randn(size=(data.shape)) * np.sqrt(2)
        e_L = e_L.to(device)
    else:
        e_L = levy.sample(alpha, 0, size=(data.shape), is_isotropic=True, clamp=20).to(device)

    data = x_coeff[:, None, None, None] * data +  e_L* sigma[:, None, None, None]
    masked_data =data*mask+noise*(1-mask)

    return masked_data


def impainted_ddim_score_update2(score_model, sde, data, mask, x_s, s, t, y=None, h=0.8, clamp=10, device='cuda', mode='approximation',
                       order=0):
    if y is not None:
        y = torch.ones((x_s.shape[0],)) * y

    alpha = sde.alpha
    score_s = score_model(x_s, s, y) * torch.pow(sde.marginal_std(s) , -(alpha - 1))[:, None, None, None]

    time_step = s - t
    beta_step = sde.beta(s) * time_step

    x_coeff = 1 + beta_step / alpha

    if alpha == 2:
        e_L = torch.randn(size=(x_s.shape)) * np.sqrt(2)
        e_L = e_L.to(device)
    else:
        e_L = levy.sample(alpha, 0, size=(x_s.shape), is_isotropic=True, clamp=20).to(device)

    e_L = impainted_noise(sde, data,e_L,mask,s)

    if alpha == 2:
        score_coeff = beta_step * 2
    noise_coeff = torch.pow(beta_step, 1 / alpha)
    if order == 0:
        score_coeff = alpha * beta_step

    x_t = x_coeff[:, None, None, None] * x_s + score_coeff[:, None, None, None] * score_s + noise_coeff[:, None, None,
                                                                                            None] * e_L

    return impainted_noise(sde, data, x_t, mask, t)


def impainted_pc_sampler2(score_models,
                sde,  data, mask,
                batch_size,
                num_steps,
                LM_steps=200,
                device='cuda',
                eps=1e-4,
                x_0=None,
                Predictor=True, mode='approximation',
                Corrector=False, trajectory=False,
                clamp=10,
                initial_clamp=10, final_clamp=1, y=None,
                datasets="CIFAR10", clamp_mode='constant', h=0.9):
    t = torch.ones(batch_size, device=device) * sde.T
    alpha = sde.alpha
    if datasets == "MNIST":
        if alpha < 2:
            x_s = levy.sample(alpha, 0, size=(batch_size, 1, 28, 28), is_isotropic=True, clamp=20).to(device)
        else:
            x_s = torch.randn(size=(batch_size, 1, 28, 28)) * np.sqrt(2)
            x_s = x_s.to(device)
    elif datasets == "CIFAR10":
        if alpha < 2:
            x_s = levy.sample(alpha, 0, size=(batch_size, 3, 32, 32), is_isotropic=True, clamp=20).to(device)
        else:
            x_s = torch.randn(size=(batch_size, 3, 32, 32)) * np.sqrt(2)
            x_s = x_s.to(device)
    elif datasets == "CelebA":
        if mode == "approximation" or "normal":
            x_s = torch.clamp(levy.sample(alpha, 0, size=(batch_size, 3, 64, 64)).to(device), -initial_clamp,
                              initial_clamp)
        if mode == "resampling":
            x_s = levy.sample(alpha, 0, clamp=initial_clamp, size=(batch_size, 3, 64, 64)).to(device)
        if mode == 'brownian':
            x_s = torch.clamp(levy.sample(1.5, 0, size=(batch_size, 3, 64, 64)).to(device), -initial_clamp,
                              initial_clamp)
    x_s = impainted_noise(sde, data,x_s,mask,t)
    if clamp_mode == "constant":
        linear_clamp = clamp
    if clamp_mode == "linear":
        linear_clamp = batch_time_step_t[0] * (clamp - final_clamp) + final_clamp
    if clamp_mode == "root":
        linear_clamp = torch.pow(batch_time_step_t[0], 1 / 2) * (clamp - final_clamp) + final_clamp
    if clamp_mode == "quad":
        linear_clamp = batch_time_step_t[0] ** 2 * (clamp - final_clamp) + final_clamp

    if trajectory:
        samples = []
        samples.append(x_s)
    time_steps = torch.pow(torch.linspace(sde.T, 1e-5, num_steps), 1)
    step_size = time_steps[0] - time_steps[1]

    batch_time_step_s = torch.ones(x_s.shape[0]) * time_steps[0]
    batch_time_step_s = batch_time_step_s.to(device)

    with torch.no_grad():
        for t in tqdm(time_steps[1:]):
            batch_time_step_t = torch.ones(x_s.shape[0]) * t
            batch_time_step_t = batch_time_step_t.to(device)

            if Predictor:
                x_s = impainted_ddim_score_update2(score_models, sde, data, mask, x_s, batch_time_step_s, batch_time_step_t, y=y,
                                         clamp=linear_clamp)
            if trajectory:
                samples.append(x_s)

            batch_time_step_s = batch_time_step_t


    if trajectory:
        return samples
    else:
        return x_s

def impainted_ode_score_update(score_model, sde, data, mask, x_s, s, t, y=None, h=0.8, clamp=10, device='cuda', mode='approximation',
                       order=0):
    alpha = sde.alpha
    if order==0:
     if y is not None:
         y = torch.ones(x_s.shape[0])*y
         y = y.type(torch.long )
     score_s = score_model(x_s, s,y)*torch.pow(sde.marginal_std(s)+1e-5,-(alpha-1))[:,None,None,None]

    x_coeff =sde.diffusion_coeff(t) * torch.pow(sde.diffusion_coeff(s), -1)
    lambda_s = sde.marginal_lambda(s)
    lambda_t = sde.marginal_lambda(t)


    h_t = lambda_t - lambda_s


    if alpha == 2:
        score_coeff = 2*torch.pow( sde.marginal_std(s), 1) * sde.marginal_std(t)*(-1+torch.exp(h_t))
        x_t = x_coeff[:, None, None, None] * x_s + score_coeff[:, None, None, None] * score_s
    else:
        if order==0:
            score_coeff= alpha*torch.pow(sde.marginal_std(s),alpha-1)*sde.marginal_std(t)*(-1+torch.exp(h_t))
            x_t = x_coeff[:, None, None, None] * x_s + score_coeff[:, None, None, None] * score_s

    # x_t = torch.clamp(x_t, -clamp,clamp)

    return impainted_noise(sde, data, x_t, mask, t)


def impainted_ode_sampler(score_model,
                sde,  data, mask,
                batch_size,
                num_steps,
                LM_steps=200,
                device='cuda',
                eps=1e-4,
                x_0=None,
                Predictor=True, mode='approximation',
                Corrector=False, trajectory=False,
                clamp=10,
                initial_clamp=10, final_clamp=1, y=None,
                datasets="CIFAR10", clamp_mode='constant', h=0.9):
    t = torch.ones(batch_size, device=device)*sde.T
    alpha = sde.alpha
    if datasets == "MNIST":
        e_L = levy.sample(alpha, 0, (batch_size, 1, 28, 28)).to(device)
        x_s = torch.clamp(e_L, -initial_clamp, initial_clamp)
    elif datasets == "CIFAR10":
        x_s = levy.sample(alpha, 0, (batch_size, 3, 32, 32), is_isotropic=True, clamp=10).to(device)


    elif datasets == "CelebA":
        e_L = levy.sample(alpha, 0, (batch_size, 3, 64, 64)).to(device)
        x_s = torch.clamp(e_L, -initial_clamp, initial_clamp)
    x_s = impainted_noise(sde, data,x_s,mask,t)
    if trajectory:
        samples = []
        samples.append(x_s)
    time_steps = torch.pow(torch.linspace(sde.T, 1e-5, num_steps),1)

    batch_time_step_s = torch.ones(x_s.shape[0]) * time_steps[0]
    batch_time_step_s = batch_time_step_s.to(device)

    with torch.no_grad():
        for t in tqdm(time_steps[1:]):
            batch_time_step_t = torch.ones(x_s.shape[0]) * t
            batch_time_step_t = batch_time_step_t.to(device)

            if clamp_mode == "constant":
                linear_clamp = clamp
            if clamp_mode == "linear":
                linear_clamp = batch_time_step_t[0] * (clamp - final_clamp) + final_clamp
            if clamp_mode == "root":
                linear_clamp = torch.pow(batch_time_step_t[0], 1 / 2) * (clamp - final_clamp) + final_clamp
            if clamp_mode == "quad":
                linear_clamp = batch_time_step_t[0] ** 2 * (clamp - final_clamp) + final_clamp

            x_s = impainted_ode_score_update(score_model, sde, data,mask, x_s, batch_time_step_s, batch_time_step_t,y=y,h=h,
                                   clamp=linear_clamp)
            if trajectory:
                samples.append(x_s)
            batch_time_step_s = batch_time_step_t

    if trajectory:
        return samples
    else:
        return x_s
