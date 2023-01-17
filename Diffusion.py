import torch
import numpy as np
import math

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

import random
import copy
import torch
import torchvision
import numpy as np
import pandas as pd
import sklearn.metrics as sm
import torch.nn.functional as F


class VPSDE:
    def __init__(self, config):
        self.beta_min = beta_min = config.Diffusion.beta_min
        self.beta_max = beta_max = config.Diffusion.beta_max
        self.cosine_s = config.Diffusion.cosine_s
        self.schedule = schedule = config.Diffusion.schedule = 'cosine'
        self.cosine_beta_max =config.Diffusion.cosine_beta_max 
        self.cosine_t_max = math.atan(self.cosine_beta_max * (1. + self.cosine_s) / math.pi) * 2. * (
                    1. + self.cosine_s) / math.pi - self.cosine_s
        if schedule == 'cosine':
            # For the cosine schedule, T = 1 will have numerical issues. So we manually set the ending time T.
            # Note that T = 0.9946 may be not the optimal setting. However, we find it works well.
            self.T = 0.9946
        else:
            self.T = 1.
        self.cosine_log_alpha_0 = math.log(math.cos(self.cosine_s / (1. + self.cosine_s) * math.pi / 2.))
        self.alpha = config.Diffusion.alpha


    def beta(self, t):
        if self.schedule =='linear':
            beta= (self.beta_1 - self.beta_0) * t + self.beta_0
        elif self.schedule == 'cosine':
            beta = math.pi/2*self.alpha/(self.cosine_s+1)*torch.tan( (t+self.cosine_s)/(1+self.cosine_s)*math.pi/2 )
        beta = torch.clamp(beta,0,20)
        return beta


    def marginal_log_mean_coeff(self, t):
        if self.schedule =='linear':
          log_alpha_t = - 1 / (2 * self.alpha) * (t ** 2) * (self.beta_1 - self.beta_0) - 1 / self.alpha * t * self.beta_0
        elif self.schedule == 'cosine':
            log_alpha_fn = lambda s: torch.log(torch.cos((s + self.cosine_s) / (1. + self.cosine_s) * math.pi / 2.))
            log_alpha_t =  log_alpha_fn(t) - self.cosine_log_alpha_0

        return log_alpha_t

    def diffusion_coeff(self, t):
        return torch.exp(self.marginal_log_mean_coeff(t))

    def marginal_std(self, t):
        sigma = torch.pow(1. - self.diffusion_coeff(t)**self.alpha,  1/ self.alpha)
        return sigma

    def marginal_lambda(self, t):
        return torch.log(self.diffusion_coeff(t)*torch.pow(self.marginal_std(t),-1))

