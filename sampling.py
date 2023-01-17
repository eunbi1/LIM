import os
from util import *
from model import *

from Diffusion import *
from sampler import *
import torch
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from models.ncsnpp import NCSNpp


def image_grid(x):
  size = x.shape[1]
  channels = x.shape[-1]
  img = x.reshape(-1, size, size, channels)
  w = int(np.sqrt(x.shape[0]))
  img = img.reshape((w, w, size, size, channels)).transpose((0, 2, 1, 3, 4)).reshape((w * size, w * size, channels))
  return img

def visualization(x, name, datasets):
  x = x.permute(0, 2, 3, 1).detach().cpu().numpy()
  img = image_grid(x)
  plt.figure(figsize=(8,8))
  plt.axis('off')
  if datasets =="MNIST":
      plt.imshow(img,cmap='gray')
  else :
      plt.imshow(img)
  plt.savefig(name, dpi= 500)
  plt.show()

def diffusion_animation(samples, name="diffusion_1.8.gif"):
    fig = plt.figure(figsize=(12, 12))
    batch_size = 64
    ims = []
    for i in range(0, len(samples), 10):
        sample = samples[i]
        sample_grid = make_grid(sample, nrow=int(np.sqrt(batch_size)))
        plt.axis('off')
        im = plt.imshow(sample_grid.permute(1, 2, 0).cpu(), vmin=0., vmax=1.)
        ims.append([im])

    animate = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
    animate.save(name)
    plt.show()


def sample(config, path=None, dir_path=None, mode="sampling"):
    sde = VPSDE(config)
    alpha = config.Diffusion.alpha
    datasets = config.datasets.dataset
    y = config.sampling.fix_class
    num_steps = config.sampling.sampling_steps
    batch_size = config.sampling.batch_size
    sampler = config.sampling.sampler
    trajectory = config.sampling.trajectory
    clamp = config.sampling.clamp
    initial_clamp  = config.sampling.initial_clamp
    device = config.sampling.device
    score_model = get_model(config)
    score_model = score_model.to(device)
    score_model.eval()
    if config.sampling.path is not None and mode !="train":
        path = config.sampling.path
    if config.sampling.dir_path is not None and mode !="train":
        dir_path = config.sampling.dir_path

    if path:
        ckpt = torch.load(path, map_location=device)
        score_model.load_state_dict(ckpt, strict=False)
        score_model.eval()

    if score_model:
        score_model = score_model

    if sampler == 'pc_sampler2':
        samples = pc_sampler2(score_model,
                              sde=sde, alpha=sde.alpha,
                              batch_size=batch_size, mode= mode,
                              num_steps=num_steps,
                              device=device, y=y,
                              trajectory=trajectory,
                              clamp=clamp, initial_clamp=initial_clamp, datasets=datasets)
    elif sampler == "ode_sampler":
        samples = ode_sampler(score_model,
                              sde=sde, alpha=sde.alpha,
                              batch_size=batch_size,
                              num_steps=num_steps,
                              device=device,y=y,
                              trajectory=trajectory,
                              clamp=clamp, initial_clamp=initial_clamp, datasets=datasets)

    if trajectory:
        for i, img in enumerate(samples):
            img = (img + 1) / 2
            img = img.clamp(0.0, 1.0)
            samples[i] = img
        last_sample = samples[-1]
        sample_grid = make_grid(last_sample, nrow=int(np.sqrt(batch_size)))
        # plt.figure(figsize=(6, 6))
        # plt.axis('off')
        # plt.imshow(sample_grid.permute(1, 2, 0).cpu(), vmin=0., vmax=1.)

    else:
        samples = (samples + 1) / 2
        samples = samples.clamp(0.0, 1.0)
        last_sample = samples
        sample_grid = make_grid(last_sample, nrow=int(np.sqrt(batch_size)))
        # plt.figure(figsize=(6, 6))
        # plt.axis('off')
        # plt.imshow(sample_grid.permute(1, 2, 0).cpu(), vmin=0., vmax=1.)

    name =  str(datasets) + str(
        time.strftime('%m%d_%H%M_', time.localtime(time.time()))) + '_' + 'alpha' + str(f'{alpha}') + str(f'{initial_clamp}_{clamp}') + '.png'
    if dir_path:
        if not os.path.isdir(dir_path):
            os.mkdir(dir_path)
    dir_path = os.path.join(dir_path, 'sample')
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)
    name = os.path.join(dir_path, name)

    visualization(last_sample, name, datasets)

    plt.show()
    plt.cla()
    plt.clf()
  
    if trajectory:
         name2 = str(datasets)+ str(time.strftime('%m%d_%H%M_', time.localtime(time.time()))) + '_' + 'alpha' + str(
             f'{alpha}') + '.gif'
         name2 = os.path.join(dir_path,name2)
         diffusion_animation(samples, name=name2)

    return samples
