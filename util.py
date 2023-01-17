from model import *
from models.ncsnpp import NCSNpp
import torchvision.transforms as transforms
from ema import EMAHelper
from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize
from torchvision.datasets import MNIST, CIFAR10
from torch.utils.data import DataLoader
from transformers import get_scheduler

def get_model(config):
    channels = config.datasets.channels
    num_classes = config.datasets.num_classes
    if config.datasets.dataset =="MNIST":
        score_model = Unet(
            dim=image_size,
            channels=channels,
            dim_mults=(1, 2, 4,), num_classes=num_classes)
    elif config.datasets.dataset == "CIFAR10":
        score_model = NCSNpp(config)
    return score_model


def get_datasets(config):
    data= config.datasets.dataset
    image_size = config.datasets.image_size

    if data == "MNIST":
        transform = Compose([
        Resize(image_size),
        ToTensor()])
        transformer = transforms.Compose([transforms.Resize((image_size, image_size)),
                                      transforms.ToTensor()], )

        dataset = MNIST('.', train=True, transform=transform, download=True)
        validation_dataset = MNIST(root='./data', train=False, download=True, transform=transformer)

        data_loader = DataLoader(dataset, batch_size=config.training.batch_size,
                             shuffle=True, num_workers=config.datasets.num_workers, generator=torch.Generator(device=config.training.device))

        validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=config.training.batch_size, shuffle=False)



    if data == 'CIFAR10':
        transform = transforms.Compose([transforms.Resize((image_size, image_size)),
                                    transforms.RandomHorizontalFlip(p=0.5),
                                    transforms.ToTensor()])

        dataset = CIFAR10('/scratch/private/eunbiyoon/Levy_motion', train=True, transform=transform, download=True)

        data_loader = DataLoader(dataset, batch_size=config.training.batch_size,
                             shuffle=True, num_workers=config.datasets.num_workers, generator=torch.Generator(device=config.training.device))

        transformer = transforms.Compose([transforms.Resize((image_size, image_size)),
                                      transforms.ToTensor()
                                      ])
        validation_dataset = CIFAR10(root='/scratch/private/eunbiyoon/Levy_motion', train=False, download=True,
                                 transform=transformer)
        validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=config.training.batch_size, shuffle=False)


    return data_loader, validation_loader