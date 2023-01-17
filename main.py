
import argparse
import os
from training import *
from sampling import *
import torch.distributed as dist
import os
import time
import yaml
from torch.backends import cudnn

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

parser = argparse.ArgumentParser(description='LIM')
parser.add_argument("--sample", action="store_true")
parser.add_argument( "--config", type=str, required=True, help="Path to the config file")
parser.add_argument("--seed", type=int, default=1234, help="Random seed")
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

with open(os.path.join("config", args.config), "r") as f:
    config = yaml.safe_load(f)

config = dict2namespace(config)


def main():
    if args.sample:
        sample(config)
    else:
        train(config)


if __name__=='__main__':
    cudnn.benchmark = False
    torch.multiprocessing.set_start_method('spawn')
    main()

