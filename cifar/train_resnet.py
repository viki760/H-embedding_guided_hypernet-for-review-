
import __init__ # pylint: disable=unused-import
import sys
import os
sys.path.append('/path/to/working/directory/')
from cifar import train_args
from cifar.train import train_Hemb as train_func
import torch
import numpy as np
import random

def set_random_seeds(seed):
    ###! Random seed setup
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    config = train_args.parse_cmd_arguments(mode='resnet_cifar', emb_reg=True)
    print(config)

    set_random_seeds(config.random_seed)
    train_func.run(config, experiment='resnet')



