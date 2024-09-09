#!/usr/bin/env python3
# Copyright 2019 Johannes von Oswald

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# @title           :train.py
# @author          :jvo
# @contact         :voswaldj@ethz.ch
# @created         :07/08/2019
# @version         :1.0
# @python_version  :3.6.8

"""
Continual learning of permutedMNIST with hypernetworks.
-------------------------------------------------------

This script is used to run PermutedMNIST continual learning experiments.
It's role is analogous to the one of the script :mod:`mnist.train_splitMNIST`.
Start training by executing the following command:

.. code-block:: console

    $ python train_permutedMNIST.py

"""
# Do not delete the following import for all executable scripts!
import __init__ # pylint: disable=unused-import
import sys
import os
sys.path.append('/mnt/d/task/research/codes/HyperNet/hypercl/')

import mnist.train_splitMNIST as train_func
from mnist import train_args 

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
    ### Get command line arguments.
    config = train_args.parse_cmd_arguments(mode="perm", emb_reg=True)
    config.data_dir = "/mnt/d/task/research/codes/HyperNet/hypercl/datasets/"
    config.infer_task_id = True
    config.temb_size = config.rp_temb_size
    # config.epochs=1
    print(config)
    set_random_seeds(config.random_seed)
    train_func.run(config)