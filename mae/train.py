import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path
import sys

import matplotlib.pyplot as plt

import torch

from engine_pretrain import train_one_epoch
from main_ViT import main

if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    print('world_size = %d' % world_size)
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '9898'
    os.environ['WORLD_SIZE'] = str(world_size)
    torch.multiprocessing.spawn(main, nprocs=world_size, args=(world_size,))