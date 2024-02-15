import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path
import sys
import cv2

import pandas as pd
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import random

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import timm

assert timm.__version__ == "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler

import models_mae

from engine_pretrain import train_one_epoch

import warnings
warnings.filterwarnings('ignore')

DATASET_PATH = '/home/placido.falqueto/IRI_Barcelona/training_data/64crop_size/1red/'
input_size = 64
mask_ratio = 0 # 0.75
prfx = 'test8'

def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=200, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='mae_vit', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=input_size, type=int,
                        help='images input size')

    parser.add_argument('--mask_ratio', default=mask_ratio, type=float, #0.75
                        help='Masking ratio (percentage of removed patches).')

    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.3,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-4, metavar='LR', # 1e-4
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=20, metavar='N', #20
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--data_path', default='./data/cifar10', type=str,
                        help='dataset path')

    parser.add_argument('--output_dir', default='/data/placido',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--f')

    return parser


class SemanticMapDataset(Dataset):
    def __init__(self, data_dirs, transform=None, target_transform=None):
        self.train_x = np.empty((0, input_size, input_size, 12))
        self.train_y = np.empty((0, input_size, input_size))
        for data_dir in data_dirs:
            train_data_dir = DATASET_PATH+data_dir
            assert os.path.exists(train_data_dir), f'data_dir {train_data_dir} does not exist'
            train_x_aux = np.loadtxt(train_data_dir+'/train_X.csv')
            sizes = train_x_aux[0:4].astype(int)
            train_x_aux = np.delete(train_x_aux, [0,1,2,3])
            train_x_aux = np.reshape(train_x_aux,sizes)

            train_y_aux = np.loadtxt(train_data_dir+'/train_Y.csv')
            sizes = train_y_aux[0:3].astype(int)
            train_y_aux = np.delete(train_y_aux, [0,1,2])
            train_y_aux = np.reshape(train_y_aux,sizes)

            self.train_x = np.append(self.train_x, train_x_aux, axis=0)
            self.train_y = np.append(self.train_y, train_y_aux, axis=0)

        self.train_x = torch.from_numpy(self.train_x).float()
        self.train_y = torch.from_numpy(self.train_y).float()

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.train_x)

    def __getitem__(self, idx):
        features = self.train_x[idx]
        target = self.train_y[idx]
        if self.transform:
            features = self.transform(features)
        if self.target_transform:
            target = self.target_transform(target)
        return features, target

imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])

def show_image(features, data, title=''):
    # features is [H, W, C]
    # plt.imshow(torch.clip((features * imagenet_std + imagenet_mean) * 255, 0, 255).int())
    data = data / data.max()
    test11 = np.stack((features[:,:,0],features[:,:,0],features[:,:,0]),axis=2)
    test11 = np.multiply(test11,np.stack((np.full(data.shape,1),1-data,1-data),axis=2))
    for i in range(10):
        alp = 0.5
        test11 = np.multiply(test11,np.stack((features[:,:,i+1],features[:,:,i+1],features[:,:,i+1]),axis=2)*alp+(1-alp))
    plt.imshow(test11)
    plt.title(title, fontsize=23)
    plt.axis('off')
    return

def run_one_image(x, target, model, epoch=None):
    # make it a batch-like
    x = x.unsqueeze(dim=0)
    target = target.unsqueeze(dim=0)

    # run MAE
    x = x.to('cuda')
    target = target.to('cuda')
    _, y, _ = model(x, mask_ratio=mask_ratio)
    x = x.detach().cpu()
    target = target.detach().cpu()
    
    y = torch.einsum('nchw->nhwc', y).detach().cpu()
    y = y.squeeze(3)

    # # visualize the mask
    # mask = mask.detach()
    # mask = mask.unsqueeze(-1).repeat(1, 1, model.module.patch_embed.patch_size[0]**2 *3)  # (N, H*W, p*p*3)
    # mask = model.module.unpatchify(mask)  # 1 is removing, 0 is keeping
    # mask = torch.einsum('nchw->nhwc', mask).detach().cpu()
    
    # x = torch.einsum('nchw->nhwc', x).detach().cpu()

    # # masked image
    # im_masked = x * (1 - mask)

    # # MAE reconstruction pasted with visible patches
    # im_paste = x * (1 - mask) + y * mask

    # make the plt figure larger
    plt.rcParams['figure.figsize'] = [24, 12] # W: 24, H: 12

    plt.subplot(1, 4, 3)
    plt.imshow(target[0])
    plt.title('target', fontsize=23)
    plt.axis('off')

    plt.subplot(1, 4, 4)
    plt.imshow(y[0])
    plt.title('y', fontsize=23)
    plt.axis('off')

    plt.subplot(1, 4, 1)
    show_image(x[0], target[0], "original")

    # plt.subplot(1, 4, 2)
    # show_image(im_masked[0], "masked")

    plt.subplot(1, 4, 2)
    show_image(x[0], y[0], "prediction")

    # plt.subplot(1, 4, 4)
    # show_image(im_paste[0], "reconstruction + visible")

    if epoch is not None:
        # if epoch % 5 == 0:
        plt.savefig(f'image_logs/vit_training_{epoch}.png')
        plt.savefig(f'image_logs/vit_training_last.png')
    else:
        plt.show()
    

def main(rank, world_size):
    os.environ['RANK'] = str(rank)
    os.environ['LOCAL_RANK'] = str(rank)

    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.abspath('')))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    split_percentage = 0.9
    data_dirs = os.listdir(DATASET_PATH)
    data_dirs.sort()
    # len_train_data = int((len(data_dirs) - 1) * split_percentage)
    # train_data_dirs = random.sample(data_dirs, len_train_data)
    # # train_data_dirs = data_dirs[-len_train_data:]
    # val_data_dirs = [x for x in data_dirs if x not in train_data_dirs]
    # train_data_dirs.sort()
    # val_data_dirs.sort()
    # test_data_dirs = val_data_dirs # [val_data_dirs.pop()]
    # # train_data_dirs = [train_data_dirs.pop()]         # UNCOMMENT FOR FAST DEBUGGING 
    # # val_data_dirs = [val_data_dirs.pop()]             # UNCOMMENT FOR FAST DEBUGGING 

    i = 0
    # if True:
    for i in range(len(data_dirs)):

        train_data_dirs = data_dirs[:]
        val_data_dirs = [train_data_dirs.pop(i)]
        test_data_dirs = val_data_dirs
        model_id = val_data_dirs[0]
        
        print(f'TRAINING MAPS: {train_data_dirs}\n')
        print(f'VALIDATION MAPS: {val_data_dirs}\n')
        print(f'TEST MAP: {test_data_dirs}\n')
        print('-------------------------------------')

        dataset_train = SemanticMapDataset(data_dirs=train_data_dirs)
        dataset_val = SemanticMapDataset(data_dirs=val_data_dirs)

        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=None,
            batch_size=args.batch_size,
            num_workers=0,
            pin_memory=False,
            drop_last=True,
        )

        dataset_test = SemanticMapDataset(data_dirs=test_data_dirs)
        data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=dataset_test.__len__(), shuffle=False)
        features, target = next(iter(data_loader_test))

        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))

        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=True,
        )

        print('------------------- DATA LOADED -------------------')

        if global_rank == 0 and args.log_dir is not None:
            os.makedirs(args.log_dir, exist_ok=True)
            log_writer = SummaryWriter(log_dir=args.log_dir)
        else:
            log_writer = None

        # define the model
        model = models_mae.__dict__[args.model](norm_pix_loss=args.norm_pix_loss)

        model.to(device)

        model_without_ddp = model
        print("Model = %s" % str(model_without_ddp))

        eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

        if args.lr is None:  # only base_lr is specified
            args.lr = args.blr * eff_batch_size / 256

        print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
        print("actual lr: %.2e" % args.lr)

        print("accumulate grad iterations: %d" % args.accum_iter)
        print("effective batch size: %d" % eff_batch_size)

        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
            model_without_ddp = model.module

        # following timm: set wd as 0 for bias and norm layers
        param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
        optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
        print(optimizer)
        loss_scaler = NativeScaler()

        misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

        print(f"Start training for {args.epochs} epochs")
        start_time = time.time()
        for epoch in range(args.start_epoch, args.epochs+1):
            if args.distributed:
                data_loader_train.sampler.set_epoch(epoch)
            train_stats = train_one_epoch(
                model, data_loader_train, data_loader_val,
                optimizer, device, epoch, loss_scaler,
                log_writer=log_writer,
                args=args
            )
            if epoch % 5 == 0 or epoch == args.epochs:    
                idx = np.random.randint(0, dataset_test.__len__())
                print(f'sample id: {idx}')
                features_test = features[idx]
                target_test = target[idx]
                run_one_image(features_test, target_test, model, epoch)
            if args.output_dir and epoch == args.epochs:
                misc.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch, id=model_id, prefix=prfx)

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                            'epoch': epoch,}

            if args.output_dir and misc.is_main_process():
                if log_writer is not None:
                    log_writer.flush()
                with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                    f.write(json.dumps(log_stats) + "\n")

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))
