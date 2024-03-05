import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import random

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import timm

assert timm.__version__ == "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler

import models_mae

from engine_pretrain import train_one_epoch


DATASET_PATH = '/data/placido/training_data/64crop_size/13labels/1red/'
STOP_TRAINING_AFTER_EPOCHS = 15
input_size = 64
mask_ratio = 0.75 # 0.75
prfx = 'xentropy_test'
Yfile = 'train_Y'

# -------------------------
# Argument Parsing
# -------------------------
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

# -------------------------
# Dataset 
# -------------------------
class SemanticMapDataset(Dataset):
    """Loads semantic map data from specified directories.
    """
    def __init__(self, data_dirs, transform=None, target_transform=None):
        self.train_x = np.empty((0, input_size, input_size, 13))
        self.train_y = np.empty((0, input_size, input_size))
        for data_dir in data_dirs:
            train_data_dir = DATASET_PATH+data_dir
            assert os.path.exists(train_data_dir), f'data_dir {train_data_dir} does not exist'
            train_x_aux = np.loadtxt(train_data_dir+'/train_X.csv')
            sizes = train_x_aux[0:4].astype(int)
            train_x_aux = np.delete(train_x_aux, [0,1,2,3])
            train_x_aux = np.reshape(train_x_aux,sizes)

            train_y_aux = np.loadtxt(train_data_dir+'/'+Yfile+'.csv')
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

# -------------------------
# Visualization Functions
# -------------------------
def show_image(features, data, title=''):
    """Processes and visualizes a single image for analysis.
    """
    # features is [H, W, C]
    data = data / data.max()
    # test11 = np.stack((np.full(data.shape,1),1-data,1-data),axis=2)
    plt.imshow(data)
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
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        _, y, mask = model(x, mask_ratio=mask_ratio)
    x = x.detach().cpu()
    target = target.detach().cpu()
    
    y = y.squeeze(3).detach().cpu()

    # Read the labelmap file
    labelmap_path = '/home/placido.falqueto/IRI_Barcelona/maps/13semantics/labelmap.txt'
    with open(labelmap_path, 'r') as f:
        labelmap = f.readlines()
    labelmap.pop(0)
    labelmap.pop(0)

    colors = []
    # Iterate over each label in the labelmap
    for i, label in enumerate(labelmap):
        label = label.strip().split(':')
        label_color = np.array(label[1].split(','), dtype=int)
        colors.append(label_color)

    sem_map = np.zeros((x.shape[1],x.shape[2],3))
    for i in range(len(colors)):
        sem = np.full((x.shape[1],x.shape[2],3),colors[i])
        sem_map = np.squeeze(np.stack((1-x[:,:,:,i],1-x[:,:,:,i],1-x[:,:,:,i]), axis=3),axis=0)*sem+sem_map
    sem_map = sem_map/255

    is_mae = 0
    if mask is not None:
        # visualize the mask
        mask = mask.detach()
        mask = mask.unsqueeze(-1).repeat(1, 1, model.module.patch_embed.patch_size[0]**2 *1)  # (N, H*W, p*p*3)
        mask = model.module.unpatchify(mask)  # 1 is removing, 0 is keeping
        mask = torch.einsum('nchw->nhwc', mask).detach().cpu()
        mask = mask.squeeze(0)

        # masked image
        msk = torch.tensor(np.full((mask.shape[0],mask.shape[1],3),[255,0,0]))*mask
        im_masked = torch.tensor(sem_map) * (1 - mask) + msk

        is_mae = 1

    # # MAE reconstruction pasted with visible patches
    # im_paste = x * (1 - mask) + y * mask

    # make the plt figure larger
    plt.rcParams['figure.figsize'] = [24, 12] # W: 24, H: 12

    plt.subplot(1, 3+is_mae, 1)
    plt.imshow(sem_map)
    plt.title("semantics", fontsize=23)
    plt.axis('off')

    if is_mae:
        plt.subplot(1, 3+is_mae, 2)
        plt.imshow(im_masked)
        plt.title("masked", fontsize=23)
        plt.axis('off')

    plt.subplot(1, 3+is_mae, 2+is_mae)
    show_image(x[0], target[0], "original")

    plt.subplot(1, 3+is_mae, 3+is_mae)
    show_image(x[0], y[0], "prediction")

    # plt.subplot(1, 4, 3)
    # plt.imshow(target[0])
    # plt.title('target', fontsize=23)
    # plt.axis('off')

    # plt.subplot(1, 4, 4)
    # plt.imshow(y[0])
    # plt.title('y', fontsize=23)
    # plt.axis('off')

    if epoch is not None:
        # if epoch % 5 == 0:
        plt.savefig(f'image_logs/vit_training_{epoch}.png')
        plt.savefig(f'image_logs/vit_training_last.png')
    else:
        plt.show()
    
# -------------------------
# Main Function
# -------------------------
def main(rank, world_size):
    """
    Main entry point for distributed training and evaluation.

    Args:
        rank (int): The rank of the current process within the distributed group.
        world_size (int): The total number of processes in the distributed group.
    """
    os.environ['RANK'] = str(rank)
    os.environ['LOCAL_RANK'] = str(rank)

    args = get_args_parser().parse_args()   # Get command-line arguments

    # Prepare output directory and logging
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    misc.init_distributed_mode(args)        # Initialize distributed environment

    print('job dir: {}'.format(os.path.abspath('')))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # Seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # -------------------------
    # Data Preparation
    # -------------------------
    data_dirs = os.listdir(DATASET_PATH)
    data_dirs.sort()
    random.shuffle(data_dirs)
    # Remove unwanted maps and insert specific ones 
    data_dirs = [f for f in data_dirs if 'gates1' not in f and 'coupa3' not in f]
    data_dirs = ['stanford_coupa3'] + ['stanford_gates1'] + data_dirs


    # if True:
    #     i = 0
    for i in range(len(data_dirs)):
        train_data_dirs = data_dirs[:]
        test_data_dirs = [train_data_dirs.pop(i)] # Pop the i-th map for testing
        random.shuffle(train_data_dirs)
        val_data_dirs = [train_data_dirs.pop(i % len(train_data_dirs)) for i in range(3)]
        
        train_data_dirs = np.sort(train_data_dirs)
        val_data_dirs = np.sort(val_data_dirs)

        model_id = test_data_dirs[0]
        
        print(f'TRAINING MAPS: {train_data_dirs}')
        print(f'VALIDATION MAPS: {val_data_dirs}')
        print(f'TEST MAP: {test_data_dirs}')
        print('-------------------------------------')

        # Create Datasets
        dataset_train = SemanticMapDataset(data_dirs=train_data_dirs)
        dataset_val = SemanticMapDataset(data_dirs=val_data_dirs)
        dataset_test = SemanticMapDataset(data_dirs=test_data_dirs)

        # -------------------------
        # Distributed Setup
        # -------------------------
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
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=None,
            batch_size=args.batch_size,
            num_workers=0,
            pin_memory=False,
            drop_last=True,
        )
        data_loader_test = torch.utils.data.DataLoader(
            dataset_test, 
            batch_size=dataset_test.__len__(), 
            shuffle=False
        )
        features, target = next(iter(data_loader_test))

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

        vlosses = np.arange(1, STOP_TRAINING_AFTER_EPOCHS+1, dtype=float)[::-1]
        for epoch in range(args.start_epoch, args.epochs+1):
            if args.distributed:
                data_loader_train.sampler.set_epoch(epoch)
            train_stats = train_one_epoch(
                model, data_loader_train, data_loader_val,
                optimizer, device, epoch, loss_scaler,
                log_writer=log_writer,
                args=args
            )

            # Update array with new value
            vlosses = np.roll(vlosses, -1)  # Shift elements to the left
            vlosses[-1] = train_stats['val_loss']  # Update the last element
            # check if vlosses stop decreasing
            stopped_decreasing = np.mean(np.diff(vlosses)) > -1e-4 and epoch > 40
            # print(f'vlosses: {vlosses}')
            # print(f'vlosses diff: {np.diff(vlosses)}')
            # print(f'stopped_decreasing: {stopped_decreasing}')

            if epoch % 5 == 0 or epoch == args.epochs:    
                idx = np.random.randint(0, dataset_test.__len__())
                print(f'sample id: {idx}')
                features_test = features[idx]
                target_test = target[idx]
                run_one_image(features_test, target_test, model, epoch)
                misc.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=102, id=model_id, prefix=prfx)
            if args.output_dir and (epoch == args.epochs or stopped_decreasing):
                misc.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch, id=model_id, prefix=prfx)
                break

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