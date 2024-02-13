# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import math
import sys
from typing import Iterable
import os
import numpy as np

import torch

import util.misc as misc
import util.lr_sched as lr_sched


def train_one_epoch(model: torch.nn.Module,
                    data_loader_train: Iterable, data_loader_val: Iterable,  optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('val_loss', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20 #20
    accum_iter = args.accum_iter

    optimizer.zero_grad()
    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, target) in enumerate(metric_logger.log_every(data_loader_train, print_freq, header)):
        
        model.train(True) # Set model to training mode

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader_train) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        
        with torch.cuda.amp.autocast():
            loss, _, _ = model(samples, target, mask_ratio=args.mask_ratio)

        del samples, target
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)
        
        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        # if os.environ.get('RANK') == '0':
        model.eval() # Set model to evaluate mode
        with torch.no_grad(): # No need to calculate gradients
            sample_val, target_val = next(iter(data_loader_val))
            sample_val = sample_val.to(device, non_blocking=True)
            target_val = target_val.to(device, non_blocking=True)
            val_loss, pred_val, _ = model(sample_val, target_val, mask_ratio=args.mask_ratio)
            val_loss_value = val_loss.item()
            del sample_val, target_val, pred_val
        
        metric_logger.update(val_loss=val_loss_value)
        val_loss_value_reduce = misc.all_reduce_mean(val_loss_value)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader_train) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('train_val_loss', val_loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}