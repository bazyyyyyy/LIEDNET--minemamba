import os
import torch
import yaml

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
import torch.amp

from ptflops import get_model_complexity_info

import time
import numpy as np
import random
from basicsr.transform.data_RGB import get_training_data,get_validation_data2
from basicsr.warmup_scheduler.scheduler import GradualWarmupScheduler
from tqdm import tqdm
from tensorboardX import SummaryWriter

from basicsr.archs.lied_arch import LIED

from basicsr.utils_llie import *

import argparse
parser = argparse.ArgumentParser(description='Hyper-parameters for LIED')
parser.add_argument('--opt', default="./training.yaml", type=str)
args = parser.parse_args()


## Set Seeds
torch.backends.cudnn.benchmark = True
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)

## Load yaml configuration file
yaml_file = args.opt

with open(yaml_file, 'r') as config:
    opt = yaml.safe_load(config)
print("load training yaml file: %s"%(yaml_file))

Train = opt['TRAINING']
OPT = opt['OPTIM']

# wandb.init(project="LLMamba", name="LLMamba_LOLv1")

## Build Model
print('==> Build the model')
device = torch.device('cuda')
model_restored = LIED()
model_restored = model_restored.cuda()#将模型移至cuda
macs, params = get_model_complexity_info(model_restored, (3, 256, 256), as_strings=True, print_per_layer_stat=False, verbose=False)

SOBEL_KX = torch.tensor([[1, 0, -1],
                         [2, 0, -2],
                         [1, 0, -1]], dtype=torch.float32).view(1, 1, 3, 3)
SOBEL_KY = SOBEL_KX.transpose(2, 3)
EDGE_LOSS_WEIGHT = Train.get('EDGE_LOSS_WEIGHT', 0.05) if 'Train' in globals() else 0.05

def gradient_edge_loss(pred, target):
    """计算预测与GT的梯度L1损失，强化边缘细节。"""
    c = pred.shape[1]
    device = pred.device
    sobel_x = SOBEL_KX.to(device).repeat(c, 1, 1, 1)
    sobel_y = SOBEL_KY.to(device).repeat(c, 1, 1, 1)
    grad_pred_x = F.conv2d(pred, sobel_x, padding=1, groups=c)
    grad_pred_y = F.conv2d(pred, sobel_y, padding=1, groups=c)
    grad_gt_x = F.conv2d(target, sobel_x, padding=1, groups=c)
    grad_gt_y = F.conv2d(target, sobel_y, padding=1, groups=c)
    grad_pred = torch.sqrt(grad_pred_x ** 2 + grad_pred_y ** 2 + 1e-6)
    grad_gt = torch.sqrt(grad_gt_x ** 2 + grad_gt_y ** 2 + 1e-6)
    return F.l1_loss(grad_pred, grad_gt)

## Training model path direction
mode = opt['MODEL']['MODE']

model_dir = os.path.join(Train['SAVE_DIR'], mode, 'models')
mkdir(model_dir)
train_dir = Train['TRAIN_DIR']
val_dir = Train['VAL_DIR']

## GPU
gpus = ','.join([str(i) for i in opt['GPU']])
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpus
print(gpus)
device_ids = [i for i in range(torch.cuda.device_count())]
print(device_ids)
if torch.cuda.device_count() > 1:
    print("\n\nLet's use", torch.cuda.device_count(), "GPUs!\n\n")
if len(device_ids) > 1:
    model_restored = nn.DataParallel(model_restored, device_ids=device_ids)
    # model_restored.cuda()
## Optimizer
start_epoch = 1
new_lr = float(OPT['LR_INITIAL'])
optimizer = optim.Adam(model_restored.parameters(), lr=new_lr, betas=(0.9, 0.999), eps=1e-8)

## Scheduler (Strategy)
warmup_epochs = 3
scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, OPT['EPOCHS'] - warmup_epochs,
                                                        eta_min=float(OPT['LR_MIN']))
scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
scheduler.step()

## Resume (Continue training by a pretrained model)
if Train['RESUME']:
    path_chk_rest = get_last_path(model_dir, '_latest.pth')
    load_checkpoint(model_restored, path_chk_rest)
    start_epoch = load_start_epoch(path_chk_rest) + 1
    load_optim(optimizer, path_chk_rest)

    for i in range(1, start_epoch):
        scheduler.step()
    new_lr = scheduler.get_lr()[0]
    print('------------------------------------------------------------------')
    print("==> Resuming Training with learning rate:", new_lr)
    print('------------------------------------------------------------------')

## Loss
Charloss = nn.SmoothL1Loss()

## Mixed Precision Training (AMP)
scaler = GradScaler()
print('==> Mixed Precision Training (AMP) enabled')

## DataLoaders
print('==> Loading datasets')
train_dataset = get_training_data(train_dir, {'patch_size': Train['TRAIN_PS']})
train_loader = DataLoader(dataset=train_dataset, batch_size=OPT['BATCH'],
                          shuffle=True, num_workers=8, drop_last=False)

val_dataset = get_validation_data2(val_dir, {'patch_size': Train['VAL_PS']})
val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=0,
                        drop_last=False)

# Show the training configuration
print(f'''==> Training details:
------------------------------------------------------------------
    Restoration mode:   {mode}
    Train patches size: {str(Train['TRAIN_PS']) + 'x' + str(Train['TRAIN_PS'])}
    Val patches size:   {str(Train['VAL_PS']) + 'x' + str(Train['VAL_PS'])}
    Model parameters:   {params}
    Start/End epochs:   {str(start_epoch) + '~' + str(OPT['EPOCHS'])}
    Batch sizes:        {OPT['BATCH']}
    Learning rate:      {OPT['LR_INITIAL']}
    GPU:                {'GPU' + str(device_ids)}
    Model FLOPs:        {macs}
    Mixed Precision:    Enabled (AMP)''')
print('------------------------------------------------------------------')

# Start training!
print('==> Training start: ')
best_psnr = 0
best_ssim = 0
best_epoch_psnr = 0
best_epoch_ssim = 0
total_start_time = time.time()

## Log
log_dir = os.path.join(Train['SAVE_DIR'], mode, 'log')
mkdir(log_dir)
writer = SummaryWriter(log_dir=log_dir, filename_suffix=f'_{mode}')

#for epoch in range(start_epoch, OPT['EPOCHS'] + 1):
for epoch in range(start_epoch, OPT['EPOCHS'] + 1):
    epoch_start_time = time.time()
    epoch_loss = 0
    train_id = 1

    model_restored.train()
    for i, data in enumerate(tqdm(train_loader), 0):
        # Forward propagation
        for param in model_restored.parameters():
            param.grad = None
        target = data[0].cuda()
        input_ = data[1].cuda()
        #print(f"Input size at iteration {i}: {input_.size()}")
        
        # Mixed Precision Training: forward pass with autocast
        with torch.amp.autocast(device_type='cuda'):
            restored = model_restored(input_)
            # Compute loss
            loss = Charloss(restored, target)
            edge_loss = gradient_edge_loss(restored, target)
            total_loss = loss + EDGE_LOSS_WEIGHT * edge_loss

        # Back propagation with gradient scaling
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        epoch_loss += total_loss.item()
        
        # Clear cache periodically to free up memory
        if i % 10 == 0:
            torch.cuda.empty_cache()

    ## Evaluation (Validation)
    if epoch % Train['VAL_AFTER_EVERY'] == 0:
        model_restored.eval()
        psnr_val_rgb = []
        ssim_val_rgb = []
        for ii, data_val in enumerate(val_loader, 0):
            target = data_val[0].cuda()
            input_ = data_val[1].cuda()
            h, w = target.shape[2], target.shape[3]
            with torch.no_grad():
                # Use autocast for validation as well (optional, but recommended)
                with torch.amp.autocast(device_type='cuda'):
                    restored = model_restored(input_)
                restored = restored[:, :, :h, :w]

            for res, tar in zip(restored, target):
                psnr_val_rgb.append(torchPSNR(res, tar))
                ssim_val_rgb.append(torchSSIM(restored, target))

        psnr_val_rgb = torch.stack(psnr_val_rgb).mean().item()
        ssim_val_rgb = torch.stack(ssim_val_rgb).mean().item()

        # Save the best PSNR model of validation
        if psnr_val_rgb > best_psnr:
            best_psnr = psnr_val_rgb
            best_epoch_psnr = epoch
            torch.save({'epoch': epoch,
                        'state_dict': model_restored.state_dict(),
                        'optimizer': optimizer.state_dict()
                        }, os.path.join(model_dir, "model_bestPSNR.pth"))
        print("[epoch %d PSNR: %.4f --- best_epoch %d Best_PSNR %.4f]" % (
            epoch, psnr_val_rgb, best_epoch_psnr, best_psnr))

        # Save the best SSIM model of validation
        if ssim_val_rgb > best_ssim:
            best_ssim = ssim_val_rgb
            best_epoch_ssim = epoch
            torch.save({'epoch': epoch,
                        'state_dict': model_restored.state_dict(),
                        'optimizer': optimizer.state_dict()
                        }, os.path.join(model_dir, "model_bestSSIM.pth"))
        print("[epoch %d SSIM: %.4f --- best_epoch %d Best_SSIM %.4f]" % (
            epoch, ssim_val_rgb, best_epoch_ssim, best_ssim))

        """
        # Save evey epochs of model
        torch.save({'epoch': epoch,
                    'state_dict': model_restored.state_dict(),
                    'optimizer': optimizer.state_dict()
                    }, os.path.join(model_dir, f"model_epoch_{epoch}.pth"))
        """
        # wandb.log({"epoch": epoch, "PSNR": psnr_val_rgb, "SSIM": ssim_val_rgb})
        writer.add_scalar('val/PSNR', psnr_val_rgb, epoch)
        writer.add_scalar('val/SSIM', ssim_val_rgb, epoch)
    scheduler.step()

    print("------------------------------------------------------------------")
    print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(epoch, time.time() - epoch_start_time,
                                                                              epoch_loss, scheduler.get_lr()[0]))
    print("------------------------------------------------------------------")

    # Save the last model
    torch.save({'epoch': epoch,
                'state_dict': model_restored.state_dict(),
                'optimizer': optimizer.state_dict()
                }, os.path.join(model_dir, "model_latest.pth"))

    writer.add_scalar('train/loss', epoch_loss, epoch)
    writer.add_scalar('train/lr', scheduler.get_lr()[0], epoch)
writer.close()

total_finish_time = (time.time() - total_start_time)  # seconds
total_hours = total_finish_time / 60 / 60
print('Total training time: {:.1f} hours'.format(total_hours))
print('Training completed at: {}'.format(time.strftime('%Y-%m-%d %H:%M:%S')))

# wandb.finish()