import datetime
import numpy as np
import os
import time
from pathlib import Path
import wandb

import torch
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from torch import distributions
import torchvision.transforms as transforms
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import destroy_process_group
from einops import rearrange

from util.frame_triplet_loader import Frameloader
from util.transforms import RandomApplyTransform, RandomShiftsAug, GaussianNoise
from util.util import ddp_setup, save_checkpoint, load_checkpoint, get_args_parser
from reward_model import Model


def main(args): 
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # Login to wandb for logging
    if args.wandb_key:
        wandb.login(key=args.wandb_key)
        wandb.init(project='video2reward', group='ddp')

    # Image augmentation: color jitter, gaussian noise, random shifts
    if not args.augment:
        transform_train = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    else:
        transform_train = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            RandomApplyTransform([
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                GaussianNoise(mean=0, std=(0.01, 0.1)),
                RandomShiftsAug(pad=4)
            ]),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    # Initialize dataset
    dataset_train = Frameloader(
        root_dir=args.data_path, 
        test=False,
        transforms=transform_train, 
        normalize_trajectory=args.normalize_prediction,
        subset_len=8,
        randomized=args.randomize)
    if args.eval_path != '': 
        dataset_eval = Frameloader(
            root_dir=args.data_path, 
            test=False,
            transforms=transform_train, 
            normalize_trajectory=args.normalize_prediction,
            subset_len=8,
            randomized=args.randomize)

    # Initialize sampler for distributed training
    if args.distributed:
        ddp_setup()
        sampler_train = torch.utils.data.DistributedSampler(dataset_train, shuffle=True)
        print('Sampler_train = %s' % str(sampler_train))
    else:
        print('single gpu mode')
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    # Initialize dataloader
    data_loader_train = DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        persistent_workers = True
    )
    if args.eval_path != '':
        data_loader_eval = DataLoader(
            dataset_eval,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
            drop_last=False,
            persistent_workers=True
        )

    # Initialize model and optimizer
    device_id = int(os.environ['LOCAL_RANK'])
    model = Model(model_type='resnet18', latent_dim=512)
    model.to(device_id)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

    # Load checkpoint if resuming training
    if args.resume == 'last':
        model, optimizer, epoch  = load_checkpoint(args.output_dir, model, optimizer, args.resume)
        args.start_epoch = epoch

    # Initialize distributed model
    if args.distributed:
        model = DDP(model, device_ids=[device_id])
        
    # Training loop
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        model.train()
        data_loader_train.sampler.set_epoch(epoch)

        # Batch loop
        epoch_loss = []
        epoch_mean_error = []
        for cur_batch, cur_lbl, delta in data_loader_train:
            cur_batch, cur_lbl, delta = cur_batch.to(device_id), cur_lbl.to(device_id), delta.to(device_id)
            # Dimensions: (batch_size, num_frames, channels, height, width)
            # True batch size is batch_size * num_frames
            cur_batch = rearrange(cur_batch, 'b i c h w -> (b i) c h w')
            cur_lbl = rearrange(cur_lbl, 'b i -> (b i)')
            delta = rearrange(delta,'b i -> (b i)')
            optimizer.zero_grad()    

            # Predicted distribution
            pred_dist = model(cur_batch)

            # Ground truth distribution
            # 'delta' is the number of frames between the initial and goal frames
            # We scale the standard deviation by delta to account for the difference in time; more frames = smaller std
            # When args.normalize_prediction is False (i.e. wno normalizing prediction 0-1), delta is 1 for all frames
            gt_std = (torch.ones(cur_lbl.shape[0]).to(device_id) / delta).unsqueeze(-1)
            gt_mean = cur_lbl.unsqueeze(-1)
            lbl = distributions.Normal(gt_mean, gt_std)
            
            # Loss calculation
            loss = distributions.kl.kl_divergence(lbl, pred_dist).sum()
            mean_error = (torch.abs(pred_dist.mean - gt_mean)).mean().item()
            epoch_loss.append(loss.item())
            epoch_mean_error.append(mean_error)
            
            loss.backward()
            optimizer.step()
        
        # Evaluation loop
        if args.eval_path != '' and (epoch % args.eval_every == 0 or epoch == args.epochs - 1):
            eval_loss = []
            eval_mean_error = []
            with torch.no_grad():
                model.eval()
                for cur_batch, cur_lbl, delta in data_loader_eval:
                    # Same as training loop without backpropagation
                    cur_batch, cur_lbl, delta = cur_batch.to(device_id), cur_lbl.to(device_id), delta.to(device_id)
                    cur_batch = rearrange(cur_batch, 'b i c h w -> (b i) c h w')
                    cur_lbl = rearrange(cur_lbl, 'b i -> (b i)')
                    delta = rearrange(delta,'b i -> (b i)')
                    
                    pred_dist = model(cur_batch)
                    gt_std = (torch.ones(cur_lbl.shape[0]).to(device_id) / delta).unsqueeze(-1)
                    gt_mean = cur_lbl.unsqueeze(-1)
                    lbl = distributions.Normal(gt_mean, gt_std)

                    loss = distributions.kl.kl_divergence(lbl, pred_dist).sum()
                    mean_error = (torch.abs(pred_dist.mean - gt_mean)).mean().item()
                    eval_loss.append(loss.item())
                    eval_mean_error.append(mean_error)

                # Log training and evaluation metrics to wandb
                if args.wandb_key:
                    wandb.log({'Loss': np.array(epoch_loss).mean(),
                               'Mean Error': np.array(epoch_mean_error).mean(),
                               'Eval Loss': np.array(eval_loss).mean(),
                               'Eval Mean Error': np.array(eval_mean_error).mean()})
                    
        # If not evaluating, log only training metrics to wandb 
        elif args.wandb_key:
            wandb.log({'Loss': np.array(epoch_loss).mean(),
                       'Mean Error': np.array(epoch_mean_error).mean()})
    
        # Update learning rate
        scheduler.step()
            
        # Save model checkpoint
        if epoch % args.save_every == 0 or epoch == args.epochs - 1:
            if args.distributed:
                if device_id == 0:
                    save_checkpoint(model.module, epoch, args.output_dir, optimizer)
            else:
                save_checkpoint(model, epoch, args.output_dir, optimizer)
            
    # Print total training time
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    destroy_process_group()

    
if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
