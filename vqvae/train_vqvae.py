import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import sys
import time
import warnings
import glob
import math
import shutil
from collections import deque
from contextlib import nullcontext
from functools import partial
from typing import List, Optional, Tuple, Dict, Any

import argparse
from datetime import datetime
import json

from vqvae_trainer import VQVAETrainer
# from tensorboard_logger import TensorboardLogger, DistLogger
# from metric_logger import MetricLogger, SmoothedValue
# from checkpoint_saver import CheckpointSaver
from misc import *
from models.vqvae_grayscale import VQVAEGrayscale
from utils.data_mri import braTS2d

def time_str(fmt: str = "%m%d_%H%M") -> str:
    """Get current time string"""
    return datetime.now().strftime(fmt)

def maybe_auto_resume(save_dir: str, pattern: str = 'vqvae_epoch_*.pth') -> Tuple[List[str], int, int, str, Dict[str, Any]]:
    """Auto-resume from checkpoint if available"""
    info = []
    resume = None
    
    # Check for checkpoints
    all_ckpt = sorted(glob.glob(os.path.join(save_dir, pattern)), key=os.path.getmtime, reverse=True)
    
    if len(all_ckpt) == 0:
        info.append(f'[auto_resume] no ckpt found @ {pattern}')
        info.append(f'[auto_resume quit]')
    else:
        resume = all_ckpt[0]
        info.append(f'[auto_resume] auto load from @ {resume} ...')
        info.append(f'[auto_resume quit]')
    
    if resume is None:
        return info, 0, 0, '[no acc str]', {}
    
    try:
        ckpt = torch.load(resume, map_location='cpu')
    except Exception as e:
        info.append(f'[auto_resume] failed, {e} @ {resume}')
        return info, 0, 0, '[no acc str]', {}
    
    ep, it = (ckpt['epoch'], ckpt.get('global_step', 0))
    acc_str = ckpt.get('acc_str', '[no acc str]')
    trainer_state = ckpt.get('trainer_state', {})
    
    info.append(f'[auto_resume success] resume from ep{ep}, it{it}, acc_str: {acc_str}')
    return info, ep, it, acc_str, trainer_state

def create_tb_logger(save_dir: str, is_master: bool = True) -> DistLogger:
    """Create tensorboard logger"""
    if is_master:
        tb_log_dir = os.path.join(save_dir, 'tensorboard_logs')
        os.makedirs(tb_log_dir, exist_ok=True)
        tb_lg = DistLogger(TensorboardLogger(
            log_dir=tb_log_dir, 
            filename_suffix=f'_{time_str("%m%d_%H%M")}'
        ))
        tb_lg.flush()
    else:
        tb_lg = DistLogger(None)
    return tb_lg

def build_things_from_args(args):
    """Build model, trainer, and data loaders from arguments"""
    # Auto-resume setup
    auto_resume_info, start_ep, start_it, acc_str, trainer_state = maybe_auto_resume(args.save_dir)
    
    # Create tensorboard logger
    tb_lg = create_tb_logger(args.save_dir, is_master=True)
    print(f'initial args:\n{str(args)}')
    
    if start_ep == args.epochs:
        print(f'[VQVAE] Training finished ({acc_str}), skipping ...\n\n')
        return args, tb_lg
    
    # Build data loaders
    print(f'[build data] ...\n')
    # num_classes, train_set, val_set = build_mri_dataset_grayscale(
    #     data_path=args.data_path,
    #     final_reso=args.final_reso,
    #     hflip=args.hflip
    # )
    num_classes, train_set, val_set = braTS2d(
        data_path=args.data_path,
        crop_shape=(128, 128),
        train_split=0.95
    )
    
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        dataset=val_set,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    iters_train = len(train_loader)
    print(f'[dataloader] batch_size={args.batch_size}, iters_train={iters_train}')
    
    # Build model
    print(f'[build model] ...\n')
    model = VQVAEGrayscale(
        vocab_size=args.vocab_size,
        z_channels=args.z_channels,
        ch=args.ch,
        beta=args.beta,
        test_mode=False,
        share_quant_resi=args.share_quant_resi,
        v_patch_nums=args.v_patch_nums
    )
    
    # Build trainer
    trainer = VQVAETrainer(
        model=model,
        device=args.device,
        lr=args.lr,
        beta1=args.beta1,
        beta2=args.beta2,
        weight_decay=args.weight_decay,
        codebook_weight=args.codebook_weight,
        perceptual_weight=args.perceptual_weight,
        tb_writer=tb_lg.logger.writer if tb_lg.logger else None,
        grad_clip=args.grad_clip,
        grad_accu=args.grad_accu,
        fp16=args.fp16,
        bf16=args.bf16,
        zero=args.zero,
        compile_model=args.compile_model
    )
    
    # Load trainer state if resuming
    if trainer_state:
        trainer.load_state_dict(trainer_state, strict=False)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'[model] total params: {total_params/1e6:.2f}M, trainable: {trainable_params/1e6:.2f}M')
    print(f'[model] downsample factor: {model.downsample}×')
    
    [print(l) for l in auto_resume_info]
    
    return (
        tb_lg, trainer,
        start_ep, start_it, acc_str, iters_train, train_loader, val_loader
    )

def train_one_ep(
    ep: int, 
    is_first_ep: bool, 
    start_it: int, 
    args, 
    tb_lg: DistLogger, 
    train_loader: DataLoader, 
    iters_train: int, 
    trainer: VQVAETrainer,
    logging_params_milestone: List[int]
):
    """Train one epoch with comprehensive logging"""
    
    step_cnt = 0
    me = MetricLogger(delimiter='  ')
    
    # Add meters for different metrics
    [me.add_meter(x, SmoothedValue(window_size=1, fmt='{value:.2g}')) for x in ['lr']]
    [me.add_meter(x, SmoothedValue(window_size=1, fmt='{median:.2f} ({global_avg:.2f})')) for x in ['gnm']]
    
    header = f'[Ep]: [{ep:4d}/{args.epochs}]'
    
    if is_first_ep:
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        warnings.filterwarnings('ignore', category=UserWarning)
    
    # Learning rate scheduling setup
    g_it = ep * iters_train
    wp_it = args.warmup_epochs * iters_train
    max_it = args.epochs * iters_train
    
    # Profiling setup
    doing_profiling = args.prof and is_first_ep
    maybe_record_function = torch.profiler.record_function if doing_profiling else nullcontext
    
    profiler = None
    if doing_profiling:
        profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(
                wait=40,
                warmup=3,
                active=2,
                repeat=1,
            ),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        )
        profiler.start()
    
    # Speed tracking
    last_t_perf = time.perf_counter()
    speed_ls: deque = deque(maxlen=128)
    FREQ = min(50, iters_train//2-1)
    
    for it, batch in me.log_every(start_it, iters_train, train_loader, max(10, iters_train // 1000), header):
        if it < start_it: 
            continue
        if is_first_ep and it == start_it: 
            warnings.resetwarnings()
        
        if doing_profiling: 
            profiler.step()
        
        # Learning rate scheduling
        g_it = ep * iters_train + it
        min_lr, max_lr, min_wd, max_wd = trainer.lr_wd_annealing(
            args.sche, trainer.optimizer, args.lr, args.weight_decay, 
            g_it, wp_it, max_it, wp0=args.wp0, wpe=args.sche_end
        )
        
        stepping = (g_it + 1) % args.grad_accu == 0
        step_cnt += int(stepping)
        
        # Training step - tensorboard logging ('train_loss' and 'codebook_usage')
        metrics = trainer.training_step(
            batch=batch,
            step=it,
            g_it=g_it,
            stepping=stepping,
            metric_lg=me,
            logging_params=stepping and step_cnt == 1 and (ep < 4 or ep in logging_params_milestone),
            tb_lg=tb_lg,
            maybe_record_function=maybe_record_function,
            args=args
        )
        
        # Speed tracking
        if (it+1) % FREQ == 0:
            speed_ls.append((time.perf_counter()-last_t_perf)/FREQ)
            iter_speed = float(sum(speed_ls) / len(speed_ls))
            img_per_sec = args.batch_size / iter_speed
            last_t_perf = time.perf_counter()
            
            if tb_lg.loggable():
                tb_lg.update(head='Profiling/speed', iter_cost=iter_speed, img_per_sec=img_per_sec)
                tb_lg.update(head='PT_opt_lr/lr_max', sche_lr=max_lr)
                tb_lg.update(head='PT_opt_lr/lr_min', sche_lr=min_lr)
                # tb_lg.update(head='PT_opt_wd/wd_max', sche_wd=max_wd)
                # tb_lg.update(head='PT_opt_wd/wd_min', sche_wd=min_wd)
        
        # Update metric logger
        me.update(lr=max_lr)
        tb_lg.set_step(step=g_it)
    
    me.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in me.meters.items()}

def main():
    parser = argparse.ArgumentParser(description='Train VQVAE with vaex-style features')
    
    # Data arguments
    parser.add_argument('--data_path', type=str, required=True, 
                       help='Path to MRI data')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--final_reso', type=int, default=128)
    parser.add_argument('--hflip', action='store_true', help='Enable horizontal flip')
    
    # Model arguments
    parser.add_argument('--vocab_size', type=int, default=128)
    parser.add_argument('--z_channels', type=int, default=8)
    parser.add_argument('--ch', type=int, default=8)
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--share_quant_resi', type=int, default=4)
    parser.add_argument('--v_patch_nums', nargs='+', type=int, default=[1, 2, 4, 8])
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.95)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--codebook_weight', type=float, default=1.0)
    parser.add_argument('--perceptual_weight', type=float, default=1.0)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--grad_accu', type=int, default=1)
    
    # Learning rate scheduling
    parser.add_argument('--sche', type=str, default='cos', choices=['cos', 'lin', 'lin0', 'lin00', 'exp'])
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--wp0', type=float, default=0.005)
    parser.add_argument('--sche_end', type=float, default=0.001)
    
    # Optimization arguments
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--bf16', action='store_true')
    parser.add_argument('--zero', action='store_true')
    parser.add_argument('--compile_model', action='store_true')
    
    # Logging and saving
    parser.add_argument('--save_dir', type=str, default='./local_output/vqvae_checkpoints')
    parser.add_argument('--val_freq', type=int, default=20)
    parser.add_argument('--prof', action='store_true', help='Enable profiling')
    
    args = parser.parse_args()
    
    # Set device
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {args.device}")
    
    # Convert v_patch_nums to tuple
    args.v_patch_nums = tuple(args.v_patch_nums)
    
    # Build everything
    ret = build_things_from_args(args)
    if len(ret) < 8:
        return ret
    
    (
        tb_lg, trainer,
        start_ep, start_it, acc_str, iters_train, train_loader, val_loader
    ) = ret
    
    # Training setup
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Initialize checkpoint saver
    saver = CheckpointSaver(args.save_dir, is_master=True)
    
    # Logging milestones
    logging_params_milestone = list(range(1, args.epochs, 10)) + [args.epochs - 1]
    
    # Training loop
    start_time = time.time()
    min_total_loss = float('inf')
    
    print(f'[training] starting from epoch {start_ep}, iteration {start_it}')
    print(f'[training] total epochs: {args.epochs}, iterations per epoch: {iters_train}')
    
    try:
        for ep in range(start_ep, args.epochs):
            print(f'\n[epoch {ep+1}/{args.epochs}] starting...')
            
            # Train one epoch ('train_loss' logged on tensorboard)
            stats = train_one_ep(
                ep=ep,
                is_first_ep=(ep == start_ep),
                start_it=start_it if ep == start_ep else 0,
                args=args,
                tb_lg=tb_lg,
                train_loader=train_loader,
                iters_train=iters_train,
                trainer=trainer,
                logging_params_milestone=logging_params_milestone
            )
            
            # Validation - Specify validation frequency (default every epoch)
            if (ep + 1) % 1 == 0 or (ep + 1) == args.epochs:
                print(f'[epoch {ep+1}] running validation...')
                trainer.model.eval()
                val_metrics = []
                
                with torch.no_grad():
                    for batch in val_loader:
                        metrics = trainer.validation_step(batch)
                        val_metrics.append(metrics)
                
                # Calculate average validation metrics
                avg_val_loss = sum(m['val_total_loss'] for m in val_metrics) / len(val_metrics)
                avg_val_recon = sum(m['val_recon_loss'] for m in val_metrics) / len(val_metrics)
                avg_val_vq = sum(m['val_vq_loss'] for m in val_metrics) / len(val_metrics)
                avg_val_perceptual = sum(m['val_perceptual_loss'] for m in val_metrics) / len(val_metrics)
                
                # Log validation metrics
                tb_lg.set_step(step=(ep + 1) * iters_train)
                tb_lg.update(head='val_loss', 
                           Total=avg_val_loss,
                           Recon=avg_val_recon,
                           VQ=avg_val_vq,
                           Perc=avg_val_perceptual)
                
                print(f'[epoch {ep+1}] validation - Total: {avg_val_loss:.6f}, Recon: {avg_val_recon:.6f}, VQ: {avg_val_vq:.6f}, Perc: {avg_val_perceptual:.6f}')
                
                # Save best model
                if avg_val_loss < min_total_loss:
                    min_total_loss = avg_val_loss
                    saver.save_checkpoint(
                        epoch=ep + 1,
                        model_state_dict=trainer.model.state_dict(),
                        optimizer_state_dict=trainer.optimizer.state_dict(),
                        trainer_state=trainer.state_dict(),
                        val_loss=avg_val_loss,
                        args=vars(args),
                        is_best=True
                    )
            
            # Save checkpoint every 10 epochs
            if (ep + 1) % 10 == 0:
                saver.save_checkpoint(
                    epoch=ep + 1,
                    model_state_dict=trainer.model.state_dict(),
                    optimizer_state_dict=trainer.optimizer.state_dict(),
                    trainer_state=trainer.state_dict(),
                    args=vars(args)
                )
            
            # Reset start_it after first epoch
            if ep == start_ep:
                start_it = 0
        
        # Save final model
        saver.save_checkpoint(
            epoch=args.epochs,
            model_state_dict=trainer.model.state_dict(),
            optimizer_state_dict=trainer.optimizer.state_dict(),
            trainer_state=trainer.state_dict(),
            args=vars(args),
            is_final=True
        )
        
        # Save model configuration to a JSON file
        model_config = {
            "vocab_size": args.vocab_size,
            "z_channels": args.z_channels,
            "ch": args.ch,
            "v_patch_nums": args.v_patch_nums
        }
        config_path = os.path.join(args.save_dir, "model_config.json")
        with open(config_path, "w") as f:
            json.dump(model_config, f, indent=4)
        print(f"Model configuration saved to {config_path}")
        
        total_time = time.time() - start_time
        print(f'\n🎉 Training completed!')
        print(f'Total time: {total_time/3600:.2f} hours')
        print(f'Final model saved: {os.path.join(args.save_dir, "vqvae_final.pth")}')
        print(f'Best model saved: {os.path.join(args.save_dir, "vqvae_best.pth")}')
        
    except KeyboardInterrupt:
        print('\n⚠️ Training interrupted by user')
        # Save interrupted checkpoint
        saver.save_interrupted_checkpoint(
            epoch=ep + 1,
            model_state_dict=trainer.model.state_dict(),
            optimizer_state_dict=trainer.optimizer.state_dict(),
            trainer_state=trainer.state_dict(),
            args=vars(args)
        )
    
    finally:
        tb_lg.flush()
        tb_lg.close()

if __name__ == '__main__':
    main()

# Example usage:
# python train_vqvae_vaex_style.py \
#     --data_path /home/yuchenliu/Dataset/IXI/train_val_test_split_single_slice \
#     --batch_size 64 \
#     --epochs 100 \
#     --final_reso 128 \
#     --vocab_size 128 \
#     --z_channels 8 \
#     --ch 8 \
#     --lr 1e-4 \
#     --sche cos \
#     --warmup_epochs 5 \
#     --save_dir ./local_output/vqvae_ixi_v128_z8_ch8
