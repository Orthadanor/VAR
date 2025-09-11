import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import time
import warnings
from collections import deque
from contextlib import nullcontext
from functools import partial
from typing import List, Optional, Tuple, Dict, Any

from vqvae.lpips_grayscale import LPIPS
from models.vqvae_grayscale import VQVAEGrayscale
# from utils.data_mri import braTS2d
from torch.utils.tensorboard import SummaryWriter
import json
from datetime import datetime

class VQVAETrainer:
    def __init__(
        self, 
        model: VQVAEGrayscale,
        device: torch.device,
        lr: float = 1e-4,
        beta1: float = 0.9,
        beta2: float = 0.95,
        weight_decay: float = 0.05,
        codebook_weight: float = 1.0,
        perceptual_weight: float = 1.0,
        tb_writer: Optional[SummaryWriter] = None,
        grad_clip: float = 1.0,
        grad_accu: int = 1,
        fp16: bool = False,
        bf16: bool = False,
        zero: bool = False,
        compile_model: bool = False
    ):
        self.model = model.to(device)
        self.device = device
        self.tb_writer = tb_writer
        self.grad_clip = grad_clip
        self.grad_accu = grad_accu
        self.fp16 = fp16
        self.bf16 = bf16
        self.zero = zero
        self.compile_model = compile_model
        
        # Loss weights
        self.codebook_weight = codebook_weight
        self.perceptual_weight = perceptual_weight
        
        # Optimizer setup matching VAR training
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=lr,
            betas=(beta1, beta2),
            weight_decay=weight_decay,
            fused=True
        )
        
        # Loss functions
        self.reconstruction_loss = nn.L1Loss()
        self.perceptual_loss = LPIPS().to(device).eval()
        
        # Compile model if requested
        if compile_model:
            self.model = torch.compile(self.model)
        
        # Gradient accumulation counter
        self.grad_accu_counter = 0
        
        # Speed tracking
        self.speed_ls = deque(maxlen=128)
        self.last_t_perf = time.perf_counter()
        
    def lr_wd_annealing(self, sche_type: str, optimizer, peak_lr: float, wd: float, 
                       cur_it: int, wp_it: int, max_it: int, wp0: float = 0.005, wpe: float = 0.001):
        """Decay the learning rate with half-cycle cosine after warmup"""
        wp_it = round(wp_it)
        
        if cur_it < wp_it:
            cur_lr = wp0 + (1-wp0) * cur_it / wp_it
        else:
            pasd = (cur_it - wp_it) / (max_it-1 - wp_it)   # [0, 1]
            rest = 1 - pasd     # [1, 0]
            if sche_type == 'cos':
                # Fix TypeError by converting 'pasd' to a tensor
                cur_lr = wpe + (1-wpe) * (0.5 + 0.5 * torch.cos(torch.tensor(torch.pi * pasd)))
            elif sche_type == 'lin':
                T = 0.15; max_rest = 1-T
                if pasd < T: cur_lr = 1
                else: cur_lr = wpe + (1-wpe) * rest / max_rest  # 1 to wpe
            elif sche_type == 'lin0':
                T = 0.05; max_rest = 1-T
                if pasd < T: cur_lr = 1
                else: cur_lr = wpe + (1-wpe) * rest / max_rest
            elif sche_type == 'lin00':
                cur_lr = wpe + (1-wpe) * rest
            elif sche_type.startswith('lin'):
                T = float(sche_type[3:]); max_rest = 1-T
                wpe_mid = wpe + (1-wpe) * max_rest
                wpe_mid = (1 + wpe_mid) / 2
                if pasd < T: cur_lr = 1 + (wpe_mid-1) * pasd / T
                else: cur_lr = wpe + (wpe_mid-wpe) * rest / max_rest
            elif sche_type == 'exp':
                T = 0.15; max_rest = 1-T
                if pasd < T: cur_lr = 1
                else:
                    expo = (pasd-T) / max_rest * torch.log(torch.tensor(wpe))
                    cur_lr = torch.exp(expo)
            else:
                raise NotImplementedError(f'unknown sche_type {sche_type}')
        
        cur_lr *= peak_lr
        inf = 1e6
        min_lr, max_lr = inf, -1
        min_wd, max_wd = inf, -1
        
        for param_group in optimizer.param_groups:
            param_group['lr'] = cur_lr * param_group.get('lr_sc', 1)
            max_lr = max(max_lr, param_group['lr'])
            min_lr = min(min_lr, param_group['lr'])
            
            param_group['weight_decay'] = wd * param_group.get('wd_sc', 1)
            max_wd = max(max_wd, param_group['weight_decay'])
            if param_group['weight_decay'] > 0:
                min_wd = min(min_wd, param_group['weight_decay'])

        if min_lr == inf: min_lr = -1
        if min_wd == inf: min_wd = -1
        return min_lr, max_lr, min_wd, max_wd
    
    def backward_clip_step(self, loss: torch.Tensor, stepping: bool = True) -> Tuple[Optional[float], Optional[float]]:
        """Backward pass with gradient clipping and optional stepping"""
        # Scale loss for gradient accumulation
        if self.grad_accu > 1:
            loss = loss / self.grad_accu
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        grad_norm = None
        scale_log2 = None
        
        if stepping:
            # Compute gradient norm
            total_norm = 0
            for p in self.model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)
            grad_norm = total_norm
            
            # Clip gradients
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            
            # Optimizer step
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
        
        return grad_norm, scale_log2
    
    def training_step(
        self, 
        batch: Tuple[torch.Tensor, torch.Tensor], 
        step: int,
        g_it: int,
        stepping: bool = True,
        metric_lg = None,
        logging_params: bool = False,
        tb_lg = None,
        maybe_record_function = nullcontext,
        args = None
    ) -> Dict[str, float]:
        """Single training step with comprehensive logging"""
        
        # Unpack batch
        x, _ = batch  # x: [B, 1, H, W], _: class labels (unused)
        x = x.to(self.device, non_blocking=True)
        
        # Forward pass
        with maybe_record_function('VQVAE_forward'):
            # Ensure model is in training mode
            reconstructed, usages, vq_loss = self.model(x, ret_usages=True)
        
        # Reconstruction loss
        with maybe_record_function('VQVAE_loss'):
            recon_loss = self.reconstruction_loss(reconstructed, x)
            
            # Perceptual loss - convert to [-1, 1] range for LPIPS
            x_normalized = 2.0 * x - 1.0  # Convert from [0,1] to [-1,1]
            reconstructed_normalized = 2.0 * reconstructed - 1.0
            p_loss = self.perceptual_loss(x_normalized, reconstructed_normalized)
            
            # Total loss
            total_loss = recon_loss + self.codebook_weight * vq_loss + self.perceptual_weight * p_loss
        
        # Backward pass
        with maybe_record_function('VQVAE_backward'):
            grad_norm, scale_log2 = self.backward_clip_step(total_loss, stepping=stepping)
        
        # Speed tracking
        if step % 50 == 0:
            self.speed_ls.append((time.perf_counter() - self.last_t_perf) / 50)
            self.last_t_perf = time.perf_counter()
        
        # Logging
        # if metric_lg and (step == 0 or step in getattr(metric_lg, 'log_iters', set())):
        if metric_lg and (step == 0 or step % 10 == 0):

            metric_lg.update(
                Recon=recon_loss.item(),
                VQ=vq_loss.item(),
                Perc=p_loss.item(),
                Total=total_loss.item(),
                gnm=grad_norm,
                usages=usages[0] if usages else 0  # Log usages
            )
        
        # Update tensorboard logging
        if tb_lg and tb_lg.loggable():
            tb_lg.update(head='train_loss', step=g_it,
                        Total=total_loss.item(),
                        Recon=recon_loss.item(),
                        VQ=vq_loss.item(),
                        Perc=p_loss.item())

            if grad_norm is not None:
                tb_lg.update(head='opt_grad', step=g_it, grad_norm=grad_norm)

            # Log codebook usage if available
            if usages:
                for i, usage in enumerate(usages):
                    tb_lg.update(head=f'codebook_usage', step=g_it, 
                               **{f'scale_{i}': usage})
        
        # Debug: Print first few steps to check if values are changing
        if step < 5:
            print(f"[DEBUG step {step}] recon: {recon_loss.item():.6f}, vq: {vq_loss.item():.6f}, perc: {p_loss.item():.6f}, total: {total_loss.item():.6f}")
        
        return {
            'total_loss': total_loss.item(),
            'recon_loss': recon_loss.item(),
            'l1_loss': recon_loss.item(),  # Also return as l1_loss for consistency
            'vq_loss': vq_loss.item(),
            'perceptual_loss': p_loss.item(),
            'grad_norm': grad_norm,
            'perplexity': usages[0] if usages else 0
        }
    
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, float]:
        """Validation step"""
        with torch.no_grad():
            x, _ = batch
            x = x.to(self.device, non_blocking=True)
            
            reconstructed, usages, vq_loss = self.model(x, ret_usages=True)
            recon_loss = self.reconstruction_loss(reconstructed, x)
            
            # Perceptual loss for validation
            x_normalized = 2.0 * x - 1.0
            reconstructed_normalized = 2.0 * reconstructed - 1.0
            p_loss = self.perceptual_loss(x_normalized, reconstructed_normalized)
            
            total_loss = recon_loss + self.codebook_weight * vq_loss + self.perceptual_weight * p_loss
            
        return {
            'val_total_loss': total_loss.item(),
            'val_recon_loss': recon_loss.item(),
            'val_vq_loss': vq_loss.item(),
            'val_perceptual_loss': p_loss.item(),
            'val_perplexity': usages[0] if usages else 0
        }
    
    def state_dict(self) -> Dict[str, Any]:
        """Save trainer state"""
        return {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': {
                'codebook_weight': self.codebook_weight,
                'perceptual_weight': self.perceptual_weight,
                'grad_clip': self.grad_clip,
                'grad_accu': self.grad_accu,
                'fp16': self.fp16,
                'bf16': self.bf16,
                'zero': self.zero,
                'compile_model': self.compile_model
            }
        }
    
    def load_state_dict(self, state: Dict[str, Any], strict: bool = True):
        """Load trainer state"""
        self.model.load_state_dict(state['model_state_dict'], strict=strict)
        self.optimizer.load_state_dict(state['optimizer_state_dict'])
        
        config = state.get('config', {})
        for key, value in config.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def get_config(self) -> Dict[str, Any]:
        """Get trainer configuration"""
        return {
            'codebook_weight': self.codebook_weight,
            'perceptual_weight': self.perceptual_weight,
            'grad_clip': self.grad_clip,
            'grad_accu': self.grad_accu,
            'fp16': self.fp16,
            'bf16': self.bf16,
            'zero': self.zero,
            'compile_model': self.compile_model
        }
