import os
import torch
import shutil
import time
from collections import defaultdict
from typing import Dict, Any, Optional, List
from torch.utils.tensorboard import SummaryWriter

class CheckpointSaver:
    """Checkpoint saver for VQVAE training"""
    
    def __init__(self, save_dir: str, is_master: bool = True):
        self.save_dir = save_dir
        self.is_master = is_master
        self.best_val_loss = float('inf')
        
        if self.is_master:
            os.makedirs(save_dir, exist_ok=True)
    
    def save_checkpoint(
        self,
        epoch: int,
        model_state_dict: Dict[str, torch.Tensor],
        optimizer_state_dict: Dict[str, Any],
        trainer_state: Dict[str, Any],
        val_loss: Optional[float] = None,
        args: Optional[Dict[str, Any]] = None,
        is_best: bool = False,
        is_final: bool = False
    ) -> str:
        """Save checkpoint"""
        if not self.is_master:
            return ""
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': optimizer_state_dict,
            'trainer_state': trainer_state,
            'args': args or {}
        }
        
        if val_loss is not None:
            checkpoint['val_loss'] = val_loss
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(self.save_dir, f'vqvae_epoch_{epoch}.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model if specified
        if is_best and val_loss is not None:
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                best_path = os.path.join(self.save_dir, 'vqvae_best.pth')
                shutil.copy(checkpoint_path, best_path)
                print(f'[checkpoint] new best model saved: {best_path} (val_loss: {val_loss:.6f})')
        
        # Save final model if specified
        if is_final:
            final_path = os.path.join(self.save_dir, 'vqvae_final.pth')
            shutil.copy(checkpoint_path, final_path)
            print(f'[checkpoint] final model saved: {final_path}')
        
        return checkpoint_path
    
    def save_interrupted_checkpoint(
        self,
        epoch: int,
        model_state_dict: Dict[str, torch.Tensor],
        optimizer_state_dict: Dict[str, Any],
        trainer_state: Dict[str, Any],
        args: Optional[Dict[str, Any]] = None
    ) -> str:
        """Save checkpoint when training is interrupted"""
        if not self.is_master:
            return ""
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': optimizer_state_dict,
            'trainer_state': trainer_state,
            'args': args or {},
            'interrupted': True
        }
        
        checkpoint_path = os.path.join(self.save_dir, 'vqvae_interrupted.pth')
        torch.save(checkpoint, checkpoint_path)
        print(f'[checkpoint] interrupted checkpoint saved: {checkpoint_path}')
        
        return checkpoint_path
    
    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """Load checkpoint"""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print(f'[checkpoint] loaded checkpoint from: {checkpoint_path}')
        print(f'[checkpoint] epoch: {checkpoint.get("epoch", "unknown")}')
        
        return checkpoint
    
    def get_latest_checkpoint(self, pattern: str = 'vqvae_epoch_*.pth') -> Optional[str]:
        """Get the latest checkpoint path"""
        import glob
        checkpoints = glob.glob(os.path.join(self.save_dir, pattern))
        if not checkpoints:
            return None
        
        # Sort by modification time (newest first)
        checkpoints.sort(key=os.path.getmtime, reverse=True)
        return checkpoints[0]
    
    def cleanup_old_checkpoints(self, keep_last: int = 5):
        """Clean up old checkpoints, keeping only the most recent ones"""
        if not self.is_master:
            return
        
        import glob
        checkpoints = glob.glob(os.path.join(self.save_dir, 'vqvae_epoch_*.pth'))
        if len(checkpoints) <= keep_last:
            return
        
        # Sort by modification time (oldest first)
        checkpoints.sort(key=os.path.getmtime)
        
        # Remove old checkpoints
        for checkpoint in checkpoints[:-keep_last]:
            os.remove(checkpoint)
            print(f'[checkpoint] removed old checkpoint: {checkpoint}')

class SmoothedValue:
    """Track a series of values and provide access to smoothed values over a window."""
    
    def __init__(self, window_size: int = 20, fmt: str = '{median:.4f} ({global_avg:.4f})'):
        self.deque = []
        self.total = 0.0
        self.count = 0
        self.window_size = window_size
        self.fmt = fmt
    
    def update(self, value: float, n: int = 1):
        self.deque.append(value)
        if len(self.deque) > self.window_size:
            self.deque.pop(0)
        self.count += n
        self.total += value * n
    
    def synchronize_between_processes(self):
        """Warning: does not synchronize the deque!"""
        pass
    
    @property
    def median(self) -> float:
        if not self.deque:
            return 0.0
        d = sorted(self.deque)
        return d[len(d) // 2]
    
    @property
    def avg(self) -> float:
        if not self.deque:
            return 0.0
        d = sorted(self.deque)
        return sum(d) / len(d)
    
    @property
    def global_avg(self) -> float:
        return self.total / self.count if self.count > 0 else 0.0
    
    @property
    def max(self) -> float:
        return max(self.deque) if self.deque else 0.0
    
    @property
    def value(self) -> float:
        return self.deque[-1] if self.deque else 0.0
    
    def __str__(self) -> str:
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )

class MetricLogger:
    def __init__(self, delimiter: str = "\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter
        self.log_iters = set()
        self.iter_time = SmoothedValue(fmt='{avg:.4f}')
        self.data_time = SmoothedValue(fmt='{avg:.4f}')
        self.space = " " * 20
    
    def add_meter(self, name: str, meter: SmoothedValue):
        self.meters[name] = meter
    
    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is not None:
                if k not in self.meters:
                    self.meters[k] = SmoothedValue()
                self.meters[k].update(v)
    
    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")
    
    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(f"{name}: {str(meter)}")
        return self.delimiter.join(loss_str)
    
    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()
    
    def add_meter(self, name: str, meter: SmoothedValue):
        self.meters[name] = meter
    
    def log_every(self, is_master, start_it: int, iters: int, iterable, print_freq: int, header: str = ""):
        """Log every print_freq iterations (only master process prints if is_master=True)"""
        i = start_it
        start_time = time.time()
        end = time.time()
        time_width = 0

        def log_msg():
            return f"{header} {i}/{iters}:" + self.delimiter + str(self)

        for obj in iterable:
            yield i, obj
            i += 1
            if i % print_freq == 0 and is_master:
                print(log_msg())
            if i >= iters:
                break

class TensorboardLogger:
    def __init__(self, log_dir: str, filename_suffix: str = ''):
        """Initialize tensorboard logger"""
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir, filename_suffix=filename_suffix)
        self.step = 0
        self._loggable = True
    
    def set_step(self, step: int):
        """Set current step for logging"""
        self.step = step
    
    def update(self, head: str, **kwargs):
        """Update metrics with head prefix"""
        for key, value in kwargs.items():
            if value is not None:
                self.writer.add_scalar(f'{head}/{key}', value, self.step)
    
    def loggable(self) -> bool:
        """Check if logging is enabled"""
        return self._loggable
    
    def flush(self):
        """Flush writer"""
        self.writer.flush()
    
    def close(self):
        """Close writer"""
        self.writer.close()

class DistLogger:
    """Distributed logger wrapper"""
    def __init__(self, logger):
        self.logger = logger
    
    def set_step(self, step: int):
        if self.logger:
            self.logger.set_step(step)
    
    def update(self, head: str, **kwargs):
        if self.logger:
            self.logger.update(head, **kwargs)
    
    def loggable(self) -> bool:
        return self.logger and self.logger.loggable()
    
    def flush(self):
        if self.logger:
            self.logger.flush()
    
    def close(self):
        if self.logger:
            self.logger.close()
