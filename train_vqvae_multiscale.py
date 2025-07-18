import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import sys
sys.path.append('/home/yuchenliu/VAR')  # Add VAR to path

# Add distributed training support
import torch.distributed as dist

# Add tqdm for progress bars
from tqdm import tqdm

from models.vqvae_grayscale import VQVAEGrayscale
from utils.data_mri import build_mri_dataset_grayscale
import argparse
from torch.utils.tensorboard import SummaryWriter

def init_distributed():
    """Initialize distributed training for single GPU"""
    if not dist.is_initialized():
        # Initialize for single GPU
        os.environ.setdefault('MASTER_ADDR', 'localhost')
        os.environ.setdefault('MASTER_PORT', '12345')
        os.environ.setdefault('RANK', '0')
        os.environ.setdefault('WORLD_SIZE', '1')
        
        dist.init_process_group(
            backend='nccl' if torch.cuda.is_available() else 'gloo',
            init_method='env://',
            rank=0,
            world_size=1
        )
        print("Initialized distributed training for single GPU")

def cleanup_distributed():
    """Cleanup distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()

class VQVAETrainer:
    def __init__(self, model, device, lr=1e-4, beta1=0.9, beta2=0.95, weight_decay=0.05, 
                 codebook_weight=1.0, tb_writer=None):
        self.model = model.to(device)
        self.device = device
        self.tb_writer = tb_writer
        
        # Optimizer setup matching VAR training
        self.optimizer = optim.AdamW(  # Changed to AdamW
            self.model.parameters(),
            lr=lr, 
            betas=(beta1, beta2),      # (0.9, 0.95) like VAR
            weight_decay=weight_decay, # 0.05 like VAR
            fused=True                 # Enable fused AdamW if available
        )
        
        # Loss functions
        self.reconstruction_loss = nn.L1Loss()  # Simplified to direct mean
        self.codebook_weight = codebook_weight
        
    def training_step(self, batch, step):
        self.optimizer.zero_grad()
        
        # Unpack batch from MRIDatasetGrayscale
        x, _ = batch  # x: [B, 1, H, W], _: class labels (unused)
        x = x.to(self.device)
        
        # Forward pass
        reconstructed, usages, vq_loss = self.model(x, ret_usages=True)
        
        # Reconstruction loss
        recon_loss = self.reconstruction_loss(reconstructed, x)
        
        # Total loss
        total_loss = recon_loss + self.codebook_weight * vq_loss
        
        total_loss.backward()
        self.optimizer.step()
        
        # Log to tensorboard
        if self.tb_writer and step % 50 == 0:
            self.tb_writer.add_scalar('Train/Total_Loss', total_loss.item(), step)
            self.tb_writer.add_scalar('Train/Reconstruction_Loss', recon_loss.item(), step)
            self.tb_writer.add_scalar('Train/VQ_Loss', vq_loss.item(), step)
            self.tb_writer.add_scalar('Train/Learning_Rate', self.optimizer.param_groups[0]['lr'], step)
            
            # Log codebook usage if available
            if usages:
                for i, usage in enumerate(usages):
                    self.tb_writer.add_scalar(f'Train/Codebook_Usage_Scale_{i}', usage, step)
        
        return {
            'total_loss': total_loss.item(),
            'recon_loss': recon_loss.item(),
            'vq_loss': vq_loss.item(),
            'perplexity': usages[0] if usages else 0  # First scale usage as proxy
        }
    
    def validation_step(self, batch):
        with torch.no_grad():
            x, _ = batch
            x = x.to(self.device)
            reconstructed, usages, vq_loss = self.model(x, ret_usages=True)
            recon_loss = self.reconstruction_loss(reconstructed, x)
            total_loss = recon_loss + self.codebook_weight * vq_loss
            
        return {
            'val_total_loss': total_loss.item(),
            'val_recon_loss': recon_loss.item(),
            'val_vq_loss': vq_loss.item()
        }

def create_model(vocab_size=512, z_channels=16, ch=128):
    """Create grayscale VQVAE model"""
    model = VQVAEGrayscale(
        vocab_size=vocab_size,
        z_channels=z_channels,
        ch=ch,
        test_mode=False,  # Enable training mode
        share_quant_resi=4,
        v_patch_nums=(1, 2, 3, 4, 5, 6, 8)  # For 128x128 with 16x downsample
    )

    # DEBUG: Check what v_patch_nums is actually being used
    print(f"Model v_patch_nums: {model.quantize.v_patch_nums}")
    print(f"Expected v_patch_nums: (1, 2, 3, 4, 5, 6, 8)")
    print(f"Downsample factor: {model.downsample}")
    
    # # Explicitly enable gradients for all parameters
    # for param in model.parameters():
    #     param.requires_grad = True
    
    # # Set to training mode
    # model.train()
    
    return model

def create_dataloaders(data_path: str, final_reso: int, batch_size: int, hflip=False):
    """Create dataloaders using existing MRI dataset infrastructure"""
    
    # Use your existing function to build datasets
    num_classes, train_set, val_set = build_mri_dataset_grayscale(
        data_path=data_path, 
        final_reso=final_reso, 
        hflip=hflip
    )
    
    # Create DataLoaders
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        dataset=val_set,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader, num_classes

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, 
                       help='Path to MRI data (should contain train/ and val/ subdirs)')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.05)  # Added weight decay
    parser.add_argument('--final_reso', type=int, default=128, 
                       help='Final resolution for images')
    parser.add_argument('--hflip', action='store_true', 
                       help='Enable horizontal flip augmentation')
    parser.add_argument('--save_dir', type=str, default='./local_output/vqvae_checkpoints')  # Changed default path
    parser.add_argument('--vocab_size', type=int, default=512)      # Reduced default
    parser.add_argument('--z_channels', type=int, default=16)       # Reduced default
    parser.add_argument('--ch', type=int, default=128)              # Reduced default
    parser.add_argument('--codebook_weight', type=float, default=1.0)
    parser.add_argument('--val_freq', type=int, default=20, help='Validation frequency in steps')
    args = parser.parse_args()
    
    # Initialize distributed training
    init_distributed()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create tensorboard writer
    tb_log_dir = os.path.join(args.save_dir, 'tensorboard_logs')
    os.makedirs(tb_log_dir, exist_ok=True)
    tb_writer = SummaryWriter(log_dir=tb_log_dir)
    
    # Create model
    model = create_model(
        vocab_size=args.vocab_size,
        z_channels=args.z_channels,
        ch=args.ch
    )
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model created with {total_params/1e6:.2f}M total parameters ({trainable_params/1e6:.2f}M trainable)")
    print(f"Downsample factor: {model.downsample}×")
    
    trainer = VQVAETrainer(
        model, device, 
        lr=args.lr, 
        beta1=0.9, beta2=0.95,           # VAR-style optimizer params
        weight_decay=args.weight_decay,
        codebook_weight=args.codebook_weight,
        tb_writer=tb_writer
    )
    
    # Create data loaders using existing MRI dataset infrastructure
    train_loader, val_loader, num_classes = create_dataloaders(
        data_path=args.data_path,
        final_reso=args.final_reso,
        batch_size=args.batch_size,
        hflip=args.hflip
    )
    
    print(f"Dataset info: {num_classes} classes")
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    print(f"Validation frequency: every {args.val_freq} steps")
    
    # Training loop
    os.makedirs(args.save_dir, exist_ok=True)
    global_step = 0
    
    # Function to run validation
    def run_validation():
        model.eval()
        val_metrics = []
        
        # Use a subset of validation set for faster validation during training
        val_subset = list(val_loader)[:10]  # Use first 10 batches for quick validation
        
        with torch.no_grad():
            for batch in val_subset:
                metrics = trainer.validation_step(batch)
                val_metrics.append(metrics)
        
        # Calculate average validation metrics
        avg_val_loss = sum(m['val_total_loss'] for m in val_metrics) / len(val_metrics)
        avg_val_recon = sum(m['val_recon_loss'] for m in val_metrics) / len(val_metrics)
        avg_val_vq = sum(m['val_vq_loss'] for m in val_metrics) / len(val_metrics)
        
        # Log validation metrics to tensorboard
        tb_writer.add_scalar('Step/Val_Total_Loss', avg_val_loss, global_step)
        tb_writer.add_scalar('Step/Val_Recon_Loss', avg_val_recon, global_step)
        tb_writer.add_scalar('Step/Val_VQ_Loss', avg_val_vq, global_step)
        
        return avg_val_loss, avg_val_recon, avg_val_vq
    
    try:
        # Create main progress bar for all epochs
        total_steps = args.epochs * len(train_loader)
        main_pbar = tqdm(total=total_steps, desc="Training Progress", position=0)
        
        for epoch in range(args.epochs):
            model.train()
            train_metrics = []
            
            # Create epoch progress bar
            epoch_pbar = tqdm(
                train_loader, 
                desc=f"Epoch {epoch+1}/{args.epochs}",
                position=1,
                leave=False
            )
            
            for batch_idx, batch in enumerate(epoch_pbar):
                # Training step
                metrics = trainer.training_step(batch, global_step)
                train_metrics.append(metrics)
                
                # Update progress bars
                main_pbar.update(1)
                epoch_pbar.set_postfix({
                    'Step': global_step,
                    'Loss': f"{metrics['total_loss']:.4f}",
                    'Recon': f"{metrics['recon_loss']:.4f}",
                    'VQ': f"{metrics['vq_loss']:.4f}",
                    'Perp': f"{metrics['perplexity']:.1f}"
                })
                
                global_step += 1
                
                # Run validation every val_freq steps
                if global_step % args.val_freq == 0:
                    val_loss, val_recon, val_vq = run_validation()
                    
                    # Print validation results alongside training metrics
                    tqdm.write(f"\n Step {global_step} Validation Results:")
                    tqdm.write(f"  Training  → Loss: {metrics['total_loss']:.6f}, Recon: {metrics['recon_loss']:.6f}, VQ: {metrics['vq_loss']:.6f}")
                    tqdm.write(f"  Validation → Loss: {val_loss:.6f}, Recon: {val_recon:.6f}, VQ: {val_vq:.6f}")
                    
                    # Update main progress bar with validation info
                    main_pbar.set_postfix({
                        'Epoch': f"{epoch+1}/{args.epochs}",
                        'Step': global_step,
                        'T_Loss': f"{metrics['total_loss']:.4f}",
                        'V_Loss': f"{val_loss:.4f}",
                        'V_Recon': f"{val_recon:.4f}",
                        'V_VQ': f"{val_vq:.4f}"
                    })
                    
                    # Set model back to training mode
                    model.train()
            
            epoch_pbar.close()
            
            # End of epoch: comprehensive validation and logging
            model.eval()
            val_metrics = []
            
            # Run full validation at end of epoch
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1} Validation", position=1, leave=False)
            with torch.no_grad():
                for batch in val_pbar:
                    metrics = trainer.validation_step(batch)
                    val_metrics.append(metrics)
                    val_pbar.set_postfix({
                        'Val_Loss': f"{metrics['val_total_loss']:.4f}",
                        'Val_Recon': f"{metrics['val_recon_loss']:.4f}",
                        'Val_VQ': f"{metrics['val_vq_loss']:.4f}"
                    })
            val_pbar.close()
            
            # Calculate epoch averages
            avg_train_loss = sum(m['total_loss'] for m in train_metrics) / len(train_metrics)
            avg_val_loss = sum(m['val_total_loss'] for m in val_metrics) / len(val_metrics)
            avg_train_recon = sum(m['recon_loss'] for m in train_metrics) / len(train_metrics)
            avg_val_recon = sum(m['val_recon_loss'] for m in val_metrics) / len(val_metrics)
            avg_train_vq = sum(m['vq_loss'] for m in train_metrics) / len(train_metrics)
            avg_val_vq = sum(m['val_vq_loss'] for m in val_metrics) / len(val_metrics)
            
            # Log epoch metrics to tensorboard
            tb_writer.add_scalar('Epoch/Train_Total_Loss', avg_train_loss, epoch)
            tb_writer.add_scalar('Epoch/Val_Total_Loss', avg_val_loss, epoch)
            tb_writer.add_scalar('Epoch/Train_Recon_Loss', avg_train_recon, epoch)
            tb_writer.add_scalar('Epoch/Val_Recon_Loss', avg_val_recon, epoch)
            tb_writer.add_scalar('Epoch/Train_VQ_Loss', avg_train_vq, epoch)
            tb_writer.add_scalar('Epoch/Val_VQ_Loss', avg_val_vq, epoch)
            
            # Print epoch summary
            tqdm.write(f"\n┌─ Epoch {epoch+1}/{args.epochs} Summary (Step {global_step}) ─┐")
            tqdm.write(f"│ Train Loss: {avg_train_loss:.6f} │ Val Loss: {avg_val_loss:.6f}   │")
            tqdm.write(f"│ Train Recon: {avg_train_recon:.6f} │ Val Recon: {avg_val_recon:.6f} │")
            tqdm.write(f"│ Train VQ: {avg_train_vq:.6f}     │ Val VQ: {avg_val_vq:.6f}     │")
            tqdm.write(f"└─────────────────────────────────────────────────────────────┘\n")
            
            # Save checkpoint
            if epoch % 10 == 0:
                checkpoint = {
                    'epoch': epoch,
                    'global_step': global_step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': trainer.optimizer.state_dict(),
                    'train_loss': avg_train_loss,
                    'val_loss': avg_val_loss,
                    'args': vars(args)
                }
                checkpoint_path = os.path.join(args.save_dir, f'vqvae_epoch_{epoch}.pth')
                torch.save(checkpoint, checkpoint_path)
                tqdm.write(f"💾 Checkpoint saved: {checkpoint_path}")
        
        main_pbar.close()
        
        # Save final model
        final_checkpoint = {
            'epoch': args.epochs,
            'global_step': global_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': trainer.optimizer.state_dict(),
            'args': vars(args)
        }
        final_path = os.path.join(args.save_dir, 'vqvae_final.pth')
        torch.save(final_checkpoint, final_path)
        
        tb_writer.close()
        print(f"\n🎉 Training completed! Final model saved: {final_path}")
        print(f"Total training steps: {global_step}")
        
    finally:
        # Cleanup distributed training
        cleanup_distributed()

if __name__ == '__main__':
    main()
    

# python train_vqvae_grayscale.py \
#     --data_path /home/yuchenliu/Dataset/IXI/t1_np_masked_128_unconditional \
#     --batch_size 64 \
#     --epochs 100 \
#     --final_reso 128 \
#     --hflip \
#     --vocab_size 256 \
#     --z_channels 16 \
#     --ch 128 \
#     --save_dir ./vqvae_mri_checkpoints

    
# python train_vqvae_multiscale.py \
#     --data_path /home/yuchenliu/Dataset/IXI/t1_np_masked_128_unconditional \
#     --batch_size 64 \
#     --epochs 100 \
#     --final_reso 128 \
#     --vocab_size 512 \
#     --z_channels 16 \
#     --ch 128 \
#     --lr 1e-4 \