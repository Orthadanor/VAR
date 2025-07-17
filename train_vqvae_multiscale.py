import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import sys
sys.path.append('/home/yuchenliu/VAR')  # Add VAR to path

from models.vqvae_grayscale import VQVAEGrayscale
from utils.data_mri import build_mri_dataset_grayscale
import argparse

class VQVAETrainer:
    def __init__(self, model, device, lr=1e-4, beta1=0.5, beta2=0.9):
        self.model = model.to(device)
        self.device = device
        
        # Optimizer setup similar to taming-transformers
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=lr, betas=(beta1, beta2)
        )
        
        # Loss functions
        self.reconstruction_loss = nn.L1Loss()  # or nn.MSELoss()
        self.perceptual_loss_weight = 1.0
        self.codebook_weight = 1.0
        
    def training_step(self, batch):
        self.optimizer.zero_grad()
        
        # Unpack batch from MRIDatasetGrayscale
        x, _ = batch  # x: [B, 1, H, W], _: class labels (unused)
        x = x.to(self.device)
        
        # Forward pass
        reconstructed, usages, vq_loss = self.model(x, ret_usages=True)
        
        # Reconstruction loss
        recon_loss = self.reconstruction_loss(reconstructed, x)
        
        # Total loss (similar to taming-transformers)
        total_loss = recon_loss + self.codebook_weight * vq_loss
        
        total_loss.backward()
        self.optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'recon_loss': recon_loss.item(),
            'vq_loss': vq_loss.item(),
            'perplexity': usages.get('perplexity', 0) if usages else 0
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

def create_model(vocab_size=4096, z_channels=32, ch=160):
    """Create grayscale VQVAE model"""
    model = VQVAEGrayscale(
        vocab_size=vocab_size,
        z_channels=z_channels,
        ch=ch,
        test_mode=False,  # Enable training mode
        share_quant_resi=4,
        v_patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
    )
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
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--final_reso', type=int, default=256, 
                       help='Final resolution for images')
    parser.add_argument('--hflip', action='store_true', 
                       help='Enable horizontal flip augmentation')
    parser.add_argument('--save_dir', type=str, default='./vqvae_checkpoints')
    parser.add_argument('--vocab_size', type=int, default=4096)
    parser.add_argument('--z_channels', type=int, default=32)
    parser.add_argument('--ch', type=int, default=160)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = create_model(
        vocab_size=args.vocab_size,
        z_channels=args.z_channels,
        ch=args.ch
    )
    trainer = VQVAETrainer(model, device, lr=args.lr)
    
    # Create data loaders using existing MRI dataset infrastructure
    train_loader, val_loader, num_classes = create_dataloaders(
        data_path=args.data_path,
        final_reso=args.final_reso,
        batch_size=args.batch_size,
        hflip=args.hflip
    )
    
    print(f"Dataset info: {num_classes} classes")
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Training loop
    os.makedirs(args.save_dir, exist_ok=True)
    
    for epoch in range(args.epochs):
        # Training
        model.train()
        train_metrics = []
        for batch_idx, batch in enumerate(train_loader):
            metrics = trainer.training_step(batch)
            train_metrics.append(metrics)
            
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}: {metrics}")
        
        # Validation
        model.eval()
        val_metrics = []
        for batch in val_loader:
            metrics = trainer.validation_step(batch)
            val_metrics.append(metrics)
        
        # Calculate average metrics
        avg_train_loss = sum(m['total_loss'] for m in train_metrics) / len(train_metrics)
        avg_val_loss = sum(m['val_total_loss'] for m in val_metrics) / len(val_metrics)
        
        print(f"Epoch {epoch}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Save checkpoint
        if epoch % 10 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'args': vars(args)
            }
            torch.save(checkpoint, os.path.join(args.save_dir, f'vqvae_epoch_{epoch}.pth'))
        
        print(f"Epoch {epoch} completed")
    
    # Save final model
    final_checkpoint = {
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': trainer.optimizer.state_dict(),
        'args': vars(args)
    }
    torch.save(final_checkpoint, os.path.join(args.save_dir, 'vqvae_final.pth'))
    print("Training completed!")

if __name__ == '__main__':
    main()
    

python train_vqvae_grayscale.py \
    --data_path /path/to/your/mri/data \
    --batch_size 16 \
    --epochs 100 \
    --final_reso 256 \
    --hflip \
    --vocab_size 4096 \
    --z_channels 32 \
    --ch 160 \
    --save_dir ./vqvae_mri_checkpoints