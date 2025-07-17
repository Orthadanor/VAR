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
from torchvision import models
from collections import namedtuple
import torch.nn.functional as F

# Simplified LPIPS implementation for grayscale images
class ScalingLayer(nn.Module):
    def __init__(self):
        super(ScalingLayer, self).__init__()
        # For grayscale, we'll replicate to 3 channels and use RGB normalization
        self.register_buffer('shift', torch.Tensor([-.030, -.088, -.188])[None, :, None, None])
        self.register_buffer('scale', torch.Tensor([.458, .448, .450])[None, :, None, None])

    def forward(self, inp):
        # Convert grayscale to RGB by replication
        if inp.size(1) == 1:
            inp = inp.repeat(1, 3, 1, 1)
        return (inp - self.shift) / self.scale

class NetLinLayer(nn.Module):
    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super(NetLinLayer, self).__init__()
        layers = [nn.Dropout()] if use_dropout else []
        layers += [nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False)]
        self.model = nn.Sequential(*layers)

class VGG16(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrained=True):
        super(VGG16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=pretrained).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
            
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        h1 = self.slice1(x)
        h2 = self.slice2(h1)
        h3 = self.slice3(h2)
        h4 = self.slice4(h3)
        h5 = self.slice5(h4)
        
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3'])
        return vgg_outputs(h1, h2, h3, h4, h5)

def normalize_tensor(x, eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(x**2, dim=1, keepdim=True))
    return x / (norm_factor + eps)

def spatial_average(x, keepdim=True):
    return x.mean([2, 3], keepdim=keepdim)

class LPIPS(nn.Module):
    def __init__(self, use_dropout=True):
        super().__init__()
        self.scaling_layer = ScalingLayer()
        self.chns = [64, 128, 256, 512, 512]  # VGG16 features
        self.net = VGG16(pretrained=True, requires_grad=False)
        
        self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout)
        self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout)
        self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout)
        self.lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout)
        self.lin4 = NetLinLayer(self.chns[4], use_dropout=use_dropout)
        
        # Freeze parameters
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, input, target):
        # Scale inputs
        input_scaled = self.scaling_layer(input)
        target_scaled = self.scaling_layer(target)
        
        # Get VGG features
        outs0 = self.net(input_scaled)
        outs1 = self.net(target_scaled)
        
        # Compute perceptual loss
        lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
        val = 0
        
        for kk in range(len(self.chns)):
            feats0 = normalize_tensor(outs0[kk])
            feats1 = normalize_tensor(outs1[kk])
            diff = (feats0 - feats1) ** 2
            val += spatial_average(lins[kk].model(diff), keepdim=True)
        
        return val

class VQVAETrainer:
    def __init__(self, model, device, lr=1e-4, beta1=0.5, beta2=0.9, 
                 perceptual_weight=1.0, pixel_weight=1.0, codebook_weight=1.0):
        self.model = model.to(device)
        self.device = device
        
        # Optimizer setup similar to taming-transformers
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=lr, betas=(beta1, beta2)
        )
        
        # Loss components
        self.pixel_loss = nn.L1Loss(reduction='none')  # Keep per-pixel for proper weighting
        self.perceptual_loss = LPIPS().to(device).eval()
        
        # Loss weights
        self.pixel_weight = pixel_weight
        self.perceptual_weight = perceptual_weight
        self.codebook_weight = codebook_weight
        
    def compute_reconstruction_loss(self, inputs, reconstructions):
        """Compute reconstruction loss with both pixel and perceptual components"""
        # Pixel loss (L1) - Answer to Q1: yes, this is equivalent to torch.abs()
        pixel_loss = self.pixel_loss(reconstructions, inputs)  # Shape: [B, C, H, W]
        pixel_loss = pixel_loss.mean()  # Average over all dimensions
        
        # Perceptual loss (LPIPS)
        if self.perceptual_weight > 0:
            perceptual_loss = self.perceptual_loss(inputs, reconstructions)  # Shape: [B, 1, 1, 1]
            perceptual_loss = perceptual_loss.mean()  # Average over batch
        else:
            perceptual_loss = torch.tensor(0.0, device=inputs.device)
        
        # Combined reconstruction loss
        total_recon_loss = (self.pixel_weight * pixel_loss + 
                           self.perceptual_weight * perceptual_loss)
        
        return total_recon_loss, pixel_loss, perceptual_loss
        
    def training_step(self, batch):
        self.optimizer.zero_grad()
        
        # Unpack batch from MRIDatasetGrayscale
        x, _ = batch  # x: [B, 1, H, W], _: class labels (unused)
        x = x.to(self.device)
        
        # Forward pass
        reconstructed, usages, vq_loss = self.model(x, ret_usages=True)
        
        # Compute reconstruction loss components
        recon_loss, pixel_loss, perceptual_loss = self.compute_reconstruction_loss(x, reconstructed)
        
        # Answer to Q2: Be careful about dimensions when combining losses
        # vq_loss is already a scalar from VectorQuantizer2.forward
        # recon_loss is a scalar from our computation above
        total_loss = recon_loss + self.codebook_weight * vq_loss
        
        total_loss.backward()
        self.optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'recon_loss': recon_loss.item(),
            'pixel_loss': pixel_loss.item(),
            'perceptual_loss': perceptual_loss.item(),
            'vq_loss': vq_loss.item(),
            'perplexity': usages.get('perplexity', 0) if usages else 0
        }
    
    def validation_step(self, batch):
        with torch.no_grad():
            x, _ = batch
            x = x.to(self.device)
            reconstructed, usages, vq_loss = self.model(x, ret_usages=True)
            
            # Compute reconstruction loss components
            recon_loss, pixel_loss, perceptual_loss = self.compute_reconstruction_loss(x, reconstructed)
            total_loss = recon_loss + self.codebook_weight * vq_loss
            
        return {
            'val_total_loss': total_loss.item(),
            'val_recon_loss': recon_loss.item(),
            'val_pixel_loss': pixel_loss.item(),
            'val_perceptual_loss': perceptual_loss.item(),
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
    # Loss weights
    parser.add_argument('--pixel_weight', type=float, default=1.0)
    parser.add_argument('--perceptual_weight', type=float, default=1.0)
    parser.add_argument('--codebook_weight', type=float, default=1.0)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = create_model(
        vocab_size=args.vocab_size,
        z_channels=args.z_channels,
        ch=args.ch
    )
    trainer = VQVAETrainer(
        model, device, lr=args.lr,
        pixel_weight=args.pixel_weight,
        perceptual_weight=args.perceptual_weight,
        codebook_weight=args.codebook_weight
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
    print(f"Loss weights: pixel={args.pixel_weight}, perceptual={args.perceptual_weight}, codebook={args.codebook_weight}")
    
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
        avg_train_pixel = sum(m['pixel_loss'] for m in train_metrics) / len(train_metrics)
        avg_train_perceptual = sum(m['perceptual_loss'] for m in train_metrics) / len(train_metrics)
        
        print(f"Epoch {epoch}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        print(f"  Pixel: {avg_train_pixel:.4f}, Perceptual: {avg_train_perceptual:.4f}")
        
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
    
python train_vqvae_multiscale.py \
    --data_path /path/to/your/mri/data \
    --batch_size 8 \
    --epochs 100 \
    --final_reso 256 \
    --pixel_weight 1.0 \
    --perceptual_weight 1.0 \
    --codebook_weight 1.0 \
    --vocab_size 4096 \
    --z_channels 32 \
    --ch 160