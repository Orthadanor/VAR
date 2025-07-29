import torch
import sys
import os
import numpy as np
from PIL import Image
import torch.distributed as dist
sys.path.append('/home/yuchenliu/VAR')  # Add VAR to path

from models.vqvae_grayscale import VQVAEGrayscale
from utils.data_mri import build_mri_dataset_grayscale

def init_distributed():
    """Initialize distributed training for single GPU"""
    if not dist.is_initialized():
        # Initialize for single GPU
        os.environ.setdefault('MASTER_ADDR', 'localhost')
        os.environ.setdefault('MASTER_PORT', '12355')  # Different port from training
        os.environ.setdefault('RANK', '0')
        os.environ.setdefault('WORLD_SIZE', '1')
        
        dist.init_process_group(
            backend='nccl' if torch.cuda.is_available() else 'gloo',
            init_method='env://',
            rank=0,
            world_size=1
        )
        print("✅ Initialized distributed training for testing")

def cleanup_distributed():
    """Cleanup distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()
        print("🧹 Cleaned up distributed training")

def test_tokenization(same_shape=True):
    # Create model with the same parameters as your training
    vqvae = VQVAEGrayscale(
        vocab_size=128,  # Changed from 4096 to match your training
        z_channels=16,   # Changed from 32 to match your training  
        ch=64,          # Changed from 160 to match your training
        test_mode=False,
        v_patch_nums=(1, 2, 4, 8)  # Add your patch configuration
    ).cuda()
    
    # Load checkpoint with proper key handling
    checkpoint_path = '/home/yuchenliu/VAR/local_output/vqvae_checkpoints_v128_z16_c64_b1/vqvae_final.pth'
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return
    
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cuda')
    
    # Print checkpoint info
    print(f"Checkpoint info:")
    print(f"  Epoch: {checkpoint.get('epoch', 'Unknown')}")
    print(f"  Global step: {checkpoint.get('global_step', 'Unknown')}")
    print(f"  Train loss: {checkpoint.get('train_loss', 'Unknown')}")
    print(f"  Val loss: {checkpoint.get('val_loss', 'Unknown')}")
    
    # Load model weights
    vqvae.load_state_dict(checkpoint['model_state_dict'])
    vqvae.eval()
    print("Model loaded successfully!")
    
    # Test with MRI data using same parameters as training
    try:
        num_classes, train_set, val_set = build_mri_dataset_grayscale(
            data_path='/home/yuchenliu/Dataset/IXI/t1_np_masked_128_unconditional', 
            final_reso=128,  # Changed from 256 to match your training
            hflip=False
        )
        print(f"Dataset loaded: {len(val_set)} validation samples")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return
    
    # Get a sample
    image, _ = val_set[9]
    print(f"Sample image shape: {image.shape}")
    image = image.unsqueeze(0).cuda()
    
    print(f"\n Input image shape: {image.shape}")
    print(f"    Image range: [{image.min().item():.3f}, {image.max().item():.3f}]")
    
    # Test tokenization
    try:
        with torch.no_grad():
            # Test encoding
            encoded = vqvae.encoder(image)
            print(f"Encoder output shape: {encoded.shape}")
            
            # Test quantization
            quant_input = vqvae.quant_conv(encoded)
            print(f"🔧 Quant input shape: {quant_input.shape}")
            
            # Test multi-scale tokenization
            gt_idx_Bl = vqvae.img_to_idxBl(image)
            
            print(f"\n🏷️  Tokenization Results:")
            print(f"    Number of scales: {len(gt_idx_Bl)}")
            
            total_tokens = 0
            for i, tokens in enumerate(gt_idx_Bl):
                scale_size = int(tokens.shape[1] ** 0.5) if tokens.shape[1] > 0 else 0
                total_tokens += tokens.shape[1]
                print(f"    Scale {i}: {scale_size}x{scale_size} = {tokens.shape[1]} tokens")
                print(f"      Token range: [{tokens.min().item()}, {tokens.max().item()}]")
                print(f"      Unique tokens: {tokens.unique().numel()}/{vqvae.vocab_size}")
                print(f"      Usage: {tokens.unique().numel()/vqvae.vocab_size*100:.1f}%")
            
            print(f"    Total tokens: {total_tokens}")
            
            # Create output directory for saving images
            output_dir = "/home/yuchenliu/VAR/local_output/vqvae_checkpoints_v128_z16_c64_b1/recon_imgs"
            os.makedirs(output_dir, exist_ok=True)
            
            # Save original image first
            original_img = image.squeeze().cpu().numpy()
            # Normalize to 0-255 range for saving
            original_img_normalized = ((original_img + 1) * 127.5).clip(0, 255).astype(np.uint8)
            original_pil = Image.fromarray(original_img_normalized, mode='L')
            original_path = os.path.join(output_dir, "original_image.png")
            original_pil.save(original_path)
            print(f"💾 Original image saved: {original_path}")
            
            # Test reconstruction - Show all reconstructed images
            if same_shape:
                reconstructed_list = vqvae.idxBl_to_img(gt_idx_Bl, True)
                print(f"\n🔄 Reconstruction test (same_shape=True):")
                print(f"    Number of reconstructed images: {len(reconstructed_list)}")
                
                for i, reconstructed in enumerate(reconstructed_list):
                    recon_loss = torch.nn.functional.l1_loss(reconstructed, image)
                    print(f"    Scale {i} reconstruction:")
                    print(f"      Shape: {reconstructed.shape}")
                    print(f"      L1 loss: {recon_loss.item():.6f}")
                    print(f"      Range: [{reconstructed.min().item():.3f}, {reconstructed.max().item():.3f}]")
                    print(f"      Mean: {reconstructed.mean().item():.3f}, Std: {reconstructed.std().item():.3f}")
                    
                    # Save each reconstructed image
                    recon_img = reconstructed.squeeze().cpu().numpy()
                    # Normalize to 0-255 range for saving
                    recon_img_normalized = ((recon_img + 1) * 127.5).clip(0, 255).astype(np.uint8)
                    recon_pil = Image.fromarray(recon_img_normalized, mode='L')
                    recon_path = os.path.join(output_dir, f"reconstruction_scale_{i}.png")
                    recon_pil.save(recon_path)
                    print(f"      💾 Saved: {recon_path}")
            else:
                reconstructed_list = vqvae.idxBl_to_img(gt_idx_Bl, False)
                print(f"\n🔄 Reconstruction test (same_shape=False - native scales):")
                print(f"    Number of reconstructed images: {len(reconstructed_list)}")
                
                for i, reconstructed in enumerate(reconstructed_list):
                    scale_size = reconstructed.shape[-1]
                    print(f"    Scale {i} reconstruction (native {scale_size}×{scale_size}):")
                    print(f"      Shape: {reconstructed.shape}")
                    print(f"      Range: [{reconstructed.min().item():.3f}, {reconstructed.max().item():.3f}]")
                    print(f"      Mean: {reconstructed.mean().item():.3f}, Std: {reconstructed.std().item():.3f}")
                    print(f"      L1 loss: Skipped (different sizes)")
                    
                    # Save each reconstructed image at native resolution
                    recon_img = reconstructed.squeeze().cpu().numpy()
                    # Normalize to 0-255 range for saving
                    recon_img_normalized = ((recon_img + 1) * 127.5).clip(0, 255).astype(np.uint8)
                    recon_pil = Image.fromarray(recon_img_normalized, mode='L')
                    recon_path = os.path.join(output_dir, f"reconstruction_scale_{i}_native_{scale_size}x{scale_size}.png")
                    recon_pil.save(recon_path)
                    print(f"      💾 Saved: {recon_path}")
            
            # Also test the standard forward pass for comparison
            print(f"\n🔄 Standard forward pass (for comparison):")
            
            # Initialize distributed training only for standard forward pass
            if not dist.is_initialized():
                init_distributed()
            
            reconstructed_std, usages, vq_loss = vqvae(image, ret_usages=True)
            recon_loss_std = torch.nn.functional.l1_loss(reconstructed_std, image)
            print(f"    Standard reconstruction shape: {reconstructed_std.shape}")
            print(f"    Standard reconstruction L1 loss: {recon_loss_std.item():.6f}")
            print(f"    Standard reconstruction range: [{reconstructed_std.min().item():.3f}, {reconstructed_std.max().item():.3f}]")
            print(f"    VQ loss: {vq_loss.item():.6f}")
            if usages:
                print(f"    Codebook usages: {[f'{u:.2f}' for u in usages]}")
            
            # Save standard reconstruction
            std_recon_img = reconstructed_std.squeeze().cpu().numpy()
            std_recon_img_normalized = ((std_recon_img + 1) * 127.5).clip(0, 255).astype(np.uint8)
            std_recon_pil = Image.fromarray(std_recon_img_normalized, mode='L')
            std_recon_path = os.path.join(output_dir, "reconstruction_standard.png")
            std_recon_pil.save(std_recon_path)
            print(f"    💾 Standard reconstruction saved: {std_recon_path}")
            
            print(f"\n📁 All images saved to: {output_dir}")
            
    except Exception as e:
        print(f"❌ Error during tokenization: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup distributed training if it was initialized
        if dist.is_initialized():
            cleanup_distributed()
        return

def test_model_properties():
    """Test model properties and architecture"""
    print("🏗️  Testing model architecture...")
    
    vqvae = VQVAEGrayscale(
        vocab_size=128,
        z_channels=16,
        ch=64,
        test_mode=False,
        v_patch_nums=(1, 2, 4, 8)
    )
    
    print(f"Model properties:")
    print(f"    Vocab size: {vqvae.vocab_size}")
    print(f"    Z channels: {vqvae.Cvae}")
    print(f"    Downsample factor: {vqvae.downsample}")
    print(f"    Patch numbers: {vqvae.quantize.v_patch_nums}")
    
    total_params = sum(p.numel() for p in vqvae.parameters())
    trainable_params = sum(p.numel() for p in vqvae.parameters() if p.requires_grad)
    print(f"    Total parameters: {total_params/1e6:.2f}M")
    print(f"    Trainable parameters: {trainable_params/1e6:.2f}M")

if __name__ == '__main__':
    print("Starting VQVAE Tokenization Test\n")
    
    # Test model architecture first
    test_model_properties()
    print()
    
    # Set whether to use same_shape reconstruction
    same_shape = True  # Change this to True for same_shape reconstruction, False for native scales
    print(f"🔧 Using same_shape={same_shape} for reconstruction test\n")
    
    # Test tokenization with checkpoint
    test_tokenization(same_shape=same_shape)
    
    print("\nTest completed!")