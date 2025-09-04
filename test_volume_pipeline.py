"""
Test script to verify the volume VQVAE pipeline works correctly
"""

import torch
import numpy as np
import sys
import os
sys.path.append('/home/yuchenliu/VAR')

from models.vqvae_vol import VQVAEVol
from utils.data_mri_volume import MRIDatasetVolume, build_mri_dataset_volume
from lpips_grayscale import LPIPS

def test_volume_model():
    """Test the volume VQVAE model"""
    print("=== Testing Volume VQVAE Model ===")
    
    # Create model
    model = VQVAEVol(
        vocab_size=512,
        z_channels=16,
        ch=128,
        beta=1.0,
        test_mode=False,
        num_slices=10
    )
    
    print(f"Model created successfully")
    print(f"Number of slices: {model.num_slices}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    batch_size = 2
    x = torch.randn(batch_size, 10, 128, 128)  # [B, num_slices, H, W]
    
    print(f"Input shape: {x.shape}")
    
    # Test forward pass
    reconstructed, usages, vq_loss = model(x, ret_usages=True)
    print(f"Output shape: {reconstructed.shape}")
    print(f"VQ loss: {vq_loss.item():.6f}")
    print(f"Codebook usages: {[u.item() for u in usages]}")
    
    # Test reconstruction
    recon_loss = torch.nn.L1Loss()(reconstructed, x)
    print(f"Reconstruction loss: {recon_loss.item():.6f}")
    
    return model

def test_volume_dataset():
    """Test the volume dataset"""
    print("\n=== Testing Volume Dataset ===")
    
    # Create synthetic data for testing
    test_dir = "/tmp/test_volume_data"
    os.makedirs(test_dir, exist_ok=True)
    
    # Create some fake volume files
    for i in range(5):
        volume = np.random.rand(128, 128, 10).astype(np.float32)
        np.save(os.path.join(test_dir, f"test_volume_{i}.npy"), volume)
    
    # Test dataset
    dataset = MRIDatasetVolume(
        data_path=test_dir,
        final_reso=128,
        hflip=True,
        num_slices=10
    )
    
    print(f"Dataset created with {len(dataset)} volumes")
    
    # Test loading
    vol_tensor, label = dataset[0]
    print(f"Loaded volume shape: {vol_tensor.shape}")
    print(f"Volume range: [{vol_tensor.min():.3f}, {vol_tensor.max():.3f}]")
    
    # Test dataloader
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    for batch_idx, (batch_x, batch_labels) in enumerate(dataloader):
        print(f"Batch {batch_idx}: shape {batch_x.shape}, labels {batch_labels}")
        break
    
    # Cleanup
    import shutil
    shutil.rmtree(test_dir)
    
    return dataset

def test_volume_lpips():
    """Test volume LPIPS"""
    print("\n=== Testing Volume LPIPS ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test volume LPIPS
    lpips_vol = LPIPS(volume_mode=True).to(device)
    
    # Create test volumes
    batch_size = 2
    num_slices = 10
    x = torch.randn(batch_size, num_slices, 128, 128).to(device)
    y = torch.randn(batch_size, num_slices, 128, 128).to(device)
    
    # Convert to [-1, 1] range
    x = 2.0 * x - 1.0
    y = 2.0 * y - 1.0
    
    # Test forward pass
    with torch.no_grad():
        loss = lpips_vol(x, y)
        print(f"Volume LPIPS loss: {loss.item():.6f}")
        
        # Test with identical volumes
        identical_loss = lpips_vol(x, x)
        print(f"Volume LPIPS identical loss: {identical_loss.item():.6f}")
    
    # Test standard LPIPS for comparison
    lpips_std = LPIPS(volume_mode=False).to(device)
    
    # Take first slice
    x_slice = x[:, 0:1, :, :]
    y_slice = y[:, 0:1, :, :]
    
    with torch.no_grad():
        std_loss = lpips_std(x_slice, y_slice)
        print(f"Standard LPIPS (single slice): {std_loss.item():.6f}")
    
    return lpips_vol

def test_full_pipeline():
    """Test the full training pipeline"""
    print("\n=== Testing Full Pipeline ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = VQVAEVol(
        vocab_size=512,
        z_channels=16,
        ch=128,
        beta=1.0,
        test_mode=False,
        num_slices=10
    ).to(device)
    
    # Create synthetic data
    batch_size = 2
    x = torch.randn(batch_size, 10, 128, 128).to(device)
    
    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Create loss functions
    recon_loss_fn = torch.nn.L1Loss()
    lpips_fn = LPIPS(volume_mode=True).to(device).eval()
    
    # Training step
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    reconstructed, usages, vq_loss = model(x, ret_usages=True)
    
    # Compute losses
    recon_loss = recon_loss_fn(reconstructed, x)
    
    # Perceptual loss
    x_normalized = 2.0 * x - 1.0
    reconstructed_normalized = 2.0 * reconstructed - 1.0
    p_loss = lpips_fn(x_normalized, reconstructed_normalized)
    
    # Total loss
    total_loss = recon_loss + vq_loss + p_loss
    
    # Backward pass
    total_loss.backward()
    optimizer.step()
    
    print(f"Training step completed successfully")
    print(f"  Reconstruction loss: {recon_loss.item():.6f}")
    print(f"  VQ loss: {vq_loss.item():.6f}")
    print(f"  Perceptual loss: {p_loss.item():.6f}")
    print(f"  Total loss: {total_loss.item():.6f}")
    
    return model

def main():
    """Run all tests"""
    print("🧪 Testing Volume VQVAE Pipeline")
    print("=" * 50)
    
    try:
        # Test individual components
        model = test_volume_model()
        dataset = test_volume_dataset()
        lpips = test_volume_lpips()
        
        # Test full pipeline
        trained_model = test_full_pipeline()
        
        print("\n✅ All tests passed successfully!")
        print("\n📋 Summary:")
        print("  • Volume VQVAE model: ✓")
        print("  • Volume dataset: ✓")
        print("  • Volume LPIPS: ✓")
        print("  • Full training pipeline: ✓")
        
        print("\n🚀 Ready to train with:")
        print("  python train_vqvae_vol.py \\")
        print("    --data_path /path/to/volume/data \\")
        print("    --batch_size 16 \\")
        print("    --epochs 100 \\")
        print("    --num_slices 10 \\")
        print("    --volume_lpips")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
