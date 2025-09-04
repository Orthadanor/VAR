import os
import sys
import glob
import numpy as np
from PIL import Image
import torch
import torch.distributed as dist

sys.path.append('/home/yuchenliu/VAR')  # Ensure project root on path

from models.vqvae_vol import VQVAEVol


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


def to_pm1(x: torch.Tensor) -> torch.Tensor:
    # Convert [0,1] to [-1,1]
    return x.mul(2.0).add_(-1.0)


def from_pm1_to_uint8(x: torch.Tensor) -> np.ndarray:
    # Convert [-1,1] to uint8 [0,255]
    return ((x.clamp(-1, 1) + 1.0) * 127.5).round().clamp(0, 255).to(torch.uint8).cpu().numpy()


def load_best_checkpoint(ckpt_path: str, device: torch.device):
    if not os.path.exists(ckpt_path):
        print(f"❌ Best checkpoint not found: {ckpt_path}")
        return None
    print(f" Loading best checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    epoch = ckpt.get('epoch', 'Unknown')
    best_val_loss = ckpt.get('best_val_loss', ckpt.get('val_loss', 'Unknown'))
    recon_enable = ckpt.get('args', {}).get('recon_enable', True)
    print(f"   • Best checkpoint epoch: {epoch}")
    print(f"   • Best val loss: {best_val_loss}")
    print(f"   • Reconstruction loss enabled: {recon_enable}")
    return ckpt, recon_enable


def build_model(device: torch.device, num_slices: int = 10) -> VQVAEVol:
    model = VQVAEVol(
        vocab_size=128,
        z_channels=8,
        ch=32,
        beta=1.0,
        test_mode=False,          # eval + no grad for inference
        share_quant_resi=4,
        v_patch_nums=(1, 2, 4, 8),
        num_slices=num_slices,
    ).to(device)
    return model


def pick_test_volume(test_dir: str) -> str:
    pattern = os.path.join(test_dir, '*.npy')
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No .npy files found in {test_dir}")
    # Pick the first volume for testing
    return files[0]


def load_volume_as_tensor(npy_path: str, device: torch.device) -> torch.Tensor:
    vol = np.load(npy_path).astype(np.float32)  # [H, W, S]
    # Normalize to [0,1] if data appears to be in [0,255]
    if vol.max() > 1.0:
        vol = vol / 255.0
    # [S, H, W]
    vol_th = torch.from_numpy(vol).permute(2, 0, 1).unsqueeze(0)  # [1, S, H, W]
    return vol_th.to(device)


def save_volume_slices(out_dir: str, base_name: str, vol_pm1: torch.Tensor, tag: str):
    # Create tag-specific directory
    tag_dir = os.path.join(out_dir, tag)
    os.makedirs(tag_dir, exist_ok=True)
    
    # vol_pm1: [1, S, H, W] in [-1,1]
    vol_uint8 = from_pm1_to_uint8(vol_pm1.squeeze(0))  # [S, H, W] uint8
    num_slices = vol_uint8.shape[0]
    saved_paths = []
    
    # Save individual slice images
    for s in range(num_slices):
        pil_img = Image.fromarray(vol_uint8[s], mode='L')
        out_path = os.path.join(tag_dir, f"{base_name}_{tag}_slice_{s:02d}.png")
        pil_img.save(out_path)
        saved_paths.append(out_path)
    
    # Create compact 2x5 grid image
    if num_slices == 10:  # Expected for volume data
        # Create a 2x5 grid
        rows, cols = 2, 5
        slice_height, slice_width = vol_uint8.shape[1], vol_uint8.shape[2]
        
        # Create blank grid image
        grid_img = np.zeros((rows * slice_height, cols * slice_width), dtype=np.uint8)
        
        # Fill the grid with slices
        for s in range(num_slices):
            row = s // cols
            col = s % cols
            y_start = row * slice_height
            y_end = (row + 1) * slice_height
            x_start = col * slice_width
            x_end = (col + 1) * slice_width
            grid_img[y_start:y_end, x_start:x_end] = vol_uint8[s]
        
        # Save the grid image in tag directory
        grid_pil = Image.fromarray(grid_img, mode='L')
        grid_path_tag = os.path.join(tag_dir, f"{base_name}_{tag}_grid_2x5.png")
        grid_pil.save(grid_path_tag)
        saved_paths.append(grid_path_tag)
        
        # Also save the grid image directly under out_dir
        grid_path_main = os.path.join(out_dir, f"{base_name}_{tag}_grid_2x5.png")
        grid_pil.save(grid_path_main)
        saved_paths.append(grid_path_main)
        
        print(f"💾 Saved {num_slices} '{tag}' slices + 2x5 grid to: {tag_dir} and {out_dir}")
    else:
        print(f"💾 Saved {num_slices} '{tag}' slices to: {tag_dir} (no grid - unexpected slice count)")
    
    return saved_paths


def test_multi_scale_tokenization(model, vol_pm1, output_dir, base_name, recon_enable=True, same_shape=True):
    """Test multi-scale tokenization and reconstruction"""
    print(f"\n🏷️  Testing multi-scale tokenization...")
    
    try:
        with torch.no_grad():
            # Test multi-scale tokenization
            gt_idx_Bl = model.img_to_idxBl(vol_pm1)
            
            print(f"🏷️  Tokenization Results:")
            print(f"    Number of scales: {len(gt_idx_Bl)}")
            
            total_tokens = 0
            for i, tokens in enumerate(gt_idx_Bl):
                # For volume data, tokens shape is [B, L] where L is the sequence length
                # We need to calculate the spatial dimensions differently for volumes
                if len(tokens.shape) == 2:  # [B, L]
                    seq_len = tokens.shape[1]
                    # For volume data, we need to calculate spatial dimensions based on volume shape
                    # This is a rough estimation - you might need to adjust based on your model's architecture
                    spatial_dim = int((seq_len / vol_pm1.shape[1]) ** 0.5)  # Assuming square spatial dimensions
                    total_tokens += seq_len
                    print(f"    Scale {i}: estimated {spatial_dim}x{spatial_dim} spatial = {seq_len} tokens")
                else:
                    total_tokens += tokens.numel()
                    print(f"    Scale {i}: {tokens.shape} = {tokens.numel()} tokens")
                
                print(f"      Token range: [{tokens.min().item()}, {tokens.max().item()}]")
                print(f"      Unique tokens: {tokens.unique().numel()}/{model.vocab_size}")
                print(f"      Usage: {tokens.unique().numel()/model.vocab_size*100:.1f}%")
            
            print(f"    Total tokens: {total_tokens}")
            
            # Test reconstruction at different scales
            if same_shape:
                reconstructed_list = model.idxBl_to_img(gt_idx_Bl, True)
                print(f"\n🔄 Reconstruction test (same_shape=True):")
                print(f"    Number of reconstructed volumes: {len(reconstructed_list)}")
                
                for i, reconstructed in enumerate(reconstructed_list):
                    print(f"    Scale {i} reconstruction:")
                    print(f"      Shape: {reconstructed.shape}")
                    if recon_enable:
                        recon_loss = torch.nn.functional.l1_loss(reconstructed, vol_pm1)
                        print(f"      L1 loss: {recon_loss.item():.6f}")
                    print(f"      Range: [{reconstructed.min().item():.3f}, {reconstructed.max().item():.3f}]")
                    print(f"      Mean: {reconstructed.mean().item():.3f}, Std: {reconstructed.std().item():.3f}")
                    
                    # Save each reconstructed volume
                    save_volume_slices(output_dir, base_name, reconstructed, tag=f'recon_scale_{i}')
            else:
                reconstructed_list = model.idxBl_to_img(gt_idx_Bl, False)
                print(f"\n🔄 Reconstruction test (same_shape=False - native scales):")
                print(f"    Number of reconstructed volumes: {len(reconstructed_list)}")
                
                for i, reconstructed in enumerate(reconstructed_list):
                    print(f"    Scale {i} reconstruction (native scale):")
                    print(f"      Shape: {reconstructed.shape}")
                    print(f"      Range: [{reconstructed.min().item():.3f}, {reconstructed.max().item():.3f}]")
                    print(f"      Mean: {reconstructed.mean().item():.3f}, Std: {reconstructed.std().item():.3f}")
                    print(f"      L1 loss: Skipped (different sizes)")
                    
                    # Save each reconstructed volume at native resolution
                    save_volume_slices(output_dir, base_name, reconstructed, tag=f'recon_scale_{i}_native')
            
            # Test standard forward pass for comparison
            print(f"\n🔄 Standard forward pass (for comparison):")
            
            # Initialize distributed training only for standard forward pass
            if not dist.is_initialized():
                init_distributed()
            
            reconstructed_std, usages, vq_loss = model(vol_pm1, ret_usages=True)
            print(f"    Standard reconstruction shape: {reconstructed_std.shape}")
            if recon_enable:
                recon_loss_std = torch.nn.functional.l1_loss(reconstructed_std, vol_pm1)
                print(f"    Standard reconstruction L1 loss: {recon_loss_std.item():.6f}")
            print(f"    Standard reconstruction range: [{reconstructed_std.min().item():.3f}, {reconstructed_std.max().item():.3f}]")
            print(f"    VQ loss: {vq_loss.item():.6f}")
            if usages:
                print(f"    Codebook usages: {[f'{u:.2f}' for u in usages]}")
            
            # Save standard reconstruction
            save_volume_slices(output_dir, base_name, reconstructed_std, tag='recon_standard')
            print(f"    💾 Standard reconstruction saved")
            
            print(f"\n📁 All volumes saved to: {output_dir}")
            
    except Exception as e:
        print(f"❌ Error during multi-scale tokenization: {e}")
        import traceback
        traceback.print_exc()


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Updated paths
    test_dir = "/home/yuchenliu/Dataset/IXI/train_val_test_split_multislice/test/mid_brain"
    best_ckpt_path = "/home/yuchenliu/VAR/local_output/vqvae_checkpoints_v128_z16_c64_lpips_idx65_75/vqvae_volume_best.pth"
    best_ckpt_path = "/home/yuchenliu/VAR/local_output/vqvae_checkpoints_v128_z16_c64_lpips_idx65_75/vqvae_volume_epoch_200.pth"
    best_ckpt_path = "/home/yuchenliu/VAR/local_output/vqvae_checkpoints_v128_z8_c32_lpips_idx65_75/vqvae_volume_best.pth"
    ckpt_name = "200pth"
    ckpt_name = "best"
        
    # Load best checkpoint metadata and print epoch
    best_ckpt, recon_enable = load_best_checkpoint(best_ckpt_path, device)

    # Build model and load weights
    model = build_model(device, num_slices=10)
    if best_ckpt is None:
        print("⚠️ Proceeding without loading weights (best checkpoint missing)")
    else:
        state = best_ckpt.get('model_state_dict', None)
        if state is None:
            print("⚠️ model_state_dict missing in checkpoint; skipping weight load")
        else:
            model.load_state_dict(state)
            print("✅ Model weights loaded from best checkpoint")
    model.eval()
    
    # Pick a test volume
    try:
        vol_path = pick_test_volume(test_dir)
    except Exception as e:
        print(f"❌ {e}")
        return

    base_name = os.path.splitext(os.path.basename(vol_path))[0]
    print(f"🧪 Testing on volume: {base_name}")

    # Load and prepare input
    with torch.no_grad():
        vol_01 = load_volume_as_tensor(vol_path, device)          # [1, S, H, W], [0,1]
        vol_pm1 = to_pm1(vol_01)                                  # [-1,1]

        print(f"Input volume shape: {vol_pm1.shape}")
        print(f"Input volume range: [{vol_pm1.min().item():.3f}, {vol_pm1.max().item():.3f}]")

        # Create output directory
        output_dir = os.path.join(
            os.path.dirname(best_ckpt_path) if best_ckpt is not None else '/home/yuchenliu/VAR/local_output',
            f"recon_multi_scale_{base_name}_{ckpt_name}"
        )
        os.makedirs(output_dir, exist_ok=True)
        
        # Save original volume
        save_volume_slices(output_dir, base_name, vol_pm1, tag='original')
        
        # Test multi-scale tokenization with same_shape=True
        same_shape = True
        print(f"\n🔧 Testing with same_shape={same_shape}")
        test_multi_scale_tokenization(model, vol_pm1, output_dir, base_name, recon_enable=recon_enable, same_shape=same_shape)
        
        # Test multi-scale tokenization with same_shape=False
        same_shape = False
        print(f"\n🔧 Testing with same_shape={same_shape}")
        test_multi_scale_tokenization(model, vol_pm1, output_dir, base_name, recon_enable=recon_enable, same_shape=same_shape)
        
        print(f"\n📁 All outputs written to: {output_dir}")


if __name__ == '__main__':
    try:
        main()
    finally:
        # Cleanup distributed training if it was initialized
        if dist.is_initialized():
            cleanup_distributed()
