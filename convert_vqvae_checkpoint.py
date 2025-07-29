#!/usr/bin/env python3
"""
Convert VQVAE training checkpoint to format expected by VAR training
"""

import torch
import os
import sys

def convert_vqvae_checkpoint():
    # Path to your training checkpoint
    training_ckpt_path = '/home/yuchenliu/VAR/local_output/vqvae_checkpoints_v128_z16_c64_b1/vqvae_final.pth'
    
    # Output path for VAR training
    output_ckpt_path = '/home/yuchenliu/VAR/vqvae_v128_z16_c64_b1.pth'
    
    print(f"Converting VQVAE checkpoint...")
    print(f"Input:  {training_ckpt_path}")
    print(f"Output: {output_ckpt_path}")
    
    if not os.path.exists(training_ckpt_path):
        print(f"❌ Training checkpoint not found: {training_ckpt_path}")
        sys.exit(1)
    
    # Load training checkpoint
    print("🔄 Loading training checkpoint...")
    checkpoint = torch.load(training_ckpt_path, map_location='cpu')
    
    # Print checkpoint info
    print(f"📊 Checkpoint info:")
    for key in checkpoint.keys():
        if key == 'model_state_dict':
            print(f"  • {key}: <state_dict with {len(checkpoint[key])} keys>")
        else:
            print(f"  • {key}: {checkpoint[key]}")
    
    # Extract model state dict
    if 'model_state_dict' in checkpoint:
        model_state_dict = checkpoint['model_state_dict']
        print(f"✅ Extracted model_state_dict with {len(model_state_dict)} parameters")
        
        # Print some key parameter shapes for verification
        print(f"🔍 Key parameter shapes:")
        for key, tensor in model_state_dict.items():
            if 'embedding' in key or 'vocab' in key or 'quantize' in key:
                pass
                # print(f"  • {key}: {tensor.shape}")
    else:
        print(f"❌ No 'model_state_dict' found in checkpoint!")
        print(f"Available keys: {list(checkpoint.keys())}")
        sys.exit(1)
    
    # Save in format expected by VAR training (direct state dict)
    print(f"💾 Saving converted checkpoint...")
    torch.save(model_state_dict, output_ckpt_path)
    
    # Verify the saved checkpoint
    print(f"🔍 Verifying saved checkpoint...")
    loaded_ckpt = torch.load(output_ckpt_path, map_location='cpu')
    print(f"✅ Verification successful - {len(loaded_ckpt)} parameters loaded")
    
    print(f"🎉 Conversion completed successfully!")
    print(f"💡 You can now use '{os.path.basename(output_ckpt_path)}' for VAR training")

if __name__ == '__main__':
    convert_vqvae_checkpoint()
