# test_vqvae_tokenization.py
import torch
from models.vqvae_grayscale import VQVAEGrayscale
from utils.data_mri import build_mri_dataset_grayscale

def test_tokenization():
    vqvae = VQVAEGrayscale(
        vocab_size=4096,
        z_channels=32,
        ch=160,
        test_mode=True
    ).cuda()
    
    # Load best checkpoint
    checkpoint = torch.load('./checkpoints_vqvae_mri/vqvae_mri_best.pth')
    vqvae.load_state_dict(checkpoint)
    
    # Test with MRI data
    _, dataset_val, _ = build_mri_dataset_grayscale(
        '/home/yuchenliu/Dataset/IXI/t1_np_masked_128_unconditional', 
        final_reso=256
    )
    
    # Get a sample
    image, _ = dataset_val[0]
    image = image.unsqueeze(0).cuda()
    
    print(f"Input image shape: {image.shape}")
    
    # Test tokenization
    with torch.no_grad():
        gt_idx_Bl = vqvae.img_to_idxBl(image)
        
        print(f"Number of scales: {len(gt_idx_Bl)}")
        for i, tokens in enumerate(gt_idx_Bl):
            scale_size = int(tokens.shape[1] ** 0.5)
            print(f"Scale {i}: {scale_size}x{scale_size} = {tokens.shape[1]} tokens")
            print(f"  Token range: [{tokens.min().item()}, {tokens.max().item()}]")
            print(f"  Unique tokens: {tokens.unique().numel()}")

if __name__ == '__main__':
    test_tokenization()