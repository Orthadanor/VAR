"""
LPIPS perceptual loss adapted for grayscale images and volume data.
Stripped version of https://github.com/richzhang/PerceptualSimilarity/tree/master/models
Modified to handle grayscale input by duplicating channels to RGB.
Added volumetric mean loss calculation for processing multiple slices.
"""

import torch
import torch.nn as nn
from torchvision import models
import os

class LPIPS(nn.Module):
    # Learned perceptual metric for grayscale images and volumes
    def __init__(self, lpips_path="/home/yuchenliu/VAR/checkpoints/vgg.pth", use_dropout=False, volume_mode=False):
        super().__init__()
        self.volume_mode = volume_mode  # Flag to enable volume processing
        
        # build models
        self.net = Vgg16(requires_grad=False)
        self.lins = nn.ModuleList([NetLinLayer(c, use_dropout=use_dropout) for c in [64, 128, 256, 512, 512]])  # c: vgg16 feature dimensions
        
        # detach parameters & set to eval mode
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
        
        # load weights if checkpoint exists
        # Replace the loading section in __init__ method

        # load weights if checkpoint exists
        if lpips_path and os.path.exists(lpips_path):
            try:
                print(f"Loading LPIPS weights from: {lpips_path}")
                checkpoint = torch.load(lpips_path, map_location='cpu')
                
                # Handle different checkpoint formats
                if isinstance(checkpoint, dict) and ('state_dict' in checkpoint or 'model' in checkpoint):
                    if 'state_dict' in checkpoint:
                        state_dict = checkpoint['state_dict']
                    else:
                        state_dict = checkpoint['model']
                else:
                    state_dict = checkpoint
                
                # Fix the key mapping for your checkpoint format
                model_dict = self.state_dict()
                filtered_dict = {}
                
                # Map lin0, lin1, etc. to lins.0, lins.1, etc.
                for key, value in state_dict.items():
                    if key.startswith('lin') and key[3].isdigit():  # lin0, lin1, etc.
                        # Extract the number and convert to lins.X format
                        lin_num = key[3]  # Get the digit after 'lin'
                        new_key = f"lins.{lin_num}.model.1.weight"
                        if new_key in model_dict and model_dict[new_key].shape == value.shape:
                            filtered_dict[new_key] = value
                            # print(f"Mapped {key} -> {new_key}")
                
                if filtered_dict:
                    self.load_state_dict(filtered_dict, strict=False)
                    print(f"Successfully loaded {len(filtered_dict)} LPIPS linear layers from checkpoint")
                    print("Using ImageNet pretrained VGG backbone")
                else:
                    print("No compatible layers found in checkpoint, using random initialization")
                    
            except Exception as e:
                print(f"Failed to load checkpoint: {e}")
                print("Using random initialization")
        else:
            print(f"Checkpoint not found at {lpips_path}, using random initialization")
        
        # register helper tensors for VGG normalization
        self.register_buffer('shift', torch.tensor([-.030, -.088, -.188], dtype=torch.float32).view(1, 3, 1, 1).contiguous())
        self.register_buffer('scale_inv', 1. / torch.tensor([.458, .448, .450], dtype=torch.float32).view(1, 3, 1, 1).contiguous())
    
    def grayscale_to_rgb(self, x):
        """
        Convert grayscale [B, 1, H, W] to RGB [B, 3, H, W] by duplicating channels
        """
        if x.shape[1] == 1:  # Grayscale input
            return x.repeat(1, 3, 1, 1)  # Duplicate channel 3 times
        elif x.shape[1] == 3:  # Already RGB
            return x
        else:
            raise ValueError(f"Expected 1 or 3 channels, got {x.shape[1]}")
    
    def volume_to_rgb(self, x):
        """
        Convert volume [B, num_slices, H, W] to RGB by processing each slice individually
        Returns: [B * num_slices, 3, H, W]
        """
        B, num_slices, H, W = x.shape

        x_reshaped = x.view(B * num_slices, 1, H, W)
        x_rgb = self.grayscale_to_rgb(x_reshaped) # Convert each slice to RGB: [B * num_slices, 3, H, W]

        return x_rgb
    
    def forward(self, inp, rec):
        """
        :param inp: grayscale image or volume for calculating LPIPS loss, [-1, 1]
                   shape [B, 1, H, W] for grayscale or [B, num_slices, H, W] for volume
        :param rec: grayscale image or volume for calculating LPIPS loss, [-1, 1]
                   shape [B, 1, H, W] for grayscale or [B, num_slices, H, W] for volume
        :return: lpips loss (scalar)
        """
        if self.volume_mode and inp.shape[1] > 1:
            # Volume mode: process each slice individually
            return self._forward_volume(inp, rec)
        else:
            # Standard grayscale mode
            return self._forward_grayscale(inp, rec)
    
    def _forward_grayscale(self, inp, rec):
        """Standard grayscale LPIPS forward pass"""
        B = inp.shape[0]
        
        # Convert grayscale to RGB by duplicating channels
        inp_rgb = self.grayscale_to_rgb(inp)  # [B, 3, H, W]
        rec_rgb = self.grayscale_to_rgb(rec)  # [B, 3, H, W]
        
        # Concatenate and apply VGG normalization
        inp_and_recs = torch.cat((inp_rgb, rec_rgb), dim=0).sub(self.shift).mul_(self.scale_inv)
        
        # Extract features
        inp_and_recs = self.net(inp_and_recs)   # inp_and_recs: List[Tensor], len(inp_and_recs) == 5
        
        # Compute perceptual difference
        diff = 0.
        for inp_and_rec, lin in zip(inp_and_recs, self.lins):
            diff += lin.model((normalize_tensor(inp_and_rec[:B]) - normalize_tensor(inp_and_rec[B:])).square_()).mean()
        return diff
    
    def _forward_volume(self, inp, rec):
        """Volume LPIPS forward pass - processes each slice individually"""
        B, num_slices, H, W = inp.shape
        
        # Convert volumes to RGB format: [B * num_slices, 3, H, W]
        inp_rgb = self.volume_to_rgb(inp)
        rec_rgb = self.volume_to_rgb(rec)
        
        # Concatenate and apply VGG normalization
        inp_and_recs = torch.cat((inp_rgb, rec_rgb), dim=0).sub(self.shift).mul_(self.scale_inv)
        
        # Extract features
        inp_and_recs = self.net(inp_and_recs)   # inp_and_recs: List[Tensor], len(inp_and_recs) == 5
        
        # Compute perceptual difference for all slices
        diff = 0.
        total_slices = B * num_slices
        for inp_and_rec, lin in zip(inp_and_recs, self.lins):
            diff += lin.model((normalize_tensor(inp_and_rec[:total_slices]) - normalize_tensor(inp_and_rec[total_slices:])).square_()).mean()
        
        return diff


class NetLinLayer(nn.Module):
    """ A single linear layer which does a 1x1 conv """
    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super(NetLinLayer, self).__init__()
        layers = [nn.Dropout(), ] if use_dropout else [nn.Identity()]
        layers += [nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False), ]
        self.model = nn.Sequential(*layers)


class Vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg16, self).__init__()
        # Use ImageNet pretrained VGG16 for better feature extraction
        # Handle both old and new torchvision API
        try:
            # New API (torchvision >= 0.13)
            vgg_pretrained_features = models.vgg16(weights='IMAGENET1K_V1').features
        except TypeError:
            # Old API (torchvision < 0.13)
            vgg_pretrained_features = models.vgg16(pretrained=True).features
            
        self.slice1 = torch.nn.Sequential(*[vgg_pretrained_features[x] for x in range(4)])
        self.slice2 = torch.nn.Sequential(*[vgg_pretrained_features[x] for x in range(4, 9)])
        self.slice3 = torch.nn.Sequential(*[vgg_pretrained_features[x] for x in range(9, 16)])
        self.slice4 = torch.nn.Sequential(*[vgg_pretrained_features[x] for x in range(16, 23)])
        self.slice5 = torch.nn.Sequential(*[vgg_pretrained_features[x] for x in range(23, 30)])
        self.N_slices = 5
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False
    
    def forward(self, x):
        h_relu1_2 = self.slice1(x)
        h_relu2_2 = self.slice2(h_relu1_2)
        h_relu3_3 = self.slice3(h_relu2_2)
        h_relu4_3 = self.slice4(h_relu3_3)
        h_relu5_3 = self.slice5(h_relu4_3)
        return h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3


def normalize_tensor(x, eps=1e-10):
    norm_factor = torch.sum(x.square(), dim=1, keepdim=True).add_(1e-9).sqrt_()
    return x / (norm_factor + eps)


def inspect_checkpoint(checkpoint_path):
    """Debug function to inspect checkpoint contents"""
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return
    
    print(f"Inspecting checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    if isinstance(checkpoint, dict):
        print("Checkpoint is a dictionary with keys:")
        for key in checkpoint.keys():
            if isinstance(checkpoint[key], dict):
                print(f"  {key}: dict with {len(checkpoint[key])} items")
            else:
                print(f"  {key}: {type(checkpoint[key])}")
                
        # Look for state_dict
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
        print("Checkpoint is direct state_dict")
    
    print(f"\nState dict has {len(state_dict)} keys:")
    for i, (key, value) in enumerate(state_dict.items()):
        if i < 10:  # Show first 10 keys
            print(f"  {key}: {tuple(value.shape) if hasattr(value, 'shape') else type(value)}")
        elif i == 10:
            print(f"  ... and {len(state_dict) - 10} more keys")
    
    # Check for LPIPS-specific keys
    lpips_keys = [k for k in state_dict.keys() if k.startswith('lins.')]
    vgg_keys = [k for k in state_dict.keys() if k.startswith('net.')]
    
    print(f"\nLPIPS linear layer keys: {len(lpips_keys)}")
    print(f"VGG network keys: {len(vgg_keys)}")
    
    if lpips_keys:
        print("Sample LPIPS keys:")
        for key in lpips_keys[:5]:
            print(f"  {key}")
    
    if vgg_keys:
        print("Sample VGG keys:")
        for key in vgg_keys[:5]:
            print(f"  {key}")


def test_lpips_grayscale():
    """Test function for grayscale LPIPS"""
    print("Testing LPIPS for grayscale images...")
    
    # First inspect the checkpoint
    checkpoint_path = "/home/yuchenliu/VAR/checkpoints/vgg.pth"
    inspect_checkpoint(checkpoint_path)
    print("\n" + "="*50 + "\n")
    
    # Create test grayscale images
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    inp = torch.randn(2, 1, 128, 128).to(device)  # Grayscale input
    rec = torch.randn(2, 1, 128, 128).to(device)  # Grayscale reconstruction
    
    # Initialize LPIPS
    lpips = LPIPS().to(device)
    
    # Test forward pass
    with torch.no_grad():
        loss = lpips(inp, rec)
    
    print(f"Input shape: {inp.shape}")
    print(f"LPIPS loss: {loss.item():.6f}")
    
    # Test with identical images (should be close to 0)
    with torch.no_grad():
        identical_loss = lpips(inp, inp)
    print(f"LPIPS loss for identical images: {identical_loss.item():.6f}")
    
    # Test gradient computation
    rec.requires_grad_(True)
    loss = lpips(inp, rec)
    loss.backward()
    print(f"Gradient computed successfully, grad norm: {rec.grad.norm().item():.6f}")


def test_lpips_volume():
    """Test function for volume LPIPS"""
    print("Testing LPIPS for volume data...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create test volume data: [B, num_slices, H, W]
    B, num_slices, H, W = 2, 10, 128, 128
    inp_vol = torch.randn(B, num_slices, H, W).to(device)  # Volume input
    rec_vol = torch.randn(B, num_slices, H, W).to(device)  # Volume reconstruction
    
    # Initialize LPIPS with volume mode
    lpips_vol = LPIPS(volume_mode=True).to(device)
    
    # Test forward pass
    with torch.no_grad():
        loss = lpips_vol(inp_vol, rec_vol)
    
    print(f"Volume input shape: {inp_vol.shape}")
    print(f"Volume LPIPS loss: {loss.item():.6f}")
    
    # Test with identical volumes (should be close to 0)
    with torch.no_grad():
        identical_loss = lpips_vol(inp_vol, inp_vol)
    print(f"Volume LPIPS loss for identical volumes: {identical_loss.item():.6f}")
    
    # Test gradient computation
    rec_vol.requires_grad_(True)
    loss = lpips_vol(inp_vol, rec_vol)
    loss.backward()
    print(f"Volume gradient computed successfully, grad norm: {rec_vol.grad.norm().item():.6f}")
    
    # Compare with standard grayscale mode
    print("\nComparing with standard grayscale mode...")
    lpips_std = LPIPS(volume_mode=False).to(device)
    
    # Take first slice for comparison
    inp_slice = inp_vol[:, 0:1, :, :]  # [B, 1, H, W]
    rec_slice = rec_vol[:, 0:1, :, :]  # [B, 1, H, W]
    
    with torch.no_grad():
        std_loss = lpips_std(inp_slice, rec_slice)
        vol_loss_single = lpips_vol(inp_slice, rec_slice)  # Should work the same for single slice
    
    print(f"Standard LPIPS (single slice): {std_loss.item():.6f}")
    print(f"Volume LPIPS (single slice): {vol_loss_single.item():.6f}")
    print(f"Difference: {abs(std_loss.item() - vol_loss_single.item()):.8f}")


if __name__ == '__main__':
    test_lpips_grayscale()
    print("\n" + "="*50 + "\n")
    test_lpips_volume()