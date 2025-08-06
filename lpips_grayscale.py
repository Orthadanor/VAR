"""
LPIPS perceptual loss adapted for grayscale images.
Stripped version of https://github.com/richzhang/PerceptualSimilarity/tree/master/models
Modified to handle grayscale input by duplicating channels to RGB.
"""

import torch
import torch.nn as nn
from torchvision import models
import os

class LPIPS(nn.Module):
    # Learned perceptual metric for grayscale images
    def __init__(self, lpips_path="/home/yuchenliu/VAR/checkpoints/vgg.pth", use_dropout=False):
        super().__init__()
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
                            print(f"Mapped {key} -> {new_key}")
                
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
        # if lpips_path and os.path.exists(lpips_path):
        #     try:
        #         print(f"Loading LPIPS weights from: {lpips_path}")
        #         # Load with strict=True to match original implementation
        #         self.load_state_dict(torch.load(lpips_path, map_location='cpu'), strict=True)
        #         print("Successfully loaded LPIPS checkpoint with strict=True")
                    
        #     except Exception as e:
        #         print(f"Failed to load checkpoint with strict=True: {e}")
        #         print("Falling back to filtered loading...")
                
        #         # Fallback to the original filtered approach
        #         try:
        #             checkpoint = torch.load(lpips_path, map_location='cpu')
                    
        #             # Handle different checkpoint formats
        #             if isinstance(checkpoint, dict):
        #                 if 'state_dict' in checkpoint:
        #                     state_dict = checkpoint['state_dict']
        #                 elif 'model' in checkpoint:
        #                     state_dict = checkpoint['model']
        #                 else:
        #                     state_dict = checkpoint
        #             else:
        #                 state_dict = checkpoint
                    
        #             # Filter to load only compatible layers
        #             model_dict = self.state_dict()
        #             filtered_dict = {}
                    
        #             for k, v in state_dict.items():
        #                 if k in model_dict and model_dict[k].shape == v.shape:
        #                     filtered_dict[k] = v
                    
        #             if filtered_dict:
        #                 self.load_state_dict(filtered_dict, strict=False)
        #                 print(f"Loaded {len(filtered_dict)} layers from checkpoint")
        #             else:
        #                 print("No compatible layers found in checkpoint, using random initialization")
                        
        #         except Exception as e2:
        #             print(f"Fallback loading also failed: {e2}")
        #             print("Using random initialization")
        # else:
        #     print(f"Checkpoint not found at {lpips_path}, using random initialization")
        
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
    
    def forward(self, inp, rec):
        """
        :param inp: grayscale image for calculating LPIPS loss, [-1, 1], shape [B, 1, H, W]
        :param rec: grayscale image for calculating LPIPS loss, [-1, 1], shape [B, 1, H, W]
        :return: lpips loss (scalar)
        """
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


if __name__ == '__main__':
    test_lpips_grayscale()