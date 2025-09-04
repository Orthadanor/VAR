
import torch
import torch.nn as nn
from models.vqvae import VQVAE, VectorQuantizer2


class VQVAEVol(VQVAE):
    """VQVAE modified for 3D volume data (multiple slices)"""
    
    def __init__(self, vocab_size=4096, z_channels=32, ch=160, beta=1.0, test_mode=False, 
                 share_quant_resi=4, v_patch_nums=(1, 2, 3, 4, 5, 6, 8), num_slices=10):
        # Store number of slices for volume processing
        self.num_slices = num_slices
        
        # Initialize parent with keyword arguments to avoid order issues
        super().__init__(
            vocab_size=vocab_size,
            z_channels=z_channels,
            ch=ch,
            beta=beta,  # Default commitment loss weight
            test_mode=test_mode,
            share_quant_resi=share_quant_resi,
            v_patch_nums=v_patch_nums
        )
        
        # Modify the encoder and decoder for volume input/output
        self._modify_for_volume()
    
    def _modify_for_volume(self):
        """Modify the first and last conv layers for volume input (num_slices channels)"""
        
        # Modify encoder input layer (RGB -> Volume slices)
        old_encoder_conv_in = self.encoder.conv_in
        self.encoder.conv_in = nn.Conv2d(
            self.num_slices,  # Change from 3 to num_slices channels
            old_encoder_conv_in.out_channels,
            kernel_size=old_encoder_conv_in.kernel_size,
            stride=old_encoder_conv_in.stride,
            padding=old_encoder_conv_in.padding
        )
        
        # Initialize new conv layer weights (average the RGB weights across channels)
        with torch.no_grad():
            # Average the weights across the 3 input channels and repeat for num_slices
            avg_weights = old_encoder_conv_in.weight.data.mean(dim=1, keepdim=True)
            # Repeat for num_slices channels
            self.encoder.conv_in.weight.data = avg_weights.repeat(1, self.num_slices, 1, 1) / self.num_slices
            if old_encoder_conv_in.bias is not None:
                self.encoder.conv_in.bias.data = old_encoder_conv_in.bias.data.clone()
        
        # Modify decoder output layer (RGB -> Volume slices)
        old_decoder_conv_out = self.decoder.conv_out
        self.decoder.conv_out = nn.Conv2d(
            old_decoder_conv_out.in_channels,
            self.num_slices,  # Change from 3 to num_slices channels
            kernel_size=old_decoder_conv_out.kernel_size,
            stride=old_decoder_conv_out.stride,
            padding=old_decoder_conv_out.padding
        )
        
        # Initialize new conv layer weights (average the RGB weights across channels)
        with torch.no_grad():
            # Average the weights across the 3 output channels and repeat for num_slices
            avg_weights = old_decoder_conv_out.weight.data.mean(dim=0, keepdim=True)
            # Repeat for num_slices channels
            self.decoder.conv_out.weight.data = avg_weights.repeat(self.num_slices, 1, 1, 1) / self.num_slices
            if old_decoder_conv_out.bias is not None:
                avg_bias = old_decoder_conv_out.bias.data.mean(dim=0, keepdim=True)
                self.decoder.conv_out.bias.data = avg_bias.repeat(self.num_slices)
    
    def encode(self, x):
        """Encode volume data"""
        assert x.shape[1] == self.num_slices, f"Expected {self.num_slices} channels, got {x.shape[1]}"
        return super().encode(x)
    
    def decode(self, h):
        """Decode to volume data"""
        result = super().decode(h)
        assert result.shape[1] == self.num_slices, f"Expected {self.num_slices} channels output, got {result.shape[1]}"
        return result
    
    def forward(self, inp, ret_usages=False):
        """
        Forward pass for volume data
        Args:
            inp: Volume input of shape [B, num_slices, H, W]
            ret_usages: Whether to return codebook usage statistics
        Returns:
            reconstructed volume, usages, vq_loss, commitment_loss, codebook_loss
        """
        # Ensure input has correct shape
        if inp.dim() == 4 and inp.shape[1] != self.num_slices:
            raise ValueError(f"Expected input with {self.num_slices} channels, got {inp.shape[1]}")
        
        return super().forward(inp, ret_usages=ret_usages)
    
    def img_to_reconstructed_img(self, x, v_patch_nums=None, last_one=False):
        """
        Reconstruct volume from input volume
        Args:
            x: Input volume [B, num_slices, H, W]
            v_patch_nums: Optional patch numbers for multi-scale
            last_one: Whether to return only the last scale
        Returns:
            Reconstructed volume(s)
        """
        f = self.quant_conv(self.encoder(x))
        ls_f_hat_BChw = self.quantize.f_to_idxBl_or_fhat(f, to_fhat=True, v_patch_nums=v_patch_nums)
        if last_one:
            return self.decoder(self.post_quant_conv(ls_f_hat_BChw[-1])).clamp_(-1, 1)
        else:
            return [self.decoder(self.post_quant_conv(f_hat)).clamp_(-1, 1) for f_hat in ls_f_hat_BChw]

