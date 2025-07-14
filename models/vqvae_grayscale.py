# Create a new file: models/vqvae_grayscale.py

import torch
import torch.nn as nn
from models.vqvae import VQVAE, VectorQuantizer2


class VQVAEGrayscale(VQVAE):
    """VQVAE modified for single-channel (grayscale) images"""
    
    def __init__(self, vocab_size=4096, z_channels=32, ch=160, test_mode=True, 
                 share_quant_resi=4, v_patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16)):
        # Initialize parent with modified parameters
        super().__init__(vocab_size, z_channels, ch, test_mode, share_quant_resi, v_patch_nums)
        
        # Modify the encoder and decoder for single channel input/output
        self._modify_for_grayscale()
    
    def _modify_for_grayscale(self):
        """Modify the first and last conv layers for grayscale (1 channel)"""
        
        # Modify encoder input layer (RGB -> Grayscale)
        old_encoder_conv_in = self.encoder.conv_in
        self.encoder.conv_in = nn.Conv2d(
            1,  # Change from 3 to 1 channel
            old_encoder_conv_in.out_channels,
            kernel_size=old_encoder_conv_in.kernel_size,
            stride=old_encoder_conv_in.stride,
            padding=old_encoder_conv_in.padding
        )
        
        # Initialize new conv layer weights (average the RGB weights)
        with torch.no_grad():
            # Average the weights across the 3 input channels
            self.encoder.conv_in.weight.data = old_encoder_conv_in.weight.data.mean(dim=1, keepdim=True)
            if old_encoder_conv_in.bias is not None:
                self.encoder.conv_in.bias.data = old_encoder_conv_in.bias.data.clone()
        
        # Modify decoder output layer (RGB -> Grayscale)
        old_decoder_conv_out = self.decoder.conv_out
        self.decoder.conv_out = nn.Conv2d(
            old_decoder_conv_out.in_channels,
            1,  # Change from 3 to 1 channel
            kernel_size=old_decoder_conv_out.kernel_size,
            stride=old_decoder_conv_out.stride,
            padding=old_decoder_conv_out.padding
        )
        
        # Initialize new conv layer weights (average the RGB weights)
        with torch.no_grad():
            # Average the weights across the 3 output channels
            self.decoder.conv_out.weight.data = old_decoder_conv_out.weight.data.mean(dim=0, keepdim=True)
            if old_decoder_conv_out.bias is not None:
                self.decoder.conv_out.bias.data = old_decoder_conv_out.bias.data.mean(dim=0, keepdim=True)
    
    def encode(self, x):
        """Encode single-channel images"""
        assert x.shape[1] == 1, f"Expected 1 channel, got {x.shape[1]}"
        return super().encode(x)
    
    def decode(self, h):
        """Decode to single-channel images"""
        result = super().decode(h)
        assert result.shape[1] == 1, f"Expected 1 channel output, got {result.shape[1]}"
        return result