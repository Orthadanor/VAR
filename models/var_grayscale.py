# Create a new file: models/var_grayscale.py

import torch
import torch.nn as nn
from models.var import VAR
from models.vqvae_grayscale import VQVAEGrayscale
from models.helpers import gumbel_softmax_with_rng, sample_with_top_k_top_p_


class VARGrayscale(VAR):
    """VAR model modified for grayscale image generation"""
    
    def __init__(self, vae_local: VQVAEGrayscale, **kwargs):
        # Force num_classes=1 and cond_drop_rate=1.0 for unconditional training
        kwargs['num_classes'] = 1
        kwargs['cond_drop_rate'] = 1.0
        super().__init__(vae_local=vae_local, **kwargs)
        
        # Override class embedding to always return the same unconditional embedding
        self.class_emb = nn.Parameter(torch.randn(1, self.C))  # Single learnable embedding
    
    def forward(self, label_B, x_BLCv_wo_first_l):
        """Forward pass - ignores class labels and uses unconditional embedding"""
        B = x_BLCv_wo_first_l.shape[0]
        bg, ed = self.begin_ends[self.prog_si] if self.prog_si >= 0 else (0, self.L)
        
        # Always use the same unconditional embedding
        cond_BD = self.class_emb.expand(B, -1)
        sos = cond_BD.unsqueeze(1).expand(B, self.first_l, -1) + self.pos_start.expand(B, self.first_l, -1)
        
        if self.prog_si == 0: 
            x_BLC = sos
        else: 
            x_BLC = torch.cat((sos, self.word_embed(x_BLCv_wo_first_l.float())), dim=1)
        x_BLC += self.lvl_embed(self.lvl_1L[:, :ed].expand(B, -1)) + self.pos_1LC[:, :ed]
        
        attn_bias = self.attn_bias_for_masking[:, :, :ed, :ed]
        cond_BD_or_gss = self.shared_ada_lin(cond_BD)
        
        # Rest of forward pass same as original
        temp = x_BLC.new_ones(8, 8)
        main_type = torch.matmul(temp, temp).dtype
        
        x_BLC = x_BLC.to(dtype=main_type)
        cond_BD_or_gss = cond_BD_or_gss.to(dtype=main_type)
        attn_bias = attn_bias.to(dtype=main_type)
        
        for i, b in enumerate(self.blocks):
            x_BLC = b(x=x_BLC, cond_BD=cond_BD_or_gss, attn_bias=attn_bias)
        x_BLC = self.get_logits(x_BLC.float(), cond_BD)
        
        return x_BLC
    
    @torch.no_grad()
    def autoregressive_infer_unconditional(self, B: int, g_seed=None, top_k=0, top_p=0.0, more_smooth=False):
        """Unconditional inference for grayscale images"""
        if g_seed is None: 
            rng = None
        else: 
            self.rng.manual_seed(g_seed)
            rng = self.rng
        
        # Use unconditional embedding for all samples
        sos = cond_BD = self.class_emb.expand(B, -1)
        
        lvl_pos = self.lvl_embed(self.lvl_1L) + self.pos_1LC
        next_token_map = sos.unsqueeze(1).expand(B, self.first_l, -1) + self.pos_start.expand(B, self.first_l, -1) + lvl_pos[:, :self.first_l]
        
        cur_L = 0
        f_hat = sos.new_zeros(B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1])
        
        for b in self.blocks: b.attn.kv_caching(True)
        for si, pn in enumerate(self.patch_nums):
            ratio = si / self.num_stages_minus_1
            cur_L += pn*pn
            
            cond_BD_or_gss = self.shared_ada_lin(cond_BD)
            x = next_token_map
            for b in self.blocks:
                x = b(x=x, cond_BD=cond_BD_or_gss, attn_bias=None)
            logits_BlV = self.get_logits(x, cond_BD)
            
            # Sample without CFG (since we're unconditional)
            idx_Bl = sample_with_top_k_top_p_(logits_BlV, rng=rng, top_k=top_k, top_p=top_p, num_samples=1)[:, :, 0]
            
            if not more_smooth:
                h_BChw = self.vae_quant_proxy[0].embedding(idx_Bl)
            else:
                gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)
                h_BChw = gumbel_softmax_with_rng(logits_BlV.mul(1 + ratio), tau=gum_t, hard=False, dim=-1, rng=rng) @ self.vae_quant_proxy[0].embedding.weight.unsqueeze(0)
            
            h_BChw = h_BChw.transpose_(1, 2).reshape(B, self.Cvae, pn, pn)
            f_hat, next_token_map = self.vae_quant_proxy[0].get_next_autoregressive_input(si, len(self.patch_nums), f_hat, h_BChw)
            
            if si != self.num_stages_minus_1:
                next_token_map = next_token_map.view(B, self.Cvae, -1).transpose(1, 2)
                next_token_map = self.word_embed(next_token_map) + lvl_pos[:, cur_L:cur_L + self.patch_nums[si+1] ** 2]
        
        for b in self.blocks: b.attn.kv_caching(False)
        # Return grayscale images (B, 1, H, W) in [0, 1]
        return self.vae_proxy[0].fhat_to_img(f_hat).add_(1).mul_(0.5)