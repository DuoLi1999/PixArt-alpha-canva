from typing import Optional, Dict, Any
import copy

import torch
import torch.nn as nn
from timm.models.layers import DropPath
from timm.models.vision_transformer import PatchEmbed, Mlp

from diffusion.model.builder import MODELS
from diffusion.model.utils import auto_grad_checkpoint, to_2tuple
from diffusion.model.nets.PixArt_blocks import t2i_modulate, CaptionEmbedder, WindowAttention, MultiHeadCrossAttention, T2IFinalLayer, TimestepEmbedder, LabelEmbedder, FinalLayer
from diffusion.utils.logger import get_root_logger
from diffusion.model.nets.PixArt import PixArtBlock

class PixartInsertGlyphBlock(PixArtBlock):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, drop_path=0., window_size=0, input_size=None, use_rel_pos=False,use_flash_attn=False,glyph_attn_insertion_type = 0, **block_kwargs):
        super().__init__(hidden_size, num_heads, mlp_ratio, drop_path, window_size, input_size, use_rel_pos,use_flash_attn, **block_kwargs)
        self.hidden_size = hidden_size
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = WindowAttention(hidden_size, num_heads=num_heads, qkv_bias=True,
                                    input_size=input_size if window_size == 0 else (window_size, window_size),
                                    use_rel_pos=use_rel_pos, 
                                    use_flash_attn=use_flash_attn,**block_kwargs)
        self.cross_attn = MultiHeadCrossAttention(hidden_size, num_heads, use_flash_attn=use_flash_attn,**block_kwargs)
        # add glyph_cross_attention:
        self.glyph_cross_attn = MultiHeadCrossAttention(hidden_size, num_heads, use_flash_attn=use_flash_attn,**block_kwargs)
        # zero conv for glyph_cross_attention
        self.glyph_out = nn.Linear(hidden_size, hidden_size)
        for p in self.glyph_out.parameters():
            nn.init.zeros_(p)
        
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        # to be compatible with lower version pytorch
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=int(hidden_size * mlp_ratio), act_layer=approx_gelu, drop=0)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.window_size = window_size
        self.scale_shift_table = nn.Parameter(torch.randn(6, hidden_size) / hidden_size ** 0.5)

    def get_inserted_modules(self):
        return (self.glyph_cross_attn, self.glyph_out)
    
    def get_inserted_modules_names(self):
        return ("glyph_cross_attn", "glyph_out")
    
    def get_origin_modules(self):
        inserted_modules = self.get_inserted_modules()
        origin_modules = []
        for module in self.children():
            if module not in inserted_modules:
                origin_modules.append(module)
        return tuple(origin_modules)
    
    @classmethod
    def from_PixArt_block(
            cls, 
            pixart_block,
            glyph_attn_insertion_type = 0,
            clone_from_cross_attn=False,
            block_count=0,
            patch_size=2,
            input_size=64,
            **block_kwargs
        ):
        window_block_indexes = block_kwargs['window_block_indexes']
        hidden_size=pixart_block.hidden_size
        num_heads=pixart_block.attn.num_heads
        mlp_ratio=pixart_block.mlp.fc1.out_features / hidden_size
        drop_path=pixart_block.drop_path.drop_prob if isinstance(pixart_block.drop_path, DropPath) else 0
        window_size= block_kwargs['window_size'] if block_count in window_block_indexes else 0,
        window_size=window_size[0] if isinstance(window_size, tuple) else window_size
        input_size= (input_size // patch_size, input_size // patch_size)
        use_rel_pos= block_kwargs['use_rel_pos'] if block_count in window_block_indexes else False,
        use_rel_pos=use_rel_pos[0] if isinstance(use_rel_pos, tuple) else use_rel_pos
        use_flash_attn=pixart_block.attn.use_flash_attn
    

        model = cls(
            hidden_size=hidden_size,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop_path=drop_path,
            input_size=input_size,
            window_size=window_size,
            use_rel_pos=use_rel_pos,
            use_flash_attn=use_flash_attn,
            glyph_attn_insertion_type=glyph_attn_insertion_type,
           
        )
        missing_keys, unexpected_keys = model.load_state_dict(
            pixart_block.state_dict(),
            strict=False,
        )
        assert len(unexpected_keys) == 0
        assert all(i.startswith('glyph') for i in missing_keys)
        if clone_from_cross_attn:
            model.glyph_cross_attn = copy.deepcopy(model.self.cross_attn)
        
        return model




    def forward(self, x, y, t, glyph, mask=None, glyph_mask = None,  **kwargs):
        B, N, C = x.shape

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (self.scale_shift_table[None] + t.reshape(B, 6, -1)).chunk(6, dim=1)
        x = x + self.drop_path(gate_msa * self.attn(t2i_modulate(self.norm1(x), shift_msa, scale_msa)).reshape(B, N, C))
        x = x + self.cross_attn(x, y, mask)
        x = x + self.glyph_out(self.glyph_cross_attn(x, glyph, glyph_mask))

        x = x + self.drop_path(gate_mlp * self.mlp(t2i_modulate(self.norm2(x), shift_mlp, scale_mlp)))

        return x

class PixArtBlock(nn.Module):
    """
    A PixArt block with adaptive layer norm (adaLN-single) conditioning.
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, drop_path=0., window_size=0, input_size=None, use_rel_pos=False,use_flash_attn=False, **block_kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = WindowAttention(hidden_size, num_heads=num_heads, qkv_bias=True,
                                    input_size=input_size if window_size == 0 else (window_size, window_size),
                                    use_rel_pos=use_rel_pos, 
                                    use_flash_attn=use_flash_attn,**block_kwargs)
        self.cross_attn = MultiHeadCrossAttention(hidden_size, num_heads, use_flash_attn=use_flash_attn,**block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        # to be compatible with lower version pytorch
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=int(hidden_size * mlp_ratio), act_layer=approx_gelu, drop=0)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.window_size = window_size
        self.scale_shift_table = nn.Parameter(torch.randn(6, hidden_size) / hidden_size ** 0.5)

    def forward(self, x, y, t, mask=None, **kwargs):
        B, N, C = x.shape

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (self.scale_shift_table[None] + t.reshape(B, 6, -1)).chunk(6, dim=1)
        x = x + self.drop_path(gate_msa * self.attn(t2i_modulate(self.norm1(x), shift_msa, scale_msa)).reshape(B, N, C))
        x = x + self.cross_attn(x, y, mask)
        x = x + self.drop_path(gate_mlp * self.mlp(t2i_modulate(self.norm2(x), shift_mlp, scale_mlp)))

        return x



