import copy

import torch
import torch.nn as nn

from diffusers import ModelMixin
from diffusers.models.attention_processor import Attention
from diffusers.models.attention import FeedForward

import logging

logger = logging.getLogger(__name__)

from .byt5_block_byt5_mapper import T5EncoderBlockByT5Mapper
from .identity_byt5_mapper import IdentityByT5Mapper

byt5_mapper_dict = [T5EncoderBlockByT5Mapper, IdentityByT5Mapper]
byt5_mapper_dict = {mapper.__name__: mapper for mapper in byt5_mapper_dict}


class AttentionBlock(nn.Module):
    def __init__(
        self,
        q_dim,
        kv_dim,
        num_heads,
        head_dim,
        ffn_act_fn='geglu',
    ):
        super().__init__()
        self.q_dim = q_dim
        self.kv_dim = kv_dim
        self.num_heads = num_heads
        self.dim_head = head_dim
        self.norm1 = nn.LayerNorm(q_dim, eps=1e-6)
        self.self_attn = Attention(
            query_dim=q_dim,
            cross_attention_dim=None,
            heads=num_heads,
            dim_head=head_dim,
        )
        self.norm2 = nn.LayerNorm(q_dim, eps=1e-6)
        self.cross_attn = Attention(
            query_dim=q_dim,
            cross_attention_dim=kv_dim,
            heads=num_heads,
            dim_head=head_dim,
        )
        self.norm3 = nn.LayerNorm(q_dim, eps=1e-6)
        self.ffn = FeedForward(
            q_dim,
            activation_fn=ffn_act_fn,
        )
    
    def forward(self, clip_feat, byt5_feat, self_attn_mask, cross_attn_mask):
        norm_clip_feat = self.norm1(clip_feat)
        attn_output = self.self_attn(
            norm_clip_feat,
            attention_mask=self_attn_mask,
        )
        clip_feat = attn_output + clip_feat
        
        norm_clip_feat = self.norm2(clip_feat)
        attn_output = self.cross_attn(
            hidden_states=norm_clip_feat,
            encoder_hidden_states=byt5_feat,
            attention_mask=cross_attn_mask,
        )
        clip_feat = attn_output + clip_feat
        
        norm_clip_feat = self.norm3(clip_feat)
        ff_output = self.ffn(norm_clip_feat)
        clip_feat = ff_output + clip_feat
        
        return clip_feat


class DetrStyleClipByT5Fuser(ModelMixin):
    def __init__(
        self, 
        byt5_config, 
        byt5_mapper_config,
        inner_dim,
        num_heads,
        num_layers, 
        clip_channels):
        super().__init__()
        byt5_mapper_config = copy.deepcopy(byt5_mapper_config)
        byt5_mapper_type = byt5_mapper_config.pop('type')
        self.byt5_mapper = byt5_mapper_dict[byt5_mapper_type](
            byt5_config=byt5_config, 
            **byt5_mapper_config
        )
        head_dim = inner_dim // num_heads
        assert head_dim * num_heads == inner_dim, "inner_dim must be divisible by num_heads"
        self.attn_blocks = nn.ModuleList(
            [
                AttentionBlock(
                    q_dim=clip_channels,
                    kv_dim=byt5_config.d_model,
                    num_heads=num_heads,
                    head_dim=head_dim,
                )
            for _ in range(num_layers)]
        )
    
    def forward(self, 
                clip_feat, 
                byt5_feat, 
                byt5_attn_mask,
                clip_text_feat_idx,
                byt5_text_feat_idx,
                refill_idx,
                self_attn_mask, 
                cross_attn_mask,
            ):
        fetch_refill_mask = self_attn_mask[:, 0, :] == 1
        self_attn_mask = (1 - self_attn_mask) * -10000.0
        cross_attn_mask = (1 - cross_attn_mask) * -10000.0
        # clip_text_idx_lists:
        # List[List[[idx1, idx2, idx3], [idx1, idx2], ...], ...]
        byt5_feat = self.byt5_mapper(byt5_feat, attention_mask=byt5_attn_mask)
        clip_text_feat = torch.cat((clip_feat, clip_feat.new_zeros(
            clip_feat.shape[0],
            1,
            clip_feat.shape[2],
        )), dim=1)
        byt5_text_feat = torch.cat((byt5_feat, byt5_feat.new_zeros(
            byt5_feat.shape[0],
            1,
            byt5_feat.shape[2],
        )), dim=1)
        clip_text_feat_idx = clip_text_feat_idx.repeat(1, 1, clip_feat.shape[2])
        
        byt5_text_feat_idx = byt5_text_feat_idx.repeat(1, 1, byt5_feat.shape[2])
        
        # b, clip_text_max_len,c
        clip_text_feat = torch.gather(clip_text_feat, dim=1, index=clip_text_feat_idx)
        # b, byt5_text_max_len,c
        byt5_text_feat = torch.gather(byt5_text_feat, dim=1, index=byt5_text_feat_idx)
        
        for block in self.attn_blocks:
            clip_text_feat = block(
                clip_text_feat,
                byt5_text_feat,
                self_attn_mask,
                cross_attn_mask,
            )
        clip_feat[refill_idx] = clip_text_feat[fetch_refill_mask]
        
        return clip_feat