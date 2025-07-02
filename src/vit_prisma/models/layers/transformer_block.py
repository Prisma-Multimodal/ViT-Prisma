from typing import Dict, Union, Optional

import einops
import torch
import torch.nn as nn
from jaxtyping import Float
from vit_prisma.configs.HookedViTConfig import HookedViTConfig
from vit_prisma.models.layers.attention import Attention, VJEPARopeAttention
from vit_prisma.models.layers.layer_norm import LayerNorm, LayerNormPre
from vit_prisma.models.layers.mlp import MLP
from vit_prisma.prisma_tools.hook_point import HookPoint


def add_head_dimension(
        tensor: Float[torch.Tensor, "batch pos d_model"],
        n_heads:int,
        clone_tensor=True,
        # `einops.repeat` uses a view in torch, so we generally clone the tensor to avoid using shared storage for each head entry
    ):
        repeated_tensor = einops.repeat(
            tensor,
            "batch pos d_model -> batch pos n_heads d_model",
            n_heads=n_heads,
        )
        if clone_tensor:
            return repeated_tensor.clone()
        else:
                return repeated_tensor

class TransformerBlock(nn.Module):
    """
    Transformer block.
    """

    def __init__(self, cfg: Union[Dict, HookedViTConfig], block_index=None):
        super().__init__()

        if isinstance(cfg, Dict):
            cfg = HookedViTConfig.from_dict(cfg)

        self.cfg = cfg

        if self.cfg.normalization_type == "LN":
            self.ln1 = LayerNorm(self.cfg)
            if not self.cfg.attn_only:
                self.ln2 = LayerNorm(self.cfg)
        elif self.cfg.normalization_type == "LNPre":
            self.ln1 = LayerNormPre(self.cfg)
            if not self.cfg.attn_only:
                self.ln2 = LayerNormPre(self.cfg)
        elif self.cfg.normalization_type is None:
            self.ln1 = nn.Identity()
            if not self.cfg.attn_only:
                self.ln2 = nn.Identity()
        else:
            raise ValueError(f"Invalid normalization type: {self.cfg.normalization_type}")
        
        if hasattr(self.cfg, "use_rope") and self.cfg.use_rope:
            self.attn = VJEPARopeAttention(self.cfg)
        else:
            self.attn  = Attention(self.cfg)

        if not self.cfg.attn_only:
            self.mlp = MLP(self.cfg)
        
        self.hook_attn_in = HookPoint()
        self.hook_q_input = HookPoint()
        self.hook_k_input = HookPoint()
        self.hook_v_input = HookPoint()
        self.hook_mlp_in = HookPoint()
        self.hook_attn_out = HookPoint()
        self.hook_mlp_out = HookPoint()
        self.hook_resid_pre = HookPoint()

        if not self.cfg.attn_only:
            self.hook_resid_mid = HookPoint()

        self.hook_resid_post = HookPoint()

        self.attn_dropout = nn.Dropout(cfg.attn_dropout_rate)
        self.mlp_dropout = nn.Dropout(cfg.mlp_dropout_rate)

    def forward(
            self,
            resid_pre: Float[torch.Tensor, "batch pos d_model"],
            attn_mask: Optional[Float[torch.Tensor, "batch pos pos"]] = None,
    ) -> Float[torch.Tensor, "batch pos d_model"]:
        
        resid_pre = self.hook_resid_pre(resid_pre)

        if self.cfg.use_attn_in or self.cfg.use_split_qkv_input:
            # We're adding a head dimension
            attn_in = add_head_dimension(resid_pre, self.cfg.n_heads, clone_tensor=False)
        else:
            attn_in = resid_pre

        if self.cfg.use_attn_in:
            attn_in = self.hook_attn_in(attn_in.clone())
        
        if self.cfg.use_split_qkv_input:
            query_input = self.hook_q_input(attn_in.clone())
            key_input = self.hook_k_input(attn_in.clone())
            value_input = self.hook_v_input(attn_in.clone())
        else:
            query_input = attn_in
            key_input = attn_in
            value_input = attn_in
        
        attn_out = self.attn(
                query_input = self.ln1(query_input),
                key_input = self.ln1(key_input),
                value_input = self.ln1(value_input),
                attention_mask=attn_mask,
            )

        attn_out = self.attn_dropout(attn_out)

        # Take hook fn
        
        attn_out = self.hook_attn_out(
            attn_out
        )

        if not self.cfg.attn_only: 
            resid_mid = self.hook_resid_mid(
                resid_pre + attn_out
            )
            mlp_in = (
                resid_mid
                if not self.cfg.use_hook_mlp_in
                else self.hook_mlp_in(resid_mid.clone())
            )
            normalized_resid_mid = self.ln2(mlp_in)
            mlp_out = self.mlp(normalized_resid_mid)
            mlp_out = self.mlp_dropout(mlp_out)
            mlp_out = self.hook_mlp_out(mlp_out)
            resid_post = self.hook_resid_post(resid_mid + mlp_out)
        else:
            resid_post = self.hook_resid_post(resid_pre + attn_out)
        
        return resid_post

    
class BertBlock(nn.Module):
    """
    
    Just like TransformerBlock but applies layernorm after attn and after mlp, not before. Necessary for some CLIP models.
    """

    def __init__(self, cfg: Union[Dict, HookedViTConfig], block_index):
        super().__init__()

        if isinstance(cfg, Dict):
            cfg = HookedViTConfig.from_dict(cfg)
        
        self.cfg = cfg

        if self.cfg.normalization_type == "LN":
            self.ln1 = LayerNorm(self.cfg)
            if not self.cfg.attn_only:
                self.ln2 = LayerNorm(self.cfg)
        elif self.cfg.normalization_type == "LNPre":
            self.ln1 = LayerNormPre(self.cfg)
            if not self.cfg.attn_only:
                self.ln2 = LayerNormPre(self.cfg)
        elif self.cfg.normalization_type is None:
            self.ln1 = nn.Identity()
            if not self.cfg.attn_only:
                self.ln2 = nn.Identity()
        else:
            raise ValueError(f"Invalid normalization type: {self.cfg.normalization_type}")
        
        self.attn  = Attention(self.cfg)

        if not self.cfg.attn_only:
            self.mlp = MLP(self.cfg)
        
        self.hook_attn_in = HookPoint()
        self.hook_q_input = HookPoint()
        self.hook_k_input = HookPoint()
        self.hook_v_input = HookPoint()
        self.hook_mlp_in = HookPoint()
        self.hook_attn_out = HookPoint()
        self.hook_mlp_out = HookPoint()
        self.hook_resid_pre = HookPoint()

        if not self.cfg.attn_only:
            self.hook_resid_mid = HookPoint()

        self.hook_resid_post = HookPoint()

    def forward(
            self,
            resid_pre: Float[torch.Tensor, "batch pos d_model"],
    ) -> Float[torch.Tensor, "batch pos d_model"]:
        
        resid_pre = self.hook_resid_pre(resid_pre)

        if self.cfg.use_attn_in or self.cfg.use_split_qkv_input:
            # We're adding a head dimension
            attn_in = add_head_dimension(resid_pre, self.cfg.n_heads, clone_tensor=False)
        else:
            attn_in = resid_pre

        if self.cfg.use_attn_in:
            attn_in = self.hook_attn_in(attn_in.clone())
        
        if self.cfg.use_split_qkv_input:
            query_input = self.hook_q_input(attn_in.clone())
            key_input = self.hook_k_input(attn_in.clone())
            value_input = self.hook_v_input(attn_in.clone())
        else:
            query_input = attn_in
            key_input = attn_in
            value_input = attn_in
        
        attn_out = self.attn(
                query_input,
                key_input,
                value_input,
            )

        # Take hook fn
        
        attn_out = self.hook_attn_out(
            attn_out
        )
        
        attn_out = self.ln1(attn_out)

        if not self.cfg.attn_only: 
            resid_mid = self.hook_resid_mid(
                resid_pre + attn_out
            )
            mlp_in = (
                resid_mid
                if not self.cfg.use_hook_mlp_in
                else self.hook_mlp_in(resid_mid.clone())
            )
            normalized_resid_mid = mlp_in
            mlp_out = self.hook_mlp_out(self.mlp(normalized_resid_mid))
            
            mlp_out = self.ln2(mlp_out)
            
            resid_post = self.hook_resid_post(resid_mid + mlp_out)
        else:
            resid_post = self.hook_resid_post(resid_pre + attn_out)
        
        return resid_post
