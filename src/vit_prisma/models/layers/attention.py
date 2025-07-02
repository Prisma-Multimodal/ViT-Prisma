import torch.nn as nn
import torch

import logging
from vit_prisma.prisma_tools import HookPoint
from vit_prisma.configs.HookedViTConfig import HookedViTConfig

from typing import Union, Dict, Optional, List, Tuple

from jaxtyping import Float, Int

import numpy as np

import einops

from vit_prisma.prisma_tools import FactoredMatrix
from fancy_einsum import einsum

import torch.nn.functional as F



import torch
import torch.nn as nn
import einops
from typing import Optional, Tuple, Union


class Attention(nn.Module):

    def __init__(
            self,
            cfg: Union[Dict, HookedViTConfig],
            layer_id: Optional[int] = None,
    ):
        super().__init__()
        if isinstance(cfg, Dict):
            cfg = HookedViTConfig.from_dict(cfg)

        self.cfg = cfg

        # Initialize parameters
        self.W_Q = nn.Parameter(
            torch.empty(
                self.cfg.n_heads,
                self.cfg.d_model,
                self.cfg.d_head,
                dtype = self.cfg.dtype
            )
        )
        self.W_K = nn.Parameter(
            torch.empty(
                self.cfg.n_heads,
                self.cfg.d_model,
                self.cfg.d_head,
                dtype = self.cfg.dtype
            )
        )
        self.W_V = nn.Parameter(
            torch.empty(
                self.cfg.n_heads,
                self.cfg.d_model,
                self.cfg.d_head,
                dtype = self.cfg.dtype
            )
        )
        self.W_O = nn.Parameter(
            torch.empty(
                self.cfg.n_heads,
                self.cfg.d_head,
                self.cfg.d_model,
                dtype = self.cfg.dtype
            )
        )

        # Initialize biases
        self.b_Q = nn.Parameter(
            torch.zeros(self.cfg.n_heads, self.cfg.d_head, dtype=self.cfg.dtype)
        )
        self.b_K = nn.Parameter(
            torch.zeros(self.cfg.n_heads, self.cfg.d_head, dtype=self.cfg.dtype)
        )
        self.b_V = nn.Parameter(
            torch.zeros(self.cfg.n_heads, self.cfg.d_head, dtype=self.cfg.dtype)
        )
        self.b_O = nn.Parameter(torch.zeros(self.cfg.d_model, dtype=self.cfg.dtype))


        # Add hook points
        self.hook_k = HookPoint()  # [batch, pos, head_index, d_head]
        self.hook_q = HookPoint()  # [batch, pos, head_index, d_head]
        self.hook_v = HookPoint()  # [batch, pos, head_index, d_head]
        self.hook_z = HookPoint()  # [batch, pos, head_index, d_head]
        self.hook_attn_scores = HookPoint()  # [batch, head_index, query_pos, key_pos]
        self.hook_pattern = HookPoint()  # [batch, head_index, query_pos, key_pos]
        self.hook_result = HookPoint()  # [batch, pos, head_index, d_model]

        self.layer_id = layer_id

        # Note to Sonia: check this.
        # attn_scale is a constant that we divide the attention scores by pre-softmax. I'm not entirely sure why it matters, but it's probably a mix of softmax not being scale invariant and numerical stability?
        if self.cfg.use_attn_scale:
            self.attn_scale = np.sqrt(self.cfg.d_head)
        else:
            self.attn_scale = 1.0

    @property
    def OV(self) -> FactoredMatrix:
        """
        OV-Circuit, as defined in A Mathematical Framework. Because there's no non-linearity between the value vector and the output of the layer, the output is purely determined by the matrix W_OV = W_V @ W_O, and not W_V or W_O individually. (Mathematically, for a single head, output == pattern @ residual @ W_V @ W_O, see the glossary for more)

        Done in the order W_V, W_O because the paper uses left-multiplying weight matrices, and TransformerLens uses right-multiplying, sorry!

        Returns a FactoredMatrix, with left matrix W_V [head_index, d_model, d_head] and right matrix W_O [head_index, d_head, d_model] - this is a low rank factorisation of the underlying [head_index, d_model, d_model]. FactoredMatrix has helper functions to deal with these large matrices efficiently. To get the OV circuit of a head k, attn.OV[k] works.
        """
        return FactoredMatrix(self.W_V, self.W_O)

    @property
    def QK(self) -> FactoredMatrix:
        """
        QK-Circuit, as defined in A Mathematical Framework. Because there's no non-linearity in the key-query dot product, the output is purely determined by the matrix W_QK = W_Q.T @ W_K, and not W_Q or W_K individually. (Mathematically, for a single head, pattern = destination_residual.T @ W_Q.T @ W_K @ source-residual, see the glossary for more).

        Done in the order Q on the left, K on the right, because the pattern has dimensions [destination_pos, source_pos]

        Returns a FactoredMatrix, with left matrix W_Q [head_index, d_model, d_head] and right matrix W_K.T [head_index, d_head, d_model] - this is a low rank factorisation of the underlying [head_index, d_model, d_model] matrix. FactoredMatrix has helper functions to deal with these large matrices efficiently. To get the QK circuit of a head k, attn.QK[k] works.
        """
        W_K_transpose = einops.rearrange(
            self.W_K, "head_index d_model d_head -> head_index d_head d_model"
        )
        return FactoredMatrix(self.W_Q, W_K_transpose)
    
    def forward(
            self,
            query_input: Union[
                Float[torch.Tensor, "batch pos d_model"],
                Float[torch.Tensor, "batch pos head_index d_model"],
            ],
            key_input: Union[
                Float[torch.Tensor, "batch pos d_model"],
                Float[torch.Tensor, "batch pos head_index d_model"],
            ],
            value_input: Union[
                Float[torch.Tensor, "batch pos d_model"],
                Float[torch.Tensor, "batch pos head_index d_model"],
            ],
            attention_mask: Optional[Float[torch.Tensor, "batch pos pos"]] = None,
    ) -> Float[torch.Tensor, "batch pos d_model"]:
        

        
        q, k, v  = self.calculate_qkv_matrices(query_input, key_input, value_input)

        attn_scores = self.calculate_attn_scores(q, k, attention_mask)
        attn_scores = self.hook_attn_scores(attn_scores)

        pattern = F.softmax(attn_scores, dim=-1) # where do I do normalization? 
        pattern = torch.where(torch.isnan(pattern), torch.zeros_like(pattern), pattern)
        pattern = self.hook_pattern(pattern)

        pattern = pattern.to(self.cfg.dtype)
        z = self.calculate_z_scores(v, pattern)

        if not self.cfg.use_attn_result:
            out = (
                (
                    einsum(
                        "batch pos head_index d_head, \
                        head_index d_head d_model -> \
                        batch pos d_model",
                        z,
                        self.W_O,
                    )
                )
                + self.b_O
            )
        else: 
            # Explicitly calculate the attention result so it can be accessed by a hook.
            # Off by default to not eat through GPU memory.
            result = self.hook_result(
                einsum(
                    "batch pos head_index d_head, \
                    head_index d_head d_model -> \
                    batch pos head_index d_model",
                    z,
                    self.W_O,
                )
            )
            out = (
                einops.reduce(result, "batch pos head_index d_model -> batch pos d_model", "sum")
                + self.b_O
            )
        return out

    def calculate_qkv_matrices(
            self,
            query_input: Union[
                Float[torch.Tensor, "batch pos d_model"],
                Float[torch.Tensor, "batch pos head_index d_model"],
            ],
            key_input: Union[
                Float[torch.Tensor, "batch pos d_model"],
                Float[torch.Tensor, "batch pos head_index d_model"],
            ],
            value_input: Union[
                Float[torch.Tensor, "batch pos d_model"],
                Float[torch.Tensor, "batch pos head_index d_model"],
            ]
    ) -> Tuple[
        Float[torch.Tensor, "batch pos head_index d_head"],
        Float[torch.Tensor, "batch pos head_index d_head"],
        Float[torch.Tensor, "batch pos head_index d_head"],
    ]:
        """
        Calculate the Q, K, V matrices for the attention layer. This is done by multiplying the input by the weight matrices and adding the biases.

        Returns a tuple of (Q, K, V) matrices, each of shape [batch, pos, head_index, d_head]
        """

        if self.cfg.use_split_qkv_input or self.cfg.use_attn_in:
            qkv_einops_string = "batch pos head_index d_model"
        else:
            qkv_einops_string = "batch pos d_model"


        q = self.hook_q(
            einsum(
                f"{qkv_einops_string}, head_index d_model d_head \
                -> batch pos head_index d_head",
                query_input,
                self.W_Q,
            )
            + self.b_Q
        )  # [batch, pos, head_index, d_head]
        k = self.hook_k(
            einsum(
                f"{qkv_einops_string}, head_index d_model d_head \
                -> batch pos head_index d_head",
                key_input,
                self.W_K,
            )
            + self.b_K
        )  # [batch, pos, head_index, d_head]
        v = self.hook_v(
            einsum(
                f"{qkv_einops_string}, head_index d_model d_head \
                -> batch pos head_index d_head",
                value_input,
                self.W_V,
            )
            + self.b_V
        )  # [batch, pos, head_index, d_head]
        return q, k, v
    
    def calculate_attn_scores(
            self,
            q: Float[torch.Tensor, "batch pos head_index d_head"],
            k: Float[torch.Tensor, "batch pos head_index d_head"],
            attention_mask: Optional[Float[torch.Tensor, "batch pos pos"]] = None,
    ) -> Float[torch.Tensor, "batch head_index query_pos key_pos"]:
        """
        Calculate the attention scores for the attention layer. This is done by multiplying the Q and K matrices together, and dividing by the square root of the dimension of the key vectors.

        Returns a tensor of shape [batch, head_index, query_pos, key_pos]
        """
        attn_scores = einsum(
            "batch query_pos head_index d_head, batch key_pos head_index d_head -> batch head_index query_pos key_pos",
            q,
            k,
        )
        attn_scores = attn_scores / self.attn_scale
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask
        return attn_scores
    
    def calculate_z_scores(
            self,
            v: Float[torch.Tensor, "batch key_pos head_index d_head"],
            pattern: Float[torch.Tensor, "batch head_index query_pos key_pos"],
    ) -> Float[torch.Tensor, "batch query_pos head_index d_head"]:
        z = self.hook_z(
            einsum(
                "batch key_pos head_index d_head, \
                batch head_index query_pos key_pos -> \
                batch query_pos head_index d_head",
                v,
                pattern,
            )
        )
        return z


def rotate_queries_or_keys(x, pos):
    B, num_heads, N, D = x.size()

    # similar to inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
    # they are computing this every time. instead HF style is to compute the inv_freq once and store it
    # -- compute angle for each position
    omega = torch.arange(D // 2, dtype=x.dtype, device=x.device)
    omega /= D / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)
    freq = torch.einsum("..., f -> ... f", pos, omega)  # (..., N, D/2), outer product

    # -- build rotation matrix and apply
    emb_sin = freq.sin()  # (..., N, D/2)
    emb_cos = freq.cos()  # (..., N, D/2)

    emb_sin = emb_sin.squeeze(-1).repeat(1, 1, 1, 2)
    emb_cos = emb_cos.squeeze(-1).repeat(1, 1, 1, 2)

    # --
    y = x.unflatten(-1, (-1, 2))
    y1, y2 = y.unbind(dim=-1)

    y = torch.stack((-y2, y1), dim=-1)
    y = y.flatten(-2)
    return (x * emb_cos) + (y * emb_sin)


class VJEPARopeAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # Set dimensions
        self.n_heads = cfg.n_heads
        self.d_model = cfg.d_model
        self.d_head = cfg.d_head
        self.dtype = cfg.dtype

        # Factorized QKV weights and biases (same naming as Prisma Attention)
        self.W_Q = nn.Parameter(torch.empty(self.n_heads, self.d_model, self.d_head, dtype=self.dtype))
        self.W_K = nn.Parameter(torch.empty(self.n_heads, self.d_model, self.d_head, dtype=self.dtype))
        self.W_V = nn.Parameter(torch.empty(self.n_heads, self.d_model, self.d_head, dtype=self.dtype))
        self.W_O = nn.Parameter(torch.empty(self.n_heads, self.d_head, self.d_model, dtype=self.dtype))

        self.b_Q = nn.Parameter(torch.zeros(self.n_heads, self.d_head, dtype=self.dtype))
        self.b_K = nn.Parameter(torch.zeros(self.n_heads, self.d_head, dtype=self.dtype))
        self.b_V = nn.Parameter(torch.zeros(self.n_heads, self.d_head, dtype=self.dtype))
        self.b_O = nn.Parameter(torch.zeros(self.d_model, dtype=self.dtype))

        # Hook points (from Prisma Attention)
        self.hook_q = HookPoint()
        self.hook_k = HookPoint()
        self.hook_v = HookPoint()
        self.hook_z = HookPoint()
        self.hook_attn_scores = HookPoint()
        self.hook_pattern = HookPoint()
        self.hook_result = HookPoint()

        # Rotary embedding params — assuming same grid and spatial structure as VJEPA
        self.grid_size = cfg.crop_size // cfg.patch_size
        self.grid_depth = cfg.frames_per_clip // cfg.tubelet_size

        # Calculate attention head size for rotary split:
        # Here d_head = total dimension per head, split into 3 dims for D, H, W.
        self.d_dim = 2 * ((self.d_head // 3) // 2)
        self.h_dim = 2 * ((self.d_head // 3) // 2)
        self.w_dim = 2 * ((self.d_head // 3) // 2)

        self.scaling = self.d_head ** -0.5
        self.is_causal = False

    def _get_frame_pos(self, ids):
        tokens_per_frame = self.grid_size * self.grid_size
        return ids // tokens_per_frame

    def _get_height_pos(self, ids):
        tokens_per_frame = self.grid_size * self.grid_size
        frame_ids = self._get_frame_pos(ids)
        ids = ids - tokens_per_frame * frame_ids
        tokens_per_row = self.grid_size
        return ids // tokens_per_row

    def get_position_ids(self, x, masks=None):
        device = x.device
        token_size = x.size(1)
        if masks is not None:
            ids = masks.unsqueeze(1).repeat(1, self.n_heads, 1)
        else:
            ids = torch.arange(token_size, device=device)
        frame_ids = self._get_frame_pos(ids)
        height_ids = self._get_height_pos(ids)
        tokens_per_frame = self.grid_size * self.grid_size
        tokens_per_row = self.grid_size
        width_ids = (ids - tokens_per_frame * frame_ids) - tokens_per_row * height_ids
        return frame_ids, height_ids, width_ids

    def apply_rotary_embeddings(self, qk, pos_ids):

        # qk: [batch, pos, head, d_head]
        d_mask, h_mask, w_mask = pos_ids

        if d_mask.dim() == 1:
            d_mask = d_mask.unsqueeze(0).unsqueeze(-1)
            h_mask = h_mask.unsqueeze(0).unsqueeze(-1)
            w_mask = w_mask.unsqueeze(0).unsqueeze(-1)

        s = 0
        qkd = rotate_queries_or_keys(qk[..., s : s + self.d_dim], pos=d_mask)
        s += self.d_dim
        qkh = rotate_queries_or_keys(qk[..., s : s + self.h_dim], pos=h_mask)
        s += self.h_dim
        qkw = rotate_queries_or_keys(qk[..., s : s + self.w_dim], pos=w_mask)
        s += self.w_dim
        if s < self.d_head:
            qkr = qk[..., s:]
            qk = torch.cat([qkd, qkh, qkw, qkr], dim=-1)
        else:
            qk = torch.cat([qkd, qkh, qkw], dim=-1)
        return qk

    def forward(
        self,
        query_input: Union[torch.Tensor, Float[torch.Tensor, "batch pos d_model"]],
        key_input: Union[torch.Tensor, Float[torch.Tensor, "batch pos d_model"]],
        value_input: Union[torch.Tensor, Float[torch.Tensor, "batch pos d_model"]],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        query_input, key_input, value_input shape: [batch, pos, d_model]
        returns: [batch, pos, d_model]
        """

        # Calculate Q, K, V with einsum
        q = self.hook_q(
            einsum(
                "batch pos d_model, head_index d_model d_head -> batch pos head_index d_head",
                query_input,
                self.W_Q,
            )
            + self.b_Q
        )
        k = self.hook_k(
            einsum(
                "batch pos d_model, head_index d_model d_head -> batch pos head_index d_head",
                key_input,
                self.W_K,
            )
            + self.b_K
        )
        v = self.hook_v(
            einsum(
                "batch pos d_model, head_index d_model d_head -> batch pos head_index d_head",
                value_input,
                self.W_V,
            )
            + self.b_V
        )

        # Rotary embeddings expect Q,K in shape [batch, pos, head, d_head]
        # Compute positional IDs (frame, height, width)
        pos_ids = self.get_position_ids(query_input)
        q = self.apply_rotary_embeddings(q, pos_ids)
        k = self.apply_rotary_embeddings(k, pos_ids)

        # Attention scores: QK^T
        attn_scores = einsum(
            "batch query_pos head_index d_head, batch key_pos head_index d_head -> batch head_index query_pos key_pos",
            q,
            k,
        )
        attn_scores = attn_scores * self.scaling
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask
        attn_scores = self.hook_attn_scores(attn_scores)

        # Softmax to get attention pattern
        pattern = torch.softmax(attn_scores, dim=-1)
        pattern = torch.where(torch.isnan(pattern), torch.zeros_like(pattern), pattern)
        pattern = self.hook_pattern(pattern)

        # Calculate z = pattern @ V
        z = self.hook_z(
            einsum(
                "batch key_pos head_index d_head, batch head_index query_pos key_pos -> batch query_pos head_index d_head",
                v,
                pattern,
            )
        )

        # Output projection with W_O and bias
        result = self.hook_result(
            einsum(
                "batch query_pos head_index d_head, head_index d_head d_model -> batch query_pos head_index d_model",
                z,
                self.W_O,
            )
        )
        # Sum over heads + add bias
        out = einops.reduce(result, "batch pos head_index d_model -> batch pos d_model", "sum") + self.b_O

        return out
