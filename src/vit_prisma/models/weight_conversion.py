"""
Prisma Repo
By Sonia Joseph

Copyright (c) Sonia Joseph. All rights reserved.

Inspired by TransformerLens. Some functions have been adapted from the TransformerLens project.
For more information on TransformerLens, visit: https://github.com/neelnanda-io/TransformerLens
"""
from open_clip import get_model_config
import open_clip
import logging
from functools import partial
from typing import Dict
from typing import Union

import einops
import timm
import torch

from vit_prisma.configs.HookedViTConfig import HookedViTConfig
from tokenizers.models import Model
from transformers import (
    AutoConfig,
    ViTForImageClassification,
    VivitForVideoClassification,
    CLIPModel,
    ViTModel,
)
from vit_prisma.configs.HookedTextTransformerConfig import HookedTextTransformerConfig
from vit_prisma.utils.enums import ModelType

try:
    from huggingface_hub import hf_hub_download

    hf_hub_download = partial(
        hf_hub_download, library_name="open_clip", library_version="2.20.0"
    )
    _has_hf_hub = True
except ImportError:
    hf_hub_download = None
    _has_hf_hub = False

import json


def convert_vjepa_weights(
        old_state_dict,
        cfg: HookedViTConfig,
        device = 'cuda',
):

    print("CONFIG", cfg)
    
    new_vision_model_state_dict = {}

    new_vision_model_state_dict["pos_embed.W_pos"] = old_state_dict["embeddings.position_embeddings"].squeeze()

    new_vision_model_state_dict["embed.proj.weight"] = old_state_dict["embeddings.patch_embeddings.proj.weight"]
    new_vision_model_state_dict["embed.proj.bias"] =  old_state_dict["embeddings.patch_embeddings.proj.bias"]

    new_vision_model_state_dict["ln_final.w"] = old_state_dict["layernorm.weight"]
    new_vision_model_state_dict["ln_final.b"] = old_state_dict["layernorm.bias"]
    
    # new_vision_model_state_dict["ln_pre.w"] = old_state_dict["pre_layrnorm.weight"] #typo in ClipModel
    # new_vision_model_state_dict["ln_pre.b"] = old_state_dict["pre_layrnorm.bias"]


    for layer in range(cfg.n_layers):
        layer_key = f"encoder.layer.{layer}"
        new_layer_key = f"blocks.{layer}"

        new_vision_model_state_dict[f"{new_layer_key}.ln1.w"] = old_state_dict[f"{layer_key}.norm1.weight"]
        new_vision_model_state_dict[f"{new_layer_key}.ln1.b"] = old_state_dict[f"{layer_key}.norm1.bias"]
        new_vision_model_state_dict[f"{new_layer_key}.ln2.w"] = old_state_dict[f"{layer_key}.norm2.weight"]
        new_vision_model_state_dict[f"{new_layer_key}.ln2.b"] = old_state_dict[f"{layer_key}.norm2.bias"]

        W_Q = old_state_dict[f"{layer_key}.attention.query.weight"]
        W_K = old_state_dict[f"{layer_key}.attention.key.weight"]
        W_V = old_state_dict[f"{layer_key}.attention.value.weight"]
        W_O = old_state_dict[f"{layer_key}.attention.proj.weight"]

        W_Q = einops.rearrange(W_Q, "(h dh) d-> h d dh", h=cfg.n_heads, d=cfg.d_model, dh=cfg.d_head)
        W_K = einops.rearrange(W_K, "(h dh) d-> h d dh", h=cfg.n_heads, d=cfg.d_model, dh=cfg.d_head)
        W_V = einops.rearrange(W_V, "(h dh) d-> h d dh", h=cfg.n_heads, d=cfg.d_model, dh=cfg.d_head)
        W_O = einops.rearrange(W_O, "d (h dh) -> h dh d", h=cfg.n_heads, d=cfg.d_model, dh=cfg.d_head)
        
        new_vision_model_state_dict[f"{new_layer_key}.attn.W_Q"] = W_Q
        new_vision_model_state_dict[f"{new_layer_key}.attn.W_K"] = W_K
        new_vision_model_state_dict[f"{new_layer_key}.attn.W_V"] = W_V
        new_vision_model_state_dict[f"{new_layer_key}.attn.W_O"] = W_O

        b_Q = old_state_dict[f"{layer_key}.attention.query.bias"]
        b_K = old_state_dict[f"{layer_key}.attention.key.bias"]
        b_V = old_state_dict[f"{layer_key}.attention.value.bias"]
        b_O = old_state_dict[f"{layer_key}.attention.proj.bias"]

        b_Q = einops.rearrange(b_Q, "(h dh) -> h dh", h=cfg.n_heads, dh=cfg.d_head)
        b_K = einops.rearrange(b_K, "(h dh) -> h dh", h=cfg.n_heads, dh=cfg.d_head)
        b_V = einops.rearrange(b_V, "(h dh) -> h dh", h=cfg.n_heads, dh=cfg.d_head)

        new_vision_model_state_dict[f"{new_layer_key}.attn.b_Q"] = b_Q
        new_vision_model_state_dict[f"{new_layer_key}.attn.b_K"] = b_K
        new_vision_model_state_dict[f"{new_layer_key}.attn.b_V"] = b_V
        new_vision_model_state_dict[f"{new_layer_key}.attn.b_O"] = b_O

        mlp_W_in = old_state_dict[f"{layer_key}.mlp.fc1.weight"]
        mlp_W_out = old_state_dict[f"{layer_key}.mlp.fc2.weight"]
        mlp_b_in = old_state_dict[f"{layer_key}.mlp.fc1.bias"]
        mlp_b_out = old_state_dict[f"{layer_key}.mlp.fc2.bias"]

        mlp_W_in = einops.rearrange(mlp_W_in, "m d -> d m")
        mlp_W_out = einops.rearrange(mlp_W_out, "d m -> m d")

        new_vision_model_state_dict[f"{new_layer_key}.mlp.W_in"] = mlp_W_in
        new_vision_model_state_dict[f"{new_layer_key}.mlp.W_out"] = mlp_W_out
        new_vision_model_state_dict[f"{new_layer_key}.mlp.b_in"] = mlp_b_in
        new_vision_model_state_dict[f"{new_layer_key}.mlp.b_out"] = mlp_b_out

    new_vision_model_state_dict["head.W_H"] = torch.eye(cfg.d_model)
    new_vision_model_state_dict["head.b_H"] = torch.zeros((cfg.d_model,))
        
    return new_vision_model_state_dict



def convert_kandinsky_clip_weights(
    old_state_dict,
    cfg: HookedViTConfig,
    device="cuda",
):
    new_vision_model_state_dict = {}

    logging.info("Convering Kandinsky Clip weights")

    # Convert embedding layers
    new_vision_model_state_dict["cls_token"] = (
        old_state_dict["vision_model.embeddings.class_embedding"]
        .unsqueeze(0)
        .unsqueeze(0)
    )
    new_vision_model_state_dict["pos_embed.W_pos"] = old_state_dict[
        "vision_model.embeddings.position_embedding.weight"
    ]

    # Patch embedding
    new_vision_model_state_dict["embed.proj.weight"] = old_state_dict[
        "vision_model.embeddings.patch_embedding.weight"
    ]
    new_vision_model_state_dict["embed.proj.bias"] = torch.zeros((cfg.d_model,))

    # Convert layer norms
    new_vision_model_state_dict["ln_final.w"] = old_state_dict[
        "vision_model.post_layernorm.weight"
    ]
    new_vision_model_state_dict["ln_final.b"] = old_state_dict[
        "vision_model.post_layernorm.bias"
    ]

    new_vision_model_state_dict["ln_pre.w"] = old_state_dict[
        "vision_model.pre_layrnorm.weight"
    ]
    new_vision_model_state_dict["ln_pre.b"] = old_state_dict[
        "vision_model.pre_layrnorm.bias"
    ]

    logging.info(
        "visual projection shape", old_state_dict["visual_projection.weight"].shape
    )

    # Convert transformer blocks
    logging.info("doing number of layers", cfg.n_layers)
    for layer in range(cfg.n_layers):
        old_layer_key = f"vision_model.encoder.layers.{layer}"
        new_layer_key = f"blocks.{layer}"

        # Layer norms
        new_vision_model_state_dict[f"{new_layer_key}.ln1.w"] = old_state_dict[
            f"{old_layer_key}.layer_norm1.weight"
        ]
        new_vision_model_state_dict[f"{new_layer_key}.ln1.b"] = old_state_dict[
            f"{old_layer_key}.layer_norm1.bias"
        ]
        new_vision_model_state_dict[f"{new_layer_key}.ln2.w"] = old_state_dict[
            f"{old_layer_key}.layer_norm2.weight"
        ]
        new_vision_model_state_dict[f"{new_layer_key}.ln2.b"] = old_state_dict[
            f"{old_layer_key}.layer_norm2.bias"
        ]

        # Attention weights
        W_Q = old_state_dict[f"{old_layer_key}.self_attn.q_proj.weight"]
        W_K = old_state_dict[f"{old_layer_key}.self_attn.k_proj.weight"]
        W_V = old_state_dict[f"{old_layer_key}.self_attn.v_proj.weight"]

        b_Q = old_state_dict[f"{old_layer_key}.self_attn.q_proj.bias"]
        b_K = old_state_dict[f"{old_layer_key}.self_attn.k_proj.bias"]
        b_V = old_state_dict[f"{old_layer_key}.self_attn.v_proj.bias"]

        # Reshape Q, K, V weights
        W_Q = einops.rearrange(
            W_Q, "(h dh) d -> h d dh", h=cfg.n_heads, d=cfg.d_model, dh=cfg.d_head
        )
        W_K = einops.rearrange(
            W_K, "(h dh) d -> h d dh", h=cfg.n_heads, d=cfg.d_model, dh=cfg.d_head
        )
        W_V = einops.rearrange(
            W_V, "(h dh) d -> h d dh", h=cfg.n_heads, d=cfg.d_model, dh=cfg.d_head
        )

        # Reshape Q, K, V biases
        b_Q = einops.rearrange(b_Q, "(h dh) -> h dh", h=cfg.n_heads, dh=cfg.d_head)
        b_K = einops.rearrange(b_K, "(h dh) -> h dh", h=cfg.n_heads, dh=cfg.d_head)
        b_V = einops.rearrange(b_V, "(h dh) -> h dh", h=cfg.n_heads, dh=cfg.d_head)

        # Output projection
        W_O = old_state_dict[f"{old_layer_key}.self_attn.out_proj.weight"]
        b_O = old_state_dict[f"{old_layer_key}.self_attn.out_proj.bias"]
        W_O = einops.rearrange(
            W_O, "d (h dh) -> h dh d", h=cfg.n_heads, d=cfg.d_model, dh=cfg.d_head
        )

        new_vision_model_state_dict[f"{new_layer_key}.attn.W_Q"] = W_Q
        new_vision_model_state_dict[f"{new_layer_key}.attn.W_K"] = W_K
        new_vision_model_state_dict[f"{new_layer_key}.attn.W_V"] = W_V
        new_vision_model_state_dict[f"{new_layer_key}.attn.W_O"] = W_O
        new_vision_model_state_dict[f"{new_layer_key}.attn.b_Q"] = b_Q
        new_vision_model_state_dict[f"{new_layer_key}.attn.b_K"] = b_K
        new_vision_model_state_dict[f"{new_layer_key}.attn.b_V"] = b_V
        new_vision_model_state_dict[f"{new_layer_key}.attn.b_O"] = b_O

        # MLP weights
        mlp_W_in = old_state_dict[f"{old_layer_key}.mlp.fc1.weight"]
        mlp_W_out = old_state_dict[f"{old_layer_key}.mlp.fc2.weight"]
        mlp_b_in = old_state_dict[f"{old_layer_key}.mlp.fc1.bias"]
        mlp_b_out = old_state_dict[f"{old_layer_key}.mlp.fc2.bias"]

        mlp_W_in = einops.rearrange(mlp_W_in, "m d -> d m")
        mlp_W_out = einops.rearrange(mlp_W_out, "d m -> m d")

        new_vision_model_state_dict[f"{new_layer_key}.mlp.W_in"] = mlp_W_in
        new_vision_model_state_dict[f"{new_layer_key}.mlp.W_out"] = mlp_W_out
        new_vision_model_state_dict[f"{new_layer_key}.mlp.b_in"] = mlp_b_in
        new_vision_model_state_dict[f"{new_layer_key}.mlp.b_out"] = mlp_b_out

    # Set final projection
    new_vision_model_state_dict["head.W_H"] = old_state_dict[
        "visual_projection.weight"
    ].T
    new_vision_model_state_dict["head.b_H"] = torch.zeros((cfg.n_classes,))

    return new_vision_model_state_dict


def convert_open_clip_weights(
    old_state_dict,
    cfg: HookedViTConfig,
):

    new_vision_model_state_dict = {}

    # Convert embedding layers
    new_vision_model_state_dict["cls_token"] = (
        old_state_dict["visual.class_embedding"].unsqueeze(0).unsqueeze(0)
    )
    new_vision_model_state_dict["pos_embed.W_pos"] = old_state_dict[
        "visual.positional_embedding"
    ].clone()

    new_vision_model_state_dict["embed.proj.weight"] = old_state_dict[
        "visual.conv1.weight"
    ]
    new_vision_model_state_dict["embed.proj.bias"] = torch.zeros((cfg.d_model,))

    # Convert layer norms
    new_vision_model_state_dict["ln_final.w"] = old_state_dict["visual.ln_post.weight"]
    new_vision_model_state_dict["ln_final.b"] = old_state_dict["visual.ln_post.bias"]

    new_vision_model_state_dict["ln_pre.w"] = old_state_dict["visual.ln_pre.weight"]
    new_vision_model_state_dict["ln_pre.b"] = old_state_dict["visual.ln_pre.bias"]

    logging.info("visual projection shape", old_state_dict["visual.proj"].shape)

    new_vision_model_state_dict["head.W_H"] = old_state_dict["visual.proj"]
    new_vision_model_state_dict["head.b_H"] = torch.zeros((cfg.n_classes,))

    old_layer_key = f"visual.transformer.resblocks"
    new_vision_model_state_dict.update(
        _load_open_clip_attention_weights(old_state_dict, cfg, old_layer_key)
    )

    return new_vision_model_state_dict


def convert_open_clip_text_weights(
    old_state_dict,
    cfg: HookedTextTransformerConfig,
):
    """Load the model weights from the text encoder of Open CLIP models."""

    new_text_model_state_dict = {}

    # Convert embedding layers
    new_text_model_state_dict["token_embed.weight"] = old_state_dict[
        "token_embedding.weight"
    ]
    new_text_model_state_dict["pos_embed"] = old_state_dict["positional_embedding"]

    new_text_model_state_dict["ln_final.w"] = old_state_dict["ln_final.weight"]
    new_text_model_state_dict["ln_final.b"] = old_state_dict["ln_final.bias"]

    # Text projection
    new_text_model_state_dict["head.W_H"] = old_state_dict["text_projection"]
    new_text_model_state_dict["head.b_H"] = torch.zeros((cfg.n_classes,))

    old_layer_key = f"transformer.resblocks"
    new_text_model_state_dict.update(
        _load_open_clip_attention_weights(old_state_dict, cfg, old_layer_key)
    )

    return new_text_model_state_dict


def _load_open_clip_attention_weights(
    old_state_dict: dict,
    cfg: Union[HookedViTConfig, HookedTextTransformerConfig],
    layer_key: str,
):
    """The Open CLIP text and vision attention weights are the same, this function
    copies the weights to a dict with the correctly named keys for the Prisma models.
    """

    new_state_dict = dict()

    # Convert transformer blocks
    for layer in range(cfg.n_layers):
        new_layer_key = f"blocks.{layer}"
        old_layer_key = f"{layer_key}.{layer}"

        # Layer norms
        new_state_dict[f"{new_layer_key}.ln1.w"] = old_state_dict[
            f"{old_layer_key}.ln_1.weight"
        ]
        new_state_dict[f"{new_layer_key}.ln1.b"] = old_state_dict[
            f"{old_layer_key}.ln_1.bias"
        ]
        new_state_dict[f"{new_layer_key}.ln2.w"] = old_state_dict[
            f"{old_layer_key}.ln_2.weight"
        ]
        new_state_dict[f"{new_layer_key}.ln2.b"] = old_state_dict[
            f"{old_layer_key}.ln_2.bias"
        ]

        # Attention weights
        in_proj_weight = old_state_dict[f"{old_layer_key}.attn.in_proj_weight"]
        in_proj_bias = old_state_dict[f"{old_layer_key}.attn.in_proj_bias"]

        # Split in_proj_weight and in_proj_bias into Q, K, V
        W_Q, W_K, W_V = in_proj_weight.chunk(3)
        b_Q, b_K, b_V = in_proj_bias.chunk(3)

        # Reshape Q, K, V weights
        W_Q = einops.rearrange(
            W_Q, "(h dh) d -> h d dh", h=cfg.n_heads, d=cfg.d_model, dh=cfg.d_head
        )
        W_K = einops.rearrange(
            W_K, "(h dh) d -> h d dh", h=cfg.n_heads, d=cfg.d_model, dh=cfg.d_head
        )
        W_V = einops.rearrange(
            W_V, "(h dh) d -> h d dh", h=cfg.n_heads, d=cfg.d_model, dh=cfg.d_head
        )

        # Reshape Q, K, V biases
        b_Q = einops.rearrange(b_Q, "(h dh) -> h dh", h=cfg.n_heads, dh=cfg.d_head)
        b_K = einops.rearrange(b_K, "(h dh) -> h dh", h=cfg.n_heads, dh=cfg.d_head)
        b_V = einops.rearrange(b_V, "(h dh) -> h dh", h=cfg.n_heads, dh=cfg.d_head)

        # Output projection
        W_O = old_state_dict[f"{old_layer_key}.attn.out_proj.weight"]
        b_O = old_state_dict[f"{old_layer_key}.attn.out_proj.bias"]
        W_O = einops.rearrange(
            W_O, "d (h dh) -> h dh d", h=cfg.n_heads, d=cfg.d_model, dh=cfg.d_head
        )

        new_state_dict[f"{new_layer_key}.attn.W_Q"] = W_Q
        new_state_dict[f"{new_layer_key}.attn.W_K"] = W_K
        new_state_dict[f"{new_layer_key}.attn.W_V"] = W_V
        new_state_dict[f"{new_layer_key}.attn.W_O"] = W_O
        new_state_dict[f"{new_layer_key}.attn.b_Q"] = b_Q
        new_state_dict[f"{new_layer_key}.attn.b_K"] = b_K
        new_state_dict[f"{new_layer_key}.attn.b_V"] = b_V
        new_state_dict[f"{new_layer_key}.attn.b_O"] = b_O

        # MLP weights
        mlp_W_in = old_state_dict[f"{old_layer_key}.mlp.c_fc.weight"]
        mlp_W_out = old_state_dict[f"{old_layer_key}.mlp.c_proj.weight"]
        mlp_b_in = old_state_dict[f"{old_layer_key}.mlp.c_fc.bias"]
        mlp_b_out = old_state_dict[f"{old_layer_key}.mlp.c_proj.bias"]

        mlp_W_in = einops.rearrange(mlp_W_in, "m d -> d m")
        mlp_W_out = einops.rearrange(mlp_W_out, "d m -> m d")

        new_state_dict[f"{new_layer_key}.mlp.W_in"] = mlp_W_in
        new_state_dict[f"{new_layer_key}.mlp.W_out"] = mlp_W_out
        new_state_dict[f"{new_layer_key}.mlp.b_in"] = mlp_b_in
        new_state_dict[f"{new_layer_key}.mlp.b_out"] = mlp_b_out

    return new_state_dict



def convert_dino_weights(
    old_state_dict,
    cfg: HookedViTConfig,
):

    new_state_dict = {}

    new_state_dict["cls_token"] = old_state_dict["embeddings.cls_token"]
    new_state_dict["pos_embed.W_pos"] = old_state_dict[
        "embeddings.position_embeddings"
    ].squeeze(0)
    new_state_dict["embed.proj.weight"] = old_state_dict[
        "embeddings.patch_embeddings.projection.weight"
    ]
    new_state_dict["embed.proj.bias"] = old_state_dict[
        "embeddings.patch_embeddings.projection.bias"
    ]
    new_state_dict["ln_final.w"] = old_state_dict["layernorm.weight"]
    new_state_dict["ln_final.b"] = old_state_dict["layernorm.bias"]

    for layer in range(cfg.n_layers):
        layer_key = f"encoder.layer.{layer}"
        new_layer_key = f"blocks.{layer}"
        new_state_dict[f"{new_layer_key}.ln1.w"] = old_state_dict[
            f"{layer_key}.layernorm_before.weight"
        ]
        new_state_dict[f"{new_layer_key}.ln1.b"] = old_state_dict[
            f"{layer_key}.layernorm_before.bias"
        ]
        new_state_dict[f"{new_layer_key}.ln2.w"] = old_state_dict[
            f"{layer_key}.layernorm_after.weight"
        ]
        new_state_dict[f"{new_layer_key}.ln2.b"] = old_state_dict[
            f"{layer_key}.layernorm_after.bias"
        ]

        W_Q = old_state_dict[f"{layer_key}.attention.attention.query.weight"]
        W_K = old_state_dict[f"{layer_key}.attention.attention.key.weight"]
        W_V = old_state_dict[f"{layer_key}.attention.attention.value.weight"]
        W_O = old_state_dict[f"{layer_key}.attention.output.dense.weight"]

        W_Q = einops.rearrange(
            W_Q, "(h dh) d-> h d dh", h=cfg.n_heads, d=cfg.d_model, dh=cfg.d_head
        )
        W_K = einops.rearrange(
            W_K, "(h dh) d-> h d dh", h=cfg.n_heads, d=cfg.d_model, dh=cfg.d_head
        )
        W_V = einops.rearrange(
            W_V, "(h dh) d-> h d dh", h=cfg.n_heads, d=cfg.d_model, dh=cfg.d_head
        )
        W_O = einops.rearrange(
            W_O, "d (h dh) -> h dh d", h=cfg.n_heads, d=cfg.d_model, dh=cfg.d_head
        )

        new_state_dict[f"{new_layer_key}.attn.W_Q"] = W_Q
        new_state_dict[f"{new_layer_key}.attn.W_K"] = W_K
        new_state_dict[f"{new_layer_key}.attn.W_V"] = W_V
        new_state_dict[f"{new_layer_key}.attn.W_O"] = W_O

        b_Q = old_state_dict[f"{layer_key}.attention.attention.query.bias"]
        b_K = old_state_dict[f"{layer_key}.attention.attention.key.bias"]
        b_V = old_state_dict[f"{layer_key}.attention.attention.value.bias"]
        b_O = old_state_dict[f"{layer_key}.attention.output.dense.bias"]

        b_Q = einops.rearrange(b_Q, "(h dh) -> h dh", h=cfg.n_heads, dh=cfg.d_head)
        b_K = einops.rearrange(b_K, "(h dh) -> h dh", h=cfg.n_heads, dh=cfg.d_head)
        b_V = einops.rearrange(b_V, "(h dh) -> h dh", h=cfg.n_heads, dh=cfg.d_head)

        new_state_dict[f"{new_layer_key}.attn.b_Q"] = b_Q
        new_state_dict[f"{new_layer_key}.attn.b_K"] = b_K
        new_state_dict[f"{new_layer_key}.attn.b_V"] = b_V
        new_state_dict[f"{new_layer_key}.attn.b_O"] = b_O

        mlp_W_in = old_state_dict[f"{layer_key}.intermediate.dense.weight"]
        mlp_W_out = old_state_dict[f"{layer_key}.output.dense.weight"]
        mlp_b_in = old_state_dict[f"{layer_key}.intermediate.dense.bias"]
        mlp_b_out = old_state_dict[f"{layer_key}.output.dense.bias"]

        mlp_W_in = einops.rearrange(mlp_W_in, "m d -> d m")
        mlp_W_out = einops.rearrange(mlp_W_out, "d m -> m d")

        new_state_dict[f"{new_layer_key}.mlp.W_in"] = mlp_W_in
        new_state_dict[f"{new_layer_key}.mlp.W_out"] = mlp_W_out
        new_state_dict[f"{new_layer_key}.mlp.b_in"] = mlp_b_in
        new_state_dict[f"{new_layer_key}.mlp.b_out"] = mlp_b_out

    return new_state_dict


def convert_clip_weights(
    old_state_dict,
    old_head_state_dict,
    cfg: HookedViTConfig,
):

    new_vision_model_state_dict = {}

    new_vision_model_state_dict["cls_token"] = (
        old_state_dict["embeddings.class_embedding"].unsqueeze(0).unsqueeze(0)
    )
    new_vision_model_state_dict["pos_embed.W_pos"] = old_state_dict[
        "embeddings.position_embedding.weight"
    ]
    new_vision_model_state_dict["embed.proj.weight"] = old_state_dict[
        "embeddings.patch_embedding.weight"
    ]
    new_vision_model_state_dict["embed.proj.bias"] = torch.zeros(
        (cfg.d_model,), device=new_vision_model_state_dict["embed.proj.weight"].device
    )
    new_vision_model_state_dict["ln_final.w"] = old_state_dict["post_layernorm.weight"]
    new_vision_model_state_dict["ln_final.b"] = old_state_dict["post_layernorm.bias"]
    new_vision_model_state_dict["ln_pre.w"] = old_state_dict[
        "pre_layrnorm.weight"
    ]  # typo in ClipModel
    new_vision_model_state_dict["ln_pre.b"] = old_state_dict["pre_layrnorm.bias"]

    for layer in range(cfg.n_layers):
        layer_key = f"encoder.layers.{layer}"
        new_layer_key = f"blocks.{layer}"

        new_vision_model_state_dict[f"{new_layer_key}.ln1.w"] = old_state_dict[
            f"{layer_key}.layer_norm1.weight"
        ]
        new_vision_model_state_dict[f"{new_layer_key}.ln1.b"] = old_state_dict[
            f"{layer_key}.layer_norm1.bias"
        ]
        new_vision_model_state_dict[f"{new_layer_key}.ln2.w"] = old_state_dict[
            f"{layer_key}.layer_norm2.weight"
        ]
        new_vision_model_state_dict[f"{new_layer_key}.ln2.b"] = old_state_dict[
            f"{layer_key}.layer_norm2.bias"
        ]

        W_Q = old_state_dict[f"{layer_key}.self_attn.q_proj.weight"]
        W_K = old_state_dict[f"{layer_key}.self_attn.k_proj.weight"]
        W_V = old_state_dict[f"{layer_key}.self_attn.v_proj.weight"]
        W_O = old_state_dict[f"{layer_key}.self_attn.out_proj.weight"]

        W_Q = einops.rearrange(
            W_Q, "(h dh) d-> h d dh", h=cfg.n_heads, d=cfg.d_model, dh=cfg.d_head
        )
        W_K = einops.rearrange(
            W_K, "(h dh) d-> h d dh", h=cfg.n_heads, d=cfg.d_model, dh=cfg.d_head
        )
        W_V = einops.rearrange(
            W_V, "(h dh) d-> h d dh", h=cfg.n_heads, d=cfg.d_model, dh=cfg.d_head
        )
        W_O = einops.rearrange(
            W_O, "d (h dh) -> h dh d", h=cfg.n_heads, d=cfg.d_model, dh=cfg.d_head
        )

        new_vision_model_state_dict[f"{new_layer_key}.attn.W_Q"] = W_Q
        new_vision_model_state_dict[f"{new_layer_key}.attn.W_K"] = W_K
        new_vision_model_state_dict[f"{new_layer_key}.attn.W_V"] = W_V
        new_vision_model_state_dict[f"{new_layer_key}.attn.W_O"] = W_O

        b_Q = old_state_dict[f"{layer_key}.self_attn.q_proj.bias"]
        b_K = old_state_dict[f"{layer_key}.self_attn.k_proj.bias"]
        b_V = old_state_dict[f"{layer_key}.self_attn.v_proj.bias"]
        b_O = old_state_dict[f"{layer_key}.self_attn.out_proj.bias"]

        b_Q = einops.rearrange(b_Q, "(h dh) -> h dh", h=cfg.n_heads, dh=cfg.d_head)
        b_K = einops.rearrange(b_K, "(h dh) -> h dh", h=cfg.n_heads, dh=cfg.d_head)
        b_V = einops.rearrange(b_V, "(h dh) -> h dh", h=cfg.n_heads, dh=cfg.d_head)

        new_vision_model_state_dict[f"{new_layer_key}.attn.b_Q"] = b_Q
        new_vision_model_state_dict[f"{new_layer_key}.attn.b_K"] = b_K
        new_vision_model_state_dict[f"{new_layer_key}.attn.b_V"] = b_V
        new_vision_model_state_dict[f"{new_layer_key}.attn.b_O"] = b_O

        mlp_W_in = old_state_dict[f"{layer_key}.mlp.fc1.weight"]
        mlp_W_out = old_state_dict[f"{layer_key}.mlp.fc2.weight"]
        mlp_b_in = old_state_dict[f"{layer_key}.mlp.fc1.bias"]
        mlp_b_out = old_state_dict[f"{layer_key}.mlp.fc2.bias"]

        mlp_W_in = einops.rearrange(mlp_W_in, "m d -> d m")
        mlp_W_out = einops.rearrange(mlp_W_out, "d m -> m d")

        new_vision_model_state_dict[f"{new_layer_key}.mlp.W_in"] = mlp_W_in
        new_vision_model_state_dict[f"{new_layer_key}.mlp.W_out"] = mlp_W_out
        new_vision_model_state_dict[f"{new_layer_key}.mlp.b_in"] = mlp_b_in
        new_vision_model_state_dict[f"{new_layer_key}.mlp.b_out"] = mlp_b_out

    new_vision_model_state_dict["head.W_H"] = einops.rearrange(
        old_head_state_dict["weight"], "c d -> d c"
    )
    new_vision_model_state_dict["head.b_H"] = torch.zeros(
        (cfg.n_classes,), device=new_vision_model_state_dict["head.W_H"].device
    )

    return new_vision_model_state_dict


def convert_timm_weights(
    old_state_dict,
    cfg: HookedViTConfig,
):
    print(f"Converting the weights of a timm model to a Prisma ViT")

    new_state_dict = {}
    new_state_dict["cls_token"] = old_state_dict["cls_token"]
    new_state_dict["pos_embed.W_pos"] = old_state_dict["pos_embed"].squeeze(0)
    new_state_dict["embed.proj.weight"] = old_state_dict["patch_embed.proj.weight"]
    new_state_dict["embed.proj.bias"] = old_state_dict["patch_embed.proj.bias"]
    new_state_dict["ln_final.w"] = old_state_dict["norm.weight"]
    new_state_dict["ln_final.b"] = old_state_dict["norm.bias"]

    for layer in range(cfg.n_layers):
        layer_key = f"blocks.{layer}"
        new_state_dict[f"{layer_key}.ln1.w"] = old_state_dict[
            f"{layer_key}.norm1.weight"
        ]
        new_state_dict[f"{layer_key}.ln1.b"] = old_state_dict[f"{layer_key}.norm1.bias"]
        new_state_dict[f"{layer_key}.ln2.w"] = old_state_dict[
            f"{layer_key}.norm2.weight"
        ]
        new_state_dict[f"{layer_key}.ln2.b"] = old_state_dict[f"{layer_key}.norm2.bias"]

        W = old_state_dict[f"{layer_key}.attn.qkv.weight"]
        W_reshape = einops.rearrange(
            W,
            "(three h dh) d ->three h d dh",
            three=3,
            h=cfg.n_heads,
            d=cfg.d_model,
            dh=cfg.d_head,
        )
        W_Q, W_K, W_V = torch.unbind(W_reshape, dim=0)
        new_state_dict[f"{layer_key}.attn.W_Q"] = W_Q
        new_state_dict[f"{layer_key}.attn.W_K"] = W_K
        new_state_dict[f"{layer_key}.attn.W_V"] = W_V

        W_O = old_state_dict[f"{layer_key}.attn.proj.weight"]
        W_O = einops.rearrange(W_O, "m (i h)->i h m", i=cfg.n_heads)
        new_state_dict[f"{layer_key}.attn.W_O"] = W_O

        attn_bias = old_state_dict[f"{layer_key}.attn.qkv.bias"]
        attn_bias_reshape = einops.rearrange(
            attn_bias,
            "(three h dh) -> three h dh",
            three=3,
            h=cfg.n_heads,
            dh=cfg.d_head,
        )
        b_Q, b_K, b_V = torch.unbind(attn_bias_reshape, dim=0)
        new_state_dict[f"{layer_key}.attn.b_Q"] = b_Q
        new_state_dict[f"{layer_key}.attn.b_K"] = b_K
        new_state_dict[f"{layer_key}.attn.b_V"] = b_V

        b_O = old_state_dict[f"{layer_key}.attn.proj.bias"]
        new_state_dict[f"{layer_key}.attn.b_O"] = b_O

        new_state_dict[f"{layer_key}.mlp.b_in"] = old_state_dict[
            f"{layer_key}.mlp.fc1.bias"
        ]
        new_state_dict[f"{layer_key}.mlp.b_out"] = old_state_dict[
            f"{layer_key}.mlp.fc2.bias"
        ]

        mlp_W_in = old_state_dict[f"{layer_key}.mlp.fc1.weight"]
        mlp_W_in = einops.rearrange(mlp_W_in, "m d -> d m")
        new_state_dict[f"{layer_key}.mlp.W_in"] = mlp_W_in

        mlp_W_out = old_state_dict[f"{layer_key}.mlp.fc2.weight"]
        mlp_W_out = einops.rearrange(mlp_W_out, "d m -> m d")
        new_state_dict[f"{layer_key}.mlp.W_out"] = mlp_W_out

    new_state_dict["head.W_H"] = einops.rearrange(
        old_state_dict["head.weight"], "c d -> d c"
    )
    new_state_dict["head.b_H"] = old_state_dict["head.bias"]

    return new_state_dict


def convert_vivet_weights(
    old_state_dict,
    cfg: HookedViTConfig,
):

    new_state_dict = {}

    new_state_dict["cls_token"] = old_state_dict["vivit.embeddings.cls_token"]
    new_state_dict["pos_embed.W_pos"] = old_state_dict[
        "vivit.embeddings.position_embeddings"
    ].squeeze(0)
    new_state_dict["embed.proj.weight"] = old_state_dict[
        "vivit.embeddings.patch_embeddings.projection.weight"
    ]
    new_state_dict["embed.proj.bias"] = old_state_dict[
        "vivit.embeddings.patch_embeddings.projection.bias"
    ]
    new_state_dict["ln_final.w"] = old_state_dict["vivit.layernorm.weight"]
    new_state_dict["ln_final.b"] = old_state_dict["vivit.layernorm.bias"]

    for layer in range(cfg.n_layers):
        layer_key = f"vivit.encoder.layer.{layer}"
        new_layer_key = f"blocks.{layer}"
        new_state_dict[f"{new_layer_key}.ln1.w"] = old_state_dict[
            f"{layer_key}.layernorm_before.weight"
        ]
        new_state_dict[f"{new_layer_key}.ln1.b"] = old_state_dict[
            f"{layer_key}.layernorm_before.bias"
        ]
        new_state_dict[f"{new_layer_key}.ln2.w"] = old_state_dict[
            f"{layer_key}.layernorm_after.weight"
        ]
        new_state_dict[f"{new_layer_key}.ln2.b"] = old_state_dict[
            f"{layer_key}.layernorm_after.bias"
        ]

        W_Q = old_state_dict[f"{layer_key}.attention.attention.query.weight"]
        W_K = old_state_dict[f"{layer_key}.attention.attention.key.weight"]
        W_V = old_state_dict[f"{layer_key}.attention.attention.value.weight"]

        new_state_dict[f"{new_layer_key}.attn.W_Q"] = einops.rearrange(
            W_Q, "(h dh) d -> h d dh", h=cfg.n_heads, d=cfg.d_model, dh=cfg.d_head
        )
        new_state_dict[f"{new_layer_key}.attn.W_K"] = einops.rearrange(
            W_K, "(h dh) d -> h d dh", h=cfg.n_heads, d=cfg.d_model, dh=cfg.d_head
        )
        new_state_dict[f"{new_layer_key}.attn.W_V"] = einops.rearrange(
            W_V, "(h dh) d -> h d dh", h=cfg.n_heads, d=cfg.d_model, dh=cfg.d_head
        )

        W_O = old_state_dict[f"{layer_key}.attention.output.dense.weight"]
        new_state_dict[f"{new_layer_key}.attn.W_O"] = einops.rearrange(
            W_O, "d (h dh) -> h dh d", h=cfg.n_heads, d=cfg.d_model, dh=cfg.d_head
        )

        b_Q = old_state_dict[f"{layer_key}.attention.attention.query.bias"]
        b_K = old_state_dict[f"{layer_key}.attention.attention.key.bias"]
        b_V = old_state_dict[f"{layer_key}.attention.attention.value.bias"]

        new_state_dict[f"{new_layer_key}.attn.b_Q"] = einops.rearrange(
            b_Q, "(h dh) -> h dh", h=cfg.n_heads, dh=cfg.d_head
        )
        new_state_dict[f"{new_layer_key}.attn.b_K"] = einops.rearrange(
            b_K, "(h dh) -> h dh", h=cfg.n_heads, dh=cfg.d_head
        )
        new_state_dict[f"{new_layer_key}.attn.b_V"] = einops.rearrange(
            b_V, "(h dh) -> h dh", h=cfg.n_heads, dh=cfg.d_head
        )

        b_O = old_state_dict[f"{layer_key}.attention.output.dense.bias"]
        new_state_dict[f"{new_layer_key}.attn.b_O"] = b_O

        mlp_W_in = old_state_dict[f"{layer_key}.intermediate.dense.weight"]
        new_state_dict[f"{new_layer_key}.mlp.W_in"] = einops.rearrange(
            mlp_W_in, "m d -> d m"
        )

        mlp_W_out = old_state_dict[f"{layer_key}.output.dense.weight"]

        new_state_dict[f"{new_layer_key}.mlp.W_out"] = einops.rearrange(
            mlp_W_out, "d m -> m d"
        )

        new_state_dict[f"{new_layer_key}.mlp.b_in"] = old_state_dict[
            f"{layer_key}.intermediate.dense.bias"
        ]
        new_state_dict[f"{new_layer_key}.mlp.b_out"] = old_state_dict[
            f"{layer_key}.output.dense.bias"
        ]

    new_state_dict["head.W_H"] = einops.rearrange(
        old_state_dict["classifier.weight"], "c d -> d c"
    )
    new_state_dict["head.b_H"] = old_state_dict["classifier.bias"]

    return new_state_dict


def convert_hf_vit_for_image_classification_weights(
    old_state_dict,
    cfg: HookedViTConfig,
):

    new_state_dict = {}

    # exit(0)
    new_state_dict["cls_token"] = old_state_dict["vit.embeddings.cls_token"]
    new_state_dict["pos_embed.W_pos"] = old_state_dict[
        "vit.embeddings.position_embeddings"
    ].squeeze(0)
    new_state_dict["embed.proj.weight"] = old_state_dict[
        "vit.embeddings.patch_embeddings.projection.weight"
    ]
    new_state_dict["embed.proj.bias"] = old_state_dict[
        "vit.embeddings.patch_embeddings.projection.bias"
    ]
    new_state_dict["ln_final.w"] = old_state_dict["vit.layernorm.weight"]
    new_state_dict["ln_final.b"] = old_state_dict["vit.layernorm.bias"]

    for layer in range(cfg.n_layers):
        layer_key = f"vit.encoder.layer.{layer}"
        new_layer_key = f"blocks.{layer}"
        new_state_dict[f"{new_layer_key}.ln1.w"] = old_state_dict[
            f"{layer_key}.layernorm_before.weight"
        ]
        new_state_dict[f"{new_layer_key}.ln1.b"] = old_state_dict[
            f"{layer_key}.layernorm_before.bias"
        ]
        new_state_dict[f"{new_layer_key}.ln2.w"] = old_state_dict[
            f"{layer_key}.layernorm_after.weight"
        ]
        new_state_dict[f"{new_layer_key}.ln2.b"] = old_state_dict[
            f"{layer_key}.layernorm_after.bias"
        ]

        W_Q = old_state_dict[f"{layer_key}.attention.attention.query.weight"]
        W_K = old_state_dict[f"{layer_key}.attention.attention.key.weight"]
        W_V = old_state_dict[f"{layer_key}.attention.attention.value.weight"]

        new_state_dict[f"{new_layer_key}.attn.W_Q"] = einops.rearrange(
            W_Q, "(h dh) d -> h d dh", h=cfg.n_heads, d=cfg.d_model, dh=cfg.d_head
        )
        new_state_dict[f"{new_layer_key}.attn.W_K"] = einops.rearrange(
            W_K, "(h dh) d -> h d dh", h=cfg.n_heads, d=cfg.d_model, dh=cfg.d_head
        )
        new_state_dict[f"{new_layer_key}.attn.W_V"] = einops.rearrange(
            W_V, "(h dh) d -> h d dh", h=cfg.n_heads, d=cfg.d_model, dh=cfg.d_head
        )

        W_O = old_state_dict[f"{layer_key}.attention.output.dense.weight"]
        new_state_dict[f"{new_layer_key}.attn.W_O"] = einops.rearrange(
            W_O, "d (h dh) -> h dh d", h=cfg.n_heads, d=cfg.d_model, dh=cfg.d_head
        )

        b_Q = old_state_dict[f"{layer_key}.attention.attention.query.bias"]
        b_K = old_state_dict[f"{layer_key}.attention.attention.key.bias"]
        b_V = old_state_dict[f"{layer_key}.attention.attention.value.bias"]

        new_state_dict[f"{new_layer_key}.attn.b_Q"] = einops.rearrange(
            b_Q, "(h dh) -> h dh", h=cfg.n_heads, dh=cfg.d_head
        )
        new_state_dict[f"{new_layer_key}.attn.b_K"] = einops.rearrange(
            b_K, "(h dh) -> h dh", h=cfg.n_heads, dh=cfg.d_head
        )
        new_state_dict[f"{new_layer_key}.attn.b_V"] = einops.rearrange(
            b_V, "(h dh) -> h dh", h=cfg.n_heads, dh=cfg.d_head
        )

        b_O = old_state_dict[f"{layer_key}.attention.output.dense.bias"]
        new_state_dict[f"{new_layer_key}.attn.b_O"] = b_O

        mlp_W_in = old_state_dict[f"{layer_key}.intermediate.dense.weight"]
        new_state_dict[f"{new_layer_key}.mlp.W_in"] = einops.rearrange(
            mlp_W_in, "m d -> d m"
        )

        mlp_W_out = old_state_dict[f"{layer_key}.output.dense.weight"]

        new_state_dict[f"{new_layer_key}.mlp.W_out"] = einops.rearrange(
            mlp_W_out, "d m -> m d"
        )

        new_state_dict[f"{new_layer_key}.mlp.b_in"] = old_state_dict[
            f"{layer_key}.intermediate.dense.bias"
        ]
        new_state_dict[f"{new_layer_key}.mlp.b_out"] = old_state_dict[
            f"{layer_key}.output.dense.bias"
        ]

    new_state_dict["head.W_H"] = einops.rearrange(
        old_state_dict["classifier.weight"], "c d -> d c"
    )
    new_state_dict["head.b_H"] = old_state_dict["classifier.bias"]


    return new_state_dict
  

def fill_missing_keys(model, state_dict):
    """Takes in a state dict from a pretrained model, and fills in any missing keys with the default initialization.

    This function is assumed to be run before weights are initialized.

    Args:
        state_dict (dict): State dict from a pretrained model

    Returns:
        dict: State dict with missing keys filled in
    """
    # Get the default state dict
    default_state_dict = model.state_dict()
    # Get the keys that are missing from the pretrained model
    missing_keys = set(default_state_dict.keys()) - set(state_dict.keys())
    # Fill in the missing keys with the default initialization
    for key in missing_keys:
        if "hf_model" in key:
            # Skip keys that are from the HuggingFace model, if loading from HF.
            continue
        if "W_" in key:
            logging.warning(
                "Missing key for a weight matrix in pretrained, filled in with an empty tensor: {}".format(
                    key
                )
            )
        state_dict[key] = default_state_dict[key]
    return state_dict


def remove_open_clip_prefix(text, prefix="open-clip:"):
    if text.startswith(prefix):
        return text[len(prefix) :]
    return text

def load_state_dict(checkpoint_path: str, map_location="cpu"):
    checkpoint = torch.load(
        checkpoint_path, map_location=map_location, weights_only=False
    )
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    if next(iter(state_dict.items()))[0].startswith("module"):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    return state_dict



def has_hf_hub(necessary=False):
    if not _has_hf_hub and necessary:
        # if no HF Hub module installed, and it is necessary to continue, raise error
        raise RuntimeError(
            "Hugging Face hub model specified but package not installed. Run `pip install huggingface_hub`."
        )
    return _has_hf_hub


def download_pretrained_from_hf(
    model_id: str,
    filename: str = "open_clip_pytorch_model.bin",
    revision=None,
    cache_dir: Union[str, None] = None,
):
    logging.info("model_id download_pretrained_from_hf:", model_id)
    has_hf_hub(True)
    cached_file = hf_hub_download(
        model_id, filename, revision=revision, cache_dir=cache_dir
    )
    return cached_file

