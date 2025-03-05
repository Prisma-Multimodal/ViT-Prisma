"""
Comprehensive Prisma Model Configurations
======================================
Configs load directly from Huggingface/OpenCLIP, sometimes with overrides to be compatible with Prisma.
"""

from typing import Dict, Any
from enum import Enum

from vit_prisma.utils.enums import ModelType

# Model categories enum
class ModelCategory(Enum):
    TIMM = "timm"
    CLIP = "clip"
    OPEN_CLIP = "open_clip"
    DINO = "dino"
    VIVIT = "vivit"
    VJEPA = "vjepa"

# ===============================
# Vision Model Configurations (Overrides; otherwise automatically uses Huggingface config Values)
# ===============================


# TIMM model configurations
TIMM_CONFIGS = {
    "vit_base_patch16_224": {
        "return_type": "class_logits",
    },
    "vit_base_patch32_224": {
        "return_type": "class_logits",
    },
    "vit_large_patch16_224": {
        "return_type": "class_logits",
    },
}



# Define a base configuration for OpenCLIP vision models
BASE_OPEN_CLIP_CONFIGS = {
    "ViT-B": {
        "n_heads": 12,
        "layer_norm_pre": True,
        "return_type": "class_logits",
        "eps": 1e-5,
        "normalize_output": True,
    },
    "ViT-L": {
        "n_heads": 16,
        "layer_norm_pre": True,
        "return_type": "class_logits",
        "eps": 1e-5,
        "normalize_output": True,
    },
    "ViT-H": {
        "n_heads": 16,
        "layer_norm_pre": True,
        "return_type": "class_logits",
        "eps": 1e-5,
        "normalize_output": True,
    },
    "ViT-g": {
        "n_heads": 16,
        "layer_norm_pre": True,
        "return_type": "class_logits",
        "eps": 1e-5,
        "normalize_output": True,
    },
    "ViT-G": {
        "n_heads": 16,
        "layer_norm_pre": True,
        "return_type": "class_logits",
        "eps": 1e-5,
        "normalize_output": True,
    },
}
# CLIP model configurations
CLIP_CONFIGS = {
    "openai/clip-vit-base-patch16": {
        **BASE_OPEN_CLIP_CONFIGS['ViT-B']

    },

    "openai/clip-vit-base-patch32": {
        "layer_norm_pre": True,
        "patch_size": 32,
        "eps": 1e-6,
        "return_type": "class_logits",
        "n_heads": 12,
        "normalize_output": False,
        "normalization_type": "LN",
        "architecture": "vit_clip_vision_encoder"

    },

    "openai/clip-vit-large-patch14": {
        "layer_norm_pre": True,
        "patch_size": 14,

    },

    "wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M": {
        "n_heads": 4,
        "layer_norm_pre": True,
        "eps": 1e-5,
        "patch_size": 16,
    },
}

# OpenCLIP Base models configurations
OPEN_CLIP_BASE_CONFIGS = {
    # ViT-B-16 models
    "open-clip:laion/CLIP-ViT-B-16-CommonPool.L-s1B-b8K": {
        **BASE_OPEN_CLIP_CONFIGS['ViT-B'],

    },
    "open-clip:laion/CLIP-ViT-B-16-CommonPool.L.basic-s1B-b8K": {
        **BASE_OPEN_CLIP_CONFIGS['ViT-B'],

    },
    "open-clip:laion/CLIP-ViT-B-16-CommonPool.L.clip-s1B-b8K": {
        **BASE_OPEN_CLIP_CONFIGS['ViT-B'],
    },
    "open-clip:laion/CLIP-ViT-B-16-CommonPool.L.image-s1B-b8K": {
        **BASE_OPEN_CLIP_CONFIGS['ViT-B'],
    },
    "open-clip:laion/CLIP-ViT-B-16-CommonPool.L.laion-s1B-b8K": {
        **BASE_OPEN_CLIP_CONFIGS['ViT-B'],
    },
    "open-clip:laion/CLIP-ViT-B-16-CommonPool.L.text-s1B-b8K": {
        **BASE_OPEN_CLIP_CONFIGS['ViT-B'],
    },
    "open-clip:laion/CLIP-ViT-B-16-DataComp.L-s1B-b8K": {
        **BASE_OPEN_CLIP_CONFIGS['ViT-B'],
    },
    "open-clip:laion/CLIP-ViT-B-16-DataComp.XL-s13B-b90K": {
        **BASE_OPEN_CLIP_CONFIGS['ViT-B'],
    },
    "open-clip:laion/CLIP-ViT-B-16-laion2B-s34B-b88K": {
        **BASE_OPEN_CLIP_CONFIGS['ViT-B'],
    },

    # ViT-B-32 models
    "open-clip:laion/CLIP-ViT-B-32-CommonPool.M-s128M-b4K": {
        "patch_size": 32,
        **BASE_OPEN_CLIP_CONFIGS['ViT-B'],
    },
    "open-clip:laion/CLIP-ViT-B-32-CommonPool.M.basic-s128M-b4K": {
        "patch_size": 32,
        **BASE_OPEN_CLIP_CONFIGS['ViT-B'],
    },
    "open-clip:laion/CLIP-ViT-B-32-CommonPool.M.clip-s128M-b4K": {
        "patch_size": 32,
        **BASE_OPEN_CLIP_CONFIGS['ViT-B'],
    },
    "open-clip:laion/CLIP-ViT-B-32-CommonPool.M.image-s128M-b4K": {
        "patch_size": 32,
        **BASE_OPEN_CLIP_CONFIGS['ViT-B'],
    },
    "open-clip:laion/CLIP-ViT-B-32-CommonPool.M.laion-s128M-b4K": {
        "patch_size": 32,
        **BASE_OPEN_CLIP_CONFIGS['ViT-B'],
    },
    "open-clip:laion/CLIP-ViT-B-32-CommonPool.M.text-s128M-b4K": {
        "patch_size": 32,
        **BASE_OPEN_CLIP_CONFIGS['ViT-B'],
    },
    "open-clip:laion/CLIP-ViT-B-32-CommonPool.S-s13M-b4K": {
        "patch_size": 32,
        **BASE_OPEN_CLIP_CONFIGS['ViT-B'],
    },
    "open-clip:laion/CLIP-ViT-B-32-CommonPool.S.basic-s13M-b4K": {
        "patch_size": 32,
        **BASE_OPEN_CLIP_CONFIGS['ViT-B'],
    },
    "open-clip:laion/CLIP-ViT-B-32-CommonPool.S.clip-s13M-b4K": {
        "patch_size": 32,
        **BASE_OPEN_CLIP_CONFIGS['ViT-B'],
    },
    "open-clip:laion/CLIP-ViT-B-32-CommonPool.S.image-s13M-b4K": {
        "patch_size": 32,
        **BASE_OPEN_CLIP_CONFIGS['ViT-B'],
    },
    "open-clip:laion/CLIP-ViT-B-32-CommonPool.S.laion-s13M-b4K": {
        "patch_size": 32,
        **BASE_OPEN_CLIP_CONFIGS['ViT-B'],
    },
    "open-clip:laion/CLIP-ViT-B-32-CommonPool.S.text-s13M-b4K": {
        "patch_size": 32,
        **BASE_OPEN_CLIP_CONFIGS['ViT-B'],
    },

    # ViT-L-14 models
    "open-clip:laion/CLIP-ViT-L-14-CommonPool.XL-s13B-b90K": {
        **BASE_OPEN_CLIP_CONFIGS['ViT-L'],
    },

    "open-clip:laion/CLIP-ViT-L-14-CommonPool.XL.clip-s13B-b90K": {
        **BASE_OPEN_CLIP_CONFIGS['ViT-L'],

    },
    "open-clip:laion/CLIP-ViT-L-14-CommonPool.XL.laion-s13B-b90K": {
        **BASE_OPEN_CLIP_CONFIGS['ViT-L'],

    },
    
    # DataComp models
    "open-clip:laion/CLIP-ViT-B-32-DataComp.M-s128M-b4K": {
        **BASE_OPEN_CLIP_CONFIGS['ViT-B'],
    },

    "open-clip:laion/CLIP-ViT-B-32-DataComp.S-s13M-b4K": {
        **BASE_OPEN_CLIP_CONFIGS['ViT-B'],

    },
    "open-clip:laion/CLIP-ViT-B-32-DataComp.XL-s13B-b90K": {
        **BASE_OPEN_CLIP_CONFIGS['ViT-B'],
    },
    "open-clip:laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K": {
        **BASE_OPEN_CLIP_CONFIGS['ViT-L'],

    },
    
    # Other LAION models
    "open-clip:laion/CLIP-ViT-B-32-laion2B-s34B-b79K": {
        **BASE_OPEN_CLIP_CONFIGS['ViT-B'],

    },
    "open-clip:laion/CLIP-ViT-L-14-laion2B-s32B-b82K": {
        **BASE_OPEN_CLIP_CONFIGS['ViT-L'],

    },
    
    # TIMM versions
    "open-clip:timm/vit_base_patch16_clip_224.laion400m_e31": {
        **BASE_OPEN_CLIP_CONFIGS['ViT-B'],

    },
    "open-clip:timm/vit_base_patch16_clip_224.laion400m_e32": {
        **BASE_OPEN_CLIP_CONFIGS['ViT-B'],

    },
    "open-clip:timm/vit_base_patch32_clip_224.laion2b_e16": {
        **BASE_OPEN_CLIP_CONFIGS['ViT-B'],

    },
    "open-clip:timm/vit_large_patch14_clip_224.laion400m_e31": {
        **BASE_OPEN_CLIP_CONFIGS['ViT-L'],

    },

    "open-clip:timm/vit_large_patch14_clip_224.laion400m_e32": {
        **BASE_OPEN_CLIP_CONFIGS['ViT-L'],

    },

    "open-clip:timm/vit_medium_patch16_clip_224.tinyclip_yfcc15m": {

        # "d_model": 640,
        # "n_heads": 10,
        # "n_layers": 16,
        # "d_mlp": 2560,
        # "d_head": 64,
        # "patch_size": 16,
        # "image_size": 224,
        # "layer_norm_pre": True,
        # "return_type": "class_logits",
        # "normalization_type": "LN",
        # "use_cls_token": True,
        # "eps": 1e-6,
        # "n_classes": 640,
    },

    "open-clip:laion/CLIP-ViT-bigG-14-laion2B-39B-b160k": {
        **BASE_OPEN_CLIP_CONFIGS['ViT-G'],
        # "d_model": 1664, 
        # "n_heads": 16,
        # "n_layers": 48,
        # "d_mlp": 8192,
        # "d_head": 104,
        # "patch_size": 14,
        # "image_size": 224,
        # "layer_norm_pre": True,
        # "return_type": "class_logits",
        # "normalization_type": "LN",
        # "use_cls_token": True,
        # "eps": 1e-6,
        # "n_classes": 1280,
    },
}

# Models that are currently failing but should be supported
OPEN_CLIP_EXTENDED_CONFIGS = {
    # Additional models
    "open-clip:timm/vit_base_patch16_clip_224.metaclip_2pt5b": {
        **BASE_OPEN_CLIP_CONFIGS['ViT-B'],

    },
    "open-clip:timm/vit_base_patch16_clip_224.metaclip_400m": {
        **BASE_OPEN_CLIP_CONFIGS['ViT-B'],

    },
    "open-clip:timm/vit_base_patch16_clip_224.openai": {
        **BASE_OPEN_CLIP_CONFIGS['ViT-B'],

    },
    "open-clip:timm/vit_base_patch32_clip_224.laion400m_e31": {
        "patch_size": 32,
        **BASE_OPEN_CLIP_CONFIGS['ViT-B'],

    },
    "open-clip:timm/vit_base_patch32_clip_224.laion400m_e32": {
        "patch_size": 32,
        **BASE_OPEN_CLIP_CONFIGS['ViT-B'],

    },
    "open-clip:timm/vit_base_patch32_clip_224.metaclip_2pt5b": {
        "patch_size": 32,
        **BASE_OPEN_CLIP_CONFIGS['ViT-B'],
    },
    "open-clip:timm/vit_base_patch32_clip_224.metaclip_400m": {
        "patch_size": 32,
        **BASE_OPEN_CLIP_CONFIGS['ViT-B'],
    },
    "open-clip:timm/vit_base_patch32_clip_224.openai": {
        "patch_size": 32,
        **BASE_OPEN_CLIP_CONFIGS['ViT-B'],

    },

    "open-clip:laion/CLIP-ViT-B-32-256x256-DataComp-s34B-b86K": {
        "patch_size": 32,
        "image_size": 256,
        "layer_norm_pre": True,
        "return_type": "class_logits",
    },
    "open-clip:laion/CLIP-ViT-B-32-xlm-roberta-base-laion5B-s13B-b90k": {
        "patch_size": 32,
        "layer_norm_pre": True,
        "return_type": "class_logits",
    },
    "open-clip:laion/CLIP-ViT-B-32-roberta-base-laion2B-s12B-b32k": {
        "patch_size": 32,
        "layer_norm_pre": True,
        "return_type": "class_logits",
    },
    "open-clip:laion/CLIP-ViT-H-14-frozen-xlm-roberta-large-laion5B-s13B-b90k": {
        "patch_size": 14,
        **BASE_OPEN_CLIP_CONFIGS['ViT-H'],

    },
    "open-clip:laion/CLIP-ViT-H-14-laion2B-s32B-b79K": {
        **BASE_OPEN_CLIP_CONFIGS['ViT-H'],

    },
    "open-clip:timm/vit_base_patch16_plus_clip_240.laion400m_e31": {
        "image_size": 240,
        "layer_norm_pre": True,
        "return_type": "class_logits",
    },
    "open-clip:timm/vit_base_patch16_plus_clip_240.laion400m_e32": {
        **BASE_OPEN_CLIP_CONFIGS['ViT-B'],
    },

    "open-clip:timm/vit_large_patch14_clip_224.metaclip_2pt5b": {
        "patch_size": 14,
        "layer_norm_pre": True,
        "return_type": "class_logits",
    },
    "open-clip:timm/vit_large_patch14_clip_224.metaclip_400m": {
        "patch_size": 14,
        "layer_norm_pre": True, 
        "return_type": "class_logits",
    },
    "open-clip:timm/vit_large_patch14_clip_224.openai": {
        "patch_size": 14,
        "layer_norm_pre": True,
        "return_type": "class_logits",
    },
    "open-clip:timm/vit_large_patch14_clip_336.openai": {
        "patch_size": 14,
        "image_size": 336,
        "layer_norm_pre": True,
        "return_type": "class_logits",
    },
    "open-clip:timm/vit_medium_patch32_clip_224.tinyclip_laion400m": {
        "d_model": 640,
        "n_heads": 10,
        "n_layers": 16,
        "d_mlp": 2560,
        "d_head": 64,
        "patch_size": 32,
        "image_size": 224,
        "layer_norm_pre": True,
        "return_type": "class_logits",
        "normalization_type": "LN",
        "use_cls_token": True,
        "eps": 1e-6,
        "n_classes": 640,
    },
    "open-clip:timm/vit_xsmall_patch16_clip_224.tinyclip_yfcc15m": {
        "d_model": 384,
        "n_heads": 6,
        "n_layers": 8,
        "d_mlp": 1536,
        "d_head": 64,
        "patch_size": 16,
        "image_size": 224,
        "layer_norm_pre": True,
        "return_type": "class_logits",
        "normalization_type": "LN",
        "use_cls_token": True,
        "eps": 1e-6,
        "n_classes": 384,
    },
    "open-clip:timm/vit_betwixt_patch32_clip_224.tinyclip_laion400m": {
        "d_model": 512,
        "n_heads": 8,
        "n_layers": 12,
        "d_mlp": 2048,
        "d_head": 64,
        "patch_size": 32,
        "image_size": 224,
        "layer_norm_pre": True,
        "return_type": "class_logits",
        "normalization_type": "LN",
        "use_cls_token": True,
        "eps": 1e-6,
        "n_classes": 512,
    },
    "open-clip:timm/vit_gigantic_patch14_clip_224.metaclip_2pt5b": {
        "d_model": 1920,
        "n_heads": 24,
        "n_layers": 48,
        "d_mlp": 7680,
        "d_head": 80,
        "patch_size": 14,
        "image_size": 224,
        "layer_norm_pre": True,
        "return_type": "class_logits",
        "normalization_type": "LN",
        "use_cls_token": True,
        "eps": 1e-6,
        "n_classes": 1024,
    },
    "open-clip:timm/vit_huge_patch14_clip_224.metaclip_2pt5b": {
        "patch_size": 14,
        "layer_norm_pre": True,
        "return_type": "class_logits",
    },
    "open-clip:laion/CoCa-ViT-B-32-laion2B-s13B-b90k": {
        "patch_size": 32,
        "layer_norm_pre": True,
        "return_type": "class_logits",
    },
    "open-clip:laion/CoCa-ViT-L-14-laion2B-s13B-b90k": {
        "patch_size": 14,
        "layer_norm_pre": True,
        "return_type": "class_logits",
    },
}

# Combine all OpenCLIP configs
OPEN_CLIP_CONFIGS = {
    **OPEN_CLIP_BASE_CONFIGS,
    **OPEN_CLIP_EXTENDED_CONFIGS,
}

# EVA02 model configurations
EVA02_CONFIGS = {
    "open-clip:timm/eva02_enormous_patch14_clip_224.laion2b_s4b_b115k": {
        "d_model": 1792,
        "n_heads": 16,
        "n_layers": 40,
        "patch_size": 14,
        "image_size": 224,
        "n_classes": 1000,
        "d_mlp": 7168,
        "layer_norm_pre": True,
        "return_type": "class_logits",
        "normalization_type": "LN",
        "use_cls_token": True,
        "eps": 1e-6
    },
    "open-clip:timm/eva02_enormous_patch14_plus_clip_224.laion2b_s9b_b144": {
        "d_model": 1792,
        "n_heads": 16,
        "n_layers": 40,
        "patch_size": 14,
        "image_size": 224,
        "n_classes": 1000,
        "d_mlp": 7168,
        "layer_norm_pre": True,
        "return_type": "class_logits",
        "normalization_type": "LN",
        "use_cls_token": True,
        "eps": 1e-6
    },
    "open-clip:timm/eva02_large_patch14_clip_224.merged2b_s4b_b131k": {
        "d_model": 1024,
        "n_heads": 16,
        "n_layers": 40,
        "patch_size": 14,
        "image_size": 224,
        "n_classes": 1024,
        "d_mlp": 4096,
        "layer_norm_pre": True,
        "return_type": "class_logits",
        "normalization_type": "LN",
        "use_cls_token": True,
        "eps": 1e-6
    },
    "open-clip:timm/eva02_large_patch14_clip_336.merged2b_s6b_b61k": {
        "d_model": 1024,
        "n_heads": 16,
        "n_layers": 40,
        "patch_size": 14,
        "image_size": 336,
        "n_classes": 1024,
        "d_mlp": 4096,
        "layer_norm_pre": True,
        "return_type": "class_logits",
        "normalization_type": "LN",
        "use_cls_token": True,
        "eps": 1e-6
    },
    "open-clip:timm/eva02_base_patch16_clip_224.merged2b_s8b_b131k": {
        "d_model": 768,
        "n_heads": 12,
        "n_layers": 12,
        "patch_size": 16,
        "image_size": 224,
        "n_classes": 512,
        "d_mlp": 3072,
        "layer_norm_pre": True,
        "return_type": "class_logits",
        "normalization_type": "LN",
        "use_cls_token": True,
        "eps": 1e-6
    },
    "open-clip:timm/eva_giant_patch14_clip_224.laion400m_s11b_b41k": {
        "d_model": 1408,
        "n_heads": 16,
        "n_layers": 40,
        "patch_size": 14,
        "image_size": 224,
        "n_classes": 1024,
        "d_mlp": 5632,
        "layer_norm_pre": True,
        "return_type": "class_logits",
        "normalization_type": "LN",
        "use_cls_token": True,
        "eps": 1e-6
    },
    "open-clip:timm/eva_giant_patch14_plus_clip_224.merged2b_s11b_b114k": {
        "d_model": 1408,
        "n_heads": 16,
        "n_layers": 40,
        "patch_size": 14,
        "image_size": 224,
        "n_classes": 1024,
        "d_mlp": 5632,
        "layer_norm_pre": True,
        "return_type": "class_logits",
        "normalization_type": "LN",
        "use_cls_token": True,
        "eps": 1e-6
    }
}

# DINO model configurations
DINO_CONFIGS = {
    "facebook/dino-vitb16": {
        "return_type": "pre_logits",
    },
    "facebook/dino-vitb8": {
        "patch_size": 8,
        "return_type": "pre_logits", 
    },
    "facebook/dino-vits16": {
        "n_heads": 6,
        "return_type": "pre_logits",
    },

    "facebook/dino-vits8": {
        "d_model": 384,
        "n_heads": 6,
        "n_layers": 12,
        "d_mlp": 1536,
        "d_head": 64,
        "patch_size": 8,
        "image_size": 224,
        "return_type": "pre_logits",
        "n_classes": 384,
        "normalization_type": "LN",
        "use_cls_token": True,
        "eps": 1e-6,
    },
}

# ViViT model configurations
VIVIT_CONFIGS = {
    "google/vivit-b-16x2-kinetics400": {
        "is_video_transformer": True,
        "video_tubelet_depth": 2,
        "video_num_frames": 16,
        "n_classes": 400,
        "return_type": "class_logits",
    },
    "google/vivit-l-16x2-kinetics400": {
        "is_video_transformer": True,
        "video_tubelet_depth": 2,
        "video_num_frames": 16,
        "n_classes": 400,
        "return_type": "class_logits",
    },
}

# VJEPA model configurations
VJEPA_CONFIGS = {
    "vjepa_v1_vit_large": {
        "use_cls_token": False,
        "layer_norm_pre": False,
        "return_type": "pre_logits",
        "eps": 1e-6,
    },
    "vjepa_v1_vit_huge": {
        "use_cls_token": False,
        "layer_norm_pre": False,
        "return_type": "pre_logits",
        "eps": 1e-6,
    },
}



# ===============================
# Text Model Configurations
# ===============================

# Base text configuration
BASE_TEXT_CONFIG = {
    "d_model": 768,
    "n_heads": 12,
    "n_layers": 12,
    "d_mlp": 3072,
    "d_head": 64,
    "vocab_size": 49408,
    "context_length": 77,
    "eps": 1e-5,
    "normalization_type": "LN",
    "layer_norm_pre": True,
}

# Text configurations for CLIP models
OPEN_CLIP_TEXT_CONFIGS = {
    # Base models
    "open-clip:laion/CLIP-ViT-B-32-DataComp.XL-s13B-b90K": {
        **BASE_TEXT_CONFIG,
    },
    "open-clip:laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K": {
        **BASE_TEXT_CONFIG,
    },
    
    # Add models that support custom text encoders
    "open-clip:laion/CLIP-ViT-B-32-xlm-roberta-base-laion5B-s13B-b90k": {
        **BASE_TEXT_CONFIG,
        "vocab_size": 250002,  # XLM-RoBERTa vocabulary size
    },
    "open-clip:laion/CLIP-ViT-B-32-roberta-base-laion2B-s12B-b32k": {
        **BASE_TEXT_CONFIG,
        "vocab_size": 50265,   # RoBERTa vocabulary size
    },
    "open-clip:laion/CLIP-ViT-H-14-frozen-xlm-roberta-large-laion5B-s13B-b90k": {
        **BASE_TEXT_CONFIG,
        "d_model": 1024,       # XLM-RoBERTa large
        "n_heads": 16,
        "n_layers": 24,
        "d_mlp": 4096,
        "vocab_size": 250002,  # XLM-RoBERTa large vocabulary size
    },
    
    # CommonPool models
    "open-clip:laion/CLIP-ViT-B-16-CommonPool.L-s1B-b8K": {
        **BASE_TEXT_CONFIG,
    },
    "open-clip:laion/CLIP-ViT-B-16-CommonPool.L.basic-s1B-b8K": {
        **BASE_TEXT_CONFIG,
    },
    "open-clip:laion/CLIP-ViT-B-16-CommonPool.L.clip-s1B-b8K": {
        **BASE_TEXT_CONFIG,
    },
    "open-clip:laion/CLIP-ViT-B-16-CommonPool.L.laion-s1B-b8K": {
        **BASE_TEXT_CONFIG,
    },
    "open-clip:laion/CLIP-ViT-B-32-CommonPool.M-s128M-b4K": {
        **BASE_TEXT_CONFIG,
    },
    "open-clip:laion/CLIP-ViT-B-32-CommonPool.S-s13M-b4K": {
        **BASE_TEXT_CONFIG,
    },
    "open-clip:laion/CLIP-ViT-L-14-CommonPool.XL-s13B-b90K": {
        **BASE_TEXT_CONFIG,
    },
    
    # CoCa models
    "open-clip:laion/CoCa-ViT-B-32-laion2B-s13B-b90k": {
        **BASE_TEXT_CONFIG,
    },
    "open-clip:laion/CoCa-ViT-L-14-laion2B-s13B-b90k": {
        **BASE_TEXT_CONFIG,
    },
}

# ===============================
# Combined Registry Dictionary
# ===============================

# Mapping model names to categories
MODEL_CATEGORIES = {
    **{name: ModelCategory.TIMM for name in TIMM_CONFIGS},
    **{name: ModelCategory.CLIP for name in CLIP_CONFIGS},
    **{name: ModelCategory.OPEN_CLIP for name in OPEN_CLIP_CONFIGS},
    **{name: ModelCategory.OPEN_CLIP for name in EVA02_CONFIGS},
    **{name: ModelCategory.DINO for name in DINO_CONFIGS},
    **{name: ModelCategory.VIVIT for name in VIVIT_CONFIGS},
    **{name: ModelCategory.VJEPA for name in VJEPA_CONFIGS},
}

# Combined configuration dictionary
MODEL_CONFIGS = {
    # Vision configurations
    ModelType.VISION: {
        **TIMM_CONFIGS,
        **CLIP_CONFIGS,
        **OPEN_CLIP_CONFIGS, 
        **EVA02_CONFIGS,
        **DINO_CONFIGS,
        **VIVIT_CONFIGS,
        **VJEPA_CONFIGS,
    },
    # Text configurations
    ModelType.TEXT: {
        **OPEN_CLIP_TEXT_CONFIGS,
    }
}

# Get the list of models that support text modality 
TEXT_SUPPORTED_MODELS = set(MODEL_CONFIGS[ModelType.TEXT].keys())
