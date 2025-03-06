import os
from typing import Any
from typing import Tuple

import torch
from vit_prisma.models.base_text_transformer import HookedTextTransformer
from vit_prisma.models.base_vit import HookedViT
# from vit_prisma.models.open_clip_models import hf_hub_download
from vit_prisma.sae import VisionModelSAERunnerConfig
from vit_prisma.sae import SparseAutoencoder
from vit_prisma.utils.constants import DEVICE
from vit_prisma.utils.enums import ModelType


def load_model(
    cfg: VisionModelSAERunnerConfig,
    model_from_pretrained_kwargs: dict[str, Any] | None = None
) -> HookedViT:
    model_from_pretrained_kwargs = model_from_pretrained_kwargs or {}

    if "n_devices" in model_from_pretrained_kwargs:
        n_devices = model_from_pretrained_kwargs["n_devices"]
        if n_devices > 1:
            print("MODEL LOADING:")
            print("Setting model device to cuda for d_devices")
            print(f"Will use cuda:0 to cuda:{n_devices-1}")
            device = "cuda"
            print("-------------")

    if cfg.model_class_name == "HookedViT":
        is_timm, is_clip, is_local = set_flags(cfg.model_name)

        if is_local:
            return HookedViT.from_local(cfg.prisma_vit_cfg, cfg.model_path).to(cfg.device)

        fold_ln = cfg.model_name != "openai/clip-vit-base-patch32"

        return HookedViT.from_pretrained(cfg.model_name, is_timm=is_timm, is_clip=is_clip, fold_ln=fold_ln).to(cfg.device)

    else:  # pragma: no cover
        raise ValueError(f"Unknown model class: {cfg.model_class_name}")


def set_flags(model_name):
    model_name_lower = model_name.lower()
    is_clip = 'clip' in model_name_lower
    is_local = 'local' in model_name_lower
    is_timm = (not is_clip) and (not is_local)
    return is_timm, is_clip, is_local


def load_remote_sae_and_model(
    repo_id: str,
    checkpoint="n_images_2600058.pt",
    config_file: str = "config.json",
    current_cfg: dict = None
) -> Tuple[SparseAutoencoder, HookedTextTransformer, HookedViT]:
    """Load and test SAE from HuggingFace."""

    print(f"Loading SAE from repo_id: {repo_id} with checkpoint: {checkpoint}")
    sae_path = hf_hub_download(repo_id, checkpoint)
    sae_config_path = hf_hub_download(repo_id, config_file)

    sae = SparseAutoencoder.load_from_pretrained(sae_path, config_path=sae_config_path, current_cfg=current_cfg, )

    print(f"Loading model name: {sae.cfg.model_name}")
    print(f"The config device is: {sae.cfg.device}")
    language_model = HookedTextTransformer.from_pretrained(sae.cfg.model_name, is_timm=False, is_clip=True, model_type=ModelType.TEXT).to(sae.cfg.device)
    model = HookedViT.from_pretrained(sae.cfg.model_name, is_timm=False, is_clip=True).to(sae.cfg.device)

    sae = sae.to(DEVICE)
    print(f"Using device: {DEVICE}")

    return sae, language_model, model
