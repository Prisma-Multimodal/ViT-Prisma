import json
import math
from abc import ABC
from dataclasses import fields, field, asdict, dataclass
from typing import Any, Optional, Literal
import logging
import inspect

import os

import torch
from vit_prisma.configs.HookedViTConfig import HookedViTConfig

import torch.distributed as dist
import logging

def setup_logger():
    """Configure logging to print only on rank 0"""
    rank = 0
    if dist.is_initialized():
        rank = dist.get_rank()
    
    # Create logger
    logger = logging.getLogger()
    
    # Only process with rank 0 should output logs
    if rank == 0:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.ERROR)  # Or logging.CRITICAL to suppress most messages
    
    return logger

# Then in your main code:
logger = setup_logger()


# Define a mapping from string to torch.dtype
dtype_mapping = {
    "float32": torch.float32,
    "float": torch.float32,  # alias
    "float64": torch.float64,
    "double": torch.float64,  # alias
    "float16": torch.float16,
    "half": torch.float16,  # alias
    "int64": torch.int64,
    "long": torch.int64,  # alias
    "int32": torch.int32,
    "int": torch.int32,  # alias
    "int16": torch.int16,
    "short": torch.int16,  # alias
    "int8": torch.int8,
    "uint8": torch.uint8,
    "bool": torch.bool,
    "torch.float32": torch.float32,
    "torch.float": torch.float32,  # alias
    "torch.float64": torch.float64,
    "torch.double": torch.float64,  # alias
    "torch.float16": torch.float16,
    "torch.half": torch.float16,  # alias
    "torch.int64": torch.int64,
    "torch.long": torch.int64,  # alias
    "torch.int32": torch.int32,
    "torch.int": torch.int32,  # alias
    "torch.int16": torch.int16,
    "torch.short": torch.int16,  # alias
    "torch.int8": torch.int8,
    "torch.uint8": torch.uint8,
    "torch.bool": torch.bool,
}


@dataclass
class VisionModelSAERunnerConfig:
    """
    Configuration for training a sparse autoencoder on a vision model.
    """

    ##################
    ### SAE Setup: ###
    ##################

    # Data Generating Function (Model + Training Distibuion)
    model_class_name: str = "HookedViT"
    model_name: str = "wkcn/TinyCLIP-ViT-40M-32-Text-19M-LAION400M"
    vit_model_cfg: Optional[HookedViTConfig] = None
    model_path: str = None
    hook_point_layer: int = 9
    layer_subtype: str = "hook_resid_post"
    hook_point_head_index: Optional[int] = None
    context_size: int = 50
    use_cached_activations: bool = False
    use_patches_only: bool = False
    cached_activations_path: Optional[str] = (
        None  # Defaults to "activations/{dataset}/{model}/{full_hook_name}_{hook_point_head_index}"
    )
    image_size: int = 224
    architecture: Literal["standard", "gated", "jumprelu"] = "gated"

    # SAE Parameters
    b_dec_init_method: str = "geometric_median"
    expansion_factor: int = 16
    from_pretrained_path: Optional[str] = None

    # Misc
    _device: str = "cpu"
    seed: int = 42
    _dtype: str = "float32"

    # SAE Parameters
    d_in: int = 512
    activation_fn_str: str = "relu"  # relu or topk
    activation_fn_kwargs: dict[str, Any] = field(default_factory=dict)
    cls_token_only: bool = False  # use only CLS token in training

    # New changes
    max_grad_norm: float = 1.0  # For gradient clipping, set to None to turn off
    initialization_method: str = "encoder_transpose_decoder"  # or independent
    normalize_activations: str = "layer_norm"

    #####################
    ### SAE Training: ###
    #####################

    is_training = True

    # Activation Store Parameters
    n_batches_in_buffer: int = 20
    store_batch_size: int = 32
    num_workers: int = 16

    # Training length parameters
    num_epochs: int = 10

    # logger
    verbose: bool = False

    # Training Parameters
    l1_coefficient: float = 0.0002  # 0.00008
    lp_norm: float = 1
    lr: float = 0.001
    lr_scheduler_name: str = (
        "cosineannealingwarmup"  # constant, constantwithwarmup, linearwarmupdecay, cosineannealing, cosineannealingwarmup
    )
    lr_warm_up_steps: int = 500

    train_batch_size: int = 1024 * 4

    # SAE Training run tolerance
    min_l0 = None
    min_explained_variance = None

    # Imagenet1k
    dataset_name: str = "imagenet1k"
    dataset_path: str = "/network/scratch/s/sonia.joseph/datasets/kaggle_datasets"
    dataset_train_path: str = (
        "/network/scratch/s/sonia.joseph/datasets/kaggle_datasets/ILSVRC/Data/CLS-LOC/train"
    )
    dataset_val_path: str = (
        "/network/scratch/s/sonia.joseph/datasets/kaggle_datasets/ILSVRC/Data/CLS-LOC/val"
    )

    # Resampling protocol args
    use_ghost_grads: bool = False
    feature_sampling_window: int = 1000  # 1000
    dead_feature_window: int = 5000  # unless this window is larger feature sampling,

    dead_feature_threshold: float = 1e-8

    # WANDB
    log_to_wandb: bool = True
    wandb_project: str = "tinyclip_sae_16_hyperparam_sweep_lr"
    wandb_entity: Optional[str] = None
    wandb_log_frequency: int = 10

    # Misc
    n_validation_runs: int = 100  # spaced linearly throughout training
    n_checkpoints: int = 10
    checkpoint_path: str = (
        "/network/scratch/s/sonia.joseph/sae_checkpoints/tinyclip_40M_mlp_out"
    )


    @property
    def device(self):
        """Device property, returns the torch device representation of the internal _device variable."""
        if isinstance(self._device, str):
            return torch.device(self._device)
        return self._device

    @device.setter
    def device(self, value: str):
        """Device setter to update the internal _device variable."""
        self._device = value

    @property
    def dtype(self):
        """Data type property for model precision."""
        return dtype_mapping[self._dtype]

    @dtype.setter
    def dtype(self, value: str):
        """Setter to update the data type property."""
        self._dtype = value

    @property
    def hook_point(self):
        """Returns the hook point identifier string for a specific layer."""
        return f"blocks.{self.hook_point_layer}.{self.layer_subtype}"

    @property
    def tokens_per_buffer(self):
        """Calculates the total number of tokens per buffer, considering context and batch settings."""
        if self.cls_token_only:  # Only use the CLS token per image
            tokens_per_image = 1
        elif self.use_patches_only:  # Exclude CLS token
            tokens_per_image = self.context_size - 1
        else:
            tokens_per_image = self.context_size

        return self.train_batch_size * tokens_per_image * self.n_batches_in_buffer

    @property
    def total_training_tokens(self):
        """Computes the total number of training tokens for the dataset and configuration."""
        if self.cls_token_only:  # Only use the CLS token per image
            tokens_per_image = 1
        elif self.use_patches_only:  # Exclude CLS token
            tokens_per_image = self.context_size - 1
        else:
            tokens_per_image = self.context_size

        return self.total_training_images * tokens_per_image

    @property
    def total_training_steps(self):
        """Calculates the total number of training steps based on token counts and batch size."""
        return self.total_training_tokens // self.train_batch_size

    @property
    def total_training_images(self):
        """Returns the total number of training images based on dataset and epochs."""
        if self.dataset_name == "imagenet1k":
            dataset_size = 1_300_000
        else:
            raise ValueError(
                "Your current dataset is not supported by the VisionModelSAERunnerConfig"
            )
        return int(dataset_size * self.num_epochs)

    @property
    def d_sae(self):
        """Calculates the SAE dimensionality based on input dimensions and expansion factor."""
        return self.d_in * self.expansion_factor

    @property
    def num_patch(self):
        """Calculates the number of patches based on the context size."""
        return int(math.sqrt(self.context_size - 1))

    def __post_init__(self):
        if self.b_dec_init_method not in ["geometric_median", "mean", "zeros"]:
            raise ValueError(
                f"b_dec_init_method must be geometric_median, mean, or zeros. Got {self.b_dec_init_method}"
            )

        if self.b_dec_init_method == "zeros":
            logger.warning(
                "Warning: We are initializing b_dec to zeros. This is probably not what you want."
            )

        if self.cls_token_only and self.use_patches_only:
            raise ValueError("cls_token_only and use_patches_only are exclusive.")

        # Autofill cached_activations_path unless the user overrode it
        # @TODO this here because I don't want to break backwards compability, but I think
        # we should refactor the activation cache fully
        if self.cached_activations_path is None:
            self.cached_activations_path = f"activations/{self.dataset_path.replace('/', '_')}/{self.model_name.replace('/', '_')}/{self.hook_point}"
            if self.hook_point_head_index is not None:
                self.cached_activations_path += f"_{self.hook_point_head_index}"
        
        if os.getenv("EVAL_MODE", "false").lower() in {"true", "1"}:
            self.is_training = False
            logger.info("Evaluation mode detected via environment variable; setting is_training to False.")

        if self.cls_token_only and self.use_patches_only:
            raise ValueError("Configuration error: 'cls_token_only' and 'use_patches_only' cannot both be True.")

        # Calculate key metrics
        n_tokens_per_buffer = (
            self.store_batch_size * self.context_size * self.n_batches_in_buffer
        )
        n_contexts_per_buffer = self.store_batch_size * self.n_batches_in_buffer
        n_feature_window_samples = (
            self.total_training_steps // self.feature_sampling_window
        )



        # fmt:off
        # Log useful information
        logger.info(f"n_tokens_per_buffer (millions): {n_tokens_per_buffer / 1e6}")
        logger.info(
            f"Lower bound: n_contexts_per_buffer (millions): {n_contexts_per_buffer / 1e6}"
        )
        logger.info(f"Total training steps: {self.total_training_steps}")
        logger.info(f"Total training images: {self.total_training_images}")
        logger.info(
            f"Total wandb updates: {self.total_training_steps // self.wandb_log_frequency}"
        )
        logger.info(f"Expansion factor: {self.expansion_factor}")

        logger.info(
            f"n_tokens_per_feature_sampling_window (millions): {self.feature_sampling_window * self.context_size * self.train_batch_size / 1e6}"
        )
        logger.info(
            f"n_tokens_per_dead_feature_window (millions): {self.dead_feature_window * self.context_size * self.train_batch_size / 1e6}"
        )
        logger.info(
            f"We will reset the sparsity calculation {n_feature_window_samples} times."
        )
        logger.info(
            f"Number tokens in sparsity calculation window: {self.feature_sampling_window * self.train_batch_size:.2e}"
        )

        # Log configuration options
        if self.use_ghost_grads:
            logger.info("Using Ghost Grads.")

        if self.max_grad_norm:
            logger.info(f"Gradient clipping with max_norm={self.max_grad_norm}")

        logger.info(f"Using SAE initialization method: {self.initialization_method}")
        # fmt:on

    def is_property(self, attr_name):
        return isinstance(getattr(self.__class__, attr_name, None), property)

    def save_config(self, path: str):
        """
        Save the configuration to a JSON file at the given path.
        """
        # Convert the dataclass to a dictionary
        data = asdict(self)

        # Function to make data JSON-serializable
        def make_serializable(obj):
            if inspect.isdatadescriptor(obj):
                return

            if isinstance(obj, (list, tuple)):
                # Recursively process lists and tuples
                return [make_serializable(item) for item in obj]
            elif isinstance(obj, dict):
                # Recursively process dictionaries
                return {
                    key: make_serializable(value)
                    for key, value in obj.items()
                    if not self.is_property(key)
                }
            else:
                return obj  # Other types are left as-is

        # Process data to make it serializable
        serializable_data = make_serializable(data)
        serializable_data["_dtype"] = self._dtype
        serializable_data["_device"] = self._device

        # Save to JSON file
        with open(path, "w") as f:
            json.dump(serializable_data, f, indent=4)

    @classmethod
    def load_config(cls, path: str):
        """
        Load the configuration from a JSON file at the given path.
        """
        with open(path, "r") as f:
            data = json.load(f)

        # Function to reconstruct data types
        def reconstruct_types(obj):
            if isinstance(obj, dict):
                if "__type__" in obj:
                    # Reconstruct torch.device from string
                    return obj["value"]
                else:
                    # Recursively process dictionaries
                    return {key: reconstruct_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                # Recursively process lists
                return [reconstruct_types(item) for item in obj]
            else:
                return obj  # Other types are left as-is

        # Reconstruct data with proper types
        data = reconstruct_types(data)

        # Get the current dataclass fields
        current_fields = {f.name for f in fields(cls)}

        # Remove legacy fields that are not part of the dataclass anymore
        # For example, old configs might have "total_training_images" and "total_training_tokens"
        # which are now computed properties. We should remove them.
        for legacy_key in ["total_training_images", "total_training_tokens", "d_sae"]:
            if legacy_key in data:
                # Log a warning about the deprecated field being ignored
                logger.warning(
                    f"Deprecated field '{legacy_key}' found in config. It will be ignored."
                )
                # If you need to do something with these values, do it here before removing
                del data[legacy_key]

        # Filter out any other keys not defined as fields in the current class
        cleaned_data = {k: v for k, v in data.items() if k in current_fields}

        # Create an instance of the class with the reconstructed data
        return cls(**cleaned_data)

    def pretty_print(self):
        print("Configuration:")
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, torch.dtype):
                value = str(value).split(".")[-1]  # Convert torch.dtype to string
            elif isinstance(value, torch.device):
                value = str(value)  # Convert torch.device to string
            print(f"  {field.name}: {value}")


@dataclass
class CacheActivationsRunnerConfig:
    """Configuration for caching activations of an LLM."""

    # Activation caching stuff
    shuffle_every_n_buffers: int = 10
    n_shuffles_with_last_section: int = 10
    n_shuffles_in_entire_dir: int = 10
    n_shuffles_final: int = 100

    def __post_init__(self):
        super().__post_init__()
        if self.use_cached_activations:
            raise ValueError(
                "Use_cached_activations should be False when running cache_activations_runner"
            )
