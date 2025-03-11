# load sae to check
from typing import Any

from vit_prisma.sae.sae import SparseAutoencoder

from huggingface_hub import hf_hub_download
import os
import torch

import io
import zipfile
from collections import OrderedDict

import torch
import io
import zipfile
from collections import OrderedDict
import pickle

import types
import sys

import tempfile



import torch
import io
import zipfile
from collections import OrderedDict
import pickle
import sys
import types

import re

from huggingface_hub import HfApi
import os

def upload_to_huggingface(
    checkpoint_path: str,
    repo_id: str,
    description: str,
    token: str = None,
    commit_message: str = "Upload checkpoint"
):
    api = HfApi()

    # First create the repo
    api.create_repo(
        repo_id=repo_id,
        private=False,
        exist_ok=True,
        token=token,
        repo_type="model"
    )

    # Create and upload README.md
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.md') as tmp:
        tmp.write(description)
        tmp.flush()

        api.upload_file(
            path_or_fileobj=tmp.name,
            path_in_repo="README.md",
            repo_id=repo_id,
            token=token,
            commit_message="Update README.md"
        )

    # Upload the checkpoint file
    api.upload_file(
        path_or_fileobj=checkpoint_path,
        path_in_repo=os.path.basename(checkpoint_path),
        repo_id=repo_id,
        token=token,
        commit_message=commit_message
    )
    print(f"Successfully uploaded checkpoint to: https://huggingface.co/{repo_id}")

def download_sae_from_huggingface(repo_name, file_id, download_dir):
    os.makedirs(download_dir, exist_ok=True)
    local_path = hf_hub_download(repo_id=repo_name, filename=file_id, local_dir=download_dir)
    print(f"File downloaded successfully to: {local_path}")
    print(f"File size: {os.path.getsize(local_path)} bytes")
    

def load_sae(cfg):
    sparse_autoencoder = SparseAutoencoder(cfg).load_from_pretrained(cfg.sae_path)
    sparse_autoencoder.to(cfg.device)
    sparse_autoencoder.eval()  # prevents error if we're expecting a dead neuron mask for who 
    return sparse_autoencoder

def map_legacy_sae_lens_2_to_prisma_repo(old_config):
    new_config = {}

    # Mapping dictionary for renamed fields
    field_mapping = {
        'hook_point': 'hook_point_layer',
        'dead_feature_threshold': 'dead_feature_threshold',
        'feature_sampling_method': None,  # This field seems to be removed
        'feature_reinit_scale': None,  # This field seems to be removed
        # Add more mappings as needed
    }

    # Default values for new fields
    default_values = {
        'model_class_name': "HookedViT",
        'model_name': "wkcn/TinyCLIP-ViT-40M-32-Text-19M-LAION400M",
        'hook_point_head_index': None,
        'context_size': 50,
        'use_cached_activations': False,
        'cached_activations_path': None,
        'activation_fn_str': "relu",
        'activation_fn_kwargs': {},
        'max_grad_norm': 1.0,
        'initialization_method': "encoder_transpose_decoder",
        'n_batches_in_buffer': 20,
        'store_batch_size': 32,
        'num_epochs': 1,
        'image_size': 224,
        'device': "cpu",
        'seed': 42,
        'dtype': torch.float32,
        'verbose': False,
        'b_dec_init_method': "geometric_median",
        'expansion_factor': 16,
        'from_pretrained_path': None,
        'd_sae': None,
        'lr_scheduler_name': "cosineannealing",
        'lr_warm_up_steps': 0,
        'dataset_name': 'imagenet1k',
        'dataset_path': "/network/scratch/s/firstname.joseph/datasets/kaggle_datasets",
        'dataset_train_path': "/network/scratch/s/firstname.joseph/datasets/kaggle_datasets/ILSVRC/Data/CLS-LOC/train",
        'dataset_val_path': "/network/scratch/s/firstname.joseph/datasets/kaggle_datasets/ILSVRC/Data/CLS-LOC/val",
        'use_ghost_grads': True,
        'feature_sampling_window': 300,
        'dead_feature_window': 5000,
        'log_to_wandb': True,
        'wandb_project': "tinyclip_sae_16_hyperparam_sweep_lr",
        'wandb_entity': None,
        'wandb_log_frequency': 100,
        'n_checkpoints': 10,
    }

    # First, set all default values
    new_config.update(default_values)

    # Then, update with old config values, mapping fields as necessary
    for old_key, old_value in old_config.items():
        new_key = field_mapping.get(old_key, old_key)
        if new_key is not None:
            new_config[new_key] = old_value

    # Special handling for hook_point
    if 'hook_point' in old_config:
        # Extract the layer number from the old hook_point string
        layer_match = re.search(r'blocks\.(\d+)\.', old_config['hook_point'])
        if layer_match:
            new_config['hook_point_layer'] = int(layer_match.group(1))

    # Calculate total_training_images and total_training_tokens
    new_config['total_training_images'] = int(1_300_000 * new_config['num_epochs'])
    new_config['total_training_tokens'] = new_config['total_training_images'] * new_config['context_size']

    # Handle any other special cases or computations here

    return new_config


def load_sae(cfg):
    """Load a local SAE."""
    sparse_autoencoder = SparseAutoencoder(cfg).load_from_pretrained(cfg.sae_path)
    sparse_autoencoder.to(cfg.device)
    sparse_autoencoder.eval()  # prevents error if we're expecting a dead neuron mask for who
    return sparse_autoencoder


def wandb_log_suffix(cfg: Any, hyperparams: Any):
    # Create a mapping from cfg list keys to their corresponding hyperparams attributes
    key_mapping = {
        "hook_point_layer": "layer",
        "l1_coefficient": "coeff",
        "lp_norm": "l",
        "lr": "lr",
    }

    # Generate the suffix by iterating over the keys that have list values in cfg
    suffix = "".join(
        f"_{key_mapping.get(key, key)}{getattr(hyperparams, key, '')}"
        for key, value in vars(cfg).items()
        if isinstance(value, list)
    )
    return suffix

def get_deep_attr(obj: Any, path: str):
    """Helper function to get a nested attribute from a object.
    In practice used to access HookedViT HookPoints (eg model.blocks[0].attn.hook_z)

    Args:
        obj: Any object. In practice, this is a HookedViT (or subclass)
        path: str. The path to the attribute you want to access. (eg "blocks.0.attn.hook_z")

    returns:
        Any. The attribute at the end of the path
    """
    parts = path.split(".")
    # Navigate to the last component in the path
    for part in parts:
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
    return obj


def set_deep_attr(obj: Any, path: str, value: Any):
    """Helper function to change the value of a nested attribute from a object.
    In practice used to swap HookedViT HookPoints (eg model.blocks[0].attn.hook_z) with HookedSAEs and vice versa

    Args:
        obj: Any object. In practice, this is a HookedViT (or subclass)
        path: str. The path to the attribute you want to access. (eg "blocks.0.attn.hook_z")
        value: Any. The value you want to set the attribute to (eg a HookedSAE object)
    """
    parts = path.split(".")
    # Navigate to the last component in the path
    for part in parts[:-1]:
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
    # Set the value on the final attribute
    setattr(obj, parts[-1], value)