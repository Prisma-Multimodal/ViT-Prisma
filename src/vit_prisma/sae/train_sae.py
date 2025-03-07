from vit_prisma.utils.data_utils.cifar.cifar_10_utils import load_cifar_10
from vit_prisma.utils.load_model import load_model
from vit_prisma.sae.config import VisionModelSAERunnerConfig
from vit_prisma.sae.sae import StandardSparseAutoencoder, GatedSparseAutoencoder
from vit_prisma.sae.training.activations_store import VisionActivationsStore, CacheVisionActivationStore

from vit_prisma.models.base_vit import HookedViT

import wandb

from vit_prisma.sae.training.geometric_median import compute_geometric_median
from vit_prisma.sae.training.get_scheduler import get_scheduler

# from vit_prisma.sae.evals import run_evals_vision
# from vit_prisma.sae.evals.evals import get_substitution_loss, get_text_embeddings, get_text_embeddings_openclip, get_text_labels

from vit_prisma.dataloaders.imagenet_index import imagenet_index

import torch
from torch.optim import Adam
from tqdm import tqdm
import re

# this should be abstracted out of this file in the long term
import open_clip

import os
import sys

import torchvision

import einops
import numpy as np

from typing import Any, cast

from dataclasses import is_dataclass, fields

import uuid

import wandb

from torch.nn.parallel import DistributedDataParallel as DDP
# distributed sampler
from torch.utils.data.distributed import DistributedSampler

# import dist
import torch.distributed as dist

# import dataloader
from torch.utils.data import DataLoader




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


class VisionSAETrainer:
    def __init__(self, 
        cfg: VisionModelSAERunnerConfig,
        model: HookedViT,
        train_dataset: torch.utils.data.Dataset,
        val_dataset: torch.utils.data.Dataset,
    ):
        self.cfg = cfg
        
        # Set up the distributed device
        if self.cfg.distributed:
            self.cfg.device = f'cuda:{self.cfg.gpu}'
            torch.cuda.set_device(self.cfg.gpu)
            
            # Calculate local batch sizes
            self.cfg.local_train_batch_size = self.cfg.train_batch_size // self.cfg.world_size
            self.cfg.local_store_batch_size = self.cfg.store_batch_size // self.cfg.world_size
            
            # Log the batch sizes if rank 0
            if self.cfg.rank == 0:
                print(f"Global training batch size: {self.cfg.train_batch_size}")
                print(f"Local training batch size per GPU: {self.cfg.local_train_batch_size}")
                print(f"Global store batch size: {self.cfg.store_batch_size}")
                print(f"Local store batch size per GPU: {self.cfg.local_store_batch_size}")
        else:
            self.cfg.device = self.cfg.device if torch.cuda.is_available() else 'cpu'
            self.cfg.local_train_batch_size = self.cfg.train_batch_size
            self.cfg.local_store_batch_size = self.cfg.store_batch_size
        
        # Move model to device
        self.model = model.to(self.cfg.device)
        
        # Wrap model with DDP if distributed
        if self.cfg.distributed:
            self.model = DDP(self.model, device_ids=[self.cfg.gpu])

        self.set_default_attributes()  # For backward compatibility

        self.bad_run_check = (
            True if self.cfg.min_l0 and self.cfg.min_explained_variance else False
        )

        # Create the SAE and move it to the device
        if self.cfg.architecture == "gated":
            self.sparse_coder = GatedSparseAutoencoder(self.cfg).to(self.cfg.device)
        elif self.cfg.architecture == "standard" or self.cfg.architecture == "vanilla":
            self.sparse_coder = StandardSparseAutoencoder(self.cfg).to(self.cfg.device)
        else:
            raise ValueError(f"Loading of {self.cfg.architecture} not supported")

        # Wrap SAE with DDP if distributed
        if self.cfg.distributed:
            self.sparse_coder = DDP(self.sparse_coder, device_ids=[self.cfg.gpu])

        # Set up the dataset and samplers
        self.dataset = train_dataset
        self.eval_dataset = val_dataset
        self.activations_store = self.initialize_activations_store(
            self.dataset, self.eval_dataset
        )

        # Create distributed samplers if in distributed mode
        if self.cfg.distributed:
            self.train_sampler = DistributedSampler(self.dataset, shuffle=True)
            self.val_sampler = DistributedSampler(self.eval_dataset, shuffle=True)
        else:
            self.train_sampler = None
            self.val_sampler = None
            
        # Set up wandb project name if not provided
        if not self.cfg.wandb_project:
            self.cfg.wandb_project = (
                self.cfg.model_name.replace("/", "-")
                + "-expansion-"
                + str(self.cfg.expansion_factor)
                + "-layer-"
                + str(self.cfg.hook_point_layer)
            )
            
        # Generate a unique hash for this run
        if not self.cfg.distributed or self.cfg.rank == 0:
            self.cfg.unique_hash = uuid.uuid4().hex[:8]
        
        # Broadcast the unique hash to all processes if distributed
        if self.cfg.distributed:
            if self.cfg.rank == 0:
                unique_hash = torch.tensor(
                    [ord(c) for c in self.cfg.unique_hash], 
                    dtype=torch.uint8, device=self.cfg.device
                )
            else:
                unique_hash = torch.zeros(8, dtype=torch.uint8, device=self.cfg.device)
            
            # Broadcast the hash from rank 0 to all processes
            dist.broadcast(unique_hash, 0)
            
            if self.cfg.rank != 0:
                self.cfg.unique_hash = ''.join([chr(i) for i in unique_hash.cpu().numpy()])
        
        self.cfg.run_name = self.cfg.unique_hash + "-" + self.cfg.wandb_project

        # Setup checkpointing
        self.checkpoint_thresholds = self.get_checkpoint_thresholds()
        self.setup_checkpoint_path()

        # Only print config on rank 0
        if not self.cfg.distributed or self.cfg.rank == 0:
            self.cfg.pretty_print() if self.cfg.verbose else None
    
    def create_data_loader(self, dataset, sampler, is_train=True):
        """Create a data loader for the given dataset."""
        return DataLoader(
            dataset,
            batch_size=self.cfg.train_batch_size,
            shuffle=(sampler is None and is_train),
            sampler=sampler,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            drop_last=is_train,
        )

    
    # @torch.no_grad()
    # def get_activations(self, images):
    #     """Compute activations for the given images."""
    #     hook_point_layers = (
    #         self.cfg.hook_point_layer
    #         if isinstance(self.cfg.hook_point_layer, list)
    #         else [self.cfg.hook_point_layer]
    #     )
        
    #     # Get the base model
    #     if hasattr(self.model, 'module'):
    #         model = self.model.module
    #     else:
    #         model = self.model
            
    #     # Define hook points to extract
    #     hook_points = [self.cfg.hook_point.format(layer=layer) for layer in hook_point_layers]
        
    #     # Run with cache to get activations
    #     with torch.cuda.amp.autocast(enabled=self.cfg.dtype == torch.float16):
    #         _, cache = model.run_with_cache(images, names_filter=hook_points)
        
    #     # Stack activations from all layers
    #     activations_list = []
    #     for hook_point in hook_points:
    #         acts = cache[hook_point]
            
    #         # Apply additional processing if needed
    #         if self.cfg.hook_point_head_index is not None:
    #             acts = acts[:, :, self.cfg.hook_point_head_index]
                
    #         if self.cfg.cls_token_only:
    #             acts = acts[:, 0:1]
    #         elif self.cfg.use_patches_only:
    #             acts = acts[:, 1:]
                
    #         activations_list.append(acts)
    #     # Create layer_acts in the format expected by train_step
    #     # Shape: [batch_size, num_layers, d_in]
    #     layer_acts = torch.stack(activations_list, dim=1)
        
    #     return layer_acts


    def set_default_attributes(self):
        """
        For backward compatability, add new attributes here
        """
        # Set default values for attributes that might not be in the loaded config
        default_attributes = ["min_l0", "min_explained_variance"]

        for attr in default_attributes:
            if not hasattr(self.cfg, attr):
                setattr(self.cfg, attr, None)

    def setup_checkpoint_path(self):
        if self.cfg.n_checkpoints:
            # Create checkpoint path with run_name, which contains unique identifier
            self.cfg.checkpoint_path = f"{self.cfg.checkpoint_path}/{self.cfg.run_name}"
            
            # Only rank 0 creates the directory
            if not self.cfg.distributed or self.cfg.rank == 0:
                os.makedirs(self.cfg.checkpoint_path, exist_ok=True)
                if self.cfg.verbose:
                    print(f"Checkpoint path: {self.cfg.checkpoint_path}")
        else:
            if not self.cfg.distributed or self.cfg.rank == 0:
                print(f"Not saving checkpoints so skipping creating checkpoint directory")


    def initialize_activations_store(self, dataset, eval_dataset):
        """Initialize the appropriate activation store with proper batch sizes."""
        # Create a copy of the config with adjusted batch sizes for distributed mode
        import copy
        store_cfg = copy.deepcopy(self.cfg)
        
        if self.cfg.distributed:
            # Use local batch sizes for the activation store
            store_cfg.train_batch_size = self.cfg.local_train_batch_size
            store_cfg.store_batch_size = self.cfg.local_store_batch_size
        
        if self.cfg.use_cached_activations:
            return CacheVisionActivationStore(store_cfg)

        return VisionActivationsStore(
            store_cfg,
            self.model,
            dataset,
            eval_dataset=eval_dataset,
            num_workers=self.cfg.num_workers,
        )
    
    # @staticmethod
    # def load_dataset(cfg):

    #     from vit_prisma.transforms.model_transforms import (
    #         get_model_transforms,
    #     )
                
    #     data_transforms = get_model_transforms(cfg.model_name)

    #     if cfg.dataset_name == "imagenet1k":
    #         (
    #             print(f"Dataset type: {cfg.dataset_name}")
    #             if cfg.verbose
    #             else None
    #         )
    #         # Imagenet-specific logic
    #         from vit_prisma.utils.data_utils.imagenet.imagenet_utils import (
    #             setup_imagenet_paths,
    #         )
    #         from vit_prisma.dataloaders.imagenet_dataset import (
    #             ImageNetValidationDataset,
    #         )

    #         imagenet_paths = setup_imagenet_paths(cfg.dataset_path)

    #         train_data = torchvision.datasets.ImageFolder(
    #             cfg.dataset_train_path, transform=data_transforms
    #         )

    #         val_data = ImageNetValidationDataset(
    #             cfg.dataset_val_path,
    #             imagenet_paths["label_strings"],
    #             imagenet_paths["val_labels"],
    #             data_transforms,
    #         )

    #     elif cfg.dataset_name == "cifar10":
    #         train_data, val_data, test_data = load_cifar_10(
    #             cfg.dataset_path, image_size=cfg.image_size
    #         )
    #     else:
    #         try:
    #             from torchvision.datasets import DatasetFolder
    #             from torchvision.datasets.folder import default_loader
    #             from torch.utils.data import random_split

    #             dataset = DatasetFolder(
    #                 root=cfg.dataset_path,
    #                 loader=default_loader,
    #                 extensions=('.jpg', '.jpeg', '.png'),
    #                 transform=data_transforms
    #             )

    #             train_size = int(0.8 * len(dataset))
    #             print("traning data size : ", train_size)
    #             cfg.total_training_images = train_size

    #             val_size = len(dataset) - train_size

    #             train_data, val_data = random_split(dataset, [train_size, val_size])
    #         except:
    #             raise ValueError("Invalid dataset")
        
    #     print(f"Train data length: {len(train_data)}") if cfg.verbose else None
    #     print(f"Validation data length: {len(val_data)}") if cfg.verbose else None
    
    #     return train_data, val_data

    def get_checkpoint_thresholds(self):
        if self.cfg.n_checkpoints > 0:
            return list(
                range(
                    0,
                    self.cfg.total_training_tokens,
                    self.cfg.total_training_tokens // self.cfg.n_checkpoints,
                )
            )[1:]
        return []

    def initialize_training_variables(self):
        # Initialize training variables on the appropriate device
        act_freq_scores = torch.zeros(int(self.cfg.d_sae), device=self.cfg.device)
        n_forward_passes_since_fired = torch.zeros(
            int(self.cfg.d_sae), device=self.cfg.device
        )
        n_frac_active_tokens = 0
        
        # For distributed training, we need to access the module directly
        if self.cfg.distributed:
            optimizer_params = self.sparse_coder.module.parameters()
        else:
            optimizer_params = self.sparse_coder.parameters()
            
        optimizer = Adam(optimizer_params, lr=self.cfg.lr)
        scheduler = get_scheduler(
            self.cfg.lr_scheduler_name,
            optimizer=optimizer,
            warm_up_steps=self.cfg.lr_warm_up_steps,
            training_steps=self.cfg.total_training_steps,
            lr_end=self.cfg.lr / 10,
        )
        return (
            act_freq_scores,
            n_forward_passes_since_fired,
            n_frac_active_tokens,
            optimizer,
            scheduler,
        )

    def initialize_geometric_medians(self):
        
        # Get the appropriate hyperparams object
        if self.cfg.distributed:
            hyperparams = self.sparse_coder.module.cfg
            sae_module = self.sparse_coder.module
        else:
            hyperparams = self.sparse_coder.cfg
            sae_module = self.sparse_coder
            
        all_layers = sae_module.cfg.hook_point_layer
        geometric_medians = {}
        if not isinstance(all_layers, list):
            all_layers = [all_layers]
        hyperparams = sae_module.cfg
        sae_layer_id = all_layers.index(hyperparams.hook_point_layer)
        
        if hyperparams.b_dec_init_method == "geometric_median":
            # Get local activation data
            local_acts = self.activations_store.storage_buffer.detach()[
                :, sae_layer_id, :
            ]
            
            if sae_layer_id not in geometric_medians:
                # For distributed training, compute local median first
                local_median = compute_geometric_median(local_acts, maxiter=100).median
                
                # Gather medians from all processes
                world_size = dist.get_world_size()
                gathered_medians = [torch.zeros_like(local_median) for _ in range(world_size)]
                dist.all_gather(gathered_medians, local_median)
                
                # Compute global geometric median from all local medians
                if dist.get_rank() == 0:
                    global_median = compute_geometric_median(
                        torch.stack(gathered_medians), maxiter=200
                    ).median
                else:
                    global_median = torch.zeros_like(local_median)
                    
                # Broadcast result to all processes
                dist.broadcast(global_median, src=0)
                geometric_medians[sae_layer_id] = global_median
                
            sae_module.initialize_b_dec_with_precalculated(
                geometric_medians[sae_layer_id]
            )
            
        elif hyperparams.b_dec_init_method == "mean":
            # For mean initialization, use distributed averaging
            local_acts = self.activations_store.storage_buffer.detach()[:, sae_layer_id, :]
            local_sum = local_acts.sum(dim=0)
            local_count = torch.tensor(local_acts.shape[0], device=local_acts.device)
            
            # Sum across all processes
            global_sum = local_sum.clone()
            global_count = local_count.clone()
            dist.all_reduce(global_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(global_count, op=dist.ReduceOp.SUM)
            
            # Compute global mean
            global_mean = global_sum / global_count
            
            sae_module.initialize_b_dec_with_precalculated(global_mean)
            
        self.sparse_coder.train()
        return geometric_medians


    def train_step(
        self,
        sparse_autoencoder,
        optimizer,
        scheduler,
        act_freq_scores,
        n_forward_passes_since_fired,
        n_frac_active_tokens,
        layer_acts,
        n_training_steps,
        n_training_tokens,
    ):
        # Access the module directly if using DDP
        if self.cfg.distributed:
            sparse_autoencoder_module = sparse_autoencoder.module
            hyperparams = sparse_autoencoder_module.cfg
        else:
            sparse_autoencoder_module = sparse_autoencoder
            hyperparams = sparse_autoencoder.cfg

        all_layers = (
            hyperparams.hook_point_layer
            if isinstance(hyperparams.hook_point_layer, list)
            else [hyperparams.hook_point_layer]
        )
        layer_id = all_layers.index(hyperparams.hook_point_layer)
        sae_in = layer_acts[:, layer_id, :]

        sparse_autoencoder.train()
        
        # If DDP, need to call the module method directly
        if self.cfg.distributed:
            sparse_autoencoder_module.set_decoder_norm_to_unit_norm()
        else:
            sparse_autoencoder.set_decoder_norm_to_unit_norm()

        # Log feature sparsity every feature_sampling_window steps
        if (n_training_steps + 1) % self.cfg.feature_sampling_window == 0:
            # All-reduce act_freq_scores and n_frac_active_tokens if distributed
            if self.cfg.distributed:
                local_act_freq = act_freq_scores.clone()
                local_active_tokens = torch.tensor(n_frac_active_tokens, device=self.cfg.device, dtype=torch.float32)
                
                # All-reduce to get global values
                dist.all_reduce(local_act_freq)
                dist.all_reduce(local_active_tokens)
                
                feature_sparsity = local_act_freq / local_active_tokens
            else:
                feature_sparsity = act_freq_scores / n_frac_active_tokens
                
            log_feature_sparsity = torch.log10(feature_sparsity + 1e-10).detach().cpu()

            if self.cfg.log_to_wandb and (not self.cfg.distributed or self.cfg.rank == 0):
                self._log_feature_sparsity(
                    sparse_autoencoder,
                    hyperparams,
                    log_feature_sparsity,
                    feature_sparsity,
                    n_training_steps,
                )

            act_freq_scores = torch.zeros(
                sparse_autoencoder_module.cfg.d_sae, device=sparse_autoencoder_module.cfg.device
            )
            n_frac_active_tokens = 0

        optimizer.zero_grad()

        ghost_grad_neuron_mask = (
            n_forward_passes_since_fired > sparse_autoencoder_module.cfg.dead_feature_window
        ).bool()

        # Forward and Backward Passes
        (
            sae_out,
            feature_acts,
            loss,
            mse_loss,
            l1_loss,
            ghost_grad_loss,
            aux_reconstruction_loss,
        ) = sparse_autoencoder(sae_in, ghost_grad_neuron_mask)

        with torch.no_grad():
            did_fire = (feature_acts > 0).float().sum(-2) > 0
            n_forward_passes_since_fired += 1
            n_forward_passes_since_fired[did_fire] = 0

            act_freq_scores += (feature_acts.abs() > 0).float().sum(0)
            batch_size = sae_in.shape[0]
            n_frac_active_tokens += batch_size

            l0 = (feature_acts > 0).float().sum(-1).mean()

            if self.cfg.log_to_wandb and (
                (n_training_steps + 1) % self.cfg.wandb_log_frequency == 0
            ) and (not self.cfg.distributed or self.cfg.rank == 0):
                self._log_metrics(
                    sparse_autoencoder,
                    hyperparams,
                    optimizer,
                    sae_in,
                    sae_out,
                    n_forward_passes_since_fired,
                    ghost_grad_neuron_mask,
                    mse_loss,
                    l1_loss,
                    aux_reconstruction_loss,
                    ghost_grad_loss,
                    loss,
                    l0,
                    n_training_steps,
                    n_training_tokens,
                )

        loss.backward()

        if self.cfg.max_grad_norm:  # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                sparse_autoencoder.parameters(), max_norm=self.cfg.max_grad_norm
            )

        # Access module method directly if using DDP
        if self.cfg.distributed:
            sparse_autoencoder_module.remove_gradient_parallel_to_decoder_directions()
        else:
            sparse_autoencoder.remove_gradient_parallel_to_decoder_directions()
            
        optimizer.step()
        scheduler.step()

        # If distributed, all-reduce metrics for consistent reporting
        if self.cfg.distributed:
            # Create tensors for reduction
            metrics = torch.tensor([loss.item(), mse_loss.item(), 
                                   l1_loss.item() if l1_loss is not None else 0.0, 
                                   l0.item()], device=self.cfg.device)
            dist.all_reduce(metrics)
            metrics /= self.cfg.world_size
            
            # Unpack the reduced metrics
            loss_item, mse_loss_item, l1_loss_item, l0_item = metrics.tolist()
            
            # Create new tensors with the reduced values
            reduced_loss = torch.tensor(loss_item, device=self.cfg.device)
            reduced_mse_loss = torch.tensor(mse_loss_item, device=self.cfg.device)
            reduced_l1_loss = torch.tensor(l1_loss_item, device=self.cfg.device) if l1_loss is not None else None
            reduced_l0 = l0_item
            
            return (
                reduced_loss,
                reduced_mse_loss,
                reduced_l1_loss,
                reduced_l0,
                act_freq_scores,
                n_forward_passes_since_fired,
                n_frac_active_tokens,
            )
        else:
            return (
                loss,
                mse_loss,
                l1_loss,
                l0,
                act_freq_scores,
                n_forward_passes_since_fired,
                n_frac_active_tokens,
            )

    # layer_acts be a poor format - need to run in ctx_len, gt_labels format
    @torch.no_grad()
    def val(self, sparse_autoencoder):
        sparse_autoencoder.eval()

        if dist.is_initialized() and self.cfg.verbose and dist.get_rank() == 0:
            print("Running validation")
        elif not dist.is_initialized() and self.cfg.verbose:
            print("Running validation")

        if self.cfg.distributed:
            model_module = self.model.module
            sparse_coder_module = sparse_autoencoder.module
        else:
            model_module = self.model
            sparse_coder_module = sparse_autoencoder

        # Initialize metrics collectors
        all_mse_losses = []
        all_explained_variances = []
        all_l0_values = []
        all_cos_sims = []
        all_scores = []
        all_sae_recon_losses = []

        for images, gt_labels in self.activations_store.image_dataloader_eval:
            images = images.to(self.cfg.device)
            gt_labels = gt_labels.to(self.cfg.device)
            
            # needs to start with batch_size dimension
            _, cache = model_module.run_with_cache(
                images, names_filter=sparse_coder_module.cfg.hook_point
            )
            hook_point_activation = cache[sparse_coder_module.cfg.hook_point].to(
                self.cfg.device
            )

            sae_out, feature_acts, loss, mse_loss, l1_loss, _, _ = sparse_coder_module(
                hook_point_activation
            )

            # explained variance
            sae_in = hook_point_activation
            per_token_l2_loss = (sae_out - sae_in).pow(2).sum(dim=-1).squeeze()
            total_variance = (sae_in - sae_in.mean(0)).pow(2).sum(-1)
            explained_variance = 1 - per_token_l2_loss / total_variance

            # L0
            l0 = (feature_acts > 0).float().sum(-1).mean()

            # Calculate cosine similarity between original activations and sae output
            cos_sim = (
                torch.cosine_similarity(
                    einops.rearrange(
                        hook_point_activation, "batch seq d_mlp -> (batch seq) d_mlp"
                    ),
                    einops.rearrange(sae_out, "batch seq d_mlp -> (batch seq) d_mlp"),
                    dim=0,
                )
                .mean(-1)
                .item()
            )

            # Track metrics locally for each batch
            all_mse_losses.append(mse_loss.detach())
            all_explained_variances.append(explained_variance.mean().detach())
            all_l0_values.append(l0.detach())
            all_cos_sims.append(cos_sim)

            # this should only run if this is a clip model
            score = None
            sae_recon_loss = None
            if self.cfg.model_name.startswith("open-clip:"):
                # create a list of all imagenet classes
                num_imagenet_classes = 1000
                batch_label_names = [
                    imagenet_index[str(int(label))][1]
                    for label in range(num_imagenet_classes)
                ]

                model_name = (
                    self.cfg.model_name
                    if not self.cfg.model_name.startswith("open-clip:")
                    else self.cfg.model_name[10:]
                )
                
                if dist.is_initialized() and dist.get_rank() == 0:
                    print(f"model_name: {model_name}")
                elif not dist.is_initialized():
                    print(f"model_name: {model_name}")
                    
                # bad bad bad
                oc_model_name = "hf-hub:" + model_name

                # should be moved to hookedvit pretrained long terms
                og_model, _, preproc = open_clip.create_model_and_transforms(
                    oc_model_name
                )
                tokenizer = open_clip.get_tokenizer("ViT-B-32")

                text_embeddings = get_text_embeddings_openclip(
                    og_model, preproc, tokenizer, batch_label_names
                )
                
                score, model_loss, sae_recon_loss, zero_abl_loss = (
                    get_substitution_loss(
                        sparse_autoencoder,
                        self.model,
                        images,
                        gt_labels,
                        text_embeddings,
                        device=self.cfg.device,
                    )
                )
                
                all_scores.append(score)
                all_sae_recon_losses.append(sae_recon_loss)

            break  # Currently runs one batch only

        # Aggregate metrics across all processes if using distributed training
        if dist.is_initialized():
            # Convert lists to tensors for easier gathering
            mse_loss_tensor = torch.tensor(all_mse_losses).mean().to(self.cfg.device)
            exp_var_tensor = torch.tensor(all_explained_variances).mean().to(self.cfg.device)
            l0_tensor = torch.tensor(all_l0_values).mean().to(self.cfg.device)
            cos_sim_tensor = torch.tensor(all_cos_sims).mean().to(self.cfg.device)
            
            # Create tensors for gathering
            gather_mse = [torch.zeros_like(mse_loss_tensor) for _ in range(dist.get_world_size())]
            gather_exp_var = [torch.zeros_like(exp_var_tensor) for _ in range(dist.get_world_size())]
            gather_l0 = [torch.zeros_like(l0_tensor) for _ in range(dist.get_world_size())]
            gather_cos_sim = [torch.zeros_like(cos_sim_tensor) for _ in range(dist.get_world_size())]
            
            # Gather metrics from all processes
            dist.all_gather(gather_mse, mse_loss_tensor)
            dist.all_gather(gather_exp_var, exp_var_tensor)
            dist.all_gather(gather_l0, l0_tensor)
            dist.all_gather(gather_cos_sim, cos_sim_tensor)
            
            # Average the gathered metrics
            mse_loss_final = torch.stack(gather_mse).mean().item()
            exp_var_final = torch.stack(gather_exp_var).mean().item()
            l0_final = torch.stack(gather_l0).mean().item()
            cos_sim_final = torch.stack(gather_cos_sim).mean().item()
            
            # Handle CLIP-specific metrics if they exist
            score_final = None
            sae_recon_loss_final = None
            if all_scores:
                score_tensor = torch.tensor(all_scores).mean().to(self.cfg.device)
                sae_recon_loss_tensor = torch.tensor(all_sae_recon_losses).mean().to(self.cfg.device)
                
                gather_score = [torch.zeros_like(score_tensor) for _ in range(dist.get_world_size())]
                gather_sae_recon = [torch.zeros_like(sae_recon_loss_tensor) for _ in range(dist.get_world_size())]
                
                dist.all_gather(gather_score, score_tensor)
                dist.all_gather(gather_sae_recon, sae_recon_loss_tensor)
                
                score_final = torch.stack(gather_score).mean().item()
                sae_recon_loss_final = torch.stack(gather_sae_recon).mean().item()
        else:
            # Single process case
            mse_loss_final = sum(all_mse_losses) / len(all_mse_losses) if all_mse_losses else 0
            exp_var_final = sum(all_explained_variances) / len(all_explained_variances) if all_explained_variances else 0
            l0_final = sum(all_l0_values) / len(all_l0_values) if all_l0_values else 0
            cos_sim_final = sum(all_cos_sims) / len(all_cos_sims) if all_cos_sims else 0
            score_final = sum(all_scores) / len(all_scores) if all_scores else None
            sae_recon_loss_final = sum(all_sae_recon_losses) / len(all_sae_recon_losses) if all_sae_recon_losses else None

        # Log metrics only on rank 0 process or in non-distributed mode
        if (dist.is_initialized() and dist.get_rank() == 0) or not dist.is_initialized():
            log_data = {
                "validation_metrics/mse_loss": mse_loss_final,
                "validation_metrics/explained_variance": exp_var_final,
                "validation_metrics/L0": l0_final,
            }
            
            if score_final is not None:
                log_data.update({
                    "validation_metrics/substitution_score": score_final,
                    "validation_metrics/substitution_loss": sae_recon_loss_final,
                })
            
            wandb.log(log_data)

            print(f"cos_sim: {cos_sim_final}")
            
        

    def _log_feature_sparsity(
        self,
        sparse_autoencoder,
        hyperparams,
        log_feature_sparsity,
        feature_sparsity,
        n_training_steps,
    ):
        # Skip logging if not rank 0 in distributed setting
        if self.cfg.distributed and self.cfg.rank != 0:
            return
            
        suffix = wandb_log_suffix(
            sparse_autoencoder.module.cfg if self.cfg.distributed else sparse_autoencoder.cfg, 
            hyperparams
        )

        # Calculate and log feature-level sparsity metrics
        log_sparsity_np = log_feature_sparsity.detach().cpu().numpy()
        log_sparsity_histogram = wandb.Histogram(log_sparsity_np)

        wandb.log(
            {
                f"metrics/mean_log10_feature_sparsity{suffix}": log_feature_sparsity.mean().item(),
                f"plots/log_feature_density_histogram{suffix}": log_sparsity_histogram,
                f"sparsity/below_1e-5{suffix}": (feature_sparsity < 1e-5).sum().item(),
                f"sparsity/below_1e-6{suffix}": (feature_sparsity < 1e-6).sum().item(),
            },
            step=n_training_steps,
        )

    def _log_metrics(
        self,
        sparse_autoencoder,
        hyperparams,
        optimizer,
        sae_in,
        sae_out,
        n_forward_passes_since_fired,
        ghost_grad_neuron_mask,
        mse_loss,
        l1_loss,
        aux_reconstruction_loss,
        ghost_grad_loss,
        loss,
        l0,
        n_training_steps,
        n_training_tokens,
    ):
        # Skip logging if not rank 0 in distributed setting
        if self.cfg.distributed and self.cfg.rank != 0:
            return
            
        current_learning_rate = optimizer.param_groups[0]["lr"]
        per_token_l2_loss = (sae_out - sae_in).pow(2).sum(dim=-1).squeeze()
        total_variance = (sae_in - sae_in.mean(0)).pow(2).sum(-1)
        explained_variance = 1 - per_token_l2_loss / total_variance

        # Check for bad run conditions
        if (
            (self.bad_run_check)
            and (l0.item()) < self.cfg.min_l0
            and (explained_variance.mean().item() < self.cfg.min_explained_variance)
        ):
            print(f"Skipping bad run. Moving to the next run.")
            wandb.finish()
            sys.exit()

        n_training_images = n_training_tokens // self.cfg.context_size

        # Handle None case for l1_loss (when using top k SAE loss)
        if l1_loss is None:
            l1_loss = torch.tensor(0.0, device=self.cfg.device)

        # Generate suffix for metrics
        suffix = wandb_log_suffix(
            sparse_autoencoder.module.cfg if self.cfg.distributed else sparse_autoencoder.cfg, 
            hyperparams
        )
        
        # Prepare metrics dict for logging
        metrics = {
            f"losses/mse_loss{suffix}": mse_loss.item(),
            f"losses/l1_loss{suffix}": l1_loss.item() / (
                sparse_autoencoder.module.l1_coefficient 
                if self.cfg.distributed else 
                sparse_autoencoder.l1_coefficient
            ),
            f"losses/ghost_grad_loss{suffix}": ghost_grad_loss.item(),
            f"losses/overall_loss{suffix}": loss.item(),
            f"metrics/explained_variance{suffix}": explained_variance.mean().item(),
            f"metrics/explained_variance_std{suffix}": explained_variance.std().item(),
            f"metrics/l0{suffix}": l0.item(),
            f"sparsity/mean_passes_since_fired{suffix}": n_forward_passes_since_fired.mean().item(),
            f"sparsity/dead_features{suffix}": ghost_grad_neuron_mask.sum().item(),
            f"details/current_learning_rate{suffix}": current_learning_rate,
            "details/n_training_tokens": n_training_tokens,
            "details/n_training_images": n_training_images,
        }

        # Add gated SAE specific metrics if applicable
        if self.cfg.architecture == "gated":
            metrics[f"losses/aux_reconstruction_loss{suffix}"] = aux_reconstruction_loss.item()

        # Log to wandb
        wandb.log(metrics, step=n_training_steps)

    def _run_evals(self, sparse_autoencoder, hyperparams, n_training_steps):
        sparse_autoencoder.eval()
        suffix = wandb_log_suffix(sparse_autoencoder.cfg, hyperparams)
        try:
            run_evals_vision(
                sparse_autoencoder,
                self.activations_store,
                self.model,
                n_training_steps,
                suffix=suffix,
            )
        except Exception as e:
            print(f"Error in run_evals_vision: {e}")
        sparse_autoencoder.train()

    def log_metrics(self, sae, hyperparams, metrics, n_training_steps):
        # Skip if not rank 0 in distributed setting or not logging to wandb
        if not self.cfg.log_to_wandb or (self.cfg.distributed and self.cfg.rank != 0):
            return
            
        # Only log at specified frequency
        if (n_training_steps + 1) % self.cfg.wandb_log_frequency == 0:
            suffix = wandb_log_suffix(
                sae.module.cfg if self.cfg.distributed else sae.cfg, 
                hyperparams
            )
            wandb.log(metrics, step=n_training_steps)

    def checkpoint(self, sae, n_training_tokens, act_freq_scores, n_frac_active_tokens):
        # Skip if not rank 0 in distributed setting
        if self.cfg.distributed:
            # Create a tensor with the current token count
            tokens_tensor = torch.tensor([n_training_tokens], device=self.cfg.device)
            # Broadcast from rank 0 to ensure all processes have the same value
            dist.broadcast(tokens_tensor, src=0)
            n_training_tokens = tokens_tensor.item()
        
        # Skip if not rank 0 in distributed setting
        if self.cfg.distributed and self.cfg.rank != 0:
            return
            
        # Save config first
        self.cfg.save_config(f"{self.cfg.checkpoint_path}/config.json")

        # Calculate number of training images
        n_training_images = n_training_tokens // self.cfg.context_size
        
        # Set up checkpoint path
        path = self.cfg.checkpoint_path + f"/n_images_{n_training_images}.pt"
        
        # Set decoder norm and save model
        if self.cfg.distributed:
            sae.module.set_decoder_norm_to_unit_norm()
            sae.module.save_model(path)
        else:
            sae.set_decoder_norm_to_unit_norm()
            sae.save_model(path)

        # Log checkpoint path to wandb
        if self.cfg.log_to_wandb:
            wandb.log({
                "details/checkpoint_path": path,
            })

        # Save log feature sparsity
        log_feature_sparsity_path = (
            self.cfg.checkpoint_path
            + f"/n_images_{n_training_images}_log_feature_sparsity.pt"
        )
        
        # Calculate and save feature sparsity
        if self.cfg.distributed:
            # All-reduce to get global values
            local_act_freq = act_freq_scores.clone()
            local_active_tokens = torch.tensor(n_frac_active_tokens, device=self.cfg.device, dtype=torch.float32)
            
            dist.all_reduce(local_act_freq)
            dist.all_reduce(local_active_tokens)
            
            feature_sparsity = local_act_freq / local_active_tokens
        else:
            feature_sparsity = act_freq_scores / n_frac_active_tokens
            
        log_feature_sparsity = torch.log10(feature_sparsity + 1e-10).detach().cpu()
        torch.save(log_feature_sparsity, log_feature_sparsity_path)

        # Save to wandb if enabled
        if self.cfg.log_to_wandb:
            hyperparams = sae.module.cfg if self.cfg.distributed else sae.cfg
            self.save_to_wandb(sae, hyperparams, path, log_feature_sparsity_path)

    def save_to_wandb(self, sae, hyperparams, path, log_feature_sparsity_path):
        # Skip if not rank 0 in distributed setting
        if self.cfg.distributed and self.cfg.rank != 0:
            return
            
        suffix = wandb_log_suffix(
            sae.module.cfg if self.cfg.distributed else sae.cfg, 
            hyperparams
        )
        name_for_log = re.sub(self.cfg.unique_hash, "_", suffix)
        try:
            # Create and log model artifact
            model_artifact = wandb.Artifact(
                f"{name_for_log}",
                type="model",
                metadata=dict(
                    sae.module.cfg.__dict__ if self.cfg.distributed else sae.cfg.__dict__
                ),
            )
            model_artifact.add_file(path)
            wandb.log_artifact(model_artifact)

            # Create and log sparsity artifact
            sparsity_artifact = wandb.Artifact(
                f"{name_for_log}_log_feature_sparsity",
                type="log_feature_sparsity",
                metadata=dict(
                    sae.module.cfg.__dict__ if self.cfg.distributed else sae.cfg.__dict__
                ),
            )
            sparsity_artifact.add_file(log_feature_sparsity_path)
            wandb.log_artifact(sparsity_artifact)
        except Exception as e:
            print(f"Error saving to wandb: {e}")
            pass

    @staticmethod
    def dataclass_to_dict(obj):
        if not is_dataclass(obj):
            return obj
        result = {}
        for field in fields(obj):
            value = getattr(obj, field.name)
            if is_dataclass(value):
                result[field.name] = VisionSAETrainer.dataclass_to_dict(value)
            else:
                result[field.name] = value
        return result

    def initalize_wandb(self):
        # Skip if not rank 0 in distributed setting
        if self.cfg.distributed and self.cfg.rank != 0:
            return
            
        config_dict = self.dataclass_to_dict(self.cfg)
        run_name = self.cfg.run_name.replace(":", "_")
        wandb_project = self.cfg.wandb_project.replace(":", "_")
        
        wandb.init(
            project=wandb_project,
            config=config_dict,
            entity=self.cfg.wandb_entity,
            name=run_name,
        )

    def run(self):
         # Initialize wandb on rank 0 only
        if self.cfg.log_to_wandb:
            self.initalize_wandb()
        
        # Initialize training variables
        (
            act_freq_scores,
            n_forward_passes_since_fired,
            n_frac_active_tokens,
            optimizer,
            scheduler,
        ) = self.initialize_training_variables()
        
        # Initialize model weights
        self.initialize_geometric_medians()
        
        # Track current epoch for samplers
        self.current_epoch = 0
        
        # Print only on rank 0
        if not self.cfg.distributed or self.cfg.rank == 0:
            print("Starting training") if self.cfg.verbose else None
            arch = 'Transcoder' if hasattr(self, 'is_transcoder') and self.is_transcoder else 'SAE'
        
        
        # Create progress bar only on rank 0
        if not self.cfg.distributed or self.cfg.rank == 0:
            pbar = tqdm(total=self.cfg.total_training_tokens, desc=f"Training {arch}", mininterval=20)
        
        # Main training loop
        global_n_training_tokens = 0
        global_n_training_steps = 0

        while global_n_training_tokens < self.cfg.total_training_tokens:
        # Get next batch of activations 
            layer_acts = self.activations_store.next_batch()
            
            # Local batch size may vary, especially at the end of dataset
            local_batch_size = layer_acts.shape[0]

            if self.cfg.distributed:
                # Get token counts from all processes
                local_tokens = local_batch_size * self.cfg.context_size
                tensor_tokens = torch.tensor([local_tokens], device=self.cfg.device)
                
                # Sum across all processes
                dist.all_reduce(tensor_tokens, op=dist.ReduceOp.SUM)
                tokens_this_step = tensor_tokens.item()
            else:
                tokens_this_step = local_batch_size * self.cfg.context_size
            
            global_n_training_steps += 1
            global_n_training_tokens += tokens_this_step
        
            # Update progress bar on rank 0
            if not self.cfg.distributed or self.cfg.rank == 0:
                pbar.update(tokens_this_step)
                
            # Initialize loss placeholders
            mse_loss = torch.tensor(0.0, device=self.cfg.device)
            l1_loss = torch.tensor(0.0, device=self.cfg.device)
            
            # Training step
            (
                loss,
                mse_loss,
                l1_loss,
                l0,
                act_freq_scores,
                n_forward_passes_since_fired,
                n_frac_active_tokens,
            ) = self.train_step(
                sparse_autoencoder=self.sparse_coder,
                optimizer=optimizer,
                scheduler=scheduler,
                layer_acts=layer_acts,
                n_training_steps=global_n_training_steps,
                n_training_tokens=global_n_training_tokens,
                act_freq_scores=act_freq_scores,
                n_forward_passes_since_fired=n_forward_passes_since_fired,
                n_frac_active_tokens=n_frac_active_tokens,
            )
            
            # Run validation at specified intervals
            validation_interval = (self.cfg.total_training_tokens // self.cfg.train_batch_size) // self.cfg.n_validation_runs
            if global_n_training_steps > 1 and validation_interval > 0 and global_n_training_tokens % validation_interval == 0:
                self.val(self.sparse_coder)
            
            # Handle None loss values
            if l1_loss is None:  # When using top k SAE loss
                l1_loss = torch.tensor(0.0, device=self.cfg.device)
                
            # Check for checkpointing
            if len(self.checkpoint_thresholds) > 0 and global_n_training_tokens > self.checkpoint_thresholds[0]:
                # Synchronize processes before checkpointing
                if self.cfg.distributed:
                    dist.barrier()
                    
                # Save checkpoint and remove the threshold from the list
                self.checkpoint(
                    self.sparse_coder, global_n_training_tokens, act_freq_scores, n_frac_active_tokens
                )
                
                if self.cfg.verbose and (not self.cfg.distributed or self.cfg.rank == 0):
                    print(f"Checkpoint saved at {global_n_training_tokens} tokens")
                    
                self.checkpoint_thresholds.pop(0)
                
                # Another barrier after checkpointing
                if self.cfg.distributed:
                    dist.barrier()
            
            # Update progress bar on rank 0
            if not self.cfg.distributed or self.cfg.rank == 0:
                pbar.set_description(
                    f"Training {arch}: Loss: {loss.item():.4f}, MSE: {mse_loss.item():.4f}, L1: {l1_loss.item():.4f}, L0: {l0:.4f}",
                    refresh=False
                )
        
        # Final synchronization before checkpointing
        if self.cfg.distributed:
            dist.barrier()
            
        # Final checkpoint
        if self.cfg.n_checkpoints:
            self.checkpoint(
                self.sparse_coder, n_training_tokens, act_freq_scores, n_frac_active_tokens
            )
            
            if self.cfg.verbose and (not self.cfg.distributed or self.cfg.rank == 0):
                print(f"Final checkpoint saved at {n_training_tokens} tokens")
        
        # Close progress bar on rank 0
        if not self.cfg.distributed or self.cfg.rank == 0:
            pbar.close()
        
        # Final sync barrier
        if self.cfg.distributed:
            dist.barrier()
            
        return self.sparse_coder