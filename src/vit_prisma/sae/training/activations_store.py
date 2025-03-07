import os
import torch
import torch.distributed as dist
from typing import Any, Iterator, cast
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from vit_prisma.models.base_vit import HookedViT
from functools import lru_cache


def collate_fn(data):
    imgs = [d[0] for d in data]
    return torch.stack(imgs, dim=0)


def collate_fn_eval(data):
    imgs = [d[0] for d in data]
    return torch.stack(imgs, dim=0), torch.tensor([d[1] for d in data])


class CacheVisionActivationStore:
    def __init__(
        self,
        cfg: Any,
    ):
        self.cfg = cfg

        # Fill the storage buffer with half the desired number of batches
        half_batches = self.cfg.n_batches_in_buffer // 2
        self.storage_buffer = self.get_buffer(half_batches)
        self.dataloader = self.get_data_loader()

        # If using cached activations
        if not self.cfg.use_cached_activations:
            raise ValueError("CacheVisionActivationStore cannot be initialized with cfg.use_cached_activations = False")

    @lru_cache(maxsize=2)
    def load_file_cached(self, file):
        # Only print message from rank 0
        if not self.cfg.distributed or self.cfg.rank == 0:
            print(f"\n\nLoad File {file}\n\"")
        return torch.load(file)

    def _load_cached_activations(self, total_size, context_size, num_layers, d_in):
        """
        Load cached activations from disk until the buffer is filled or no more files are found.
        For distributed training, each rank loads a different subset of the files
        based on rank and world size.
        """
        buffer_size = total_size * context_size
        new_buffer = torch.zeros(
            (buffer_size, num_layers, d_in),
            dtype=self.cfg.dtype,
            device=self.cfg.device,
        )
        n_tokens_filled = 0
        
        # Distribute file loading among ranks
        if self.cfg.distributed:
            world_size = self.cfg.world_size
            rank = self.cfg.rank
            next_cache_idx = rank  # Start with files corresponding to rank
        else:
            next_cache_idx = 0
            
        # Load from cached files one by one
        while n_tokens_filled < buffer_size:
            cache_file = f"{self.cfg.cached_activations_path}/{next_cache_idx}.pt"
            if not os.path.exists(cache_file):
                # If no more files for this rank, move to the next available rank's files
                if self.cfg.distributed:
                    original_next_idx = next_cache_idx
                    found_file = False
                    
                    # Try files from other ranks if needed
                    for offset in range(1, self.cfg.world_size):
                        next_cache_idx = (rank + offset * self.cfg.world_size) % (world_size * 100)  # Limit search
                        cache_file = f"{self.cfg.cached_activations_path}/{next_cache_idx}.pt"
                        if os.path.exists(cache_file):
                            found_file = True
                            break
                            
                    if not found_file:
                        # If still no files, we're done
                        if n_tokens_filled == 0:
                            raise ValueError(f"No cache files found for rank {rank}")
                        new_buffer = new_buffer[:n_tokens_filled, ...]
                        return new_buffer
                else:
                    # For non-distributed training, we're done when we run out of files
                    new_buffer = new_buffer[:n_tokens_filled, ...]
                    return new_buffer

            activations = self.load_file_cached(cache_file)
            if n_tokens_filled + activations.shape[0] > buffer_size:
                # Take only the needed subset
                activations = activations[: buffer_size - n_tokens_filled, ...]
                taking_subset_of_file = True
            else:
                taking_subset_of_file = False

            new_buffer[
                n_tokens_filled : n_tokens_filled + activations.shape[0], ...
            ] = activations
            n_tokens_filled += activations.shape[0]

            if taking_subset_of_file:
                self.next_idx_within_buffer = activations.shape[0]
            else:
                if self.cfg.distributed:
                    # Skip to next file assigned to this rank
                    next_cache_idx += self.cfg.world_size
                else:
                    next_cache_idx += 1
                self.next_idx_within_buffer = 0

        return new_buffer

    def get_data_loader(self) -> Iterator[Any]:
        """
        Create a new DataLoader from a mixed buffer of half "stored" and half "new" activations.
        This ensures variety and mixing each time the DataLoader is refreshed.
        """
        batch_size = self.cfg.train_batch_size
        half_batches = self.cfg.n_batches_in_buffer // 2

        # Mix current storage buffer with new buffer
        mixing_buffer = torch.cat(
            [self.get_buffer(half_batches), self.storage_buffer], dim=0
        )
        
        # Use the same random permutation across ranks if distributed
        if self.cfg.distributed:
            # Generate same permutation on all ranks
            if self.cfg.rank == 0:
                perm = torch.randperm(mixing_buffer.shape[0])
            else:
                perm = torch.zeros(mixing_buffer.shape[0], dtype=torch.long, device=self.cfg.device)
                
            # Broadcast permutation from rank 0
            dist.broadcast(perm, src=0)
            mixing_buffer = mixing_buffer[perm]
        else:
            mixing_buffer = mixing_buffer[torch.randperm(mixing_buffer.shape[0])]

        # Half of the mixed buffer is stored again
        self.storage_buffer = mixing_buffer[: mixing_buffer.shape[0] // 2]

        # The other half is used as the new training DataLoader
        data_for_loader = mixing_buffer[mixing_buffer.shape[0] // 2 :]

        dataloader = iter(
            DataLoader(
                cast(Any, data_for_loader),
                batch_size=batch_size,
                shuffle=not self.cfg.distributed,  # Don't shuffle if distributed
            )
        )
        return dataloader

    def next_batch(self) -> torch.Tensor:
        """
        Get the next batch from the current DataLoader. If the DataLoader is exhausted,
        refill the buffer and create a new DataLoader, then fetch the next batch.
        """
        try:
            return next(self.dataloader)
        except StopIteration:
            self.dataloader = self.get_data_loader()
            return next(self.dataloader)

    def get_buffer(self, n_batches_in_buffer: int) -> torch.Tensor:
        """
        Creates and returns a buffer of activations by loading from cached activations.
        """
        context_size = self.cfg.context_size
        batch_size = self.cfg.store_batch_size
        d_in = self.cfg.d_in
        total_size = batch_size * n_batches_in_buffer

        num_layers = (
            len(self.cfg.hook_point_layer)
            if isinstance(self.cfg.hook_point_layer, list)
            else 1
        )

        return self._load_cached_activations(
            total_size, context_size, num_layers, d_in
        )


class VisionActivationsStore:
    """
    Class for streaming tokens and generating and storing activations
    while training SAEs with distributed training support.
    """

    def __init__(
        self,
        cfg: Any,
        model: HookedViT,
        dataset,
        create_dataloader: bool = True,
        eval_dataset=None,
        num_workers=0,
    ):
        self.cfg = cfg
        self.model = model.to(cfg.device)
        self.dtype = cfg.dtype
        
        if self.dtype == torch.float16:
            self.model = self.model.half()
            
        self.dataset = dataset

        # Create samplers for distributed training if needed
        train_sampler = None
        eval_sampler = None
        
        if cfg.distributed:
            train_sampler = DistributedSampler(
                self.dataset, 
                num_replicas=cfg.world_size,
                rank=cfg.rank,
                shuffle=True
            )
            
            if eval_dataset is not None:
                eval_sampler = DistributedSampler(
                    eval_dataset,
                    num_replicas=cfg.world_size,
                    rank=cfg.rank,
                    shuffle=True
                )

        # Main dataset loader
        self.image_dataloader = DataLoader(
            self.dataset,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=num_workers,
            batch_size=self.cfg.store_batch_size,
            collate_fn=collate_fn,
            drop_last=True,
        )

        # Evaluation dataset loader
        self.image_dataloader_eval = DataLoader(
            eval_dataset,
            shuffle=(eval_sampler is None),
            sampler=eval_sampler,
            num_workers=num_workers,
            batch_size=self.cfg.store_batch_size,
            collate_fn=collate_fn_eval,
            drop_last=True,
        )

        # Keep track of epochs for distributed sampler
        self.current_epoch = 0

        # Infinite iterators for training and eval data
        self.image_dataloader_iter = self._batch_stream(
            self.image_dataloader, device=self.cfg.device
        )
        self.image_dataloader_eval_iter = self._eval_batch_stream(
            self.image_dataloader_eval, device=self.cfg.device
        )

        # Initialize buffer and main dataloader if requested
        if create_dataloader:
            # Fill the storage buffer with half the desired number of batches
            half_batches = self.cfg.n_batches_in_buffer // 2
            self.storage_buffer = self.get_buffer(half_batches)
            self.dataloader = self.get_data_loader()

    def _batch_stream(self, dataloader: DataLoader, device: torch.device) -> Iterator[torch.Tensor]:
    while True:
        if self.cfg.distributed and hasattr(dataloader, 'sampler') and isinstance(dataloader.sampler, DistributedSampler):
            dataloader.sampler.set_epoch(self.current_epoch)  # Sync shuffling
            torch.manual_seed(self.current_epoch) 
                
            for batch in dataloader:
                batch.requires_grad_(False)
                yield batch.to(device)
                
            self.current_epoch += 1

    def _eval_batch_stream(
        self, dataloader: DataLoader, device: torch.device
    ) -> Iterator[torch.Tensor]:
        """
        Infinite iterator over (image_data, labels) from an evaluation dataloader.
        For distributed training, also handles epoch updates for samplers.
        """
        eval_epoch = 0
        while True:
            # Set epoch for distributed sampler if needed
            if self.cfg.distributed and hasattr(dataloader, 'sampler') and isinstance(dataloader.sampler, DistributedSampler):
                dataloader.sampler.set_epoch(eval_epoch)
                
            for image_data, labels in dataloader:
                image_data.requires_grad_(False)
                labels.requires_grad_(False)
                yield image_data.to(device), labels.to(device)
                
            eval_epoch += 1

    @torch.no_grad()
    def get_activations(self, batch_tokens: torch.Tensor) -> torch.Tensor:
        """
        Returns layerwise activations from the HookedViT model according to config.
        Modified to work with DDP-wrapped models.
        """
        layers = (
            self.cfg.hook_point_layer
            if isinstance(self.cfg.hook_point_layer, list)
            else [self.cfg.hook_point_layer]
        )
        act_names = [self.cfg.hook_point.format(layer=layer) for layer in layers]
        stop_layer = max(layers) + 1

        # Get the base model if it's wrapped in DDP
        if hasattr(self.model, 'module'):
            base_model = self.model.module
        else:
            base_model = self.model
            
        # Run model and get cached activations using the base model
        with torch.cuda.amp.autocast(enabled=self.dtype == torch.float16):
            _, layerwise_activations = base_model.run_with_cache(
                batch_tokens, names_filter=act_names, stop_at_layer=stop_layer
            )

        activations_list = []
        for act_name in act_names:
            acts = layerwise_activations[act_name]

            # Select heads if specified
            if self.cfg.hook_point_head_index is not None:
                acts = acts[:, :, self.cfg.hook_point_head_index]

            # Select only CLS token if specified
            if self.cfg.cls_token_only:
                acts = acts[:, 0:1]
                
            # Select only patch tokens if specified
            elif self.cfg.use_patches_only:
                acts = acts[:, 1:]

            activations_list.append(acts)

        return torch.stack(activations_list, dim=2)

    def get_buffer(self, n_batches_in_buffer: int) -> torch.Tensor:
        """
        Creates and returns a buffer of activations by either:
            - Loading from cached activations if `use_cached_activations` is True
            - Or generating them from the model if not cached.
        """
        context_size = self.cfg.context_size
        batch_size = self.cfg.store_batch_size
        d_in = self.cfg.d_in
        total_size = batch_size * n_batches_in_buffer

        num_layers = (
            len(self.cfg.hook_point_layer)
            if isinstance(self.cfg.hook_point_layer, list)
            else 1
        )

        # If using cached activations
        if self.cfg.use_cached_activations:
            return self._load_cached_activations(
                total_size, context_size, num_layers, d_in
            )

        # Otherwise, generate activations from the model
        return self._generate_activations_buffer(
            n_batches_in_buffer, batch_size, context_size, num_layers, d_in
        )

    def load_file_cached(self, file):
        # Only print from rank 0 if distributed
        if not self.cfg.distributed or self.cfg.rank == 0:
            print(f"\n\nLoad File {file}\n")
            
        data = torch.load(file, map_location=self.cfg.device, weights_only=True)
        
        if not self.cfg.distributed or self.cfg.rank == 0:
            print(data.shape)
            
        return data

    def _load_cached_activations(self, total_size, context_size, num_layers, d_in):
        """
        Load cached activations from disk until the buffer is filled or no more files are found.
        In distributed mode, each rank loads different files to maximize throughput.
        """
        buffer_size = total_size * context_size
        new_buffer = torch.zeros(
            (buffer_size, num_layers, d_in),
            dtype=self.cfg.dtype,
            device=self.cfg.device,
        )
        n_tokens_filled = 0
        
        # Distribute file loading among ranks if applicable
        if self.cfg.distributed:
            next_cache_idx = self.cfg.rank  # Start with rank-specific files
            step = self.cfg.world_size     # Skip by world_size
        else:
            next_cache_idx = 0
            step = 1

        # Load from cached files one by one
        while n_tokens_filled < buffer_size:
            cache_file = f"{self.cfg.cached_activations_path}/{next_cache_idx}.pt"
            
            if not os.path.exists(cache_file):
                # In distributed mode, try other files if one is missing
                if self.cfg.distributed:
                    # Try to find any available file
                    found_file = False
                    for i in range(100):  # Limit search to avoid infinite loop
                        next_cache_idx = (next_cache_idx + step) % 1000  # Limit to 1000 files
                        cache_file = f"{self.cfg.cached_activations_path}/{next_cache_idx}.pt"
                        if os.path.exists(cache_file):
                            found_file = True
                            break
                    
                    if not found_file:
                        # If still no files found, we're done
                        break
                else:
                    # In non-distributed mode, we're done when no more files
                    break

            # Only print from rank 0 if distributed
            if not self.cfg.distributed or self.cfg.rank == 0:
                print(f"\n\nLoad next buffer from file {cache_file}\n\"")
                
            activations = self.load_file_cached(cache_file)
            
            if n_tokens_filled + activations.shape[0] > buffer_size:
                # Take only the needed subset
                activations = activations[: buffer_size - n_tokens_filled, ...]
                taking_subset_of_file = True
            else:
                taking_subset_of_file = False

            new_buffer[
                n_tokens_filled : n_tokens_filled + activations.shape[0], ...
            ] = activations
            n_tokens_filled += activations.shape[0]

            if taking_subset_of_file:
                self.next_idx_within_buffer = activations.shape[0]
            else:
                # Only print from rank 0 if distributed
                if not self.cfg.distributed or self.cfg.rank == 0:
                    print(f"Increase cache idx")
                next_cache_idx += step
                self.next_idx_within_buffer = 0
                
        # If we didn't fill the buffer completely, resize it
        if n_tokens_filled < buffer_size:
            new_buffer = new_buffer[:n_tokens_filled, ...]
            
        # If we're distributed, make sure all ranks end up with same size buffer
        if self.cfg.distributed:
            # First get the minimum size across all ranks
            local_size = torch.tensor([n_tokens_filled], device=self.cfg.device)
            global_sizes = [torch.zeros_like(local_size) for _ in range(self.cfg.world_size)]
            dist.all_gather(global_sizes, local_size)
            min_size = min([size.item() for size in global_sizes])
            
            # Resize to minimum size
            if min_size < n_tokens_filled:
                new_buffer = new_buffer[:min_size, ...]
                
        dist.barrier()  # Wait for all ranks before proceeding

        return new_buffer

    def _generate_activations_buffer(
        self, n_batches_in_buffer, batch_size, context_size, num_layers, d_in
    ):
        """
        Generate a buffer of activations by repeatedly fetching batches from the model.
        In distributed mode, each rank generates its own portion of activations.
        """
        total_size = batch_size * n_batches_in_buffer
        new_buffer = torch.zeros(
            (total_size, context_size, num_layers, d_in),
            dtype=self.cfg.dtype,
            device=self.cfg.device,
        )

        # Only show progress bar on rank 0 if distributed
        use_tqdm = not self.cfg.distributed or self.cfg.rank == 0
        range_iter = tqdm(range(0, total_size, batch_size)) if use_tqdm else range(0, total_size, batch_size)
        
        for start_idx in range_iter:
            batch_tokens = next(self.image_dataloader_iter)
            batch_activations = self.get_activations(batch_tokens)

            if self.cfg.use_patches_only:
                # Remove the CLS token if we only need patches
                batch_activations = batch_activations[:, 1:, :, :]

            new_buffer[start_idx : start_idx + batch_size, ...] = batch_activations

        # Reshape to (buffer_size, num_layers, d_in)
        new_buffer = new_buffer.reshape(-1, num_layers, d_in)
        
        # Use same permutation across all ranks for consistency if distributed
        if self.cfg.distributed:
            if self.cfg.rank == 0:
                # Generate permutation on rank 0
                perm = torch.randperm(new_buffer.shape[0], device=self.cfg.device)
            else:
                # Create empty tensor on other ranks
                perm = torch.zeros(new_buffer.shape[0], dtype=torch.long, device=self.cfg.device)
                
            # Broadcast permutation from rank 0 to all ranks
            dist.broadcast(perm, 0)
            
            # Apply the same permutation on all ranks
            new_buffer = new_buffer[perm]
        else:
            # Just shuffle normally for non-distributed case
            new_buffer = new_buffer[torch.randperm(new_buffer.shape[0])]
            
        return new_buffer

    def get_data_loader(self) -> Iterator[Any]:
        """
        Create a new DataLoader from a mixed buffer of half "stored" and half "new" activations.
        Ensures consistent shuffling across ranks in distributed training.
        """
        batch_size = self.cfg.train_batch_size
        half_batches = self.cfg.n_batches_in_buffer // 2

        # Mix current storage buffer with new buffer
        mixing_buffer = torch.cat(
            [self.get_buffer(half_batches), self.storage_buffer], dim=0
        )
        
        # Consistent shuffling in distributed mode
        if self.cfg.distributed:
            if self.cfg.rank == 0:
                # Generate permutation on rank 0
                perm = torch.randperm(mixing_buffer.shape[0], device=self.cfg.device)
            else:
                # Create empty tensor on other ranks
                perm = torch.zeros(mixing_buffer.shape[0], dtype=torch.long, device=self.cfg.device)
                
            # Broadcast permutation from rank 0 to all ranks
            dist.broadcast(perm, 0)
            mixing_buffer = mixing_buffer[perm]
        else:
            mixing_buffer = mixing_buffer[torch.randperm(mixing_buffer.shape[0])]

        # Half of the mixed buffer is stored again
        self.storage_buffer = mixing_buffer[: mixing_buffer.shape[0] // 2]

        # The other half is used as the new training DataLoader
        data_for_loader = mixing_buffer[mixing_buffer.shape[0] // 2 :]

        dataloader = iter(
            DataLoader(
                cast(Any, data_for_loader),
                batch_size=batch_size,
                shuffle=not self.cfg.distributed,  # Don't shuffle if distributed since we already shuffled
            )
        )
        return dataloader

    def next_batch(self) -> torch.Tensor:
        """
        Get the next batch from the current DataLoader. If the DataLoader is exhausted,
        refill the buffer and create a new DataLoader, then fetch the next batch.
        """
        try:
            return next(self.dataloader)
        except StopIteration:
            self.dataloader = self.get_data_loader()
            return next(self.dataloader)

    def generate_cached_activations_from_dataset(
        self,
        tokens_per_file: int = 1_000_000,
        shuffle_data: bool = False,
    ):
        """
        Generate cached activation tensors from the dataset and save them to disk.
        In distributed mode, each rank processes a different subset of the data.
        """
        save_dir = self.cfg.cached_activations_path
        os.makedirs(save_dir, exist_ok=True)

        # Create dataset sampler for distributed generation
        sampler = None
        if self.cfg.distributed:
            sampler = DistributedSampler(
                self.dataset,
                num_replicas=self.cfg.world_size,
                rank=self.cfg.rank,
                shuffle=shuffle_data
            )
            shuffle_data = False  # Don't shuffle if using sampler
        
        loader = DataLoader(
            self.dataset,
            batch_size=self.cfg.store_batch_size,
            shuffle=shuffle_data and sampler is None,
            sampler=sampler,
            num_workers=self.cfg.num_workers,
            drop_last=False,
        )

        device = self.cfg.device
        context_size = self.cfg.context_size
        num_layers = (
            len(self.cfg.hook_point_layer)
            if isinstance(self.cfg.hook_point_layer, list)
            else 1
        )
        d_in = self.cfg.d_in

        buffer = []
        tokens_stored = 0
        
        # In distributed mode, each rank starts with a different file index
        if self.cfg.distributed:
            file_idx = self.cfg.rank
            file_step = self.cfg.world_size
        else:
            file_idx = 0
            file_step = 1

        # Only show progress bar on rank 0 if distributed
        use_tqdm = not self.cfg.distributed or self.cfg.rank == 0
        loader_iter = tqdm(loader) if use_tqdm else loader
        
        for batch in loader_iter:
            if isinstance(batch, tuple) or isinstance(batch, list):
                batch = batch[0]
                
            batch = batch.to(device)
            batch.requires_grad_(False)

            # Get activations using the base model if wrapped in DDP
            if hasattr(self.model, 'module'):
                base_model = self.model.module
            else:
                base_model = self.model
                
            # Get activations using run_with_cache from the base model
            with torch.cuda.amp.autocast(enabled=self.dtype == torch.float16):
                layers = (
                    self.cfg.hook_point_layer
                    if isinstance(self.cfg.hook_point_layer, list)
                    else [self.cfg.hook_point_layer]
                )
                act_names = [self.cfg.hook_point.format(layer=layer) for layer in layers]
                stop_layer = max(layers) + 1
                
                _, layerwise_activations = base_model.run_with_cache(
                    batch, names_filter=act_names, stop_at_layer=stop_layer
                )
            
            # Process activations from each layer
            activations_list = []
            for act_name in act_names:
                acts = layerwise_activations[act_name]
                
                # Select heads if specified
                if self.cfg.hook_point_head_index is not None:
                    acts = acts[:, :, self.cfg.hook_point_head_index]
                    
                # Select only CLS token if specified
                if self.cfg.cls_token_only:
                    acts = acts[:, 0:1]
                    
                activations_list.append(acts)
                
            batch_acts = torch.stack(activations_list, dim=2).half()

            if getattr(self.cfg, "use_patches_only", False):
                batch_acts = batch_acts[:, 1:, :, :]  # remove CLS token if applicable

            batch_size = batch_acts.shape[0]
            flat_acts = batch_acts.reshape(batch_size * context_size, num_layers, d_in)

            buffer.append(flat_acts)
            tokens_stored += flat_acts.shape[0]

            while tokens_stored >= tokens_per_file:
                combined = torch.cat(buffer, dim=0)
                to_save = combined[:tokens_per_file]

                save_path = os.path.join(save_dir, f"{file_idx}.pt")
                torch.save(to_save.cpu(), save_path)
                
                # Only print message from rank 0 if distributed
                if not self.cfg.distributed or self.cfg.rank == 0:
                    print(f"Saved {tokens_per_file} tokens to {save_path}")

                file_idx += file_step  # Skip by world_size in distributed mode
                combined = combined[tokens_per_file:]
                tokens_stored = combined.shape[0]
                buffer = [combined] if tokens_stored > 0 else []

        # Save leftovers
        if tokens_stored > 0:
            combined = torch.cat(buffer, dim=0)
            save_path = os.path.join(save_dir, f"{file_idx}.pt")
            torch.save(combined.cpu(), save_path)
            
            # Only print message from rank 0 if distributed
            if not self.cfg.distributed or self.cfg.rank == 0:
                print(f"Saved {tokens_stored} leftover tokens to {save_path}")
        
        # Synchronize all processes before continuing
        if self.cfg.distributed:
            dist.barrier()