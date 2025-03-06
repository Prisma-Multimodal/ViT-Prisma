import os
from typing import Any, Iterator, cast

import torch
from torch.utils.data import DataLoader
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
            raise ValueError("CacheVisionActivationStore cannot be initialized with cfg.use_cached_activations = False ")

    @lru_cache(maxsize=2)
    def load_file_cached(self, file):
        print(f"\n\nLoad File {file}\n\”")
        return torch.load(file)

    def _load_cached_activations(self, total_size, context_size, num_layers, d_in):
        """
        Load cached activations from disk until the buffer is filled or no more files are found.
        """
        buffer_size = total_size * context_size
        new_buffer = torch.zeros(
            (buffer_size, num_layers, d_in),
            dtype=self.cfg.dtype,
            device=self.cfg.device,
        )
        n_tokens_filled = 0
        next_cache_idx = 0

        # Load from cached files one by one
        while n_tokens_filled < buffer_size:
            cache_file = f"{self.cfg.cached_activations_path}/{next_cache_idx}.pt"
            if not os.path.exists(cache_file):
                # self._print_cache_warning(n_tokens_filled, buffer_size)
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
                next_cache_idx += 1
                self.next_idx_within_buffer = 0

        return new_buffer

    def get_data_loader(self) -> Iterator[Any]:
        """
        Create a new DataLoader from a mixed buffer of half "stored" and half "new" activations.
        This ensures variety and mixing each time the DataLoader is refreshed.

        Steps:
            1. Create a mixing buffer by combining newly generated buffer and existing storage buffer.
            2. Shuffle and split the mixed buffer: half goes back to storage, half is used to form a DataLoader.
            3. Return an iterator from the new DataLoader.
        """
        batch_size = self.cfg.train_batch_size
        half_batches = self.cfg.n_batches_in_buffer // 2

        # Mix current storage buffer with new buffer
        mixing_buffer = torch.cat(
            [self.get_buffer(half_batches), self.storage_buffer], dim=0
        )
        mixing_buffer = mixing_buffer[torch.randperm(mixing_buffer.shape[0])]

        # Half of the mixed buffer is stored again
        self.storage_buffer = mixing_buffer[: mixing_buffer.shape[0] // 2]

        # The other half is used as the new training DataLoader
        data_for_loader = mixing_buffer[mixing_buffer.shape[0] // 2 :]

        dataloader = iter(
            DataLoader(
                cast(Any, data_for_loader),
                batch_size=batch_size,
                shuffle=True,
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
        Creates and returns a buffer of activations by either:
            - Loading from cached activations if `use_cached_activations` is True
            - Or generating them from the model if not cached.

        Returns a tensor of shape (total_size * context_size, num_layers, d_in) if cached,
        or (total_size, context_size, num_layers, d_in) reshaped and shuffled otherwise.
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
    while training SAEs.
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

        # Main dataset loader
        self.image_dataloader = DataLoader(
            self.dataset,
            shuffle=True,
            num_workers=num_workers,
            batch_size=self.cfg.store_batch_size,
            collate_fn=collate_fn,
            drop_last=True,
        )

        # Evaluation dataset loader
        self.image_dataloader_eval = DataLoader(
            eval_dataset,
            shuffle=True,
            num_workers=num_workers,
            batch_size=self.cfg.store_batch_size,
            collate_fn=collate_fn_eval,
            drop_last=True,
        )

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

    def _batch_stream(
        self, dataloader: DataLoader, device: torch.device
    ) -> Iterator[torch.Tensor]:
        """
        Infinite iterator over batches of images from a given dataloader.
        Ensures that `.requires_grad_(False)` is set and that data is moved to the specified device.
        """
        while True:
            for batch in dataloader:
                batch.requires_grad_(False)
                yield batch.to(device)

    def _eval_batch_stream(
        self, dataloader: DataLoader, device: torch.device
    ) -> Iterator[torch.Tensor]:
        """
        Infinite iterator over (image_data, labels) from an evaluation dataloader.
        Ensures that `.requires_grad_(False)` is set on both data and labels, and moves them to the specified device.
        """
        while True:
            for image_data, labels in dataloader:
                image_data.requires_grad_(False)
                labels.requires_grad_(False)
                yield image_data.to(device), labels.to(device)

    @torch.no_grad
    def get_activations(self, batch_tokens: torch.Tensor) -> torch.Tensor:
        """
        Returns layerwise activations from the HookedViT model according to config.

        Shapes:
            - If cls_token and head_index: (batch, 1, num_layers, head_dim)
            - If cls_token only: (batch, 1, num_layers, d_model)
            - If head_index only: (batch, seq_len, num_layers, head_dim)
            - If neither: (batch, seq_len, num_layers, d_model)
        """
        layers = (
            self.cfg.hook_point_layer
            if isinstance(self.cfg.hook_point_layer, list)
            else [self.cfg.hook_point_layer]
        )
        act_names = [self.cfg.hook_point.format(layer=layer) for layer in layers]
        stop_layer = max(layers) + 1

        # Run model and get cached activations
        with torch.cuda.amp.autocast(enabled=self.dtype == torch.float16):
            _, layerwise_activations = self.model.run_with_cache(
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

            activations_list.append(acts)

        return torch.stack(activations_list, dim=2)

    def get_buffer(self, n_batches_in_buffer: int) -> torch.Tensor:
        """
        Creates and returns a buffer of activations by either:
            - Loading from cached activations if `use_cached_activations` is True
            - Or generating them from the model if not cached.

        Returns a tensor of shape (total_size * context_size, num_layers, d_in) if cached,
        or (total_size, context_size, num_layers, d_in) reshaped and shuffled otherwise.
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

    # @lru_cache(maxsize=2)
    def load_file_cached(self, file):
        print(f"\n\nLoad File {file}\n")
        data = torch.load(file, map_location=self.cfg.device, weights_only=True)
        print(data.shape)
        return data

    def _load_cached_activations(self, total_size, context_size, num_layers, d_in):
        """
        Load cached activations from disk until the buffer is filled or no more files are found.
        """
        buffer_size = total_size * context_size
        new_buffer = torch.zeros(
            (buffer_size, num_layers, d_in),
            dtype=self.cfg.dtype,
            device=self.cfg.device,
        )
        n_tokens_filled = 0
        next_cache_idx = 0

        # Load from cached files one by one
        while n_tokens_filled < buffer_size:
            cache_file = f"{self.cfg.cached_activations_path}/{next_cache_idx}.pt"
            if not os.path.exists(cache_file):
                # self._print_cache_warning(n_tokens_filled, buffer_size)
                new_buffer = new_buffer[:n_tokens_filled, ...]
                return new_buffer
            

            print(f"\n\nLoad next buffer from file {cache_file}\n\”")
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

                print(f"Increase cache idx")
                next_cache_idx += 1
                self.next_idx_within_buffer = 0

        return new_buffer

    def _generate_activations_buffer(
        self, n_batches_in_buffer, batch_size, context_size, num_layers, d_in
    ):
        """
        Generate a buffer of activations by repeatedly fetching batches from the model.
        """
        total_size = batch_size * n_batches_in_buffer
        new_buffer = torch.zeros(
            (total_size, context_size, num_layers, d_in),
            dtype=self.cfg.dtype,
            device=self.cfg.device,
        )

        for start_idx in range(0, total_size, batch_size):
            batch_tokens = next(self.image_dataloader_iter)
            batch_activations = self.get_activations(batch_tokens)


            if self.cfg.use_patches_only:
                # Remove the CLS token if we only need patches
                batch_activations = batch_activations[:, 1:, :, :]

            new_buffer[start_idx : start_idx + batch_size, ...] = batch_activations

        # Reshape to (buffer_size, num_layers, d_in) and shuffle
        new_buffer = new_buffer.reshape(-1, num_layers, d_in)
        new_buffer = new_buffer[torch.randperm(new_buffer.shape[0])]
        return new_buffer

    def get_data_loader(self) -> Iterator[Any]:
        """
        Create a new DataLoader from a mixed buffer of half "stored" and half "new" activations.
        This ensures variety and mixing each time the DataLoader is refreshed.

        Steps:
            1. Create a mixing buffer by combining newly generated buffer and existing storage buffer.
            2. Shuffle and split the mixed buffer: half goes back to storage, half is used to form a DataLoader.
            3. Return an iterator from the new DataLoader.
        """
        batch_size = self.cfg.train_batch_size
        half_batches = self.cfg.n_batches_in_buffer // 2

        # Mix current storage buffer with new buffer
        mixing_buffer = torch.cat(
            [self.get_buffer(half_batches), self.storage_buffer], dim=0
        )
        mixing_buffer = mixing_buffer[torch.randperm(mixing_buffer.shape[0])]

        # Half of the mixed buffer is stored again
        self.storage_buffer = mixing_buffer[: mixing_buffer.shape[0] // 2]

        # The other half is used as the new training DataLoader
        data_for_loader = mixing_buffer[mixing_buffer.shape[0] // 2 :]

        dataloader = iter(
            DataLoader(
                cast(Any, data_for_loader),
                batch_size=batch_size,
                shuffle=True,
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
        Generate cached activation tensors from the already loaded dataset (self.dataset)
        and store them in .pt files that can be later loaded by `_load_cached_activations`.
        """
        save_dir = self.cfg.cached_activations_path

        os.makedirs(save_dir, exist_ok=True)

        self.model = self.model

        loader = DataLoader(
            self.dataset,
            batch_size=self.cfg.store_batch_size,
            shuffle=shuffle_data,
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
        file_idx = 0

        for batch in tqdm(loader):
            batch = batch[0].to(device)
            batch.requires_grad_(False)

            # Get activations
            batch_acts = self.get_activations(batch).half()

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
                # print(f"Saved {tokens_per_file} tokens to {save_path}")

                file_idx += 1
                combined = combined[tokens_per_file:]
                tokens_stored = combined.shape[0]
                buffer = [combined] if tokens_stored > 0 else []

        # Save leftovers
        if tokens_stored > 0:
            combined = torch.cat(buffer, dim=0)
            save_path = os.path.join(save_dir, f"{file_idx}.pt")
            torch.save(combined.cpu(), save_path)
            print(f"Saved {tokens_stored} leftover tokens to {save_path}")
