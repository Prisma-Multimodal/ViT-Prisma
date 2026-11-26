"""Load instance segmentation annotations as labels for ViT image patches."""

import json
from dataclasses import dataclass
from operator import index
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor


_DATASET_REPO = "Prisma-Multimodal/segmented-imagenet1k-subset"


@dataclass(frozen=True)
class PatchLabels:
    """Which instances intersect each patch in a (rows, columns) grid."""

    grid_size: tuple[int, int]
    _labels: tuple[str, ...]
    _hits: np.ndarray

    def labels_for_patch(self, patch_number: int) -> list[str]:
        """Return unique labels in annotation order for a zero-based, row-major patch."""
        patch_number = index(patch_number)
        if not 0 <= patch_number < self._hits.shape[1]:
            raise IndexError(f"Patch number {patch_number} is outside grid {self.grid_size}")
        return list(dict.fromkeys(
            label for label, hit in zip(self._labels, self._hits[:, patch_number]) if hit
        ))


class PatchLabelDataset(Dataset):
    """Yield an RGB image tensor and PatchLabels for each segmentation record.

    Records have ``image``, ``masks`` and ``labels`` fields. The image and masks
    can be PIL images or paths. ``image_size`` is (height, width), and both
    dimensions must be divisible by ``patch_size``. Images are resized to this
    grid and scaled to [0, 1]. A patch gets a label if any pixel of the resized
    instance mask is foreground. Patch indices do not include a CLS token.
    Use ``collate_patch_labels`` to batch samples with different instance counts.
    """

    def __init__(
        self, records, *, patch_size=16, image_size=(224, 224),
        image_root=None, mask_root=None,
    ):
        patch_size = index(patch_size)
        height, width = map(index, image_size)
        if min(patch_size, height, width) <= 0 or height % patch_size or width % patch_size:
            raise ValueError("Image dimensions must be positive multiples of patch_size")

        self.records = records
        self.patch_size = patch_size
        self.grid_size = (height // patch_size, width // patch_size)
        self._size = (width, height)  # PIL takes (width, height).
        self._image_root = Path(image_root).resolve() if image_root is not None else None
        self._mask_root = Path(mask_root).resolve() if mask_root is not None else None

    @classmethod
    def from_json(cls, manifest, *, image_root=None, mask_root=None, **kwargs):
        """Read a split JSON file. Paths default to being relative to its directory."""
        manifest = Path(manifest)
        with manifest.open(encoding="utf-8") as file:
            records = json.load(file)
        return cls(
            records,
            image_root=manifest.parent if image_root is None else image_root,
            mask_root=manifest.parent if mask_root is None else mask_root,
            **kwargs,
        )

    @classmethod
    def from_huggingface(cls, split="test", **kwargs):
        """Load the public split from cached archives, downloading ~8.5 GB if needed."""
        if split == "validation":
            split = "val"
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be 'train', 'validation' (or 'val'), or 'test'")

        from datasets import DownloadManager
        from huggingface_hub import hf_hub_download

        manifest, images, masks = (
            hf_hub_download(_DATASET_REPO, filename, repo_type="dataset")
            for filename in (f"{split}.json", "images.tar.gz", "masks.tar.gz")
        )
        image_root, mask_root = DownloadManager().extract([images, masks])
        return cls.from_json(manifest, image_root=image_root, mask_root=mask_root, **kwargs)

    def __len__(self):
        return len(self.records)

    def _resize(self, source, root, mode, resample):
        if isinstance(source, Image.Image):
            return source.convert(mode).resize(self._size, resample)
        if root is not None:
            source = (root / source).resolve()
            if not source.is_relative_to(root):
                raise ValueError(f"Image or mask path is outside its dataset directory: {source}")
        with Image.open(source) as image:
            return image.convert(mode).resize(self._size, resample)

    def __getitem__(self, idx):
        record = self.records[idx]
        labels, masks = tuple(record["labels"]), record["masks"]
        if len(labels) != len(masks):
            raise ValueError(f"Record {idx} must have one label per mask")

        image = self._resize(record["image"], self._image_root, "RGB", Image.Resampling.BILINEAR)
        rows, cols = self.grid_size
        patch_size = self.patch_size
        hits = np.empty((len(labels), rows * cols), dtype=bool)
        for i, source in enumerate(masks):
            mask = self._resize(source, self._mask_root, "L", Image.Resampling.NEAREST)
            pixels = np.asarray(mask, dtype=bool)
            hits[i] = pixels.reshape(rows, patch_size, cols, patch_size).any(axis=(1, 3)).ravel()
        return to_tensor(image), PatchLabels(self.grid_size, labels, hits)


def collate_patch_labels(batch):
    """Stack images and retain one independently queryable PatchLabels per image."""
    images, labels = zip(*batch)
    return torch.stack(images), list(labels)
