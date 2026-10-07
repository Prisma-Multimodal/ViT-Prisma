"""Reproducible splits and provenance for generated datasets."""

import json
import random
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split


def validate_seed(seed: int) -> None:
    """Require a seed supported by both Python and scikit-learn."""
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")


def validate_fraction(value: float, name: str) -> None:
    """Require a split fraction that leaves both partitions nonempty."""
    if not 0 < value < 1:
        raise ValueError(f"{name} must be between 0 and 1 (exclusive)")


def check_cache_manifest(directory, config: dict, filenames: tuple) -> bool:
    """Return whether all splits exist, rejecting unverified or incompatible caches.

    Existing files are never overwritten by a dataset constructor when their
    provenance is missing or their configuration differs from the request.
    """
    directory = Path(directory)
    manifest_path = directory / "split_manifest.json"
    if manifest_path.exists():
        with manifest_path.open() as handle:
            manifest = json.load(handle)
        if manifest.get("config") != config:
            raise ValueError(
                f"Cache configuration mismatch in {directory}. "
                "Use a different cache directory for different generation settings."
            )
    elif any((directory / name).exists() for name in filenames):
        raise ValueError(
            f"Cache in {directory} has no split_manifest.json; its seed and "
            "split membership cannot be verified. Use a new cache directory."
        )
    return manifest_path.exists() and all(
        (directory / name).exists() for name in filenames
    )


def write_split_manifest(directory, config: dict, splits: dict, **details) -> None:
    """Save generation settings and ordered split membership as JSON."""
    with (Path(directory) / "split_manifest.json").open("w") as handle:
        json.dump({"config": config, "splits": splits, **details}, handle, indent=2)
        handle.write("\n")


def save_balanced_splits(directory, categories: dict, config: dict) -> None:
    """Balance categories and save seeded splits, retaining raw sample indices.

    ``categories`` maps the legacy category filenames to raw dataset indices.
    Manifest indices refer to rows in ``induction_dataset.npz`` and preserve
    the order of examples in each saved split.
    """
    directory = Path(directory)
    rng = random.Random(config["seed"])
    with np.load(directory / "induction_dataset.npz", allow_pickle=True) as data:
        images, metadata, labels = data["images"], data["metadata"], data["labels"]
    sample_size = min(len(indices) for indices in categories.values())
    if sample_size < 2:
        raise ValueError("Each category needs at least two samples to split")

    train_indices, test_indices = [], []
    for category, candidates in categories.items():
        indices = rng.sample(candidates, sample_size)
        train, test = train_test_split(
            indices, test_size=config["test_size"], random_state=config["seed"]
        )
        np.savez(
            directory / f"{category}.npz",
            images=images[indices], metadata=metadata[indices], labels=labels[indices],
        )
        train_indices.extend(train)
        test_indices.extend(test)

    rng.shuffle(train_indices)
    rng.shuffle(test_indices)
    splits = {"train": train_indices, "test": test_indices}
    for split, indices in splits.items():
        np.savez(
            directory / f"all_{split}.npz",
            images=images[indices], metadata=metadata[indices], labels=labels[indices],
        )
    write_split_manifest(
        directory, config, splits, source="induction_dataset.npz",
        source_size=len(labels), samples_per_category=sample_size,
    )
