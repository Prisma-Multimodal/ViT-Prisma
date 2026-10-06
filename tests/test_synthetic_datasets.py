"""Offline CPU regression tests for synthetic data and split provenance."""

import json
import random

import numpy as np
import pytest

from vit_prisma.dataloaders import induction, polygenic_induction
from vit_prisma.dataloaders.circle import (
    CircleDataset, draw_circle_with_points, get_circle_metadata, get_train_test_data,
)


def read_manifest(directory):
    return json.loads((directory / "split_manifest.json").read_text())


@pytest.fixture(params=[induction, polygenic_induction])
def raw_dataset(request, tmp_path):
    """Small raw data with unequal class sizes exercises both balancing paths."""
    module = request.param
    n_classes = 4 if module is induction else 12
    labels = np.concatenate([np.full(20 + label, label) for label in range(n_classes)])
    metadata = np.array([
        {"Same": int(label) in (0, 2), "Vertical": int(label) in (0, 1)}
        for label in labels
    ])
    images = np.arange(len(labels), dtype=np.float32).reshape(-1, 1, 1)
    np.savez(tmp_path / "induction_dataset.npz", images=images, metadata=metadata, labels=labels)
    return module, tmp_path, images, labels


def test_balanced_splits_are_seeded_and_match_manifest(raw_dataset):
    module, directory, images, labels = raw_dataset
    random.seed(999)
    state = random.getstate()
    module.create_balanced_dataset(directory, seed=7, test_size=0.25)
    first = read_manifest(directory)
    assert random.getstate() == state
    train = first["splits"]["train"]
    test = first["splits"]["test"]
    assert not set(train) & set(test)
    assert len(set(train + test)) == len(train + test)
    for split, expected_count in [("train", 15), ("test", 5)]:
        indices = first["splits"][split]
        with np.load(directory / f"all_{split}.npz") as data:
            np.testing.assert_array_equal(data["images"], images[indices])
            np.testing.assert_array_equal(data["labels"], labels[indices])
            assert set(np.unique(data["labels"], return_counts=True)[1]) == {expected_count}

    random.seed(123)
    module.create_balanced_dataset(directory, seed=7, test_size=0.25)
    assert read_manifest(directory) == first
    module.create_balanced_dataset(directory, seed=8, test_size=0.25)
    assert read_manifest(directory)["splits"] != first["splits"]


def test_induction_cache_reloads_and_checks_settings(raw_dataset, monkeypatch):
    module, directory, _, _ = raw_dataset
    module.create_balanced_dataset(directory, seed=7, test_size=0.25)
    dataset_type = module.InductionDataset if module is induction else module.PolygenicInductionDataset

    def unexpected_generation(*args, **kwargs):
        pytest.fail("A compatible cache should not be regenerated")

    monkeypatch.setattr(module, "generate_dataset", unexpected_generation)
    dataset = dataset_type("train", directory, seed=7, test_size=0.25, use_metadata=True)
    image, _ = dataset[0]
    assert image.device.type == "cpu"
    assert image.shape == (1, 1, 1)
    assert len(dataset.metadata) == len(dataset)
    for kwargs in ({"seed": 8, "test_size": 0.25}, {"seed": 7, "test_size": 0.1}):
        with pytest.raises(ValueError, match="configuration mismatch"):
            dataset_type("train", directory, **kwargs)
    (directory / "split_manifest.json").unlink()
    with pytest.raises(ValueError, match="no split_manifest"):
        dataset_type("train", directory)


def test_induction_generates_from_empty_directory(tmp_path):
    directory = tmp_path / "nested" / "induction"
    train = induction.InductionDataset("train", directory, seed=5)
    test = induction.InductionDataset("test", directory, seed=5)
    assert train[0][0].shape == (1, 32, 32)
    assert test[0][0].device.type == "cpu"
    manifest = read_manifest(directory)
    assert len(manifest["splits"]["train"]) == len(train)
    assert len(manifest["splits"]["test"]) == len(test)


def test_circle_generation_reload_and_reproducibility(tmp_path, monkeypatch):
    directory = tmp_path / "nested" / "circle"
    state = random.getstate()
    train = CircleDataset("train", directory, seed=7)
    test = CircleDataset("test", directory, seed=7)
    assert random.getstate() == state
    assert train.imgs.shape == (885, 1, 32, 32)
    assert train.imgs.dtype == np.float32
    assert np.isfinite(train.imgs).all()
    assert train.imgs.min() >= -1 and train.imgs.max() <= 1
    pairs_train = set(map(tuple, train.data_points))
    pairs_test = set(map(tuple, test.data_points))
    assert not pairs_train & pairs_test
    assert len(pairs_train | pairs_test) == 60 * 59 // 2
    np.testing.assert_array_equal(train.labels, train.data_points.sum(axis=1) % 60)
    manifest = read_manifest(directory)
    assert manifest["splits"]["train"] == train.data_points.tolist()
    assert manifest["splits"]["test"] == test.data_points.tolist()

    again = CircleDataset("train", tmp_path / "other", seed=7)
    np.testing.assert_array_equal(train.imgs, again.imgs)
    np.testing.assert_array_equal(train.data_points, again.data_points)
    assert read_manifest(tmp_path / "other") == manifest

    def unexpected_generation(self):
        pytest.fail("A compatible cache should not be regenerated")

    monkeypatch.setattr(CircleDataset, "_generate_and_cache", unexpected_generation)
    reloaded = CircleDataset("train", directory, seed=7, transform=lambda image: image + 1)
    np.testing.assert_array_equal(reloaded[0][0], train[0][0] + 1)
    for kwargs in ({"seed": 8}, {"seed": 7, "split_ratio": 0.8},
                   {"seed": 7, "model_type": "pretrained_transformer"}):
        with pytest.raises(ValueError, match="configuration mismatch"):
            CircleDataset("train", directory, **kwargs)
    (directory / "split_manifest.json").unlink()
    with pytest.raises(ValueError, match="no split_manifest"):
        CircleDataset("train", directory)


def test_circle_split_seed_and_pretrained_render():
    metadata = get_circle_metadata()
    a = get_train_test_data(metadata, seed=7)
    b = get_train_test_data(metadata, seed=8)
    assert a != b
    image = draw_circle_with_points(0, 1, metadata, model_type="pretrained_transformer")
    assert image.shape == (3, 224, 224)
    assert image.device.type == "cpu"
    assert (image[:, 0, 0] == 1).all()


@pytest.mark.parametrize("dataset_type", [
    induction.InductionDataset, polygenic_induction.PolygenicInductionDataset, CircleDataset,
])
@pytest.mark.parametrize("kwargs", [{"train_or_test": "validation"}, {"seed": -1}, {"seed": None}])
def test_invalid_dataset_arguments(dataset_type, kwargs):
    with pytest.raises(ValueError):
        dataset_type(**({"train_or_test": "train"} | kwargs))


@pytest.mark.parametrize("fraction", [0, 1, -0.1, float("nan")])
def test_invalid_split_fractions(tmp_path, fraction):
    with pytest.raises(ValueError):
        CircleDataset("train", tmp_path, split_ratio=fraction)
    with pytest.raises(ValueError):
        induction.InductionDataset("train", tmp_path, test_size=fraction)
    with pytest.raises(ValueError):
        polygenic_induction.PolygenicInductionDataset("train", tmp_path, test_size=fraction)


def test_trainer_split_is_seeded_lazy_and_saved(tmp_path, monkeypatch):
    import torch
    from types import SimpleNamespace
    from vit_prisma.configs.HookedViTConfig import HookedViTConfig
    from vit_prisma.training import trainer

    class UnreadDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 30

        def __getitem__(self, index):
            pytest.fail("Creating a split should not load samples or apply transforms")

    def split_for(seed, name, use_wandb=False):
        directory = tmp_path / name
        cfg = HookedViTConfig(
            seed=seed, use_wandb=use_wandb, parent_dir=str(directory), device="cpu",
            num_epochs=0, scheduler_type="WarmupThenStepLR", save_checkpoints=False,
        )
        trainer.train(lambda cfg: torch.nn.Linear(1, 2), cfg, UnreadDataset())
        return json.loads((directory / "train_val_split.json").read_text())

    first = split_for(7, "first")
    assert first == split_for(7, "second")
    assert first["val_indices"] != split_for(8, "third")["val_indices"]
    assert not set(first["train_indices"]) & set(first["val_indices"])
    assert sorted(first["train_indices"] + first["val_indices"]) == list(range(30))
    assert split_for(None, "default") == split_for(666, "explicit_default")

    # A sweep's effective seed must control the split, without a tracking account.
    monkeypatch.setattr(trainer, "wandb", SimpleNamespace(
        init=lambda **kwargs: None, finish=lambda: None,
        config=SimpleNamespace(_items={"seed": 7}, update=lambda values: None),
    ))
    assert split_for(99, "sweep", use_wandb=True) == first
