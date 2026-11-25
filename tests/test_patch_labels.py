import json
import tarfile

import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from vit_prisma.dataloaders.patch_labels import PatchLabelDataset, collate_patch_labels


@pytest.mark.parametrize("patch_size", [8, 16, 32])
def test_patch_labels_are_row_major_and_deduplicated(patch_size):
    size = (96, 64)
    first = Image.new("L", size)
    second = Image.new("L", size)
    first.putpixel((0, 0), 255)
    for mask in (first, second):
        mask.putpixel((2 * patch_size, patch_size), 255)
    dataset = PatchLabelDataset(
        [{"image": Image.new("L", size), "masks": [first, second, second],
          "labels": ["bird", "branch", "bird"]}],
        patch_size=patch_size, image_size=(64, 96),
    )

    image, labels = dataset[0]
    assert image.shape == (3, 64, 96)
    assert labels.grid_size == (64 // patch_size, 96 // patch_size)
    assert labels.labels_for_patch(0) == ["bird"]
    assert labels.labels_for_patch(labels.grid_size[1] + 2) == ["bird", "branch"]
    assert labels.labels_for_patch(1) == []
    with pytest.raises(IndexError):
        labels.labels_for_patch(-1)
    with pytest.raises(IndexError):
        labels.labels_for_patch(labels.grid_size[0] * labels.grid_size[1])


def test_json_dataset_batches_images_with_different_instance_counts(tmp_path):
    image_dir = tmp_path / "images" / "test_images"
    mask_dir = tmp_path / "masks" / "test_masks"
    image_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)
    image = Image.new("RGB", (4, 2), "white")
    image.save(image_dir / "one.png")
    image.save(image_dir / "two.png")
    mask = Image.new("L", (4, 2))
    mask.putpixel((2, 1), 255)
    mask.save(mask_dir / "one.png")
    manifest = tmp_path / "test.json"
    manifest.write_text(json.dumps([
        {"image": "images/test_images/one.png", "masks": ["masks/test_masks/one.png"],
         "labels": ["bird"]},
        {"image": "images/test_images/two.png", "masks": [], "labels": []},
    ]), encoding="utf-8")

    dataset = PatchLabelDataset.from_json(manifest, patch_size=8, image_size=(16, 32))
    images, patches = next(iter(DataLoader(dataset, batch_size=2, collate_fn=collate_patch_labels)))
    assert torch.equal(images, torch.ones((2, 3, 16, 32)))
    assert len(patches) == 2
    assert [patches[0].labels_for_patch(i) for i in range(8)] == [
        [], [], [], [], [], [], ["bird"], [],
    ]
    assert patches[1].labels_for_patch(6) == []


def test_huggingface_manifest_and_archive_roots(monkeypatch, tmp_path):
    import huggingface_hub
    from datasets import config

    for directory, image in (("images", Image.new("RGB", (8, 8))),
                             ("masks", Image.new("L", (8, 8), 255))):
        root = tmp_path / directory
        root.mkdir()
        image.save(root / "file.png")
        with tarfile.open(tmp_path / f"{directory}.tar.gz", "w:gz") as archive:
            archive.add(root, arcname=directory)

    manifest = tmp_path / "val.json"
    manifest.write_text(json.dumps([
        {"image": "images/file.png", "masks": ["masks/file.png"], "labels": ["flower"]},
    ]), encoding="utf-8")

    def download(repo_id, filename, *, repo_type):
        assert repo_id == "Prisma-Multimodal/segmented-imagenet1k-subset"
        assert repo_type == "dataset"
        return str(tmp_path / filename)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    monkeypatch.setattr(config, "DOWNLOADED_DATASETS_PATH", tmp_path / "cache")
    monkeypatch.setattr(config, "EXTRACTED_DATASETS_PATH", tmp_path / "cache" / "extracted")
    dataset = PatchLabelDataset.from_huggingface("validation", patch_size=8, image_size=(8, 8))
    assert dataset[0][1].labels_for_patch(0) == ["flower"]


@pytest.mark.parametrize("size,patch_size", [((15, 16), 8), ((16, 15), 8), ((16, 16), 0)])
def test_invalid_patch_grid(size, patch_size):
    with pytest.raises(ValueError, match="multiples of patch_size"):
        PatchLabelDataset([], patch_size=patch_size, image_size=size)


def test_unpaired_masks_and_labels_are_rejected():
    dataset = PatchLabelDataset([{"image": None, "masks": [], "labels": ["bird"]}])
    with pytest.raises(ValueError, match="one label per mask"):
        dataset[0]


def test_json_paths_stay_within_the_dataset_directory(tmp_path):
    manifest = tmp_path / "test.json"
    manifest.write_text(json.dumps([
        {"image": "../elsewhere.png", "masks": [], "labels": []},
    ]), encoding="utf-8")
    dataset = PatchLabelDataset.from_json(manifest)
    with pytest.raises(ValueError, match="outside its dataset directory"):
        dataset[0]
