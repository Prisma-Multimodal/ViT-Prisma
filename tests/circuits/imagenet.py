import random

import numpy as np
from PIL import Image
from torchvision import datasets
from transformers import AutoImageProcessor
from torch.utils.data import Dataset, DataLoader

from vit_prisma.circuit.circuit_analyzer import CircuitAnalyzer
from vit_prisma.models.base_vit import HookedViT
import torch

class SubsetFolder:
    def __init__(self, dataset, class_id, num_example=None):
        if isinstance(class_id, int):
            if num_example:
                samples = np.array(dataset.samples)[
                    np.random.choice(np.where(np.array(dataset.targets) == class_id)[0], size=num_example, replace=False)
                ]
            else:
                samples = np.array(dataset.samples, )[np.array(dataset.targets) == class_id]
        elif isinstance(class_id, list):
            class_id = np.array(class_id)
            mask = np.isin(np.array(dataset.targets), class_id)
            if num_example:
                masked_samples = np.array(dataset.samples)[mask]
                rand_indices = np.random.choice(len(masked_samples), size=num_example, replace=False)
                samples = masked_samples[rand_indices]
            else:
                samples = np.array(dataset.samples, )[mask]
        self.samples = [(row[0], int(row[1])) for row in samples.tolist()]

    def __len__(self):
        return len(self.samples)

class ImageNetDataset(Dataset):
    def __init__(self, root_dir, processor=None, transform=None, select_class=None, ctft_class_ranges=None, num_examples=None):
        dataset = datasets.ImageFolder(root=root_dir)
        self.processor = processor
        self.transform = transform
        self.ctft_class_ranges = ctft_class_ranges
        if select_class is not None:
            subset_dataset = SubsetFolder(dataset, select_class, num_examples)
            self.dataset = subset_dataset
            ctft_subset_dataset = SubsetFolder(dataset, ctft_class_ranges, num_examples)
            self.ctft_dataset = ctft_subset_dataset
            self.ctft_class_index = self._build_class_index(self.ctft_dataset) if ctft_subset_dataset else None
        else:
            self.dataset = dataset

    def _build_class_index(self, dataset):
        class_index = {}
        if hasattr(dataset, 'target_name'):
            for idx, label in enumerate(getattr(dataset, dataset.target_name)):  # or dataset.labels
                label = int(label)
                if label not in class_index:
                    class_index[label] = []
                class_index[label].append(idx)
        else:
            for idx, data in enumerate(dataset.samples):
                if len(data) == 2:
                    image, label = data
                    _ = None
                elif len(data) == 3:
                    image, label, _ = data
                else:
                    raise ValueError("Unexpected number of items returned by dataset")
                label = int(label)  # Ensure consistent key type
                if label not in class_index:
                    class_index[label] = []
                class_index[label].append(idx)
        return class_index

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image_path, label = self.dataset.samples[idx]
        image = Image.open(image_path).convert("RGB")  # Ensure RGB mode
        if self.processor is not None:
            inputs = self.processor(images=image, return_tensors="pt")  # Use processor
            inputs = inputs["pixel_values"].squeeze(0)
        if self.transform is not None:
            inputs = self.transform(image)
        correct_idx = label
        incorrect_idx = random.choice(self.ctft_class_ranges)
        ctft_image_path = self.ctft_dataset.samples[random.choice(self.ctft_class_index[incorrect_idx])][0]
        ctft_image = Image.open(ctft_image_path).convert("RGB")
        if self.processor is not None:
            ctft_image = self.processor(images=ctft_image, return_tensors="pt")["pixel_values"].squeeze(
                0)  # Use processor
        if self.transform is not None:
            ctft_image = self.transform(ctft_image)
        return inputs, ctft_image, [correct_idx, incorrect_idx]

model = HookedViT.from_pretrained(
                    'vit_base_patch16_224',
                    center_writing_weights=False,
                    fold_ln=False,
                    refactor_factored_attn_matrices=False,
                    allow_failing=True,
                )
model.cfg.use_split_qkv_input = True
model.cfg.use_attn_result = True
model.cfg.use_hook_mlp_in = True
model.cfg.ungroup_grouped_query_attention = True
model.to('cuda')

class_ranges = [208, 212, 263, 189, 245]
ctft_class_ranges = [407, 656, 436, 468, 511]
train_dataset = ImageNetDataset(root_dir=f'/home/yxpengcs/Datasets/imagenet/train',
                                      processor=AutoImageProcessor.from_pretrained("google/vit-base-patch16-224"),
                                      select_class=class_ranges, ctft_class_ranges=ctft_class_ranges, num_examples=2000)

val_dataset = ImageNetDataset(root_dir=f'/home/yxpengcs/Datasets/imagenet/val',
                                      processor=AutoImageProcessor.from_pretrained("google/vit-base-patch16-224"),
                                      select_class=class_ranges, ctft_class_ranges=ctft_class_ranges)

train_dataloader = DataLoader(train_dataset, batch_size=64, num_workers=4)
val_dataloader = DataLoader(val_dataset, batch_size=64, num_workers=4)

analyzer = CircuitAnalyzer(
    model=model,
    task="waterbirds",
    method="EAP",
    metric_name="logit_diff",
    level="edge",
    ablation="patching"
)

graph, perexample_scores = analyzer.run_analysis(train_dataloader)
eval_results = analyzer.run_evaluation(val_dataloader)

# Unpack results
weighted_edge_counts, percentages, auc, auc_from_1, avg_faithfulness, faithfulnesses = eval_results