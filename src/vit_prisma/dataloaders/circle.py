from itertools import combinations
import math
from pathlib import Path
import random

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from vit_prisma.dataloaders.synthetic_cache import (
    check_cache_manifest, validate_fraction, validate_seed, write_split_manifest,
)

def check_if_valid_angle(angle, angle_range):
    if angle < 0 or angle > 360:
        raise ValueError('Angle must be between 0 and 360')
    
    # Check if in range
    if not np.isin(angle, angle_range):
        raise ValueError('Angle must at correct interval of angle_range')
    
def get_train_test_data(circle_metadata, split_ratio=0.5, seed=42):
    """Return disjoint, seeded angle-pair lists; ``split_ratio`` is the train fraction."""
    validate_seed(seed)
    validate_fraction(split_ratio, 'split_ratio')
    data = list(combinations(range(circle_metadata['mod_arith']), 2))
    random.Random(seed).shuffle(data)
    split_idx = int(len(data) * split_ratio)
    if not 0 < split_idx < len(data):
        raise ValueError("split_ratio must leave at least one pair in each split")
    return data[:split_idx], data[split_idx:]


def get_circle_metadata():
    circle_metadata = {
    "mod_arith": 60,
    "image_size": 32,
    "center": (16, 16),
    "radius": 15.5,
    "multiplier": 6,
    "angle_range": np.arange(0, 60, 1)
    } 
    return circle_metadata

def draw_circle_with_points(angle1=None, angle2=None, metadata=None, model_type=None):

    # Create a new image with white background
    image_size = metadata['image_size']
    img = Image.new('L', (image_size, image_size), color=255) # white image

    pixels = img.load()

    center = metadata['center']
    radius = metadata['radius']

    # Transform angle to get point in circle
    MULTIPLIER = metadata['multiplier']
    angle_range = metadata['angle_range']
    transformed_angle_range = angle_range * MULTIPLIER

    # Draw circle in black.
    for i in transformed_angle_range:
        x = center[0] + radius * math.cos(math.radians(i))
        y = center[1] + radius * math.sin(math.radians(i))
        pixels[int(x), int(y)] = 0

    # Specify the angle
    def _draw_point_(angle, color=128):
        angle_rad = math.radians(angle) # x3 b/c representing the circle in intervals of 3
        x = center[0] + radius * math.cos(angle_rad)
        y = center[1] + radius * math.sin(angle_rad)
        pixels[int(x), int(y)] = color

    if angle1 is not None:
        check_if_valid_angle(angle1, angle_range)
        _draw_point_(angle1*MULTIPLIER)
    if angle2 is not None:
        check_if_valid_angle(angle2, angle_range)
        _draw_point_(angle2*MULTIPLIER)


    if model_type == 'pretrained_transformer':

        # Define padding dimensions
        left_padding = (224 - 32) // 2
        top_padding = (224 - 32) // 2
        right_padding = 224 - 32 - left_padding
        bottom_padding = 224 - 32 - top_padding

        transform = transforms.Compose([
        transforms.Pad((left_padding, top_padding, right_padding, bottom_padding), fill=255),  # White padding on the uint8 PIL image
        transforms.Grayscale(num_output_channels=3),  
        transforms.ToTensor()           # Convert the PIL Image to a tensor
    ])
    else:
        transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
    ])

    img = transform(img)
    
    return img 

class CircleDataset(Dataset):
    """Circle addition data with reproducible train/test partitions and disk caching.

    ``cache_path`` is a directory. Changing ``seed``, ``split_ratio`` (the train
    fraction), or ``model_type`` requires a different directory. ``transform``
    is applied on access and does not alter cached data.
    """

    def __init__(self, train_or_test='test', cache_path='../data/circle',
                 transform=None, *, seed=42, split_ratio=0.5, model_type=None):
        if train_or_test not in ('train', 'test'):
            raise ValueError("train_or_test must be 'train' or 'test'")
        validate_seed(seed)
        validate_fraction(split_ratio, 'split_ratio')
        if model_type not in (None, 'pretrained_transformer'):
            raise ValueError("model_type must be None or 'pretrained_transformer'")
        self.cache_path = Path(cache_path)
        self.train_or_test = train_or_test
        self.transform = transform
        self.seed = seed
        self.split_ratio = split_ratio
        self.model_type = model_type
        self.circle_metadata = get_circle_metadata()
        self.mod_arith = self.circle_metadata['mod_arith']
        self.cache_config = {
            'dataset': 'circle', 'version': 1, 'seed': seed,
            'split_ratio': split_ratio, 'model_type': model_type,
        }

        if not check_cache_manifest(
            self.cache_path, self.cache_config, ('train.npz', 'test.npz')
        ):
            self._generate_and_cache()
        self._load_from_cache()

    def _load_from_cache(self):
        with np.load(self.cache_path / f'{self.train_or_test}.npz') as loaded:
            self.imgs = loaded['imgs']
            self.labels = loaded['labels']
            self.data_points = loaded['data_points']

    def __len__(self):
        return len(self.imgs)

    def _generate_and_cache(self):
        self.cache_path.mkdir(parents=True, exist_ok=True)
        train, test = get_train_test_data(
            self.circle_metadata, self.split_ratio, self.seed
        )
        splits = {'train': train, 'test': test}
        for split, pairs in splits.items():
            # Preallocate to avoid holding a list of tensors plus a stacked copy.
            shape = (3, 224, 224) if self.model_type == 'pretrained_transformer' else (1, 32, 32)
            imgs = np.empty((len(pairs), *shape), dtype=np.float32)
            for index, (a, b) in enumerate(pairs):
                imgs[index] = draw_circle_with_points(
                    a, b, self.circle_metadata, model_type=self.model_type
                ).numpy()
            labels = np.array([sum(pair) % self.mod_arith for pair in pairs], dtype=np.int64)
            np.savez(
                self.cache_path / f'{split}.npz', imgs=imgs, labels=labels,
                data_points=np.array(pairs, dtype=np.int64),
            )
            del imgs
        write_split_manifest(
            self.cache_path, self.cache_config, splits,
            membership='angle_pairs', source_size=len(train) + len(test),
        )

    def __getitem__(self, idx):
        image = self.imgs[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label
