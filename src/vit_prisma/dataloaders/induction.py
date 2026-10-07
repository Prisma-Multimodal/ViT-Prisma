import numpy as np
import os
from torch.utils.data import Dataset
import torch
from vit_prisma.dataloaders.synthetic_cache import (
    check_cache_manifest, save_balanced_splits, validate_fraction, validate_seed,
)

class InductionDataset(Dataset):
    def __init__(self, train_or_test, dir_path='../data/induction', use_metadata=False,
                 transform=None, *, seed=42, test_size=0.1):
        """Load a seeded split, generating both partitions in ``dir_path`` if needed."""
        if train_or_test not in ('train', 'test'):
            raise ValueError("train_or_test must be 'train' or 'test'")
        validate_seed(seed)
        validate_fraction(test_size, 'test_size')
        self.seed = seed
        self.test_size = test_size
        config = _cache_config(seed, test_size)

        self.dir_path = dir_path
        self.cache_path = f'{dir_path}/all_{train_or_test}.npz'

        self.use_metadata = use_metadata
        self.transform = transform

        if not check_cache_manifest(dir_path, config, ('all_train.npz', 'all_test.npz')):
            print("Generating and saving new induction dataset...")
            self._generate_and_cache()

        print("Loading induction dataset from cache...", self.cache_path)
        self._load_from_cache()
        # self._normalize_and_to_tensor()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx][np.newaxis, :, :]
        label = self.labels[idx]
        # meta_data = self.metadata[idx]
        image = torch.from_numpy(image).float()
        if self.transform:
            image = self.transform(image)
        return image, label

    def _load_from_cache(self):
        with np.load(self.cache_path, allow_pickle=True) as loaded:
            self.images = loaded['images']
            if self.use_metadata:
                self.metadata = loaded['metadata']
            self.labels = loaded['labels']

    def _generate_and_cache(self):
        generate_dataset(self.dir_path, seed=self.seed, test_size=self.test_size)

    # def _normalize_and_to_tensor(self):
    #     self.images = 2 * torch.tensor(self.images, dtype=torch.float32) / 255.0 - 1 # normalize [-1,]
    #     self.labels = torch.tensor(self.labels, dtype=torch.int64)

## Functionality to generate dataset ##

def draw_circle(image, center_row, center_col, radius=2, im_size=32):
        """Draw a circle on the given image."""
        for r in range(center_row-radius, center_row+radius+1):
            for c in range(center_col-radius, center_col+radius+1):
                if (r - center_row)**2 + (c - center_col)**2 <= radius**2 and 0 <= r < im_size and 0 <= c < im_size:
                    image[r, c] = 1
        return image

def draw_line(image, center_row, center_col, line_length=4, im_size=32):
    for i in range(-line_length // 2, line_length // 2 + 1):
        if 0 <= center_row + i < im_size and 0 <= center_col < im_size:
            image[center_row + i, center_col] = 1
    return image

def draw_x(image, center_row, center_col, x_length=5, im_size=32):
    # Drawing the X centered around the center_col
# Drawing the X centered around the start_col
    for i in range(x_length):
        image[center_row - x_length // 2 + i, center_col - x_length // 2 + i] = 1
        image[center_row - x_length // 2 + i, center_col + x_length // 2 - i] = 1
    return image


def draw_diagonal(image, center_row, center_col, line_length=4, im_size=32):
    for i in range(-line_length // 2, line_length // 2 + 1):
        if 0 <= center_row + i < im_size and 0 <= center_col + i < im_size:
            image[center_row + i, center_col + i] = 1

    return image

def plot_two_objects(A, B, Ax, Ay, Bx, By, vertical=False):

    image = np.zeros((32, 32))

    # List of available drawing functions
    draw_functions = [draw_circle, draw_line, draw_x, draw_diagonal]

    # Call the first chosen function
    A(image, Ax, Ay)

    # Call the second chosen function right next to the first
    B(image, Bx, By)

    if vertical:
        image = image.T

    return image

def _cache_config(seed, test_size):
    return {'dataset': 'induction', 'version': 1, 'seed': seed, 'test_size': test_size}


def generate_dataset(dir_path='../data/induction', *, seed=42, test_size=0.1):
    """Generate raw images and reproducible balanced splits at ``dir_path``."""
    validate_seed(seed)
    validate_fraction(test_size, 'test_size')

    # generate one of each image combo, make sure spacing makes sense
    draw_functions = [draw_circle, draw_line, draw_x, draw_diagonal]
    padding = 4
    offset = 7

    images = []
    metadata = []
    labels = [] # from 0 to 3

    for vertical in [True, False]:
        for a in range(padding, 32 - padding):
            for b in range(padding, 32 - padding - offset):
                Ax = a
                Ay = b
                Bx = Ax
                By = Ay + offset

                # Example of how to use it:
                for A in draw_functions:
                    for B in draw_functions:
                        img = plot_two_objects(A, B, Ax, Ay, Bx, By, vertical=vertical)

                        if A == B:
                            same = True
                        else:
                             same = False

                        images.append(img)
                        m = {
                            "Ax": Ax,
                            "Ay": Ay,
                            "Bx": Bx,
                            "By": By,
                            "A": A.__name__,
                            "B": B.__name__,
                            "Same": same,
                            "Vertical": vertical
                        }
                        metadata.append(m)

                        if vertical and same:
                            l = 0
                        elif vertical and not same:
                            l = 1
                        elif not vertical and same:
                            l = 2
                        elif not vertical and not same:
                            l = 3

                        labels.append(l)
    
    path = f'{dir_path}/induction_dataset.npz'
    print(f"Saving raw dataset to {path}...")
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    np.savez(path, images=images, metadata=metadata, labels=labels)

    print("Balancing dataset...")
    create_balanced_dataset(dir_path, seed=seed, test_size=test_size)


def create_balanced_dataset(dir_path='../data/induction', *, seed=42, test_size=0.1):
    """Save balanced splits and a manifest of their raw row indices."""
    validate_seed(seed)
    validate_fraction(test_size, 'test_size')
    with np.load(f'{dir_path}/induction_dataset.npz', allow_pickle=True) as data:
        metadata = data['metadata']
    categories = {
        'same_T_vertical_T': [i for i, m in enumerate(metadata) if m['Same'] and m['Vertical']],
        'same_F_vertical_F': [i for i, m in enumerate(metadata) if not m['Same'] and not m['Vertical']],
        'same_T_vertical_F': [i for i, m in enumerate(metadata) if m['Same'] and not m['Vertical']],
        'same_F_vertical_T': [i for i, m in enumerate(metadata) if not m['Same'] and m['Vertical']],
    }
    save_balanced_splits(dir_path, categories, _cache_config(seed, test_size))
