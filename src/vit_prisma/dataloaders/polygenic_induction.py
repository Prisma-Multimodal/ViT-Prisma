import numpy as np
import os
from torch.utils.data import Dataset
import torch
from vit_prisma.dataloaders.synthetic_cache import (
    check_cache_manifest, save_balanced_splits, validate_fraction, validate_seed,
)
from vit_prisma.dataloaders.induction import draw_circle, draw_line, draw_x, draw_diagonal

class PolygenicInductionDataset(Dataset):
    def __init__(self, train_or_test, dir_path='../data/polygenic_induction', use_metadata=False,
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
            print("Generating and saving new polygenic induction dataset...")
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

def plot_four_objects(A, B, C, D, Ax, Ay, Bx, By, Cx, Cy, Dx, Dy, vertical=False):  

    image = np.zeros((64, 64))
    
    A(image, Ax, Ay, im_size=64)
    B(image, Bx, By, im_size=64)
    C(image, Cx, Cy, im_size=64)
    D(image, Dx, Dy, im_size=64)

    if vertical:
        image = image.T
    return image

def _cache_config(seed, test_size):
    return {'dataset': 'polygenic_induction', 'version': 1, 'seed': seed, 'test_size': test_size}


def generate_dataset(dir_path='../data/polygenic_induction', *, seed=42, test_size=0.1):
    """Generate raw images and reproducible balanced splits at ``dir_path``."""
    validate_seed(seed)
    validate_fraction(test_size, 'test_size')
    draw_functions = [draw_circle, draw_line, draw_x, draw_diagonal]
    padding = 4
    offset = 7
    max_shape_size = 5  

    images = []
    metadata = []
    labels = [] 

    max_a = 64 - 3 * offset - 2 * (padding + max_shape_size)
    max_b = 64 - padding - max_shape_size

    arrangements = {
        'A A A A' : 0,
        'A B A B' : 1,
        'A B B A' : 2,
        'A A B B' : 3,
        'A B B B' : 4,
        'A A A B' : 5,
    }

    for vertical in [True, False]:
        for a in range(padding + max_shape_size, max_a):
            for b in range(padding + max_shape_size, max_b):
                Ax = a
                Ay = b
                Bx = Ax + offset
                By = Ay
                Cx = Bx + offset
                Cy = Ay
                Dx = Cx + offset
                Dy = By

                for A in draw_functions:
                    for B in draw_functions:
                        for arrangement in arrangements.keys():

                            text_label = arrangement
                            l = arrangements[arrangement] + (0 if vertical else 6)
                            arrangement = arrangement.split()

                            ta = locals()[arrangement[0]]
                            tb = locals()[arrangement[1]]
                            tc = locals()[arrangement[2]]
                            td = locals()[arrangement[3]]

                            if len(set(text_label.split())) != len(set([ta, tb, tc, td])):
                                continue

                            img = plot_four_objects(ta, tb, tc, td, Ax, Ay, Bx, By, Cx, Cy, Dx, Dy, vertical=vertical)

                            if ta == tb == tc == td:
                                all_same = True
                            else:
                                all_same = False

                            images.append(img)

                            m = {
                                "Ax": Ax,
                                "Ay": Ay,
                                "Bx": Bx,
                                "By": By,
                                "Cx": Cx,
                                "Cy": Cy,
                                "Dx": Dx,
                                "Dy": Dy,
                                "A": ta.__name__,
                                "B": tb.__name__,
                                "C": tc.__name__,
                                "D": td.__name__,
                                "Same": all_same,
                                "Vertical": vertical,
                                "pattern": text_label,
                            }

                            metadata.append(m)
                            labels.append(l)

    path = f'{dir_path}/induction_dataset.npz'
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    np.savez(path, images=images, metadata=metadata, labels=labels)

    print("Balancing dataset...")
    create_balanced_dataset(dir_path, seed=seed, test_size=test_size)


def create_balanced_dataset(dir_path='../data/polygenic_induction', *, seed=42, test_size=0.1):
    """Save balanced splits and a manifest of their raw row indices."""
    validate_seed(seed)
    validate_fraction(test_size, 'test_size')
    with np.load(f'{dir_path}/induction_dataset.npz', allow_pickle=True) as data:
        labels = data['labels']
    categories = {label: np.flatnonzero(labels == label).tolist() for label in range(12)}
    save_balanced_splits(dir_path, categories, _cache_config(seed, test_size))
