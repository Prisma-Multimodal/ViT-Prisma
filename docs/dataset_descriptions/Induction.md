## Induction Dataset

The Induction Mice are trained on the synthetically generated Induction dataset, which comes in two flavours: monogenic and polygenic. This dataset is designed to enable experiments that could potentially uncover induction heads in ViTs, akin to the induction heads found in language models.

## Monogenic Induction dataset

![Sample from each class](assets/images/monogenic_induction.png)

**Classes** 

0. **Vertical-Same**
1. **Vertical-Not Same**
2. **Horizontal-Same**
3. **Horizontal-Not same**

Here, the same/not-same indicates whether the structures present in the image are the same or not.

## Polygenic Induction dataset

![Sample from each class](assets/images/polygenic_induction.png)

**Classes**

| ID | Orientation | Pattern |
|----|-------------|---------|
| 0  | H           | AAAA    |
| 1  | H           | ABAB    |
| 2  | H           | ABBA    |
| 3  | H           | AABB    |
| 4  | H           | ABBB    |
| 5  | H           | AAAB    |
| 6  | V           | AAAA    |
| 7  | V           | ABAB    |
| 8  | V           | ABBA    |
| 9  | V           | AABB    |
| 10 | V           | ABBB    |
| 11 | V           | AAAB    |

## Reproducible generation and caching

Both `InductionDataset` and `PolygenicInductionDataset` accept keyword-only
`seed` (default `42`) and `test_size` (default `0.1`) arguments. Use the same
settings and directory when loading the training and test partitions:

```python
from vit_prisma.dataloaders.induction import InductionDataset

train = InductionDataset("train", dir_path="data/induction-seed42", seed=42)
test = InductionDataset("test", dir_path="data/induction-seed42", seed=42)
```

Generation and loading run on CPU and require no model downloads or tracking
account. Sampling uses a local random generator, so unrelated calls to Python's
or NumPy's global random generators do not change the partitions. Each class is
balanced before splitting. The first load generates both partitions; subsequent
loads reuse the cache.

`split_manifest.json` records the dataset name, generation version, seed, test
fraction, source size, and the ordered raw row indices for each partition.
Indices refer to `induction_dataset.npz` and match the order of examples in
`all_train.npz` and `all_test.npz`. These are random example splits, not held-out
shape or position splits; distinct raw rows may still depict the same image.

Changing a seed or split fraction requires a new cache directory. Constructors
raise `ValueError` if the stored settings differ, or if existing split files lack
a manifest. For older caches, keep the original files and choose a new directory
to generate a verified split. The explicit `generate_dataset` and
`create_balanced_dataset` functions also accept `seed` and `test_size`; unlike
constructors, these generation functions overwrite their output files.

Polygenic generation enumerates many 64×64 images and is substantially larger
than monogenic generation. Start with the monogenic dataset for a small CPU run.

## Circle addition dataset

`CircleDataset` now generates both partitions from a fresh directory. Its default
cache directory is `../data/circle`; an explicit path is recommended:

```python
from vit_prisma.dataloaders.circle import CircleDataset

train = CircleDataset("train", cache_path="data/circle-seed42", seed=42)
test = CircleDataset("test", cache_path="data/circle-seed42", seed=42)
```

The dataset contains the 1,770 unordered pairs of distinct angles from 0 to 59.
The target is their sum modulo 60. `split_ratio` is the **training** fraction
(default `0.5`), yielding 885 examples in each partition. Images have shape
`(1, 32, 32)` and float32 values in `[-1, 1]`; the optional `transform` runs when
an example is accessed. The manifest stores the ordered angle pairs themselves.
The same cache validation rules apply to the seed, split ratio, and model type.

`model_type="pretrained_transformer"` produces `(3, 224, 224)` images in `[0, 1]`
with white padding. This larger cached representation needs about 1 GiB for the
full dataset; leave it unset for small CPU experiments.

## Trainer validation splits

When `train(..., val_dataset=None)` creates a validation partition, it applies
the effective configuration seed (including tracking sweep overrides) first.
It saves `train_val_split.json` in `config.parent_dir`, containing the seed,
source size, and ordered training/validation indices. These indices refer to the
input training dataset. Splitting uses lazy subsets, preserving transforms that
run on access. A supplied validation dataset is used unchanged and produces no
new split manifest. Use a separate output directory for each run to retain its
manifest.

The Orientation column indicates whether the structures are arranged horizontally (H) or vertically (V). The Pattern column denotes the sequence in which the structures are arranged.
