# CPU quickstart

This example generates synthetic images, trains a tiny vision transformer, and
uses activation patching to compare attention heads. Everything runs locally on
CPU, including on machines with CUDA or Apple MPS available. It downloads no
models or datasets and requires no tracking account.

## Install

Use Python 3.10 or newer in a virtual environment, from the repository root:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

On Windows, activate with `.venv\Scripts\activate` instead. On Linux, to avoid
installing CUDA libraries, install CPU PyTorch before installing Prisma:

```sh
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e .
```

Package installation needs network access; the example itself runs offline.

## Run

```sh
prisma-cpu-quickstart --output-dir results/cpu-quickstart
```

The equivalent module command is:

```sh
python -m vit_prisma.examples.cpu_quickstart --output-dir results/cpu-quickstart
```

Defaults are seed 42, 200 optimizer steps, batch size 64, and two CPU threads.
The model has two layers, four heads, width 32, and a 64-unit MLP. It processes
32×32 grayscale images with 8×8 patches. Training uses the first 2,048 examples
from the seeded, shuffled training split; evaluation uses 256 test examples.
The generator still creates the complete monogenic induction cache, taking
roughly 200 MiB of disk space. Runtime depends on the CPU and thread settings.

For a smaller smoke run:

```sh
prisma-cpu-quickstart --output-dir results/cpu-smoke \
  --steps 10 --train-samples 128 --eval-samples 32 --threads 1
```

Use `--help` for all budgets. Counts larger than the available split are capped
at the split size. Use a new output directory when changing the seed; the dataset
cache checks its seed against its saved manifest. Repeating a run in the same
directory reuses the compatible dataset cache and replaces the model/results.

## Results

The command writes:

| File | Contents |
| --- | --- |
| `metrics.json` | Initial/final training loss and accuracy, test metrics, seed, budgets, parameter count, PyTorch version, and elapsed time. |
| `patching.json` | Clean/corrupted logit margins, the effect of restoring each first-layer attention head, and control errors. |
| `config.json` | The tiny model configuration. |
| `model.pt` | CPU model state dictionary and configuration. |
| `data/split_manifest.json` | The generated dataset's seed and exact split membership. |

The four labels combine horizontal/vertical orientation with same/different
objects. The patching demo takes 16 same-object examples from the test split and
changes just the second object's shape. It caches the clean activations, then
restores each first-layer attention head separately during corrupted inference.
A positive margin change means the intervention increased preference for the
original clean label. All four heads are reported; they are not selected using
test accuracy.

Two assertions check the mechanics: restoring the corrupted run's own cached
head activations leaves its logits unchanged, and restoring the complete clean
residual after the first layer recovers the clean logits. Passing these controls
does not require the model to classify the examples correctly. The short run is
an API demonstration, not a benchmark of relational reasoning or a claim that a
particular head implements equality.

Reload the checkpoint on CPU:

```python
import torch
from vit_prisma.models.base_vit import HookedViT

saved = torch.load("results/cpu-quickstart/model.pt", map_location="cpu", weights_only=True)
config = saved["config"]
config["dtype"] = torch.float32
model = HookedViT(config).to("cpu")
model.load_state_dict(saved["model_state_dict"])
model.eval()
```

## Optional features and tests

Plotting and tracking packages are optional. Install the extras you need:

```sh
python -m pip install -e ".[test]"           # pytest
python -m pip install -e ".[visualization]"  # Plotly, Matplotlib, Kaleido
python -m pip install -e ".[tracking]"       # Weights & Biases
python -m pip install -e ".[all]"            # visualization + tracking
```

Existing `sae` and `arrow` extras remain available. SAE training/evaluation entry
points may also need the visualization and tracking extras. The general trainer
can run without tracking by setting `config.use_wandb=False`.

The visualization extra pairs Plotly 6.1.1+ with Kaleido 1.x. Static image export
with Kaleido also requires Chrome; it is not needed for this quickstart.

Run the focused offline CPU checks:

```sh
HF_HUB_OFFLINE=1 python -m pytest tests/test_cpu_quickstart.py tests/test_synthetic_datasets.py -q
```

The quickstart test blocks plotting/tracking imports and checks the saved CPU
checkpoint and patching controls. CI runs these checks with CPU PyTorch on Python
3.10 and 3.12. The broader existing suite includes pretrained-model downloads and
GPU-dependent tests, so `pytest tests` is not the offline CPU test command.
