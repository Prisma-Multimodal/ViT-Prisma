"""Exercise the offline CPU entry point with optional packages unavailable."""

import json
import os
import subprocess
import sys

import torch


def test_cpu_quickstart_without_optional_dependencies(tmp_path):
    script = r'''
import importlib.abc
import importlib.util
import sys

optional = {'wandb', 'plotly', 'matplotlib', 'kaleido'}
original_find_spec = importlib.util.find_spec
def find_spec(name, package=None):
    # Dependency probes should see an absent package, not an import failure.
    if name.split('.')[0] in optional:
        return None
    return original_find_spec(name, package)
importlib.util.find_spec = find_spec

class BlockExtras(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in optional:
            raise ModuleNotFoundError(f'Optional dependency blocked: {fullname}', name=fullname)

sys.meta_path.insert(0, BlockExtras())
import vit_prisma
assert 'vit_prisma.sae.train_sae' not in sys.modules
from vit_prisma import load_hooked_model, get_model_transforms
from vit_prisma.sae import SparseAutoencoder, VisionModelSAERunnerConfig
assert callable(load_hooked_model) and callable(get_model_transforms)
from vit_prisma.examples.cpu_quickstart import main
main(['--output-dir', sys.argv[1], '--steps', '2', '--train-samples', '32',
      '--eval-samples', '16', '--batch-size', '8', '--threads', '1'])
from vit_prisma.training import trainer
assert trainer.wandb is None
try:
    from vit_prisma.sae import VisionSAETrainer
except ImportError as error:
    assert 'vit-prisma[tracking]' in str(error)
else:
    raise AssertionError('SAE training should identify its missing tracking dependency')
assert not {'wandb', 'plotly', 'matplotlib', 'kaleido'} & sys.modules.keys()
'''
    environment = {**os.environ, "HF_HUB_OFFLINE": "1", "WANDB_MODE": "disabled"}
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)], env=environment,
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert metrics["device"] == "cpu"
    assert metrics["steps"] == 2
    assert metrics["train_after"]["samples"] == 32
    assert metrics["test"]["samples"] == 16
    assert 0 <= metrics["test"]["accuracy"] <= 1
    assert metrics["train_before"]["loss"] != metrics["train_after"]["loss"]
    patches = json.loads((tmp_path / "patching.json").read_text())
    assert len(patches["heads"]) == 4
    assert patches["pairs"] == 16
    assert patches["identity_max_abs_error"] < 1e-5
    assert patches["full_restoration_max_abs_error"] < 1e-5
    checkpoint = torch.load(tmp_path / "model.pt", weights_only=True)
    assert all(value.device.type == "cpu" for value in checkpoint["model_state_dict"].values())
    assert checkpoint["config"]["device"] == "cpu"
    assert (tmp_path / "data" / "split_manifest.json").exists()


def test_cpu_quickstart_rejects_empty_run(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "vit_prisma.examples.cpu_quickstart", "--steps", "0",
         "--output-dir", str(tmp_path / "unused")],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 2
    assert "positive integer" in result.stderr
    assert not (tmp_path / "unused").exists()
