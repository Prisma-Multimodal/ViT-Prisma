"""Train a tiny vision transformer and patch its activations entirely on CPU.

Run with ``python -m vit_prisma.examples.cpu_quickstart --help``.
No pretrained weights, model downloads, plotting, or tracking services are used.
"""

import argparse
import json
from pathlib import Path
import time

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from vit_prisma.configs.HookedViTConfig import HookedViTConfig
from vit_prisma.dataloaders import induction
from vit_prisma.dataloaders.synthetic_cache import validate_seed
from vit_prisma.models.base_vit import HookedViT
from vit_prisma.utils.saving_utils import object_to_dict


def positive_int(value):
    """Parse a positive CLI budget, rejecting empty runs."""
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results/cpu-quickstart"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=positive_int, default=200)
    parser.add_argument("--train-samples", type=positive_int, default=2048)
    parser.add_argument("--eval-samples", type=positive_int, default=256)
    parser.add_argument("--batch-size", type=positive_int, default=64)
    parser.add_argument("--threads", type=positive_int, default=2)
    return parser


@torch.no_grad()
def evaluate(model, loader):
    """Compute loss per example and accuracy over a bounded CPU loader."""
    model.eval()
    total_loss, correct, count = 0.0, 0, 0
    for images, labels in loader:
        logits = model(images)
        total_loss += F.cross_entropy(logits, labels, reduction="sum").item()
        correct += (logits.argmax(dim=-1) == labels).sum().item()
        count += len(labels)
    return {"loss": total_loss / count, "accuracy": correct / count, "samples": count}


@torch.no_grad()
def patching_demo(model, dataset, max_pairs=16):
    """Swap individual attention-head outputs between matched clean/corrupt images.

    Clean images contain identical objects. Corruption changes just the second
    object, changing the same/different label while preserving its orientation.
    """
    clean, corrupt, clean_labels, corrupt_labels, sample_indices = [], [], [], [], []
    shapes = [induction.draw_circle, induction.draw_line, induction.draw_x, induction.draw_diagonal]
    shape_by_name = {shape.__name__: shape for shape in shapes}
    for index, metadata in enumerate(dataset.metadata):
        if not metadata["Same"]:
            continue
        first = shape_by_name[metadata["A"]]
        second = shapes[(shapes.index(first) + 1) % len(shapes)]
        image = induction.plot_two_objects(
            first, second, metadata["Ax"], metadata["Ay"],
            metadata["Bx"], metadata["By"], vertical=metadata["Vertical"],
        )
        clean.append(dataset[index][0])
        corrupt.append(torch.from_numpy(image.copy()).float().unsqueeze(0))
        clean_labels.append(int(dataset.labels[index]))
        corrupt_labels.append(1 if metadata["Vertical"] else 3)
        sample_indices.append(index)
        if len(clean) == max_pairs:
            break
    clean, corrupt = torch.stack(clean), torch.stack(corrupt)
    clean_labels, corrupt_labels = torch.tensor(clean_labels), torch.tensor(corrupt_labels)
    rows = torch.arange(len(clean))

    def margin(logits):
        return (logits[rows, clean_labels] - logits[rows, corrupt_labels]).mean().item()

    model.eval()
    head_hook = "blocks.0.attn.hook_z"
    residual_hook = "blocks.0.hook_resid_post"
    clean_logits, clean_cache = model.run_with_cache(
        clean, names_filter=[head_hook, residual_hook]
    )
    corrupt_logits, corrupt_cache = model.run_with_cache(corrupt, names_filter=[head_hook])
    head_results = []
    for head in range(model.cfg.n_heads):
        def restore_head(activation, hook, head=head):
            activation = activation.clone()
            activation[:, :, head, :] = clean_cache[head_hook][:, :, head, :]
            return activation

        patched_logits = model.run_with_hooks(corrupt, fwd_hooks=[(head_hook, restore_head)])
        head_results.append({
            "head": head, "patched_margin": margin(patched_logits),
            "margin_change": margin(patched_logits) - margin(corrupt_logits),
        })

    # Negative and positive controls check the patching implementation itself.
    identity = model.run_with_hooks(corrupt, fwd_hooks=[
        (head_hook, lambda activation, hook: corrupt_cache[head_hook])
    ])
    restored = model.run_with_hooks(corrupt, fwd_hooks=[
        (residual_hook, lambda activation, hook: clean_cache[residual_hook])
    ])
    torch.testing.assert_close(identity, corrupt_logits)
    torch.testing.assert_close(restored, clean_logits)
    return {
        "pairs": len(clean), "test_sample_indices": sample_indices,
        "hook": head_hook, "cached_shape": list(clean_cache[head_hook].shape),
        "margin_definition": "clean-label logit minus corrupted-label logit, averaged over pairs",
        "clean_margin": margin(clean_logits), "corrupted_margin": margin(corrupt_logits),
        "heads": head_results,
        "identity_max_abs_error": (identity - corrupt_logits).abs().max().item(),
        "full_restoration_max_abs_error": (restored - clean_logits).abs().max().item(),
    }


def run(args):
    """Run a bounded CPU demo and return the same metrics written to disk."""
    validate_seed(args.seed)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    started = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = args.output_dir / "data"
    train_data = induction.InductionDataset("train", data_dir, seed=args.seed)
    test_data = induction.InductionDataset("test", data_dir, seed=args.seed, use_metadata=True)
    train_count = min(args.train_samples, len(train_data))
    eval_count = min(args.eval_samples, len(test_data))
    train_subset = Subset(train_data, range(train_count))
    train_loader = DataLoader(
        train_subset, batch_size=args.batch_size, shuffle=True, num_workers=0,
        generator=torch.Generator(device="cpu").manual_seed(args.seed),
    )
    train_eval_loader = DataLoader(train_subset, batch_size=args.batch_size, num_workers=0)
    test_loader = DataLoader(
        Subset(test_data, range(eval_count)), batch_size=args.batch_size, num_workers=0,
    )
    cfg = HookedViTConfig(
        n_layers=2, d_model=32, d_head=8, n_heads=4, d_mlp=64,
        image_size=32, patch_size=8, n_channels=1, n_classes=4,
        return_type="logits", device="cpu", dtype=torch.float32,
        seed=args.seed, use_wandb=False,
    )
    model = HookedViT(cfg).to("cpu")
    before = evaluate(model, train_eval_loader)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.train()
    iterator = iter(train_loader)
    last_loss = None
    for step in range(args.steps):
        try:
            images, labels = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            images, labels = next(iterator)
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(images), labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        last_loss = loss.item()
        if (step + 1) % 50 == 0 or step + 1 == args.steps:
            print(f"Step {step + 1}/{args.steps}: training loss {last_loss:.4f}")

    metrics = {
        "device": "cpu", "seed": args.seed, "steps": args.steps,
        "threads": args.threads, "batch_size": args.batch_size,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "torch_version": str(torch.__version__), "train_before": before,
        "train_after": evaluate(model, train_eval_loader),
        "test": evaluate(model, test_loader), "last_batch_loss": last_loss,
    }
    patching = patching_demo(model, test_data)
    metrics["elapsed_seconds"] = time.perf_counter() - started
    config = object_to_dict(cfg)
    torch.save({"model_state_dict": model.state_dict(), "config": config}, args.output_dir / "model.pt")
    for filename, payload in (("metrics.json", metrics), ("patching.json", patching), ("config.json", config)):
        (args.output_dir / filename).write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(f"CPU test accuracy: {metrics['test']['accuracy']:.1%} on {eval_count} examples")
    print(f"Patching controls passed; results saved in {args.output_dir.resolve()}")
    return metrics


def main(argv=None):
    args = make_parser().parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
