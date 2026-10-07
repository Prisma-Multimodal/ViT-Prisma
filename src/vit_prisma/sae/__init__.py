"""SAE classes; training and evaluation dependencies load only on demand."""

from importlib import import_module

_EXPORTS = {
    "StandardSparseAutoencoder": ".sae",
    "GatedSparseAutoencoder": ".sae",
    "SparseAutoencoder": ".sae",
    "VisionModelSAERunnerConfig": ".config",
    "CacheActivationsRunnerConfig": ".config",
    "VisionSAETrainer": ".train_sae",
    "SparsecoderEval": ".evals.model_eval",
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    try:
        value = getattr(import_module(_EXPORTS[name], __name__), name)
    except ModuleNotFoundError as error:
        extras = {"wandb": "tracking", "matplotlib": "visualization",
                  "plotly": "visualization", "kaleido": "visualization"}
        extra = extras.get(error.name.split(".")[0]) if error.name else None
        if extra is None:
            raise
        raise ImportError(
            f"{name} requires optional dependencies: pip install 'vit-prisma[{extra}]'"
        ) from error
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
