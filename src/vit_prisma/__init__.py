"""Prisma's public API, with feature modules imported only when requested."""

from importlib import import_module

_SUBMODULES = (
    "configs", "dataloaders", "model_eval", "models", "prisma_tools", "sae",
    "training", "transforms", "utils", "visualization", "vjepa_hf",
)
_FUNCTIONS = {
    "get_model_transforms": ".transforms.model_transforms",
    "load_hooked_model": ".models.model_loader",
}
__all__ = [*_SUBMODULES, *_FUNCTIONS]


def __getattr__(name):
    if name in _SUBMODULES:
        value = import_module(f".{name}", __name__)
    elif name in _FUNCTIONS:
        value = getattr(import_module(_FUNCTIONS[name], __name__), name)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
