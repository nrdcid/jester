"""Optional PyTorch models. Requires the ``torch`` extra: pip install jester-ml[torch]."""

__all__ = ["Autoencoder", "Config", "sparse_loss"]


def __getattr__(name: str):
    if name in __all__:
        try:
            from . import autoencoder as _ae
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "jester.torch_models requires the torch extra: "
                'pip install "jester-ml[torch]"'
            ) from exc
        return getattr(_ae, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
