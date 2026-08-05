"""Registry of named zero-argument estimator factories."""

_REGISTRY = {}


def variant(name):
    """Register a fresh-estimator factory under ``name``."""
    def decorator(factory):
        if name in _REGISTRY:
            raise ValueError(f"variant {name!r} is already registered")
        _REGISTRY[name] = factory
        return factory
    return decorator


def get_variant(name):
    """Return the factory registered under ``name``."""
    try:
        return _REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"unknown variant {name!r}; registered: {list_variants()}"
        ) from None


def list_variants():
    """Return registered variant names in sorted order."""
    return sorted(_REGISTRY)
