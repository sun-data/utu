import importlib.metadata

__all__ = [
    "__version__",
]

__version__ = importlib.metadata.version("utu")
"""The version of this package, taken from the tag it was built from."""
