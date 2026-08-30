"""The spectrum of an optically thin plasma, from the CHIANTI atomic database."""

from ._lines import (
    contribution_function,
    ions,
    lines,
)

__all__ = [
    "contribution_function",
    "ions",
    "lines",
]
