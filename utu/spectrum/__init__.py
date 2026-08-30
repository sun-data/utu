"""
The spectrum of an optically thin plasma, from the CHIANTI atomic database.
"""

from ._lines import (
    ions,
    contribution_function,
    lines,
)

__all__ = [
    "ions",
    "contribution_function",
    "lines",
]
