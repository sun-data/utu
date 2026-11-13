"""The spectrum of an optically thin plasma, from the CHIANTI atomic database."""

from ._lines import (
    contribution_function,
    contribution_function_line,
    ions,
    lines,
)
from ._names import spectroscopic
from ._plots import stem
from ._synthesis import synthesize

__all__ = [
    "contribution_function",
    "contribution_function_line",
    "ions",
    "lines",
    "spectroscopic",
    "stem",
    "synthesize",
]
