"""The names of ions, written the way spectroscopists write them."""

import numpy as np

import named_arrays as na

__all__ = [
    "spectroscopic",
]

_numeral = (
    (1000, "M"),
    (900, "CM"),
    (500, "D"),
    (400, "CD"),
    (100, "C"),
    (90, "XC"),
    (50, "L"),
    (40, "XL"),
    (10, "X"),
    (9, "IX"),
    (5, "V"),
    (4, "IV"),
    (1, "I"),
)


def _roman(number: int) -> str:
    """Write a positive integer as a Roman numeral."""
    result = ""
    for value, numeral in _numeral:
        count, number = divmod(number, value)
        result += numeral * count
    return result


def _spectroscopic(name: str, latex: bool) -> str:
    element, _, stage = str(name).partition(" ")
    element = element.capitalize()
    numeral = _roman(int(stage))
    if latex:
        return rf"{element}\,\textsc{{{numeral.lower()}}}"
    return f"{element} {numeral}"


def spectroscopic(
    ion: str | na.AbstractScalar,
    latex: bool = False,
) -> str | na.AbstractScalar:
    r"""
    Write the name of an ion the way a spectroscopist writes it.

    The charge state is a Roman numeral, one greater than the charge, so
    that the neutral atom is ``I``. This is how :mod:`fiasco` numbers its
    ions as well, only in Arabic numerals, so ``O 5`` becomes ``O V``.

    Parameters
    ----------
    ion
        The name of an ion, or an array of them, as :mod:`fiasco` writes it.
    latex
        Whether to write the numeral as LaTeX small capitals, which is how
        it is set in print.

    Examples
    --------
    The ion ESIS was built to observe.

    .. jupyter-execute::

        import utu

        utu.spectrum.spectroscopic("O 5")

    And as it would be set in a journal.

    .. jupyter-execute::

        utu.spectrum.spectroscopic("O 5", latex=True)
    """
    if isinstance(ion, str):
        return _spectroscopic(ion, latex=latex)

    ion = na.as_named_array(ion)
    result = np.array([_spectroscopic(i, latex=latex) for i in ion.ndarray.flat])

    return na.ScalarArray(
        ndarray=result.reshape(ion.ndarray.shape),
        axes=ion.axes,
    )
