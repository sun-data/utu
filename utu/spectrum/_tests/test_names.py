import numpy as np
import pytest

import named_arrays as na
import utu


@pytest.mark.parametrize(
    argnames="ion,expected,expected_latex",
    argvalues=[
        ("He 1", "He I", r"He\,\textsc{i}"),
        ("O 5", "O V", r"O\,\textsc{v}"),
        ("Mg 10", "Mg X", r"Mg\,\textsc{x}"),
        ("S 4", "S IV", r"S\,\textsc{iv}"),
        ("Fe 24", "Fe XXIV", r"Fe\,\textsc{xxiv}"),
    ],
)
def test_spectroscopic(ion: str, expected: str, expected_latex: str):
    assert utu.spectrum.spectroscopic(ion) == expected
    assert utu.spectrum.spectroscopic(ion, latex=True) == expected_latex


def test_spectroscopic_array():
    """An array of names keeps its shape and its axes."""
    ion = na.ScalarArray(
        ndarray=np.array([["O 5", "Mg 10"], ["He 1", "S 4"]]),
        axes=("x", "y"),
    )

    result = utu.spectrum.spectroscopic(ion)

    assert na.shape(result) == na.shape(ion)
    assert result[{"x": 0, "y": 0}].ndarray == "O V"
    assert result[{"x": 1, "y": 1}].ndarray == "S IV"
