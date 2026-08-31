import astropy.units as u
import matplotlib
import matplotlib.pyplot as plt
import named_arrays as na
import numpy as np
import pytest

import utu

matplotlib.use("agg")

# A spectrum written down rather than computed, so that these tests say
# nothing about the atomic database and do not need it to run.
spectrum = na.FunctionArray(
    inputs=na.CartesianNdVectorArray(
        components={
            "wavelength": na.ScalarArray(
                ndarray=np.array([584.3, 609.8, 629.7]) * u.AA,
                axes=("line",),
            ),
            "ion": na.ScalarArray(
                ndarray=np.array(["He 1", "Mg 10", "O 5"]),
                axes=("line",),
            ),
        },
    ),
    outputs=na.ScalarArray(
        ndarray=np.array([10.0, 30.0, 100.0]) * u.erg / u.s / u.cm**2,
        axes=("line",),
    ),
)


@pytest.mark.parametrize("num_label", [None, 1, 2])
def test_stem(num_label: None | int):
    fig, ax = plt.subplots()

    result = utu.spectrum.stem(spectrum, ax=ax, num_label=num_label)

    expected = num_label if num_label is not None else spectrum.shape["line"]
    assert len(result) == expected

    # the brightest line is labelled first, whichever order it was given in
    assert result[0].get_text().startswith("O V")

    # every line is drawn, whether or not it is labelled
    segments = [s for c in ax.collections for s in c.get_segments()]
    assert len(segments) == spectrum.shape["line"]

    # and there is room above the tallest of them for the labels
    assert ax.get_ylim()[1] > 100

    plt.close(fig)


def test_stem_latex():
    """The name of an ion, as it would be set in a journal."""
    fig, ax = plt.subplots()

    result = utu.spectrum.stem(spectrum, ax=ax, num_label=1, latex=True)

    assert result[0].get_text().startswith(r"O\,\textsc{v}")

    plt.close(fig)


def test_stem_kwargs():
    """Every default the caller might disagree with can be replaced."""
    fig, ax = plt.subplots()

    result = utu.spectrum.stem(
        spectrum,
        ax=ax,
        num_label=1,
        kwargs_line={"color": "red"},
        kwargs_text={"fontsize": 5},
        kwargs_adjust={"time_lim": 1},
    )

    assert result[0].get_fontsize() == 5
    assert np.all(ax.collections[0].get_color() == np.array([[1, 0, 0, 1]]))

    plt.close(fig)
