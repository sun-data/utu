import astropy.units as u
import fiasco
import named_arrays as na
import numpy as np
import pytest

import utu


def _database() -> bool:
    """Whether the CHIANTI database has been downloaded on this machine."""
    try:
        return len(fiasco.list_ions()) > 0
    except Exception:  # pragma: nocover
        return False


needs_database = pytest.mark.skipif(
    not _database(),
    reason="the CHIANTI database is not available",
)

# a window around the O V line, narrow enough that few ions qualify
wavelength = [629.5, 630.0] * u.AA

temperature = na.ScalarArray(
    ndarray=10 ** np.arange(4.0, 7.0, 0.5) * u.K,
    axes=("temperature",),
)


@needs_database
def test_ions():
    result = utu.spectrum.ions(wavelength=wavelength)
    assert result
    for name in result:
        assert isinstance(name, str)

    # the line this window was chosen for
    assert "O 5" in result

    # and asking for every ion gives more of them
    assert len(result) < len(utu.spectrum.ions())


@needs_database
def test_contribution_function():
    ion = fiasco.Ion("O 5", temperature.ndarray)

    density = na.ScalarArray(
        ndarray=1e9 / u.cm**3 * np.ones(temperature.shape["temperature"]),
        axes=("temperature",),
    )

    result = utu.spectrum.contribution_function(
        ion=ion,
        density=density,
        axis_temperature="temperature",
    )

    assert isinstance(result, na.FunctionArray)
    assert na.unit(result.inputs).is_equivalent(u.AA)
    assert "temperature" in na.shape(result.outputs)
    assert "line" in na.shape(result.outputs)

    # a density which shares the axis of the temperature is one density per
    # temperature, so the result carries no axis of its own for it
    assert "_density" not in na.shape(result.outputs)


@needs_database
def test_contribution_function_grid():
    """A density on its own axis is every density at every temperature."""
    ion = fiasco.Ion("O 5", temperature.ndarray)

    density = na.ScalarArray(
        ndarray=np.array([1e9, 1e10]) / u.cm**3,
        axes=("density",),
    )

    result = utu.spectrum.contribution_function(
        ion=ion,
        density=density,
        axis_temperature="temperature",
    )

    shape = na.shape(result.outputs)
    assert shape["temperature"] == temperature.shape["temperature"]
    assert shape["_density"] == 2


@needs_database
def test_lines():
    density = na.ScalarArray(
        ndarray=1e15 * u.K / u.cm**3 / temperature.ndarray,
        axes=("temperature",),
    )
    emission_measure = na.ScalarArray(
        ndarray=1e27 / u.cm**5 * np.ones(temperature.shape["temperature"]),
        axes=("temperature",),
    )

    result = utu.spectrum.lines(
        temperature=temperature,
        density=density,
        emission_measure=emission_measure,
        wavelength=wavelength,
    )

    assert isinstance(result, na.FunctionArray)

    w = result.inputs.wavelength
    ion = result.inputs.ion
    intensity = result.outputs

    # the wavelength, the ion, and the intensity of a line are one array
    assert na.shape(w) == na.shape(ion) == na.shape(intensity)

    assert np.all(w > wavelength.min())
    assert np.all(w < wavelength.max())

    # brightest first
    d = np.diff(na.value(intensity).ndarray)
    assert np.all(d <= 0)

    # and the brightest line in this window is the one it was chosen for
    brightest = {"line": 0}
    assert str(ion[brightest].ndarray) == "O 5"
    assert np.isclose(w[brightest].ndarray.to_value(u.AA), 629.733, atol=1e-2)
