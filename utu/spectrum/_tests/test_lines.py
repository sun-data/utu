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

# A window around the O V line, narrow enough that only a handful of ions
# qualify. Every ion which does costs a level population solve, and six of
# them prove as much about sorting and filtering as twenty five do.
# Given as named scalars rather than as plain quantities, since both are
# allowed and only one of them would otherwise be tried.
wavelength_min = na.ScalarArray(629.72 * u.AA)
wavelength_max = na.ScalarArray(629.75 * u.AA)

temperature = na.ScalarArray(
    ndarray=10 ** np.arange(4.0, 7.0, 0.5) * u.K,
    axes=("temperature",),
)

density = na.ScalarArray(
    ndarray=1e15 * u.K / u.cm**3 / temperature.ndarray,
    axes=("temperature",),
)

emission_measure = na.ScalarArray(
    ndarray=1e27 / u.cm**5 * np.ones(temperature.shape["temperature"]),
    axes=("temperature",),
)


# Each of the fixtures below walks the whole database, which takes an order
# of magnitude longer than anything the tests do with the result. They are
# scoped to the session so that the suite pays for each walk once.


@pytest.fixture(scope="session")
def ions_all() -> list[str]:
    """Every ion the database describes."""
    return utu.spectrum.ions()


@pytest.fixture(scope="session")
def ions_window() -> list[str]:
    """Every ion with a line in the window above."""
    return utu.spectrum.ions(
        wavelength_min=wavelength_min,
        wavelength_max=wavelength_max,
    )


@pytest.fixture(scope="session")
def proton_electron_ratio() -> u.Quantity:
    """The ratio of protons to electrons at each temperature above."""
    return fiasco.proton_electron_ratio(temperature.ndarray)


@needs_database
def test_ions(ions_window: list[str], ions_all: list[str]):
    assert ions_window
    for name in ions_window:
        assert isinstance(name, str)

    # the line this window was chosen for
    assert "O 5" in ions_window

    # and asking for every ion gives more of them
    assert len(ions_window) < len(ions_all)


@needs_database
def test_contribution_function(proton_electron_ratio: u.Quantity):
    ion = fiasco.Ion("O 5", temperature.ndarray)

    result = utu.spectrum.contribution_function(
        ion=ion,
        density=density,
        axis_temperature="temperature",
        proton_electron_ratio=proton_electron_ratio,
    )

    assert isinstance(result, na.FunctionArray)
    assert na.unit(result.inputs).is_equivalent(u.AA)
    assert "temperature" in na.shape(result.outputs)
    assert "line" in na.shape(result.outputs)

    # a density which shares the axis of the temperature is one density per
    # temperature, so the result carries no axis of its own for it
    assert "_density" not in na.shape(result.outputs)


@needs_database
def test_contribution_function_grid(proton_electron_ratio: u.Quantity):
    """A density on its own axis is every density at every temperature."""
    ion = fiasco.Ion("O 5", temperature.ndarray)

    density_grid = na.ScalarArray(
        ndarray=np.array([1e9, 1e10]) / u.cm**3,
        axes=("density",),
    )

    result = utu.spectrum.contribution_function(
        ion=ion,
        density=density_grid,
        axis_temperature="temperature",
        proton_electron_ratio=proton_electron_ratio,
    )

    shape = na.shape(result.outputs)
    assert shape["temperature"] == temperature.shape["temperature"]
    assert shape["_density"] == 2


@needs_database
def test_lines():
    """The whole of it, with nothing given that can be worked out."""
    result = utu.spectrum.lines(
        temperature=temperature,
        density=density,
        emission_measure=emission_measure,
        wavelength_min=wavelength_min,
        wavelength_max=wavelength_max,
    )

    assert isinstance(result, na.FunctionArray)

    w = result.inputs.wavelength
    ion = result.inputs.ion
    intensity = result.outputs

    # the wavelength, the ion, and the intensity of a line are one array
    assert na.shape(w) == na.shape(ion) == na.shape(intensity)

    assert np.all(w > wavelength_min)
    assert np.all(w < wavelength_max)

    # brightest first
    d = np.diff(na.value(intensity).ndarray)
    assert np.all(d <= 0)

    # and the brightest line in this window is the one it was chosen for
    brightest = {"line": 0}
    assert str(ion[brightest].ndarray) == "O 5"
    assert np.isclose(w[brightest].ndarray.to_value(u.AA), 629.733, atol=1e-2)


@needs_database
def test_lines_ions(
    ions_window: list[str],
    proton_electron_ratio: u.Quantity,
):
    """Naming the ions is what keeps a result from depending on the database."""
    result = utu.spectrum.lines(
        temperature=temperature,
        density=density,
        emission_measure=emission_measure,
        wavelength_min=wavelength_min,
        wavelength_max=wavelength_max,
        ions=["O 5"],
        proton_electron_ratio=proton_electron_ratio,
    )

    assert np.all(result.inputs.ion == "O 5")

    # and there were other ions to be had in this window
    assert len(ions_window) > 1
