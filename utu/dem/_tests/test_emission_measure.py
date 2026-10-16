import astropy.units as u
import named_arrays as na
import numpy as np
import pytest

import utu
from utu.dem._plowman import _matrices

_logt = np.round(np.arange(5.5, 7.0 + 1e-9, 0.05), 2)

_logt_uneven = np.array([5.5, 5.7, 5.8, 5.85, 5.9, 6.0, 6.2, 6.5, 7.0])
"""An uneven grid of temperatures, closest together near 1 MK."""


def _mass(logt: np.ndarray) -> np.ndarray:
    """The mass matrix :func:`utu.dem.plowman` fits with, from its own code."""
    return _matrices(logt, np.eye(logt.size), smoothness=1)[0]


def _temperature(logt: np.ndarray) -> na.ScalarArray:
    """The temperatures of a DEM, from their logarithms."""
    return na.ScalarArray(10**logt * u.K, axes="temperature")


def _dem(
    logt: np.ndarray,
    dem: np.ndarray,
    axes: tuple[str, ...] = ("temperature",),
) -> na.FunctionArray:
    """A DEM in units of :math:`\\mathrm{cm^{-5}}` from plain arrays."""
    return na.FunctionArray(
        inputs=_temperature(logt),
        outputs=na.ScalarArray(dem / u.cm**5, axes=axes),
    )


@pytest.mark.parametrize(
    argnames="outputs",
    argvalues=[
        na.ScalarArray(np.full(_logt.size, 3e27) / u.cm**5, axes="temperature"),
        # without the axis of temperature, broadcast along it
        na.ScalarArray(3e27 / u.cm**5),
    ],
)
def test_emission_measure_constant(outputs: na.ScalarArray) -> None:
    """
    A constant DEM holds its value times the span of :math:`\\log_{10} T`,
    and its ends stand for half a step.
    """
    dem = na.FunctionArray(inputs=_temperature(_logt), outputs=outputs)

    result = utu.dem.emission_measure(dem, axis_temperature="temperature")

    unit = 1 / u.cm**5
    assert result.shape == dict(temperature=_logt.size)
    assert np.allclose(result.sum("temperature"), 3e27 * unit * 1.5)
    assert np.allclose(result[{"temperature": 0}], 3e27 * unit * 0.025)
    assert np.allclose(result[{"temperature": 5}], 3e27 * unit * 0.05)
    assert np.allclose(result[{"temperature": -1}], 3e27 * unit * 0.025)


@pytest.mark.parametrize(
    argnames="logt",
    argvalues=[_logt, _logt_uneven],
)
def test_emission_measure_integral(logt: np.ndarray) -> None:
    """
    Weighted by a response, it is the integral of the product of the
    response and the DEM, each linear between the temperatures.
    """
    rng = np.random.default_rng(0)
    dem = rng.uniform(1e26, 1e28, logt.size)
    response = rng.uniform(0, 1e-25, logt.size)

    result = utu.dem.emission_measure(_dem(logt, dem), axis_temperature="temperature")
    intensity = (na.ScalarArray(response, axes="temperature") * result).sum(
        "temperature"
    )

    # the trapezoid rule on a grid fine enough to be exact to about 1e-10
    fine = np.linspace(logt[0], logt[-1], 200_001)
    product = np.interp(fine, logt, dem) * np.interp(fine, logt, response)
    expected = np.trapezoid(product, fine) / u.cm**5

    assert np.allclose(intensity.ndarray, expected, rtol=1e-8)


def test_emission_measure_matrix() -> None:
    """
    It is the mass matrix of :func:`utu.dem.plowman` times the DEM, for
    every pixel of a map.
    """
    rng = np.random.default_rng(1)
    dem = rng.uniform(1e26, 1e28, (2, 3, _logt_uneven.size))

    result = utu.dem.emission_measure(
        _dem(_logt_uneven, dem, axes=("y", "x", "temperature")),
        axis_temperature="temperature",
    )

    expected = na.ScalarArray(
        dem @ _mass(_logt_uneven) / u.cm**5,
        axes=("y", "x", "temperature"),
    )
    assert result.shape == dict(y=2, x=3, temperature=_logt_uneven.size)
    assert np.allclose(result, expected, rtol=1e-14)


def test_emission_measure_grid_per_pixel() -> None:
    """Each pixel may have its own temperatures, and is weighted by its own."""
    grids = np.stack([_logt_uneven, _logt_uneven + 0.1])
    rng = np.random.default_rng(2)
    dem = rng.uniform(1e26, 1e28, grids.shape)

    result = utu.dem.emission_measure(
        na.FunctionArray(
            inputs=na.ScalarArray(10**grids * u.K, axes=("x", "temperature")),
            outputs=na.ScalarArray(dem / u.cm**5, axes=("x", "temperature")),
        ),
        axis_temperature="temperature",
    )

    expected = na.ScalarArray(
        np.stack([d @ _mass(g) for g, d in zip(grids, dem)]) / u.cm**5,
        axes=("x", "temperature"),
    )
    assert np.allclose(result, expected, rtol=1e-14)


def test_emission_measure_uncertain_temperature() -> None:
    """
    Uncertain temperatures weight the DEM by the grid of the nominal
    temperatures and of each sample.
    """
    axis = na.UncertainScalarArray.axis_distribution
    grids = np.stack([_logt_uneven + 0.02, _logt_uneven - 0.03])
    dem = np.linspace(1e27, 2e27, _logt_uneven.size)

    result = utu.dem.emission_measure(
        na.FunctionArray(
            inputs=na.UncertainScalarArray(
                nominal=_temperature(_logt_uneven),
                distribution=na.ScalarArray(
                    10**grids * u.K,
                    axes=(axis, "temperature"),
                ),
            ),
            outputs=na.ScalarArray(dem / u.cm**5, axes="temperature"),
        ),
        axis_temperature="temperature",
    )

    expected = na.UncertainScalarArray(
        nominal=na.ScalarArray(
            dem @ _mass(_logt_uneven) / u.cm**5,
            axes="temperature",
        ),
        distribution=na.ScalarArray(
            np.stack([dem @ _mass(g) for g in grids]) / u.cm**5,
            axes=(axis, "temperature"),
        ),
    )
    assert isinstance(result, na.UncertainScalarArray)
    assert np.allclose(result, expected, rtol=1e-14)


def test_emission_measure_uncertain() -> None:
    """The nominal DEM and each sample of its distribution are weighted alike."""
    axis = na.UncertainScalarArray.axis_distribution
    nominal = 1e27 * np.exp(-np.square((_logt - 6.2) / 0.2) / 2)
    samples = np.stack(
        [
            0.5 * nominal,
            1e27 * np.exp(-np.square((_logt - 6.0) / 0.1) / 2),
            np.full(_logt.size, 2e26),
        ]
    )
    dem = na.FunctionArray(
        inputs=_temperature(_logt),
        outputs=na.UncertainScalarArray(
            nominal=na.ScalarArray(nominal / u.cm**5, axes="temperature"),
            distribution=na.ScalarArray(
                samples / u.cm**5,
                axes=(axis, "temperature"),
            ),
        ),
    )

    result = utu.dem.emission_measure(dem, axis_temperature="temperature")

    mass = _mass(_logt)
    expected = na.UncertainScalarArray(
        nominal=na.ScalarArray(nominal @ mass / u.cm**5, axes="temperature"),
        distribution=na.ScalarArray(
            samples @ mass / u.cm**5,
            axes=(axis, "temperature"),
        ),
    )
    assert isinstance(result, na.UncertainScalarArray)
    assert np.allclose(result, expected, rtol=1e-14)


def test_emission_measure_plowman() -> None:
    """
    The intensities it predicts from the DEM :func:`utu.dem.plowman` finds
    give the reduced :math:`\\chi^2` the inversion reports.
    """
    peaks = np.array([5.7, 5.95, 6.15, 6.3, 6.5, 6.9])
    response = na.FunctionArray(
        inputs=_temperature(_logt),
        outputs=na.ScalarArray(
            ndarray=1e-26
            * np.exp(-np.square((_logt[:, None] - peaks) / 0.12))
            * u.DN
            * u.cm**5
            / u.s,
            axes=("temperature", "channel"),
        ),
    )

    amplitude = na.ScalarArray(np.array([1e27, 3e27, 1e28]), axes="pixel")
    logt = na.ScalarArray(_logt, axes="temperature")
    dem_true = na.FunctionArray(
        inputs=response.inputs,
        outputs=amplitude * np.exp(-np.square((logt - 6.2) / 0.15) / 2) / u.cm**5,
    )
    # the intensities of the functions themselves, as the docstring advises,
    # which the sum times the step integrates for these smooth functions
    intensity = (response.outputs * dem_true.outputs).sum("temperature") * 0.05

    # noise, so that the fit stops short of the data
    rng = np.random.default_rng(0)
    uncertainty = 0.05 * intensity
    noise = na.ScalarArray(rng.normal(size=(6, 3)), axes=("channel", "pixel"))
    intensity = intensity + noise * uncertainty

    dem, chi2 = utu.dem.plowman(
        intensity=intensity,
        uncertainty=uncertainty,
        response=response,
        axis_channel="channel",
        axis_temperature="temperature",
    )

    model = (response.outputs * utu.dem.emission_measure(dem, "temperature")).sum(
        "temperature"
    )
    expected = np.square((intensity - model) / uncertainty).mean("channel")

    assert np.all(chi2 >= 0)
    assert np.allclose(chi2, expected, rtol=1e-6)


@pytest.mark.parametrize(
    argnames="dem,match",
    argvalues=[
        (
            na.FunctionArray(
                inputs=na.ScalarArray(10**_logt * u.K, axes="t"),
                outputs=na.ScalarArray(np.ones(_logt.size) / u.cm**5, axes="t"),
            ),
            "no axis 'temperature'",
        ),
        (_dem(_logt[:1], np.ones(1)), "at least two temperatures"),
        (_dem(_logt[::-1], np.ones(_logt.size)), "must be increasing"),
        (
            na.FunctionArray(
                inputs=na.ScalarArray(
                    np.array([0, 1e5, 1e6]) * u.K, axes="temperature"
                ),
                outputs=na.ScalarArray(np.ones(3) / u.cm**5, axes="temperature"),
            ),
            "must be positive",
        ),
        (
            na.FunctionArray(
                inputs=na.ScalarArray(
                    np.array([-1, 1e5, 1e6]) * u.K, axes="temperature"
                ),
                outputs=na.ScalarArray(np.ones(3) / u.cm**5, axes="temperature"),
            ),
            "must be positive",
        ),
        (_dem(_logt, np.ones(1)), "have 1 temperatures, but its inputs have 31"),
        (_dem(_logt, np.ones(5)), "have 5 temperatures, but its inputs have 31"),
    ],
)
def test_emission_measure_invalid(dem: na.FunctionArray, match: str) -> None:
    """A DEM whose temperatures cannot be integrated over is an error."""
    with pytest.raises(ValueError, match=match):
        utu.dem.emission_measure(dem, axis_temperature="temperature")
