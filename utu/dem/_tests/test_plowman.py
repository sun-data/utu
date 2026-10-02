from typing import Any, cast

import astropy.units as u
import named_arrays as na
import numpy as np
import pytest

import utu
from utu.dem import _plowman_kernel
from utu.dem._plowman import _matrices

_logt = np.round(np.arange(5.5, 7.0 + 1e-9, 0.05), 2)
_peaks = np.array([5.7, 5.95, 6.15, 6.3, 6.5, 6.9])

_unit = u.DN / u.s
"""The units of the intensities of these tests."""

_unit_response = u.DN * u.cm**5 / u.s
"""The units of the responses of these tests."""


def _response() -> np.ndarray:
    """
    Six responses shaped like those of AIA, ``(temperature, channel)``.

    Gaussians in log temperature at different peaks, with a cool secondary
    peak on the hottest channel, as 94 and 131 angstroms have. The
    inversion does not care what the responses are, only that the
    reference and the kernel are given the same ones.
    """
    result = np.exp(-(((_logt[:, None] - _peaks) / 0.12) ** 2))
    result[:, -1] += 0.2 * np.exp(-(((_logt - 6.0) / 0.1) ** 2))
    return 1e-26 * result


def _observation(
    num: int,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Intensities and uncertainties, ``(pixel, channel)``, of random DEMs.

    The random DEMs of Section 3.1 of the paper: each a sum of five
    log-normals, with total emission measures from 5e27 to 5e28 per square
    centimeter, widths from 0.05 to 0.15 dex, and peaks from 6.0 to 7.0, with
    Poisson noise for a gain of one photon per DN.
    """
    rng = np.random.default_rng(seed)
    total = 10 ** rng.uniform(np.log10(5e27), np.log10(5e28), (num, 5, 1))
    width = rng.uniform(0.05, 0.15, (num, 5, 1))
    center = rng.uniform(6.0, 7.0, (num, 5, 1))
    norm = total / (np.sqrt(2 * np.pi) * width)
    dem = (norm * np.exp(-0.5 * ((_logt - center) / width) ** 2)).sum(1)
    intensity = np.trapezoid(dem[:, :, None] * _response(), _logt, axis=1)
    uncertainty = np.sqrt(intensity + 1)
    intensity = intensity + rng.normal(size=intensity.shape) * uncertainty
    return intensity, uncertainty


def _reference(
    data: np.ndarray,
    errors: np.ndarray,
    logt: np.ndarray,
    tresps: np.ndarray,
    kmax: int = 100,
    kcon: int = 5,
    steps: tuple[float, float] = (0.1, 0.5),
    drv_con: float = 8.0,
    chi2_th: float = 1.0,
    tol: float = 0.1,
    floor: float = 1.0e-2,
) -> tuple[np.ndarray, np.ndarray]:
    """
    ``simple_reg_dem`` from EMToolKit, line for line, with the exposure
    times set to one and :mod:`numpy.linalg` in place of
    :mod:`scipy.linalg`. ``data`` and ``errors`` are ``(pixel, channel)``.

    A factorization which fails is a step which fails, and the reference
    stops the iteration there, keeping the last good step.
    """
    nt, nd = tresps.shape
    nt_ones = np.ones(nt)
    dT = logt[1:nt] - logt[0 : nt - 1]
    dTleft, dTright = np.diag(np.hstack([dT, 0])), np.diag(np.hstack([0, dT]))
    idTleft = np.diag(np.hstack([1.0 / dT, 0]))
    idTright = np.diag(np.hstack([0, 1.0 / dT]))
    Bij = (
        (dTleft + dTright) * 2.0
        + np.roll(dTright, -1, axis=0)
        + np.roll(dTleft, 1, axis=0)
    ) / 6.0
    Rij = np.matmul(tresps.T, Bij)
    Dij = (
        idTleft + idTright - np.roll(idTright, -1, axis=0) - np.roll(idTleft, 1, axis=0)
    )
    regmat = Dij * nd / (drv_con**2 * (logt[nt - 1] - logt[0]))
    rvec = np.sum(Rij, axis=1)

    dems = np.zeros((len(data), nt))
    chi2 = np.zeros(len(data)) - 1.0
    for p in range(len(data)):
        err = errors[p]
        dat0 = np.clip(data[p], 0.0, None)
        s = np.log(
            np.sum(rvec * (np.clip(dat0, floor, None) / err**2))
            / np.sum((rvec / err) ** 2)
            / nt_ones
        )
        for k in range(kmax):
            dat = (dat0 - np.matmul(Rij, ((1 - s) * np.exp(s)))) / err
            mmat = Rij * np.outer(1.0 / err, np.exp(s))
            amat = np.matmul(mmat.T, mmat) + regmat
            # where scipy.linalg.cho_factor raises, since it checks for
            # infinities and NaNs before factoring, and the reference
            # catches whatever it raises
            if not np.all(np.isfinite(amat)):
                break
            try:
                low = np.linalg.cholesky(amat)
            except np.linalg.LinAlgError:
                break
            c2p = np.mean((dat0 - np.dot(Rij, np.exp(s))) ** 2 / err**2)
            solution = np.linalg.solve(low.T, np.linalg.solve(low, mmat.T @ dat))
            deltas = solution - s
            deltas *= np.clip(np.max(np.abs(deltas)), None, 0.5 / steps[0]) / np.max(
                np.abs(deltas)
            )
            ds = 1 - 2 * (c2p < chi2_th)
            c20 = np.mean(
                (dat0 - np.dot(Rij, np.exp(s + deltas * ds * steps[0]))) ** 2 / err**2
            )
            c21 = np.mean(
                (dat0 - np.dot(Rij, np.exp(s + deltas * ds * steps[1]))) ** 2 / err**2
            )
            interp_step = (steps[0] * (c21 - chi2_th) + steps[1] * (chi2_th - c20)) / (
                c21 - c20
            )
            s = s + deltas * ds * np.clip(interp_step, steps[0], steps[1])
            chi2[p] = np.mean((dat0 - np.dot(Rij, np.exp(s))) ** 2 / err**2)
            if (ds * (c2p - c20) / steps[0] < tol) * (k > kcon) or np.abs(
                chi2[p] - chi2_th
            ) < tol:
                break
        dems[p] = np.exp(s)
    return dems, chi2


def _arguments(**overrides: Any) -> dict[str, Any]:
    """
    The arguments of :func:`utu.dem.plowman` for two random pixels, with any
    of them replaced by ``overrides``.
    """
    intensity, uncertainty = _observation(num=2)
    arguments = dict(
        intensity=na.ScalarArray(intensity * _unit, axes=("pixel", "channel")),
        uncertainty=na.ScalarArray(uncertainty * _unit, axes=("pixel", "channel")),
        response=na.FunctionArray(
            inputs=na.ScalarArray(10**_logt * u.K, axes="temperature"),
            outputs=na.ScalarArray(
                _response() * _unit_response,
                axes=("temperature", "channel"),
            ),
        ),
        axis_channel="channel",
        axis_temperature="temperature",
    )
    return arguments | overrides


def _plowman(
    intensity: np.ndarray,
    uncertainty: np.ndarray,
    tresp: np.ndarray | None = None,
    **kwargs: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """
    :func:`utu.dem.plowman` on ``(pixel, channel)`` arrays in DN/s, with the
    responses ``tresp``, or those of :func:`_response`.

    Returns the DEMs, ``(pixel, temperature)``, in inverse centimeters to the
    fifth, and the reduced :math:`\\chi^2` of each pixel.
    """
    if tresp is None:
        tresp = _response()
    dem, chi2 = utu.dem.plowman(
        intensity=na.ScalarArray(intensity * _unit, axes=("pixel", "channel")),
        uncertainty=na.ScalarArray(uncertainty * _unit, axes=("pixel", "channel")),
        response=na.FunctionArray(
            inputs=na.ScalarArray(10**_logt * u.K, axes="temperature"),
            outputs=na.ScalarArray(
                tresp * _unit_response,
                axes=("temperature", "channel"),
            ),
        ),
        axis_channel="channel",
        axis_temperature="temperature",
        **kwargs,
    )
    outputs = cast(na.ScalarArray, dem.outputs)
    found = u.Quantity(outputs.ndarray_aligned(("pixel", "temperature")))
    chi2 = cast(na.ScalarArray, chi2)
    return cast(np.ndarray, found.to_value(u.cm**-5)), chi2.ndarray_aligned(("pixel",))


@pytest.mark.parametrize(
    argnames="kwargs",
    argvalues=[
        dict(),
        dict(smoothness=4, steps=(0.1, 0.75), iterations_min=15),
        dict(chi2_target=2, tolerance=0.01),
    ],
)
def test_plowman_reference(kwargs: dict) -> None:
    """
    The kernel reproduces the reference, to rounding error.

    Not to the last bit, since the two factor their matrices in different
    orders and the iteration amplifies the difference, but to about
    ten to the minus eleven relative on these DEMs, against a tolerance of
    ten to the minus six.
    """
    intensity, uncertainty = _observation(num=40)

    dem, chi2 = _plowman(intensity, uncertainty, **kwargs)

    names = dict(
        smoothness="drv_con",
        steps="steps",
        iterations_min="kcon",
        chi2_target="chi2_th",
        tolerance="tol",
    )
    kwargs_reference = {names[key]: value for key, value in kwargs.items()}
    dem_expected, chi2_expected = _reference(
        intensity,
        uncertainty,
        _logt,
        _response(),
        **kwargs_reference,
    )

    assert np.allclose(dem, dem_expected, rtol=1e-6, atol=0)
    assert np.allclose(chi2, chi2_expected, rtol=1e-6, atol=0)


def test_plowman_recovers() -> None:
    """
    Without noise, the DEM is positive, fits the data, and lands near the truth.

    Near, and no nearer. Six channels do not determine a DEM, which is the
    whole difficulty, and these six leave gaps in temperature between their
    peaks where the data say nothing and the regularization decides. What
    the method promises is a positive DEM which fits the data to the target
    :math:`\\chi^2`. Where the plasma is and how much of it there is follow
    only roughly: within a quarter of a decade, and within 40 percent, for
    these DEMs and these responses.
    """
    rng = np.random.default_rng(1)
    center = rng.uniform(6.0, 6.8, (20, 1))
    dem_true = 1e27 / 0.1 * np.exp(-0.5 * ((_logt - center) / 0.1) ** 2)
    intensity = np.asarray(
        np.trapezoid(dem_true[:, :, None] * _response(), _logt, axis=1)
    )
    uncertainty = 0.05 * intensity + 1

    found, chi2 = _plowman(intensity, uncertainty)

    em_found = np.trapezoid(found, _logt, axis=-1)
    em_true = np.trapezoid(dem_true, _logt, axis=-1)
    peak = _logt[np.argmax(found, axis=-1)]
    assert np.all(found > 0)
    assert np.all(chi2 < 1.1)
    assert np.all(np.abs(peak - center[:, 0]) < 0.25)
    assert np.allclose(em_found, em_true, rtol=0.4)


def test_plowman_axes() -> None:
    """Every axis but the channel is a separate pixel, and they come back."""
    intensity, uncertainty = _observation(num=12)

    dem, chi2 = utu.dem.plowman(
        intensity=na.ScalarArray(
            intensity.reshape(3, 4, 6) * _unit,
            axes=("y", "x", "channel"),
        ),
        # one uncertainty per channel, broadcast against every pixel
        uncertainty=na.ScalarArray(uncertainty.mean(0) * _unit, axes="channel"),
        response=na.FunctionArray(
            inputs=na.ScalarArray(10**_logt * u.K, axes="temperature"),
            outputs=na.ScalarArray(
                _response().T * _unit_response,
                axes=("channel", "temperature"),
            ),
        ),
        axis_channel="channel",
        axis_temperature="temperature",
    )

    assert dem.outputs.shape == dict(y=3, x=4, temperature=_logt.size)
    assert dem.inputs.shape == dict(temperature=_logt.size)
    assert na.unit(dem.outputs) == u.cm**-5
    assert chi2.shape == dict(y=3, x=4)


@pytest.mark.parametrize(
    argnames="factor",
    argvalues=[np.nan, 0, -1],
)
def test_plowman_failure(factor: float) -> None:
    """
    A pixel whose uncertainty is NaN, zero, or negative in one channel fails:
    its :math:`\\chi^2` is -1 and its DEM is NaN. The other pixels are
    unaffected.

    The reference fails on a NaN or zero uncertainty the same way, but takes
    a negative one as positive.
    """
    intensity, uncertainty = _observation(num=3)
    uncertainty[1, 2] *= factor

    dem, chi2 = _plowman(intensity, uncertainty)

    good = [0, 2]
    dem_expected, chi2_expected = _reference(
        intensity[good],
        uncertainty[good],
        _logt,
        _response(),
    )
    assert chi2[1] == -1
    assert np.all(np.isnan(dem[1]))
    assert np.allclose(dem[good], dem_expected, rtol=1e-6, atol=0)
    assert np.allclose(chi2[good], chi2_expected, rtol=1e-6, atol=0)
    if not factor < 0:
        with np.errstate(divide="ignore", invalid="ignore"):
            dem_reference, chi2_reference = _reference(
                intensity,
                uncertainty,
                _logt,
                _response(),
            )
        assert chi2_reference[1] == -1
        assert np.all(np.isnan(dem_reference[1]))


def test_plowman_negative_response() -> None:
    """
    Negative responses put the logarithm of a negative number into the
    initial guess, which fails every pixel, as in the reference.

    Compiled with fast math, ``exp(log(x))`` is simplified to ``x``, which
    once turned that NaN back into a number for the pixels to iterate on.
    """
    intensity, uncertainty = _observation(num=4)

    dem, chi2 = _plowman(intensity, uncertainty, tresp=-_response())

    with np.errstate(invalid="ignore"):
        dem_expected, chi2_expected = _reference(
            intensity,
            uncertainty,
            _logt,
            -_response(),
        )
    assert np.all(chi2 == -1)
    assert np.all(np.isnan(dem))
    assert np.array_equal(chi2, chi2_expected)
    assert np.array_equal(dem, dem_expected, equal_nan=True)


def test_plowman_singular() -> None:
    """
    A pixel whose linear system is singular in floating point fails at that
    step, as in the reference, keeping the last good step.

    With an uncertainty of :math:`10^{-12}` of the signal, the regularization
    is too small to register against the data, so the first system is
    singular and the DEM is the initial guess.
    """
    intensity, uncertainty = _observation(num=4)
    uncertainty = 1e-12 * np.abs(intensity)

    dem, chi2 = _plowman(intensity, uncertainty)

    dem_expected, chi2_expected = _reference(
        intensity,
        uncertainty,
        _logt,
        _response(),
    )
    assert np.all(chi2 == -1)
    assert np.array_equal(chi2, chi2_expected)
    assert np.allclose(dem, dem_expected, rtol=1e-12, atol=0)


def test_plowman_cholesky() -> None:
    """
    The factorization fails on a finite matrix which is not positive
    definite, where :func:`numpy.linalg.cholesky` raises, and agrees with it
    on one which is.
    """
    definite = np.array([[2.0, 1.0], [1.0, 2.0]])
    indefinite = np.array([[1.0, 2.0], [2.0, 1.0]])

    with pytest.raises(np.linalg.LinAlgError):
        np.linalg.cholesky(indefinite)
    assert not _plowman_kernel._cholesky(indefinite.copy())

    factor = definite.copy()
    assert _plowman_kernel._cholesky(factor)
    assert np.allclose(np.tril(factor), np.linalg.cholesky(definite))


def test_plowman_chunks() -> None:
    """
    Pixels which share a chunk, and so its work arrays, give the same results
    as pixels which have one each, including those after a pixel which fails.

    The kernel reuses the work arrays of a chunk from one pixel to the next,
    so a step which read one before writing it would leak from a pixel into
    its neighbour.
    """
    num = 12
    intensity, uncertainty = _observation(num=num)
    uncertainty[5, 2] = np.nan
    uncertainty[6] *= 1e-12
    rmat, regmat, rvec = _matrices(_logt, _response(), smoothness=8)

    results = []
    for num_chunks in (1, num):
        dems = np.zeros((num, _logt.size))
        chi2 = np.full(num, -1.0)
        _plowman_kernel.plowman(
            intensity,
            uncertainty,
            rmat,
            regmat,
            rvec,
            100,
            5,
            0.1,
            0.5,
            1.0,
            0.1,
            0.01,
            num_chunks,
            dems,
            chi2,
        )
        results.append((dems, chi2))

    (dems_shared, chi2_shared), (dems_own, chi2_own) = results
    assert np.array_equal(dems_shared, dems_own, equal_nan=True)
    assert np.array_equal(chi2_shared, chi2_own)


def test_plowman_floor() -> None:
    """
    The floor of the initial guess is converted to the units of the
    intensities, and used as the reference uses its own.

    Half the channels of these pixels see less than one DN per second, so a
    floor of one DN per second moves where the iteration starts, and with it
    the result.
    """
    intensity, uncertainty = _observation(num=20)
    intensity[:, :3] = 0.5

    dem, chi2 = _plowman(intensity, uncertainty, floor=60 * u.DN / u.min)
    dem_default, _ = _plowman(intensity, uncertainty)
    dem_expected, chi2_expected = _reference(
        intensity,
        uncertainty,
        _logt,
        _response(),
        floor=1,
    )

    assert np.allclose(dem, dem_expected, rtol=1e-6, atol=0)
    assert np.allclose(chi2, chi2_expected, rtol=1e-6, atol=0)
    assert not np.allclose(dem, dem_default, rtol=1e-3, atol=0)


def test_plowman_units() -> None:
    """
    Intensities and responses per pixel give the same DEM as those without,
    the default floor included, which is 0.01 in the units of the
    intensities whatever those are.
    """
    intensity, uncertainty = _observation(num=4)
    intensity[:, :3] = 0.005
    per_pixel = 1 / u.pix

    dem, chi2 = utu.dem.plowman(
        **_arguments(
            intensity=na.ScalarArray(
                intensity * _unit * per_pixel,
                axes=("pixel", "channel"),
            ),
            uncertainty=na.ScalarArray(
                uncertainty * _unit * per_pixel,
                axes=("pixel", "channel"),
            ),
            response=na.FunctionArray(
                inputs=na.ScalarArray(10**_logt * u.K, axes="temperature"),
                outputs=na.ScalarArray(
                    _response() * _unit_response * per_pixel,
                    axes=("temperature", "channel"),
                ),
            ),
        )
    )

    dem_expected, chi2_expected = _plowman(intensity, uncertainty)
    outputs = cast(na.ScalarArray, dem.outputs)
    found = u.Quantity(outputs.ndarray_aligned(("pixel", "temperature")))
    chi2 = cast(na.ScalarArray, chi2)
    assert np.array_equal(found.to_value(u.cm**-5), dem_expected)
    assert np.array_equal(chi2.ndarray_aligned(("pixel",)), chi2_expected)


@pytest.mark.parametrize(
    argnames="floor",
    argvalues=[None, 0.01, 0.01 * u.dimensionless_unscaled],
)
def test_plowman_unitless(floor: None | float | u.Quantity) -> None:
    """Plain numbers go in, with a plain floor, and plain numbers come out."""
    intensity, uncertainty = _observation(num=2)

    dem, chi2 = utu.dem.plowman(
        **_arguments(
            intensity=na.ScalarArray(intensity, axes=("pixel", "channel")),
            uncertainty=na.ScalarArray(uncertainty, axes=("pixel", "channel")),
            response=na.FunctionArray(
                inputs=na.ScalarArray(10**_logt * u.K, axes="temperature"),
                outputs=na.ScalarArray(_response(), axes=("temperature", "channel")),
            ),
            floor=floor,
        )
    )

    outputs = cast(na.ScalarArray, dem.outputs)
    found = outputs.ndarray_aligned(("pixel", "temperature"))
    expected, _ = _plowman(intensity, uncertainty)
    assert not isinstance(found, u.Quantity)
    assert np.allclose(found, expected)


def test_plowman_uncertain() -> None:
    """
    An uncertain intensity gives an uncertain DEM and :math:`\\chi^2`, whose
    nominal values are those of the nominal intensity, and whose samples are
    those of the samples of the intensity.
    """
    num_sample = 4
    axis = na.UncertainScalarArray.axis_distribution
    intensity, uncertainty = _observation(num=3)
    rng = np.random.default_rng(2)
    samples = intensity + rng.normal(size=(num_sample, *intensity.shape)) * uncertainty

    dem, chi2 = utu.dem.plowman(
        **_arguments(
            intensity=na.UncertainScalarArray(
                nominal=na.ScalarArray(intensity * _unit, axes=("pixel", "channel")),
                distribution=na.ScalarArray(
                    samples * _unit,
                    axes=(axis, "pixel", "channel"),
                ),
            ),
            uncertainty=na.ScalarArray(uncertainty * _unit, axes=("pixel", "channel")),
        )
    )

    dem_nominal, chi2_nominal = _plowman(intensity, uncertainty)
    dem_distribution, chi2_distribution = _plowman(
        samples.reshape(-1, _peaks.size),
        np.broadcast_to(uncertainty, samples.shape).reshape(-1, _peaks.size),
    )
    dem_expected = na.UncertainScalarArray(
        nominal=na.ScalarArray(dem_nominal / u.cm**5, axes=("pixel", "temperature")),
        distribution=na.ScalarArray(
            dem_distribution.reshape(num_sample, -1, _logt.size) / u.cm**5,
            axes=(axis, "pixel", "temperature"),
        ),
    )
    chi2_expected = na.UncertainScalarArray(
        nominal=na.ScalarArray(chi2_nominal, axes="pixel"),
        distribution=na.ScalarArray(
            chi2_distribution.reshape(num_sample, -1),
            axes=(axis, "pixel"),
        ),
    )
    assert isinstance(dem.outputs, na.UncertainScalarArray)
    assert np.all(dem.outputs == dem_expected)
    assert np.all(chi2 == chi2_expected)


_intensity, _uncertainty = _observation(num=2)
_temperature = u.Quantity(10**_logt, u.K)


def _function(
    temperature: u.Quantity | np.ndarray,
    response: np.ndarray,
    axes: tuple[str, ...] = ("temperature", "channel"),
    axes_temperature: tuple[str, ...] = ("temperature",),
) -> na.FunctionArray:
    """A response for :func:`utu.dem.plowman` from plain arrays."""
    return na.FunctionArray(
        inputs=na.ScalarArray(temperature, axes=axes_temperature),
        outputs=na.ScalarArray(response * _unit_response, axes=axes),
    )


@pytest.mark.parametrize(
    argnames="overrides,match",
    argvalues=[
        (
            dict(response=_function(_temperature, _response()[:, :5])),
            "channels",
        ),
        (
            dict(
                response=_function(
                    _temperature,
                    _response()[None],
                    axes=("extra", "temperature", "channel"),
                ),
            ),
            "no others",
        ),
        (
            dict(
                response=_function(
                    _temperature[:, None] + np.zeros(6) * u.K,
                    _response(),
                    axes_temperature=("temperature", "channel"),
                ),
            ),
            "alone",
        ),
        (
            dict(response=_function(_temperature[:30], _response())),
            "30 temperatures but 31",
        ),
        (
            dict(response=_function(_temperature[::-1], _response())),
            "increasing",
        ),
        (
            dict(response=_function(_temperature[:1], _response()[:1])),
            "at least two",
        ),
        (
            dict(
                response=_function(
                    u.Quantity(np.concatenate([[0], 10 ** _logt[1:]]), u.K),
                    _response(),
                ),
            ),
            "positive",
        ),
        # temperatures given as log10 T, without units
        (
            dict(response=_function(_logt, _response())),
            "not convertible",
        ),
        # an intensity without channels, which the uncertainty has
        (
            dict(intensity=na.ScalarArray(_intensity[:, 0] * _unit, axes="pixel")),
            "no axis 'channel'",
        ),
        (
            dict(
                intensity=na.ScalarArray(
                    _intensity * _unit,
                    axes=("temperature", "channel"),
                ),
                uncertainty=na.ScalarArray(
                    _uncertainty * _unit,
                    axes=("temperature", "channel"),
                ),
            ),
            "must not have the axis 'temperature'",
        ),
        (dict(smoothness=0), "`smoothness` must be positive"),
        (dict(floor=0 * _unit), "`floor` must be positive"),
        # a floor, or an uncertainty, in units other than the intensity's
        (dict(floor=0.01), "not convertible"),
        (dict(floor=0.01 * u.cm), "not convertible"),
        (
            dict(uncertainty=na.ScalarArray(_uncertainty, axes=("pixel", "channel"))),
            "not convertible",
        ),
        # plain numbers, with the default floor of the units of another test
        (
            dict(
                intensity=na.ScalarArray(_intensity, axes=("pixel", "channel")),
                uncertainty=na.ScalarArray(_uncertainty, axes=("pixel", "channel")),
                floor=0.01 * _unit,
            ),
            "not convertible",
        ),
    ],
)
def test_plowman_invalid(overrides: dict[str, Any], match: str) -> None:
    """Arguments which cannot be inverted are errors, before anything is."""
    with pytest.raises(ValueError, match=match):
        utu.dem.plowman(**_arguments(**overrides))
