from typing import Any, cast

import astropy.units as u
import named_arrays as na
import numpy as np
import pytest

import utu

_logt = np.round(np.arange(5.5, 7.0 + 1e-9, 0.05), 2)
_peaks = np.array([5.7, 5.95, 6.15, 6.3, 6.5, 6.9])


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
            # infinities and NaNs before factoring
            if not np.all(np.isfinite(amat)):
                break
            low = np.linalg.cholesky(amat)
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


def _plowman(
    intensity: np.ndarray,
    uncertainty: np.ndarray,
    **kwargs: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """
    :func:`utu.dem.plowman` on ``(pixel, channel)`` arrays in DN/s.

    Returns the DEMs, ``(pixel, temperature)``, in inverse centimeters to the
    fifth, and the reduced :math:`\\chi^2` of each pixel.
    """
    unit = u.DN / u.s
    dem, chi2 = utu.dem.plowman(
        intensity=na.ScalarArray(intensity * unit, axes=("pixel", "channel")),
        uncertainty=na.ScalarArray(uncertainty * unit, axes=("pixel", "channel")),
        response=na.FunctionArray(
            inputs=na.ScalarArray(10**_logt * u.K, axes="temperature"),
            outputs=na.ScalarArray(
                _response() * u.DN * u.cm**5 / u.s,
                axes=("temperature", "channel"),
            ),
        ),
        axis_channel="channel",
        axis_temperature="temperature",
        **kwargs,
    )
    found = u.Quantity(dem.outputs.ndarray_aligned(("pixel", "temperature")))
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
    assert np.all(np.abs(peak - center[:, 0]) < 0.3)
    assert np.allclose(em_found, em_true, rtol=0.5)


def test_plowman_axes() -> None:
    """Every axis but the channel is a separate pixel, and they come back."""
    intensity, uncertainty = _observation(num=12)
    unit = u.DN / u.s

    dem, chi2 = utu.dem.plowman(
        intensity=na.ScalarArray(
            intensity.reshape(3, 4, 6) * unit,
            axes=("y", "x", "channel"),
        ),
        # one uncertainty per channel, broadcast against every pixel
        uncertainty=na.ScalarArray(uncertainty.mean(0) * unit, axes="channel"),
        response=na.FunctionArray(
            inputs=na.ScalarArray(10**_logt * u.K, axes="temperature"),
            outputs=na.ScalarArray(
                _response().T * u.DN * u.cm**5 / u.s,
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


def test_plowman_failure() -> None:
    """
    A pixel whose uncertainty is not a number fails, as in the reference.

    Its first step cannot be taken, so its :math:`\\chi^2` is left at -1
    and its DEM at the exponential of a NaN initial guess. The other pixels
    are unaffected.
    """
    intensity, uncertainty = _observation(num=3)
    uncertainty[1, 2] = np.nan

    dem, chi2 = _plowman(intensity, uncertainty)
    dem_expected, chi2_expected = _reference(intensity, uncertainty, _logt, _response())

    assert chi2[1] == -1
    assert np.all(np.isnan(dem[1]))
    assert np.all(chi2[[0, 2]] > 0)
    assert np.allclose(dem, dem_expected, rtol=1e-6, atol=0, equal_nan=True)
    assert np.allclose(chi2, chi2_expected, rtol=1e-6, atol=0)


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


def test_plowman_no_channel() -> None:
    """An intensity without the axis of the channels is an error."""
    _, uncertainty = _observation(num=2)
    with pytest.raises(ValueError, match="no axis"):
        utu.dem.plowman(
            intensity=na.ScalarArray(np.ones(2), axes="pixel"),
            uncertainty=na.ScalarArray(uncertainty[:, 0], axes="pixel"),
            response=na.FunctionArray(
                inputs=na.ScalarArray(10**_logt, axes="temperature"),
                outputs=na.ScalarArray(_response(), axes=("temperature", "channel")),
            ),
            axis_channel="channel",
            axis_temperature="temperature",
        )


@pytest.mark.parametrize(
    argnames="floor",
    argvalues=[0.01, 0.01 * u.dimensionless_unscaled],
)
def test_plowman_unitless(floor: float | u.Quantity) -> None:
    """Plain numbers go in, with a plain floor, and plain numbers come out."""
    intensity, uncertainty = _observation(num=2)

    dem, chi2 = utu.dem.plowman(
        intensity=na.ScalarArray(intensity, axes=("pixel", "channel")),
        uncertainty=na.ScalarArray(uncertainty, axes=("pixel", "channel")),
        response=na.FunctionArray(
            inputs=na.ScalarArray(10**_logt, axes="temperature"),
            outputs=na.ScalarArray(_response(), axes=("temperature", "channel")),
        ),
        axis_channel="channel",
        axis_temperature="temperature",
        floor=floor,
    )

    found = dem.outputs.ndarray_aligned(("pixel", "temperature"))
    expected, _ = _plowman(intensity, uncertainty)
    assert not isinstance(found, u.Quantity)
    assert np.allclose(found, expected)


@pytest.mark.parametrize(
    argnames="intensity_unit,floor",
    argvalues=[
        # the default floor is in DN/s, which plain numbers are not
        (None, 0.01 * u.DN / u.s),
        (u.DN / u.s, 0.01),
        (u.DN / u.s, 0.01 * u.cm),
    ],
)
def test_plowman_floor_units(
    intensity_unit: u.UnitBase | None,
    floor: float | u.Quantity,
) -> None:
    """A floor in units other than those of the intensities is an error."""
    intensity, uncertainty = _observation(num=2)
    unit = 1 if intensity_unit is None else intensity_unit
    with pytest.raises(u.UnitConversionError):
        utu.dem.plowman(
            intensity=na.ScalarArray(intensity * unit, axes=("pixel", "channel")),
            uncertainty=na.ScalarArray(uncertainty * unit, axes=("pixel", "channel")),
            response=na.FunctionArray(
                inputs=na.ScalarArray(10**_logt, axes="temperature"),
                outputs=na.ScalarArray(_response(), axes=("temperature", "channel")),
            ),
            axis_channel="channel",
            axis_temperature="temperature",
            floor=floor,
        )


@pytest.mark.parametrize(
    argnames="response",
    argvalues=[
        # five channels for six intensities
        na.FunctionArray(
            inputs=na.ScalarArray(10**_logt, axes="temperature"),
            outputs=na.ScalarArray(_response()[:, :5], axes=("temperature", "channel")),
        ),
        # an extra axis on the response
        na.FunctionArray(
            inputs=na.ScalarArray(10**_logt, axes="temperature"),
            outputs=na.ScalarArray(
                _response()[None],
                axes=("extra", "temperature", "channel"),
            ),
        ),
        # temperatures which vary along the channels
        na.FunctionArray(
            inputs=na.ScalarArray(
                10 ** (_logt[:, None] + np.zeros(6)),
                axes=("temperature", "channel"),
            ),
            outputs=na.ScalarArray(_response(), axes=("temperature", "channel")),
        ),
    ],
)
def test_plowman_invalid(response: na.FunctionArray) -> None:
    intensity, uncertainty = _observation(num=2)
    with pytest.raises(ValueError):
        utu.dem.plowman(
            intensity=na.ScalarArray(intensity, axes=("pixel", "channel")),
            uncertainty=na.ScalarArray(uncertainty, axes=("pixel", "channel")),
            response=response,
            axis_channel="channel",
            axis_temperature="temperature",
        )
