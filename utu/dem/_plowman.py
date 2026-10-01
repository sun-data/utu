"""The regularized DEM inversion of Plowman & Caspi (2020)."""

import astropy.units as u
import named_arrays as na
import numba
import numpy as np

from . import _plowman_kernel

__all__ = [
    "plowman",
]


def _matrices(
    logt: np.ndarray,
    response: np.ndarray,
    smoothness: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    The matrices of the inversion, from the temperature grid and the responses.

    ``response`` is ``(temperature, channel)``. Returns the matrix mapping
    the coefficients of the DEM to the data, ``(channel, temperature)``, the
    regularization matrix, ``(temperature, temperature)``, and the sum of
    the first over temperature, ``(channel,)``.

    The DEM and the responses are both taken to be piecewise linear between
    the temperatures of the grid, so the mapping is the response times the
    mass matrix of those triangle functions, and the regularization is their
    stiffness matrix. Both are tridiagonal, and both are written with the
    operations of the reference in the same order, since a rounding error in
    a matrix is carried through every step of every pixel.
    """
    num_temperature, num_channel = response.shape
    dt = logt[1:] - logt[:-1]
    left = np.concatenate([dt, [0]])
    right = np.concatenate([[0], dt])

    mass = np.diag((left + right) * 2.0) / 6.0
    mass += np.diag(dt, k=1) / 6.0 + np.diag(dt, k=-1) / 6.0

    inverse_left = np.concatenate([1.0 / dt, [0]])
    inverse_right = np.concatenate([[0], 1.0 / dt])
    stiffness = np.diag(inverse_left + inverse_right)
    stiffness -= np.diag(1.0 / dt, k=1) + np.diag(1.0 / dt, k=-1)

    rmat = np.matmul(response.T, mass)
    span = logt[num_temperature - 1] - logt[0]
    regmat = stiffness * num_channel / (smoothness**2 * span)
    rvec = np.sum(rmat, axis=1)

    return np.ascontiguousarray(rmat), np.ascontiguousarray(regmat), rvec


def plowman(
    intensity: u.Quantity | na.AbstractScalar,
    uncertainty: u.Quantity | na.AbstractScalar,
    response: na.FunctionArray[na.AbstractScalar, na.AbstractScalar],
    axis_channel: str,
    axis_temperature: str,
    smoothness: float = 8,
    chi2_target: float = 1,
    tolerance: float = 0.1,
    steps: tuple[float, float] = (0.1, 0.5),
    iterations_max: int = 100,
    iterations_min: int = 5,
) -> tuple[na.FunctionArray[na.ScalarArray, na.ScalarArray], na.ScalarArray]:
    r"""
    Invert intensities for a differential emission measure (DEM), the way
    Plowman & Caspi (2020) do.

    The DEM is piecewise linear in :math:`\log_{10} T` between the
    temperatures of ``response``, and is found as the exponential of a
    piecewise linear function, which makes it positive everywhere without a
    constraint. The fit is regularized by the square of the derivative of
    the logarithm of the DEM, which penalizes changes of more than about
    ``smoothness`` e-foldings per decade of temperature, and is stopped once
    the reduced :math:`\chi^2` reaches ``chi2_target`` rather than driven
    below it.

    Each pixel is independent, and they are inverted in parallel by a
    compiled kernel which reproduces the reference implementation to
    rounding error. See the notes below.

    Parameters
    ----------
    intensity
        The observed intensity in each channel. Every axis other than
        ``axis_channel`` is a separate inversion.
    uncertainty
        The one-sigma uncertainty of ``intensity``, in the same units.
    response
        The temperature response of each channel: its inputs are the
        temperatures along ``axis_temperature``, and its outputs are the
        responses along ``axis_temperature`` and ``axis_channel``, in units
        of ``intensity`` per unit emission measure. The channels must be in
        the same order as those of ``intensity``.
    axis_channel
        The name of the axis along the channels of ``intensity`` and
        ``response``.
    axis_temperature
        The name of the axis along the temperatures of ``response``.
    smoothness
        The number of e-foldings per decade of temperature which the DEM may
        change by before the regularization starts to resist, called
        :math:`\delta_0` (and ``drv_con``) in the paper. Smaller is smoother.
        The paper finds that anything from 4 to 16 fits AIA data.
    chi2_target
        The reduced :math:`\chi^2` the iteration aims for.
    tolerance
        How close to ``chi2_target`` is close enough, and how little
        improvement per unit step counts as having stalled.
    steps
        The small and the large fraction of the way to the linearized
        solution to try at each step. The step taken is interpolated between
        them to land on ``chi2_target``.
    iterations_max
        The most steps to take for any one pixel.
    iterations_min
        The number of steps to take before giving up on a pixel whose
        :math:`\chi^2` has stalled.

    Returns
    -------
    dem
        The DEM at each temperature of ``response``, per unit
        :math:`\log_{10} T`, in units of ``intensity`` over the units of the
        outputs of ``response``: :math:`\mathrm{cm^{-5}}` for responses in
        :math:`\mathrm{DN\,cm^5\,s^{-1}\,pix^{-1}}` and intensities in
        :math:`\mathrm{DN\,s^{-1}\,pix^{-1}}`.
    chi2
        The reduced :math:`\chi^2` of each pixel, or :math:`-1` where the
        first step failed (a NaN or non-positive uncertainty, for instance),
        in which case the DEM is NaN.

    Notes
    -----
    This is ``simple_reg_dem`` from the appendix of `Plowman & Caspi (2020)
    <https://doi.org/10.3847/1538-4357/abc260>`_, as distributed in
    `EMToolKit <https://github.com/jeplowman/EMToolKit>`_. Its defaults are
    those of the code listing
    in that appendix, which are not the ones its text describes: the text
    gives 15 initial steps, steps of 0.1 and 0.75, a smoothness of 4, and a
    tolerance of :math:`10^{-4}`, and describes choosing whichever of the two
    trial steps has the lower :math:`\chi^2`, whereas the code interpolates
    between them. The code is what produced the published results, and what
    is reproduced here.

    Against the reference, the DEMs agree to about :math:`10^{-11}` relative
    on AIA data and on the random test DEMs of the paper. They do not agree
    to the last bit, and cannot: the regularization matrix is singular on
    its own (it does not penalize a constant), and where the data say little
    the linear systems are close to it, so the rounding errors of any two
    implementations of the Cholesky factorization grow by orders of
    magnitude over the iteration. A port of the reference with
    :mod:`numpy.linalg` in place of :mod:`scipy.linalg` disagrees with it by
    as much. The worst cases are pixels with no signal, at about
    :math:`10^{-6}`, and pixels the model cannot fit at all, whose
    :math:`\chi^2` stays far above the target and whose DEM is meaningless
    in either implementation.

    Negative intensities are set to zero before the fit, as in the
    reference. One more detail of the reference is kept although it depends
    on the units: the initial guess, a flat DEM fit by least squares, floors
    the intensities at 0.01 in the units of ``intensity``. The floor changes
    the result only for pixels with almost no signal, and only through where
    the iteration starts.

    The kernel takes about 20 to 30 microseconds per pixel on one core, some
    40 times faster than the reference, and runs on every core: a whole AIA
    image of :math:`4096^2` pixels takes about 12 seconds on 24 cores,
    against about four hours for the reference. The first call compiles
    the kernel, which takes a few seconds, and caches it to disk for later
    sessions.

    Examples
    --------
    Six channels whose responses peak at different temperatures, observing a
    plasma whose DEM peaks near 2 MK, with the uncertainty of a second of
    photon counting, and the DEM recovered from them.

    .. jupyter-execute::

        import astropy.units as u
        import matplotlib.pyplot as plt
        import named_arrays as na
        import numpy as np
        import utu

        temperature = 10 ** na.linspace(5.5, 7, axis="temperature", num=31) * u.K
        logt = np.log10(temperature.value)

        # Gaussian responses in log T, one per channel
        peak = na.linspace(5.7, 6.9, axis="channel", num=6)
        response = na.FunctionArray(
            inputs=temperature,
            outputs=1e-25 * np.exp(-(((logt - peak) / 0.15) ** 2)) * u.DN * u.cm**5 / u.s,
        )

        # the true DEM, and the intensities it would produce
        dem_true = 3e28 * np.exp(-(((logt - 6.3) / 0.12) ** 2)) / u.cm**5
        intensity = (response.outputs * dem_true).sum("temperature") * 0.05
        uncertainty = np.sqrt(intensity * u.DN / u.s) + 1 * u.DN / u.s

        dem, chi2 = utu.dem.plowman(
            intensity=intensity,
            uncertainty=uncertainty,
            response=response,
            axis_channel="channel",
            axis_temperature="temperature",
        )

        fig, ax = plt.subplots(constrained_layout=True)
        na.plt.plot(temperature, dem_true, ax=ax, axis="temperature", label="true")
        na.plt.plot(dem.inputs, dem.outputs, ax=ax, axis="temperature", label="recovered")
        ax.set_xscale("log")
        ax.set_xlabel(f"temperature ({temperature.unit:latex_inline})")
        ax.set_ylabel(f"DEM ({dem.outputs.unit:latex_inline})")
        ax.set_title(f"reduced $\\chi^2$ = {chi2.ndarray:.2f}")
        ax.legend();
    """
    intensity = na.as_named_array(intensity)
    uncertainty = na.as_named_array(uncertainty)

    temperature = response.inputs
    shape_temperature = na.shape(temperature)
    if tuple(shape_temperature) != (axis_temperature,):
        raise ValueError(
            f"the temperatures of `response` must vary along {axis_temperature!r} "
            f"alone, got axes {tuple(shape_temperature)}"
        )
    shape_response = na.shape(response.outputs)
    if set(shape_response) != {axis_channel, axis_temperature}:
        raise ValueError(
            f"the outputs of `response` must have axes {axis_channel!r} and "
            f"{axis_temperature!r} and no others, got {tuple(shape_response)}"
        )

    shape = na.shape_broadcasted(intensity, uncertainty)
    if axis_channel not in shape:
        raise ValueError(f"`intensity` has no axis {axis_channel!r}")
    if shape[axis_channel] != shape_response[axis_channel]:
        raise ValueError(
            f"`intensity` has {shape[axis_channel]} channels but `response` "
            f"has {shape_response[axis_channel]}"
        )
    shape_pixel = {axis: num for axis, num in shape.items() if axis != axis_channel}
    axes = (*shape_pixel, axis_channel)

    unit = na.unit(intensity)
    unit_response = na.unit(response.outputs)

    def value(a: na.AbstractScalar, axes: tuple[str, ...], unit) -> np.ndarray:
        shape_a = {ax: shape[ax] if ax in shape else shape_response[ax] for ax in axes}
        x = na.broadcast_to(na.as_named_array(a), shape_a).ndarray_aligned(axes)
        if isinstance(x, u.Quantity):
            x = x.to_value(unit if unit is not None else u.dimensionless_unscaled)
        return np.ascontiguousarray(x, dtype=np.float64)

    data = value(intensity, axes, unit).reshape(-1, shape[axis_channel])
    errors = value(uncertainty, axes, unit).reshape(-1, shape[axis_channel])
    tresp = value(response.outputs, (axis_temperature, axis_channel), unit_response)

    t = na.as_named_array(temperature).ndarray_aligned((axis_temperature,))
    if isinstance(t, u.Quantity):
        t = t.to_value(u.K)
    logt = np.log10(np.asarray(t, dtype=np.float64))

    rmat, regmat, rvec = _matrices(logt, tresp, smoothness)

    num_pixel = data.shape[0]
    dems = np.zeros((num_pixel, logt.size))
    chi2 = np.full(num_pixel, -1.0)
    _plowman_kernel.plowman(
        data,
        errors,
        rmat,
        regmat,
        rvec,
        iterations_max,
        iterations_min,
        float(steps[0]),
        float(steps[1]),
        float(chi2_target),
        float(tolerance),
        max(1, min(num_pixel, 64 * numba.get_num_threads())),
        dems,
        chi2,
    )

    unit_dem = None
    if unit is not None or unit_response is not None:
        unit_dem = (unit or u.dimensionless_unscaled) / (
            unit_response or u.dimensionless_unscaled
        )

    shape_dem = (*shape_pixel.values(), logt.size)
    dems = dems.reshape(shape_dem)
    if unit_dem is not None:
        dems = dems << unit_dem

    return (
        na.FunctionArray(
            inputs=temperature,
            outputs=na.ScalarArray(dems, axes=(*shape_pixel, axis_temperature)),
        ),
        na.ScalarArray(
            chi2.reshape(tuple(shape_pixel.values())), axes=tuple(shape_pixel)
        ),
    )
