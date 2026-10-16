"""The emission measure each temperature of a piecewise-linear DEM stands for."""

from typing import cast

import astropy.units as u
import named_arrays as na
import numpy as np

__all__ = [
    "emission_measure",
]


def emission_measure(
    dem: na.FunctionArray[na.AbstractScalar, na.AbstractScalar],
    axis_temperature: str,
) -> na.AbstractScalar:
    r"""
    The emission measure each temperature of a DEM stands for, if the DEM is
    linear in :math:`\log_{10} T` between its temperatures, as the DEMs of
    :func:`utu.dem.plowman` are.

    A DEM which is linear in :math:`\log_{10} T` between its temperatures is
    a sum of triangle functions, one at each temperature. If a temperature
    response or a contribution function is taken to be linear between the
    same temperatures too, as :func:`utu.dem.plowman` takes its responses,
    the integral of the product of the two is :math:`R^T M D`, where
    :math:`D` and :math:`R` are their values at the temperatures and
    :math:`M` is the mass matrix of the triangles. This function returns
    :math:`M D`. At a temperature whose neighbors are :math:`\Delta_-` below
    and :math:`\Delta_+` above it, in :math:`\log_{10} T`, that is

    .. math::

        (MD)_i = \frac{\Delta_- (D_{i-1} + 2 D_i) + \Delta_+ (2 D_i + D_{i+1})}{6},

    where the :math:`\Delta` beyond either end is zero, so that the first
    and last temperatures stand for :math:`\Delta_+ (2 D_0 + D_1) / 6` and
    :math:`\Delta_- (D_{n-2} + 2 D_{n-1}) / 6`. For an even step
    :math:`\Delta`, a temperature inside the grid stands for
    :math:`\Delta (D_{i-1} + 4 D_i + D_{i+1}) / 6`.

    This is the integral :func:`utu.dem.plowman` fits the intensities with,
    so the intensity of a channel predicted from these emission measures and
    its response, ``(response.outputs * result).sum(axis_temperature)``, is
    the intensity the inversion found. Neither the sum of the DEM times the
    step nor the trapezoid rule applied to the product is: both weight each
    temperature by the sum of its row of :math:`M`, which moves the
    off-diagonal terms of :math:`M` onto its diagonal, and that matters
    wherever the DEM and the response both change across a step. The result
    can also be given to :func:`utu.spectrum.synthesize` as the emission
    measure of each bin.

    This is the integral of the functions as interpolated linearly between
    the temperatures, not of the functions themselves, and linear
    interpolation is off by about :math:`\Delta^2 / 8` times the curvature
    of each. For smooth functions which the grid resolves, the plain sum of
    their product times the step is much the closer of the two to their
    integral: in the example below, this function gives an intensity 6%
    higher than the integral of the two Gaussians, which the plain sum
    matches to rounding error. So use this function to predict intensities
    consistently with the fit of :func:`utu.dem.plowman`, and, to simulate
    the intensities of a known DEM to test an inversion against, integrate
    the functions themselves rather than this model of them.

    Summed over a range of temperatures, the result is not the integral of
    the DEM over that range, since each temperature stands for plasma up to
    one step either side of it. To integrate the DEM from its ``i``-th to
    its ``j``-th temperature, give it :math:`\log_{10} T` as its inputs and
    use :meth:`~named_arrays.AbstractFunctionArray.integrate`, whose
    trapezoid rule is exact for a DEM which is linear between its
    temperatures:

    .. code-block:: python

        dem_logt = dem.replace(inputs=np.log10(dem.inputs / u.K))
        dem_logt[{axis_temperature: slice(i, j + 1)}].integrate(axis_temperature)

    A range whose ends fall between the temperatures needs the DEM
    interpolated at its ends first.

    Parameters
    ----------
    dem
        The DEM per unit :math:`\log_{10} T`, as :func:`utu.dem.plowman`
        returns it: its inputs are temperatures along `axis_temperature`,
        positive, increasing and with units of temperature. Its outputs may
        be uncertain, and have any other axes, such as the two axes of a map.
        If they vary along `axis_temperature`, they must have one value for
        each temperature.
    axis_temperature
        The name of the axis along the temperatures of `dem`.

    Returns
    -------
    :
        The emission measure each temperature stands for, in the units of
        the outputs of `dem`, along `axis_temperature` and every other axis
        of `dem`.

    Examples
    --------
    A Gaussian DEM holding :math:`10^{27}\,\mathrm{cm^{-5}}` of plasma in
    all, and the intensity it gives in a channel whose response peaks at a
    hotter temperature.

    .. jupyter-execute::

        import astropy.units as u
        import named_arrays as na
        import numpy as np
        import utu

        temperature = 10 ** na.linspace(5.5, 7, axis="temperature", num=31) * u.K
        logt = np.log10(temperature / u.K)

        width = 0.1
        dem = na.FunctionArray(
            inputs=temperature,
            outputs=1e27 / u.cm**5 * np.exp(-np.square((logt - 6.2) / width) / 2)
            / (np.sqrt(2 * np.pi) * width),
        )

        emission_measure = utu.dem.emission_measure(dem, axis_temperature="temperature")

        # the total emission measure, 1e27 cm^-5
        emission_measure.sum("temperature")

    .. jupyter-execute::

        response = 1e-25 * np.exp(-np.square((logt - 6.5) / 0.15)) * u.DN * u.cm**5 / u.s

        # the intensity, as utu.dem.plowman models it
        (response * emission_measure).sum("temperature")

    The plain sum of the product times the step, 6% less, which for these
    smooth functions is the integral of the functions themselves.

    .. jupyter-execute::

        (response * dem.outputs).sum("temperature") * 0.05
    """
    temperature = dem.inputs
    shape = na.shape(temperature)
    if axis_temperature not in shape:
        raise ValueError(
            f"the inputs of `dem` have no axis {axis_temperature!r}, "
            f"got {tuple(shape)}"
        )
    num = shape[axis_temperature]
    if num < 2:
        raise ValueError("`dem` needs at least two temperatures")
    num_outputs = na.shape(dem.outputs).get(axis_temperature, num)
    if num_outputs != num:
        raise ValueError(
            f"the outputs of `dem` have {num_outputs} temperatures, "
            f"but its inputs have {num}"
        )

    temperature = temperature.to(u.K)
    if not np.all(temperature > 0 * u.K):
        raise ValueError("the temperatures of `dem` must be positive")
    logt = np.log10(temperature.value)

    diagonal, off_diagonal = _mass(logt, axis_temperature)
    if not np.all(off_diagonal > 0):
        raise ValueError("the temperatures of `dem` must be increasing")

    d = na.broadcast_to(dem.outputs, na.shape_broadcasted(temperature, dem.outputs))

    # M D, accumulated in place since a map of DEMs can be large: each
    # temperature's own term, then the term of its neighbor above and of its
    # neighbor below
    lower = {axis_temperature: slice(None, -1)}
    upper = {axis_temperature: slice(1, None)}
    result = d * diagonal
    result[lower] += d[upper] * off_diagonal
    result[upper] += d[lower] * off_diagonal
    return cast(na.AbstractScalar, result)


def _mass(
    logt: na.AbstractScalar,
    axis: str,
) -> tuple[na.AbstractScalar, na.AbstractScalar]:
    r"""
    The diagonal and the off-diagonal of the mass matrix of the triangle
    functions on a grid of :math:`\log_{10} T` along `axis`.

    The matrix is tridiagonal and symmetric. This is the one definition of it
    in the package: :func:`emission_measure` weights a DEM with it, and
    :func:`utu.dem.plowman` fits the intensities with it. Its arithmetic is
    that of the reference implementation of :func:`utu.dem.plowman`, in the
    same order, since a rounding error in the matrix is carried through every
    step of the inversion of every pixel.
    """
    step = logt[{axis: slice(1, None)}] - logt[{axis: slice(None, -1)}]

    # the width of the interval above and below each temperature, zero
    # beyond either end
    zero = 0 * step[{axis: slice(0, 1)}]
    above = na.concatenate([step, zero], axis=axis)
    below = na.concatenate([zero, step], axis=axis)

    diagonal = (above + below) * 2.0 / 6.0
    off_diagonal = step / 6.0
    return cast(na.AbstractScalar, diagonal), cast(na.AbstractScalar, off_diagonal)
