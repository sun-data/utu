"""The spectral radiance of an optically thin plasma."""

from typing import cast

import astropy.constants
import astropy.units as u
import named_arrays as na
import numpy as np
import scipy.special

__all__ = [
    "synthesize",
]

_size_block = 2**25
"""
The most elements :func:`synthesize` puts in any one array along the way, a
quarter of a gigabyte in double precision.

The largest of them is the product of the contribution function, the
emission measure, and the profile of a block of lines in one bin of
velocity, at every temperature and in every pixel. The lines are synthesized
as many at a time as fit in this, and at least one.
"""


def synthesize(
    contribution: na.FunctionArray,
    temperature: na.AbstractScalar,
    mass: u.Quantity | na.AbstractScalar,
    emission_measure: u.Quantity | na.AbstractScalar,
    velocity: na.AbstractScalar,
    shift: u.Quantity | na.AbstractScalar = 0 * u.km / u.s,
    width_nonthermal: u.Quantity | na.AbstractScalar = 0 * u.km / u.s,
    axis_temperature: str = "temperature",
    axis_velocity: str = "velocity",
    axis: str = "line",
) -> na.FunctionArray[na.DopplerVectorArray, na.AbstractScalar]:
    r"""
    Compute the spectral radiance of an optically thin plasma, line by line.

    Each line is spread over a range of velocities around its rest
    wavelength, as a Gaussian whose center is the line-of-sight velocity of
    the plasma and whose width is that of the thermal motion of its ions
    added in quadrature to a nonthermal width. The plasma at each temperature
    has a profile of its own, and the spectrum is their sum,

    .. math::

        L(\lambda) = \frac{1}{4 \pi} \sum_T G(T) \, \mathrm{EM}(T) \,
            \phi(\lambda; T),

    where :math:`G` is the contribution function, :math:`\mathrm{EM}` the
    emission measure, and :math:`\phi` the profile, normalized to unit area.
    Its standard deviation, in velocity, is

    .. math::

        \sigma^2 = \frac{k_B T}{m} + \sigma_\mathrm{nt}^2,

    with :math:`m` the mass of the ion and :math:`\sigma_\mathrm{nt}` the
    nonthermal width.

    The radiance of each bin of velocity is the integral of the profile
    across it divided by its width in wavelength, not the profile sampled at
    its center. So the sum of the radiance over the bins is the radiance of
    the line, however coarse the bins, as long as they span the line. The
    integral is taken from whichever tail of the profile the bin is in, so
    it keeps its precision far into the wings.

    Parameters
    ----------
    contribution
        The contribution functions of the lines, as
        :func:`~utu.spectrum.contribution_function` returns them: the rest
        wavelength of each line along `axis` as the inputs, and its
        contribution function along `axis_temperature` and `axis` as the
        outputs. :mod:`fiasco` leaves the ratio of hydrogen to electrons out
        of these, so the emission measure they go with is that of
        :math:`n_e n_H`.
    temperature
        The temperatures of the plasma, along `axis_temperature`: the ones
        `contribution` was computed at. They must be positive.
    mass
        The mass of the ion each line belongs to, such as
        :attr:`fiasco.Ion.mass`. Give it along `axis` to synthesize the lines
        of several ions at once.
    emission_measure
        How much plasma there is in each bin of temperature,
        :math:`\int n_e n_H \, dh` through the column, along
        `axis_temperature`. Every other axis is carried through to the
        result, such as the two axes of a map.

        This is the emission measure of each bin, not the differential
        emission measure: a DEM per unit :math:`\log_{10} T`, such as
        :func:`utu.dem.plowman` returns, has the same units, and has to be
        multiplied by the width of each bin in :math:`\log_{10} T` first.
    velocity
        The vertices of the bins of velocity, along `axis_velocity`. Positive
        velocities are away from the observer, to longer wavelengths.
    shift
        The line-of-sight velocity of the plasma. It may vary along any axis
        of `emission_measure`, including `axis_temperature`, and along
        `axis`.
    width_nonthermal
        The standard deviation of the nonthermal motions of the plasma along
        the line of sight. Often quoted instead as the most probable speed
        :math:`\xi`, which is :math:`\sqrt{2}` times larger. It may vary
        along the same axes as `shift`.
    axis_temperature
        The name of the axis of `temperature`, summed over.
    axis_velocity
        The name of the axis of `velocity`.
    axis
        The name of the axis along the lines of `contribution`. Any other
        argument which varies along it must have one element along it, or as
        many as `contribution`.

    Returns
    -------
    :
        The spectral radiance of each line in each bin of velocity, along
        `axis`, `axis_velocity`, and every axis of the other arguments but
        `axis_temperature`. The inputs are the vertices of the bins, as
        wavelengths with the rest wavelength of each line beside them.

    Raises
    ------
    ValueError
        If `velocity` has no axis `axis_velocity` or varies along
        `axis_temperature`, if `temperature` or the
        outputs of `contribution` have no axis `axis_temperature`, if a
        temperature is not positive, or if an argument varies along `axis`
        differently than `contribution` does.

    Notes
    -----
    Nothing here is particular to an instrument: the profile is that of the
    plasma alone, with no instrumental width, and every line in
    `contribution` is synthesized, so choosing which lines are worth
    including is left to the caller. Natural and pressure broadening are
    left out, which in the transition region and the corona is a small
    fraction of a milliangstrom, and so is opacity.

    The result is filled in one bin of velocity and one block of lines at a
    time, with as many lines in a block as keep the largest array along the
    way under :data:`_size_block` elements. So a map of many lines needs
    little more memory than the result itself. Where the profile is the same
    for every line, because none of `mass`, `shift`, and `width_nonthermal`
    varies along `axis`, it is computed once for all of them.

    Examples
    --------
    The :math:`\mathrm{Ne\,VII}\;465\,\AA` line of plasma near half a million
    kelvin, at rest and moving toward the observer at
    :math:`20\,\mathrm{km\,s^{-1}}`.

    .. jupyter-execute::

        import astropy.units as u
        import fiasco
        import matplotlib.pyplot as plt
        import named_arrays as na
        import numpy as np
        import utu

        fiasco.log.setLevel("ERROR")

        axis = "temperature"
        temperature = na.geomspace(1e5, 1e6, axis=axis, num=21) * u.K

        # a Gaussian in the logarithm of the temperature, holding
        # 1e27 cm^-5 of plasma in all
        weight = np.exp(-np.square((np.log10(temperature / u.K) - 5.7) / 0.1) / 2)
        emission_measure = 1e27 / u.cm**5 * weight / weight.sum(axis)

        ion = fiasco.Ion("Ne 7", temperature.ndarray)

        contribution = utu.spectrum.contribution_function(
            ion=ion,
            density=1e15 * u.K / u.cm**3 / temperature,
            axis_temperature=axis,
        )

        # the strongest line of the ion
        brightest = np.argmax(contribution.outputs.max(axis), axis="line")
        contribution = contribution[brightest]

        shift = na.ScalarArray([0, -20] * u.km / u.s, axes="shift")

        result = utu.spectrum.synthesize(
            contribution=contribution,
            temperature=temperature,
            mass=ion.mass,
            emission_measure=emission_measure,
            velocity=na.linspace(-100, 100, axis="velocity", num=81) * u.km / u.s,
            shift=shift,
            axis_temperature=axis,
        )

        fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
        na.plt.stairs(
            result.inputs.velocity,
            result.outputs,
            ax=ax,
            axis="velocity",
            label=shift,
        )
        ax.set_xlabel(f"velocity ({u.km / u.s:latex_inline})")
        ax.set_ylabel(f"spectral radiance\n({result.outputs.unit:latex_inline})")
        ax.legend(title="shift");
    """
    temperature = _scalar(temperature)
    mass = _scalar(mass)
    emission_measure = _scalar(emission_measure)
    velocity = _scalar(velocity).explicit
    shift = _scalar(shift)
    width_nonthermal = _scalar(width_nonthermal)

    g = contribution.outputs
    wavelength_rest = contribution.inputs

    if axis_velocity not in na.shape(velocity):
        raise ValueError(
            f"`velocity` must have the axis {axis_velocity!r}, "
            f"got {na.shape(velocity)}."
        )
    if axis_temperature in na.shape(velocity):
        raise ValueError(
            f"`velocity` must not vary along {axis_temperature!r}, which is "
            f"summed over, got {na.shape(velocity)}."
        )
    for name, a in dict(temperature=temperature, contribution=g).items():
        if axis_temperature not in na.shape(a):
            raise ValueError(
                f"`{name}` must have the axis {axis_temperature!r}, "
                f"got {na.shape(a)}."
            )
    if not np.all(temperature > 0 * u.K):
        raise ValueError("every temperature must be positive.")

    # The number of lines, or None for a single line without an axis of
    # lines. Every other argument has to agree with it, or a line would be
    # profiled with the mass or the motions of another.
    num_line = na.shape(g).get(axis)
    profile = dict(
        mass=mass,
        velocity=velocity,
        shift=shift,
        width_nonthermal=width_nonthermal,
    )
    for name, a in (profile | dict(emission_measure=emission_measure)).items():
        num = na.shape(a).get(axis)
        if num is None:
            continue
        if num_line is None:
            raise ValueError(
                f"`{name}` varies along {axis!r}, "
                f"but `contribution` is a single line."
            )
        if num not in (1, num_line):
            raise ValueError(
                f"`{name}` has {num} elements along {axis!r}, "
                f"but `contribution` has {num_line} lines."
            )

    shape = na.shape_broadcasted(
        g,
        wavelength_rest,
        temperature,
        emission_measure,
        *profile.values(),
    )
    num_temperature = shape.pop(axis_temperature)
    num_bin = shape[axis_velocity] - 1
    shape[axis_velocity] = num_bin

    unit = u.erg / u.cm**2 / u.s / u.sr / u.AA
    result = na.ScalarArray(
        ndarray=np.zeros(tuple(shape.values())) << unit,
        axes=tuple(shape),
    )

    # Enough lines at a time that the product of the contribution function,
    # the emission measure, and the profile in one bin, at every temperature
    # and in every pixel, stays under the size of a block.
    size_line = num_temperature
    for a, n in shape.items():
        if a not in (axis, axis_velocity):
            size_line *= n
    num_block = max(1, _size_block // size_line)
    if num_line is None:
        blocks = [None]
    else:
        blocks = [slice(i, i + num_block) for i in range(0, num_line, num_block)]

    # The profile is the same for every line unless something it depends on
    # varies along the axis of lines, and is then computed once per bin.
    shared = all(axis not in na.shape(a) for a in profile.values())

    for k in range(num_bin):
        index_bin = {axis_velocity: slice(k, k + 2)}

        if shared:
            fraction_shared = _fraction(
                temperature=temperature,
                **{name: a[index_bin] for name, a in profile.items()},
                axis_velocity=axis_velocity,
            )

        for block in blocks:
            index = {} if block is None else {axis: block}

            if shared:
                fraction = fraction_shared
            else:
                fraction = _fraction(
                    temperature=temperature,
                    **{
                        name: _select(a[index_bin], index)
                        for name, a in profile.items()
                    },
                    axis_velocity=axis_velocity,
                )

            radiance = g[index] * _select(emission_measure, index) * fraction
            radiance = radiance.sum(axis_temperature)

            # The width of the bin in wavelength, which turns the radiance in
            # it into a spectral radiance. The Doppler shift is linear in
            # velocity, so this is exact, and does not take the difference of
            # two wavelengths which agree in most of their digits.
            width = np.diff(_select(velocity[index_bin], index), axis=axis_velocity)
            width = wavelength_rest[index] * width / astropy.constants.c

            radiance = radiance / width / (4 * np.pi * u.sr)

            result[index | {axis_velocity: slice(k, k + 1)}] = radiance

    return na.FunctionArray(
        inputs=na.DopplerVectorArray.from_velocity(
            velocity=velocity,
            wavelength_rest=wavelength_rest,
        ),
        outputs=result,
    )


def _fraction(
    temperature: na.AbstractScalar,
    mass: na.AbstractScalar,
    velocity: na.AbstractScalar,
    shift: na.AbstractScalar,
    width_nonthermal: na.AbstractScalar,
    axis_velocity: str,
) -> na.AbstractScalar:
    """
    The fraction of the profile of a line which falls in one bin of velocity.

    `velocity` is the two vertices of the bin, along `axis_velocity`, and the
    result keeps that axis, with one element.

    The fraction is the difference of the cumulative distribution of the
    profile at the two vertices. Far into either wing that distribution is
    within rounding of zero or of one, and the difference of two values near
    one loses every digit, so it is taken instead from the complementary
    error function of the tail the bin is in, which keeps its relative
    precision there.
    """
    width = np.sqrt(
        astropy.constants.k_B * temperature / mass + np.square(width_nonthermal)
    )

    z = (velocity - shift) / (np.sqrt(2) * width)
    z = z.to(u.dimensionless_unscaled).value

    z0 = z[{axis_velocity: slice(0, 1)}]
    z1 = z[{axis_velocity: slice(1, 2)}]

    # +1 if the middle of the bin is above the center of the profile, -1 if
    # below, so that the error function is evaluated in its own tail
    side = np.where(z0 + z1 >= 0, 1, -1)

    erfc0 = scipy.special.erfc(side * z0)
    erfc1 = scipy.special.erfc(side * z1)

    return side * (erfc0 - erfc1) / 2


def _select(
    a: na.AbstractScalar,
    index: dict[str, slice],
) -> na.AbstractScalar:
    """
    Select a block of lines from an argument of :func:`synthesize`, unless
    it broadcasts along the axis of lines.

    An argument with one element along that axis is the same for every
    line, and indexing it with a block past its first element would fail.
    """
    for axis in index:
        if na.shape(a).get(axis, 1) == 1:
            return a
    return a[index]


def _scalar(
    a: u.Quantity | na.AbstractScalar,
) -> na.AbstractScalar:
    """A quantity as a named scalar, so that it can be indexed by axis."""
    return cast(na.AbstractScalar, na.as_named_array(a))
