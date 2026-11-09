"""The spectral radiance of an optically thin plasma."""

import math
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
The most elements :func:`synthesize` computes at once, a quarter of a
gigabyte in double precision, where it computes the result in blocks.

The kernel is the spectral radiance of each line in each bin of velocity per
unit emission measure in each bin of temperature. If it fits in this for
every line and bin at once, the result is computed in one piece. Otherwise,
as where the profile or the contribution function varies from pixel to pixel
and the kernel is as large as the map, the kernel and the radiance it gives
are computed for as many bins of velocity and lines at a time as fit in
this, and at least one of each.
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
    axis_shift: None | str = None,
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

    An emission measure resolved in line-of-sight velocity as well as in
    temperature, such as the velocity-resolved DEM of a simulation, is a sum
    over both, each bin of velocity :math:`s` with a profile centered on its
    own shift,

    .. math::

        L(\lambda) = \frac{1}{4 \pi} \sum_T \sum_s G(T) \,
            \mathrm{EM}(T, s) \, \phi(\lambda; T, s).

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
        `axis_temperature`, and in each bin of line-of-sight velocity too,
        along `axis_shift`, if it is given. Every other axis is carried
        through to the result, such as the two axes of a map.

        This is the emission measure of each bin, not the differential
        emission measure: a DEM per unit :math:`\log_{10} T`, such as
        :func:`utu.dem.plowman` returns, has the same units, and has to be
        converted first: multiplied by the width of each bin in
        :math:`\log_{10} T`, or, to integrate it the way
        :func:`utu.dem.plowman` fits the intensities, weighted with
        :func:`utu.dem.emission_measure`.
    velocity
        The vertices of the bins of velocity, along `axis_velocity`. Positive
        velocities are away from the observer, to longer wavelengths.
    shift
        The line-of-sight velocity of the plasma. It may vary along any axis
        of `emission_measure`, including `axis_temperature` and
        `axis_shift`, and along `axis`.
    width_nonthermal
        The standard deviation of the nonthermal motions of the plasma along
        the line of sight. Often quoted instead as the most probable speed
        :math:`\xi`, which is :math:`\sqrt{2}` times larger. It may vary
        along the same axes as `shift`.
    axis_temperature
        The name of the axis of `temperature`, summed over.
    axis_shift
        The name of an axis of `emission_measure` along which the plasma is
        binned by line-of-sight velocity, as in a velocity-resolved DEM,
        summed over like `axis_temperature`. `shift` is then the velocity of
        each bin, and `width_nonthermal` can hold the spread of velocities
        within it. The profile of each bin is a Gaussian, so plasma spread
        evenly across a bin is approximated by a Gaussian of the same
        variance, with a nonthermal width of the width of the bin over
        :math:`\sqrt{12}`. `emission_measure` must have every element along
        this axis that the other arguments have. If :obj:`None`, the
        default, only `axis_temperature` is summed over.
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
        `axis_temperature` and `axis_shift`. The inputs are the vertices of
        the bins, as
        wavelengths with the rest wavelength of each line beside them.

    Raises
    ------
    ValueError
        If `velocity` has no axis `axis_velocity` or varies along
        `axis_temperature` or `axis_shift`, if `temperature` or the
        outputs of `contribution` have no axis `axis_temperature`, if
        `axis_shift` is `axis_temperature`, `axis_velocity`, or `axis`, if
        `emission_measure` has no axis `axis_shift` or fewer elements along
        it than another argument, if a temperature is not positive, or if an
        argument varies along `axis` differently than `contribution` does.

    Notes
    -----
    Nothing here is particular to an instrument: the profile is that of the
    plasma alone, with no instrumental width, and every line in
    `contribution` is synthesized, so choosing which lines are worth
    including is left to the caller. Natural and pressure broadening are
    left out, which in the transition region and the corona is a small
    fraction of a milliangstrom, and so is opacity.

    The spectral radiance of each line per unit emission measure in each bin
    of temperature, the kernel, is computed first, and then summed with the
    emission measure over `axis_temperature` and `axis_shift` by
    :func:`named_arrays.vecdot`. Unless the profile or the contribution
    function varies from pixel to pixel, the kernel has no axes of the map,
    and the sum is a single matrix product, so a map of many lines takes
    little more memory than the result itself, and is many times faster than
    forming the product of the two. Where one of them does vary from pixel
    to pixel, the kernel is as large as the map, and it and the radiance it
    gives are computed for as many bins of velocity and lines at a time as
    fit in :data:`_size_block` elements. Where the profile is the
    same for every line, because none of `mass`, `shift`, and
    `width_nonthermal` varies along `axis`, it is computed once for all of
    them.

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

    The same plasma, resolved in line-of-sight velocity as well: half of it
    at rest and a quarter in each of two bins :math:`40\,\mathrm{km\,s^{-1}}`
    to either side. The plasma of each bin is spread evenly across its
    width, which the profile approximates with a Gaussian of the same
    variance.

    .. jupyter-execute::

        # the vertices of three bins of line-of-sight velocity
        vertices = na.linspace(-60, 60, axis="vdop", num=4) * u.km / u.s
        center = vertices.cell_centers("vdop")
        width = vertices.volume_cell("vdop")

        fraction = na.ScalarArray(np.array([0.25, 0.5, 0.25]), axes="vdop")

        result = utu.spectrum.synthesize(
            contribution=contribution,
            temperature=temperature,
            mass=ion.mass,
            emission_measure=emission_measure * fraction,
            velocity=na.linspace(-100, 100, axis="velocity", num=81) * u.km / u.s,
            shift=center,
            width_nonthermal=width / np.sqrt(12),
            axis_temperature=axis,
            axis_shift="vdop",
        )

        fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
        na.plt.stairs(
            result.inputs.velocity,
            result.outputs,
            ax=ax,
            axis="velocity",
        )
        ax.set_xlabel(f"velocity ({u.km / u.s:latex_inline})")
        ax.set_ylabel(f"spectral radiance\n({result.outputs.unit:latex_inline})");
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
    axis_sum = (axis_temperature,)
    if axis_shift is not None:
        if axis_shift in (axis_temperature, axis_velocity, axis):
            raise ValueError(
                f"`axis_shift` must be an axis of its own, not `axis_temperature`, "
                f"`axis_velocity`, or `axis`, got {axis_shift!r}."
            )
        if axis_shift not in na.shape(emission_measure):
            raise ValueError(
                f"`emission_measure` must have the axis {axis_shift!r}, "
                f"got {na.shape(emission_measure)}."
            )
        axis_sum = axis_sum + (axis_shift,)
    for a in axis_sum:
        if a in na.shape(velocity):
            raise ValueError(
                f"`velocity` must not vary along {a!r}, which is "
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

    # An emission measure with one element along `axis_shift` would be
    # broadcast against the shifts and summed, counting its plasma once for
    # each of them.
    if axis_shift is not None:
        num_shift = na.shape(emission_measure)[axis_shift]
        if num_shift != shape[axis_shift]:
            raise ValueError(
                f"`emission_measure` must have as many elements along "
                f"{axis_shift!r} as every other argument, {shape[axis_shift]}, "
                f"got {num_shift}."
            )

    for a in axis_sum:
        shape.pop(a)
    num_bin = shape[axis_velocity] - 1
    shape[axis_velocity] = num_bin

    # The kernel is summed with the emission measure as plain numbers, in
    # the units which make the sum a spectral radiance, so that the result,
    # which may be very large, is never converted.
    unit = u.erg / u.cm**2 / u.s / u.sr / u.AA
    unit_kernel = unit / na.unit_normalized(emission_measure)
    emission_measure = emission_measure.value

    # The kernel varies along every axis the contribution function and the
    # profile do, which are axes of the map only where one of them varies
    # from pixel to pixel. If the kernel of every line in every bin of
    # velocity fits in a block, the result is computed in one piece.
    # Otherwise, as many lines at a time as fit in a block in one bin of
    # velocity, with the radiance they give, and as many bins at a time as
    # fit with them.
    shape_kernel = na.shape_broadcasted(
        g, wavelength_rest, temperature, *profile.values()
    )
    size_kernel = math.prod(
        n for a, n in shape_kernel.items() if a not in (axis, axis_velocity)
    )
    size_radiance = math.prod(
        n for a, n in shape.items() if a not in (axis, axis_velocity)
    )
    num_line_all = 1 if num_line is None else num_line
    if size_kernel * num_line_all * num_bin <= _size_block:
        num_line_block = max(1, num_line_all)
        num_bin_block = max(1, num_bin)
    else:
        size = size_kernel + size_radiance
        num_line_block = max(1, min(num_line_all, _size_block // size))
        num_bin_block = max(1, _size_block // (size * num_line_block))

    blocks_bin = _blocks(num_bin, num_bin_block)
    if num_line is None:
        blocks_line = [None]
    else:
        blocks_line = _blocks(num_line, num_line_block)

    # One block is the result itself, and any other number of them fills it.
    single = len(blocks_bin) * len(blocks_line) == 1
    result = None

    # The profile is the same for every line unless something it depends on
    # varies along the axis of lines, and is then computed once per block of
    # bins.
    shared = all(axis not in na.shape(a) for a in profile.values())

    for bins in blocks_bin:
        # the vertices of the bins in this block
        index_bin = {axis_velocity: slice(bins.start, bins.stop + 1)}

        if shared:
            fraction_shared = _fraction(
                temperature=temperature,
                **{name: a[index_bin] for name, a in profile.items()},
                axis_velocity=axis_velocity,
            )

        for block in blocks_line:
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

            # The width of each bin in wavelength, which turns the radiance
            # in it into a spectral radiance. The Doppler shift is linear in
            # velocity, so this is exact, and does not take the difference of
            # two wavelengths which agree in most of their digits.
            width = np.diff(_select(velocity[index_bin], index), axis=axis_velocity)
            width = wavelength_rest[index] * width / astropy.constants.c

            # The units are converted before the fraction, which has every
            # axis of the kernel, is multiplied in.
            factor = g[index] / width / (4 * np.pi * u.sr)
            kernel = factor.to(unit_kernel).value * fraction

            radiance = na.vecdot(
                kernel,
                _select(emission_measure, index),
                axis=axis_sum,
            )
            radiance = np.transpose(radiance << unit, axes=tuple(shape))

            if single:
                result = radiance
            else:
                if result is None:
                    result = _allocate(radiance, shape)
                result[index | {axis_velocity: bins}] = radiance

    # no lines, or no bins of velocity
    if result is None:
        result = na.ScalarArray(
            ndarray=np.zeros(tuple(shape.values())) << unit,
            axes=tuple(shape),
        )

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
    The fraction of the profile of a line which falls in each of a block of
    bins of velocity.

    `velocity` is the vertices of a block of bins, along `axis_velocity`,
    and the result has one element fewer along that axis, one for each bin.

    The fraction is the difference of the cumulative distribution of the
    profile at the two vertices of a bin. Far into either wing that
    distribution is within rounding of zero or of one, and the difference of
    two values near one loses every digit, so it is taken instead from the
    complementary error function of the tail the bin is in, which keeps its
    relative precision there.
    """
    width = np.sqrt(
        astropy.constants.k_B * temperature / mass + np.square(width_nonthermal)
    )

    z = (velocity - shift) / (np.sqrt(2) * width)
    z = z.to(u.dimensionless_unscaled).value

    z0 = z[{axis_velocity: slice(None, -1)}]
    z1 = z[{axis_velocity: slice(1, None)}]

    # +1 if the middle of the bin is above the center of the profile, -1 if
    # below, so that the error function is evaluated in its own tail
    side = np.where(z0 + z1 >= 0, 1, -1)

    erfc0 = scipy.special.erfc(side * z0)
    erfc1 = scipy.special.erfc(side * z1)

    return side * (erfc0 - erfc1) / 2


def _blocks(
    num: int,
    num_block: int,
) -> list[slice]:
    """
    Split `num` elements into blocks of `num_block` elements, the last of
    which may have fewer.
    """
    return [slice(k, min(k + num_block, num)) for k in range(0, num, num_block)]


def _allocate(
    block: na.AbstractScalar,
    shape: dict[str, int],
) -> na.AbstractScalar:
    """
    An array of the type and the unit of a block of the result of
    :func:`synthesize`, scalar or uncertain, with the shape of the whole
    result, every element of which the blocks overwrite.
    """
    first = {a: slice(0, 1) for a in na.shape(block)}
    return na.broadcast_to(block[first], shape).copy()


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
