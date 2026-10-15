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
    the line, however coarse the bins, as long as they span the line.

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
        `contribution` was computed at.
    mass
        The mass of the ion each line belongs to, such as
        :attr:`fiasco.Ion.mass`. Give it along `axis` to synthesize the lines
        of several ions at once.
    emission_measure
        How much plasma there is at each temperature,
        :math:`\int n_e n_H \, dh` through the column in each bin of
        temperature, along `axis_temperature`. Every other axis is carried
        through to the result, such as the two axes of a map.
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
        The name of the axis along the lines of `contribution`.

    Returns
    -------
    :
        The spectral radiance of each line in each bin of velocity, along
        `axis`, `axis_velocity`, and every axis of `emission_measure` but
        `axis_temperature`. The inputs are the vertices of the bins, as
        wavelengths with the rest wavelength of each line beside them.

    Notes
    -----
    Nothing here is particular to an instrument: the profile is that of the
    plasma alone, with no instrumental width, and every line in
    `contribution` is synthesized, so choosing which lines are worth
    including is left to the caller. Natural and pressure broadening are
    left out, which in the transition region and the corona is a small
    fraction of a milliangstrom, and so is opacity.

    The lines are synthesized one at a time, so that the largest array held
    at once is the profile of one line at every temperature and in every
    pixel, rather than that of every line.

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
    wavelength_rest = contribution.inputs

    arrays = dict(
        temperature=_scalar(temperature),
        mass=_scalar(mass),
        emission_measure=_scalar(emission_measure),
        velocity=_scalar(velocity),
        shift=_scalar(shift),
        width_nonthermal=_scalar(width_nonthermal),
    )

    shape = na.shape(contribution.outputs)

    # A single line, such as one picked out of the lines of an ion with an
    # integer index, has no axis of lines to loop over.
    if axis not in shape:
        result = _line(
            contribution=contribution,
            **arrays,
            axis_temperature=axis_temperature,
            axis_velocity=axis_velocity,
        )
    else:
        result = na.stack(
            [
                _line(
                    contribution=contribution[{axis: i}],
                    **{name: a[{axis: i}] for name, a in arrays.items()},
                    axis_temperature=axis_temperature,
                    axis_velocity=axis_velocity,
                )
                for i in range(shape[axis])
            ],
            axis=axis,
        )

    return na.FunctionArray(
        inputs=na.DopplerVectorArray.from_velocity(
            velocity=velocity,
            wavelength_rest=wavelength_rest,
        ),
        outputs=result,
    )


def _line(
    contribution: na.FunctionArray,
    temperature: na.AbstractScalar,
    mass: na.AbstractScalar,
    emission_measure: na.AbstractScalar,
    velocity: na.AbstractScalar,
    shift: na.AbstractScalar,
    width_nonthermal: na.AbstractScalar,
    axis_temperature: str,
    axis_velocity: str,
) -> na.AbstractScalar:
    """
    The spectral radiance of a single line in each bin of velocity.

    The arguments are those of :func:`synthesize`, with any axis along the
    lines already indexed away.
    """
    width = np.sqrt(
        astropy.constants.k_B * temperature / mass + np.square(width_nonthermal)
    )

    # The fraction of the line which falls in each bin, as the difference of
    # the cumulative distribution of the profile across the bin.
    z = (velocity - shift) / (np.sqrt(2) * width)
    cdf = scipy.special.erf(z.to(u.dimensionless_unscaled)) / 2
    fraction = _diff(cdf, axis_velocity)

    radiance = (contribution.outputs * emission_measure * fraction).sum(
        axis_temperature
    )

    # The width of each bin in wavelength, which is what turns the radiance in
    # a bin into a spectral radiance.
    wavelength = (1 + velocity / astropy.constants.c) * contribution.inputs
    width_bin = _diff(wavelength, axis_velocity)

    result = radiance / width_bin / (4 * np.pi * u.sr)

    return result.to(u.erg / u.cm**2 / u.s / u.sr / u.AA)


def _scalar(
    a: u.Quantity | na.AbstractScalar,
) -> na.AbstractScalar:
    """A quantity as a named scalar, so that it can be indexed by axis."""
    return cast(na.AbstractScalar, na.as_named_array(a))


def _diff(
    a: na.AbstractScalar,
    axis: str,
) -> na.AbstractScalar:
    """The difference between neighboring vertices along an axis."""
    return a[{axis: slice(1, None)}] - a[{axis: slice(None, -1)}]
