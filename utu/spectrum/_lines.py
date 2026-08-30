"""The emission lines of an optically thin plasma."""

import astropy.units as u
import fiasco
import numpy as np

import named_arrays as na

__all__ = [
    "ions",
    "contribution_function",
    "lines",
]


def ions(
    wavelength: None | u.Quantity = None,
    abundance_min: float = 1e-5,
    **kwargs,
) -> list[str]:
    """
    The ions worth computing over a range of wavelengths.

    An ion is worth computing if its element is abundant enough to contribute
    and if it has a line in the range at all. Reading the line list of an ion
    is cheap; solving its level populations is not, and this is how the
    second is avoided for ions which cannot matter.

    Parameters
    ----------
    wavelength
        The range of wavelengths to look in.
        If :obj:`None` (the default), every ion in the database is returned.
    abundance_min
        The abundance, relative to hydrogen, below which an element is not
        worth including.
    kwargs
        Additional arguments passed to :class:`fiasco.Ion`.

    Examples
    --------
    The ions with a line in the passband of ESIS.

    .. jupyter-execute::

        import astropy.units as u
        import utu

        len(utu.spectrum.ions(wavelength=[550, 680] * u.AA))
    """
    result = []

    for name in fiasco.list_ions():
        try:
            ion = fiasco.Ion(name, 1 * u.MK, **kwargs)

            if ion.abundance is None or ion.abundance < abundance_min:
                continue

            transitions = ion.transitions
            if transitions is None:
                continue

            if wavelength is not None:
                w = transitions.wavelength
                if not np.any((w > wavelength.min()) & (w < wavelength.max())):
                    continue

        except Exception:
            # an ion the database cannot describe is an ion which cannot
            # contribute, and there are a handful of them
            continue

        result.append(str(name))

    return result


def contribution_function(
    ion: fiasco.Ion,
    density: u.Quantity | na.AbstractScalar,
    axis_temperature: str,
    axis: str = "line",
) -> na.FunctionArray:
    """
    The contribution function of every line of an ion.

    Returned as a function of wavelength, so that a line and its strength
    cannot come apart. They are separate arrays underneath, of different
    lengths whenever an ion has a two-photon transition, and pairing them by
    hand is a way to label every line with its neighbor's wavelength.

    Whether the temperatures and the densities are taken in pairs or as a
    grid is decided by their axes. A density which shares the axis of the
    temperature describes one density per temperature, an isobaric
    atmosphere for instance, and is computed as such. A density on an axis of
    its own describes every density at every temperature, and costs as many
    times more.

    Parameters
    ----------
    ion
        The ion to compute the contribution function of.
    density
        The number density of electrons.
    axis_temperature
        The name of the axis of the temperature of ``ion``.
    axis
        The name to give the axis along the lines of the result.
    """
    axis_density = tuple(na.shape(density))

    coupled = tuple(axis_density) == (axis_temperature,)

    result = ion.contribution_function(
        na.as_named_array(density).ndarray,
        couple_density_to_temperature=coupled,
    )

    # the last axis of the result runs over the bound-bound transitions, so
    # that is the wavelength which belongs to it
    transitions = ion.transitions
    wavelength = transitions.wavelength[transitions.is_bound_bound]

    axes = (axis_temperature, "_density", axis)
    result = na.ScalarArray(result, axes=axes)
    if coupled:
        result = result[{"_density": 0}]

    return na.FunctionArray(
        inputs=na.ScalarArray(wavelength, axes=(axis,)),
        outputs=result,
    )


def lines(
    temperature: na.AbstractScalar,
    density: u.Quantity | na.AbstractScalar,
    emission_measure: u.Quantity | na.AbstractScalar,
    wavelength: None | u.Quantity = None,
    axis_temperature: str = "temperature",
    axis: str = "line",
    **kwargs,
) -> na.FunctionArray:
    """
    The emission lines of an optically thin plasma, brightest first.

    Every line of every ion abundant enough to contribute, with the intensity
    it would have from a plasma with the given emission measure.

    The wavelength and the ion of a line are components of the inputs of the
    result, and its intensity is the output, so that sorting or selecting
    lines carries all three together.

    Parameters
    ----------
    temperature
        The temperatures of the plasma.
    density
        The number density of electrons. A density which shares the axis of
        the temperature is one density per temperature, an isobaric
        atmosphere for instance; a density on its own axis is every density
        at every temperature.
    emission_measure
        How much plasma there is at each temperature.
    wavelength
        The range of wavelengths to compute over.
        If :obj:`None` (the default), every line of every ion is returned.
    axis_temperature
        The name of the axis of ``temperature``.
    axis
        The name to give the axis along the lines of the result.
    kwargs
        Additional arguments passed to :class:`fiasco.Ion`.
    """
    t = na.as_named_array(temperature).ndarray

    # The ratio of protons to electrons depends on the temperature and on
    # nothing else, and computing it walks the entire database. Computed
    # once here and primed into the cache of every ion, which is most of
    # what makes this bearable.
    ratio = fiasco.fiasco.proton_electron_ratio(t, **kwargs)

    wavelength_all = []
    intensity_all = []
    ion_all = []

    for name in ions(wavelength=wavelength, **kwargs):
        ion = fiasco.Ion(name, t, **kwargs)
        ion.__dict__["proton_electron_ratio"] = ratio

        try:
            g = contribution_function(
                ion=ion,
                density=density,
                axis_temperature=axis_temperature,
                axis=axis,
            )
        except Exception:
            # an ion whose atomic model the database cannot complete
            continue

        intensity = (g.outputs * emission_measure).sum(axis_temperature)

        w = g.inputs
        if wavelength is not None:
            where = (w > wavelength.min()) & (w < wavelength.max())
            w, intensity = w[where], intensity[where]

        wavelength_all.append(w)
        intensity_all.append(intensity)
        ion_all.append(na.ScalarArray(np.array([name] * w.size), axes=(axis,)))

    result = na.FunctionArray(
        inputs=na.CartesianNdVectorArray(
            components=dict(
                wavelength=na.concatenate(wavelength_all, axis=axis),
                ion=na.concatenate(ion_all, axis=axis),
            ),
        ),
        outputs=na.concatenate(intensity_all, axis=axis),
    )

    # Brightest first, carrying the wavelength and the ion of each line along
    # with its intensity. `argsort` gives back the index of each axis by
    # name, which is what `__getitem__` takes.
    order = np.argsort(result.outputs, axis=axis)
    result = result[order]

    return result[{axis: slice(None, None, -1)}]
