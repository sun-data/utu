"""The emission lines of an optically thin plasma."""

import functools

import astropy.units as u
import fiasco
import named_arrays as na
import numpy as np

__all__ = [
    "contribution_function",
    "ions",
    "lines",
]


def ions(
    wavelength_min: None | u.Quantity | na.AbstractScalar = None,
    wavelength_max: None | u.Quantity | na.AbstractScalar = None,
    abundance_min: float = 1e-5,
    **kwargs: object,
) -> list[str]:
    r"""
    Find the ions worth computing over a range of wavelengths.

    An ion is worth computing if its element is abundant enough to contribute
    and if it has a line in the range at all. Reading the line list of an ion
    is cheap; solving its level populations is not, and this is how the
    second is avoided for ions which cannot matter.

    Parameters
    ----------
    wavelength_min
        The shortest wavelength worth looking at.
        If :obj:`None` (the default), there is no lower bound.
    wavelength_max
        The longest wavelength worth looking at.
        If :obj:`None` (the default), there is no upper bound.
    abundance_min
        The abundance, relative to hydrogen, below which an element is not
        worth including.
    kwargs
        Additional arguments passed to :class:`fiasco.Ion`.

    Notes
    -----
    Which ions come back is a statement about the database as much as about
    the wavelengths: an ion the database does not describe cannot be
    returned, and the databases built for a documentation page or a test
    suite describe only a few.

    The database is read on the first call and remembered afterwards, so the
    first call takes a few seconds and the rest take none. What is
    remembered is the wavelengths of the lines of the ions abundant enough
    to pass ``abundance_min``, a few megabytes.

    Examples
    --------
    The line ESIS was built to observe is a line of :math:`\mathrm{O\,V}`.

    .. jupyter-execute::

        import astropy.units as u
        import utu

        "O 5" in utu.spectrum.ions(
            wavelength_min=629 * u.AA,
            wavelength_max=630 * u.AA,
        )
    """
    wavelength_min = _quantity(wavelength_min)
    wavelength_max = _quantity(wavelength_max)

    result = []

    for name, w in _catalog(abundance_min, **kwargs).items():
        where = np.ones(w.shape, dtype=bool)
        if wavelength_min is not None:
            where = where & (w > wavelength_min)
        if wavelength_max is not None:
            where = where & (w < wavelength_max)
        if not np.any(where):
            continue
        result.append(name)

    return result


def _quantity(
    value: None | u.Quantity | na.AbstractScalar,
) -> None | u.Quantity:
    """
    A value as a plain quantity, however it was given.

    :mod:`fiasco` is not a named-arrays library, and neither are the line
    lists it returns, so anything handed to it or compared against it has to
    shed its axes on the way.
    """
    if isinstance(value, na.AbstractArray):
        return value.ndarray
    return value


@functools.cache
def _catalog(
    abundance_min: float,
    **kwargs: object,
) -> dict[str, u.Quantity]:
    """
    The wavelengths of every line of every ion abundant enough to matter.

    Read once and remembered afterwards. The database does not change while
    a program runs, and reading it is where nearly all the time of
    :func:`ions` goes: five hundred ions at about thirty milliseconds each,
    almost none of it spent on the file.
    """
    result = {}

    for name in fiasco.list_ions():
        try:
            ion = fiasco.Ion(name, 1 * u.MK, **kwargs)

            if ion.abundance is None or ion.abundance < abundance_min:
                continue

            transitions = ion.transitions
            if transitions is None:  # pragma: nocover
                continue

            wavelength = transitions.wavelength

        except Exception:
            # an ion the database cannot describe is an ion which cannot
            # contribute, and there are a handful of them
            continue

        result[str(name)] = wavelength

    return result


_ions = ions
"""
A private alias for :func:`ions`, so that :func:`lines` can take a parameter
of that name without hiding the function it falls back on.
"""


def contribution_function(
    ion: fiasco.Ion,
    density: u.Quantity | na.AbstractScalar,
    axis_temperature: str,
    axis: str = "line",
    proton_electron_ratio: None | u.Quantity | na.AbstractScalar = None,
) -> na.FunctionArray:
    """
    Compute the contribution function of every line of an ion.

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
    proton_electron_ratio
        The ratio of protons to electrons at each temperature of ``ion``.
        If :obj:`None` (the default), :mod:`fiasco` computes it, which walks
        the whole database and takes an order of magnitude longer than the
        rest of this function put together. It depends on the temperature
        and on nothing else, so a caller with more than one ion should
        compute it once with :func:`fiasco.proton_electron_ratio` and pass
        it here. Doing so primes the cache of ``ion`` with the value it
        would otherwise have computed for itself.

    Examples
    --------
    The contribution function of the line ESIS was built to observe, which
    peaks at the temperature the line is formed at.

    .. jupyter-execute::

        import astropy.units as u
        import fiasco
        import matplotlib.pyplot as plt
        import named_arrays as na
        import numpy as np
        import utu

        axis = "temperature"
        temperature = na.geomspace(1e4, 1e7, axis=axis, num=61) * u.K

        ion = fiasco.Ion("O 5", temperature.ndarray)

        result = utu.spectrum.contribution_function(
            ion=ion,
            density=1e15 * u.K / u.cm ** 3 / temperature,
            axis_temperature=axis,
        )

        # the strongest line of the ion, at 629.7 angstroms
        index = np.argmax(result.outputs.max(axis), axis="line")

        fig, ax = plt.subplots(constrained_layout=True)
        na.plt.plot(
            temperature,
            result.outputs[index],
            ax=ax,
            axis=axis,
        )
        ax.set_xscale("log")
        ax.set_xlabel(f"temperature ({temperature.unit:latex_inline})")
        ax.set_ylabel(f"$G(T)$ ({result.outputs.unit:latex_inline})")
    """
    if proton_electron_ratio is not None:
        ion.__dict__["proton_electron_ratio"] = _quantity(proton_electron_ratio)

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
    wavelength_min: None | u.Quantity | na.AbstractScalar = None,
    wavelength_max: None | u.Quantity | na.AbstractScalar = None,
    ions: None | list[str] = None,
    proton_electron_ratio: None | u.Quantity | na.AbstractScalar = None,
    axis_temperature: str = "temperature",
    axis: str = "line",
    **kwargs: object,
) -> na.FunctionArray:
    """
    Compute the emission lines of an optically thin plasma, brightest first.

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
    wavelength_min
        The shortest wavelength worth computing.
        If :obj:`None` (the default), there is no lower bound.
    wavelength_max
        The longest wavelength worth computing.
        If :obj:`None` (the default), there is no upper bound.
    ions
        The ions to compute the lines of, named as :mod:`fiasco` names them.
        If :obj:`None` (the default), they are found with :func:`ions`, which
        is every ion the database describes with a line in ``wavelength``.
        Naming them is how a result is made to depend on the ions rather
        than on which of them the database at hand happens to hold.
    proton_electron_ratio
        The ratio of protons to electrons at each temperature.
        If :obj:`None` (the default), it is computed here, once, and given
        to every ion. Pass it to compute more than one spectrum over one
        grid of temperatures without paying for it again.
    axis_temperature
        The name of the axis of ``temperature``.
    axis
        The name to give the axis along the lines of the result.
    kwargs
        Additional arguments passed to :class:`fiasco.Ion`.

    Examples
    --------
    The brightest lines of two ions, from a plasma spread evenly over a
    decade of temperature.

    .. jupyter-execute::

        import astropy.units as u
        import named_arrays as na
        import utu

        temperature = na.geomspace(1e5, 1e6, axis="temperature", num=11) * u.K

        result = utu.spectrum.lines(
            temperature=temperature,
            density=1e15 * u.K / u.cm ** 3 / temperature,
            emission_measure=1e27 / u.cm ** 5,
            wavelength_min=550 * u.AA,
            wavelength_max=680 * u.AA,
            ions=["O 5", "Mg 10"],
        )

        result[{"line": slice(4)}]
    """
    t = na.as_named_array(temperature).ndarray

    # The ratio of protons to electrons depends on the temperature and on
    # nothing else, and computing it walks the entire database. Computed
    # once here and handed to every ion, which is most of what makes this
    # bearable.
    if proton_electron_ratio is None:
        proton_electron_ratio = fiasco.proton_electron_ratio(t, **kwargs)

    wavelength_all = []
    intensity_all = []
    ion_all = []

    if ions is None:
        ions = _ions(
            wavelength_min=wavelength_min,
            wavelength_max=wavelength_max,
            **kwargs,
        )

    for name in ions:
        try:
            g = contribution_function(
                ion=fiasco.Ion(name, t, **kwargs),
                density=density,
                axis_temperature=axis_temperature,
                axis=axis,
                proton_electron_ratio=proton_electron_ratio,
            )
        except Exception:  # pragma: nocover
            # an ion whose atomic model the database cannot complete
            continue

        intensity = (g.outputs * emission_measure).sum(axis_temperature)

        w = g.inputs
        where = None
        if wavelength_min is not None:
            where = w > wavelength_min
        if wavelength_max is not None:
            below = w < wavelength_max
            where = below if where is None else where & below
        if where is not None:
            w, intensity = w[where], intensity[where]

        wavelength_all.append(w)
        intensity_all.append(intensity)
        ion_all.append(na.ScalarArray(np.array([name] * w.size), axes=(axis,)))

    result = na.FunctionArray(
        inputs=na.CartesianNdVectorArray(
            components={
                "wavelength": na.concatenate(wavelength_all, axis=axis),
                "ion": na.concatenate(ion_all, axis=axis),
            },
        ),
        outputs=na.concatenate(intensity_all, axis=axis),
    )

    # Brightest first, carrying the wavelength and the ion of each line along
    # with its intensity. `argsort` gives back the index of each axis by
    # name, which is what `__getitem__` takes.
    order = np.argsort(result.outputs, axis=axis)
    result = result[order]

    return result[{axis: slice(None, None, -1)}]
