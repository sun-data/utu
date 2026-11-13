"""The emission lines of an optically thin plasma."""

import functools
import logging

import astropy.units as u
import fiasco
import named_arrays as na
import numpy as np

__all__ = [
    "contribution_function",
    "contribution_function_line",
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
        If ``hdf5_dbase_root`` is among them, the ions are listed from that
        database too.

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

    # the ions of the database they are read from, not of the default one
    database = kwargs.get("hdf5_dbase_root")

    # Only the abundance and the transitions of each ion are read, not its
    # ratio of protons to electrons, which an ion built without one computes
    # by walking the whole database. So each ion is given one.
    kwargs = {"proton_electron_ratio": 1} | kwargs

    for name in fiasco.list_ions(hdf5_dbase_root=database):
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
    r"""
    Compute the contribution function of every line of an ion.

    Returned as a function of wavelength, so that a line and its strength
    cannot come apart. They are separate arrays underneath, of different
    lengths whenever an ion has a two-photon transition, and pairing them by
    hand is a way to label every line with its neighbor's wavelength.

    The contribution function is that of :mod:`fiasco`, which leaves out the
    ratio of the density of hydrogen to that of electrons. So the emission
    measure it is multiplied by to find an intensity is
    :math:`\int n_e n_H \, dh`, not :math:`\int n_e^2 \, dh`, which would
    overstate the intensity by :math:`n_e / n_H`, about 1.2 in the corona.

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
        If :obj:`None` (the default), the ratio ``ion`` was built with.
        Given here, it replaces that ratio, in ``ion`` and in the ion it
        recombines from. :class:`fiasco.Ion` computes the ratio as an ion is
        built, unless it is given one, which walks the whole database. The
        ratio depends on the temperature and on nothing else, so a caller
        with more than one ion should compute it once with
        :func:`fiasco.proton_electron_ratio` and pass it to
        :class:`fiasco.Ion` as each ion is built.

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
        ion.proton_electron_ratio = _quantity(proton_electron_ratio)

    axis_density = tuple(na.shape(density))

    coupled = tuple(axis_density) == (axis_temperature,)

    fiasco.log.addFilter(_not_single_ion)
    try:
        result = ion.contribution_function(
            na.as_named_array(density).ndarray,
            couple_density_to_temperature=coupled,
        )
    finally:
        fiasco.log.removeFilter(_not_single_ion)

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


def _not_single_ion(record: logging.LogRecord) -> bool:
    """
    Whether a record of the log of :mod:`fiasco` is anything but its warning
    that an ion is solved with the single-ion model.

    :mod:`fiasco` asks for the two-ion model by default, and solves an ion
    without the data that model needs with the single-ion model instead,
    which is the right model for it. It warns each time, and that is most
    ions, while the only way to tell which ions they are beforehand is
    private to :mod:`fiasco`.
    """
    return "Using single-ion model" not in record.getMessage()


def contribution_function_line(
    ion: str | na.AbstractScalar,
    wavelength: u.Quantity | na.AbstractScalar,
    temperature: na.AbstractScalar,
    density: u.Quantity | na.AbstractScalar,
    axis_temperature: str = "temperature",
    proton_electron_ratio: None | u.Quantity | na.AbstractScalar = None,
    tolerance: u.Quantity = 0.01 * u.mAA,
    size_chunk: None | int = None,
    **kwargs: object,
) -> na.FunctionArray:
    r"""
    Compute the contribution function of an observed line, named by its ion
    and its wavelength.

    An observed line is every transition of its ion at its wavelength, of
    which there can be more than one, so the contribution functions of all
    the transitions within ``tolerance`` of the wavelength are summed. This
    is the difference from :func:`contribution_function`, which returns
    every transition of an ion and leaves picking one out to the caller.

    The result is laid out as :func:`contribution_function` lays it out, so
    that it can be handed to :func:`synthesize`: the wavelength of each line
    as the inputs, and its contribution function along ``axis_temperature``
    and the axes of the lines as the outputs. A density along an axis of its
    own keeps the name of that axis, which :func:`contribution_function`
    calls ``_density``. Its units are those of :mod:`fiasco`, which leaves
    the ratio of the density of hydrogen to that of electrons out, so the
    emission measure it goes with is :math:`\int n_e n_H \, dh`.

    Parameters
    ----------
    ion
        The ion which emits each line, named as :mod:`fiasco` names it,
        ``"O 5"`` for instance. It broadcasts against ``wavelength``, so
        several lines are an ion and a wavelength for each along the same
        axis, or one ion and several wavelengths. An ion with more than one
        of the lines is built once and solved once at each temperature, even
        if it is named two ways, ``"O 5"`` and ``"O V"`` for instance.
    wavelength
        The wavelength of each line, as CHIANTI lists it.
    temperature
        The temperatures of the plasma, along ``axis_temperature`` and no
        other axis.
    density
        The number density of electrons. A scalar is the same density at
        every temperature. A density along ``axis_temperature`` is one
        density per temperature, an isobaric atmosphere for instance. A
        density along one axis of its own is every density at every
        temperature, and costs as many times more. It may not share an axis
        with the lines.
    axis_temperature
        The name of the axis of ``temperature``.
    proton_electron_ratio
        The ratio of protons to electrons at each temperature.
        If :obj:`None` (the default), it is computed here, once, which walks
        the whole database. It is given to each ion as the ion is built,
        and passed on by the ion to the ion it recombines from. Pass it to
        compute lines over one grid of temperatures more than once without
        paying for it again.
    tolerance
        How close to ``wavelength`` a transition must be to belong to the
        line. Two transitions of an ion which CHIANTI lists at different
        wavelengths are at least a tenth of a milliangstrom apart, and many
        are only half a milliangstrom apart, so the default is a tenth of
        that least spacing. It gathers
        every transition listed at the wavelength and none beside it, as
        long as the wavelength is given as CHIANTI lists it.
    size_chunk
        How many temperatures to solve at once.
        If :obj:`None` (the default), all of them. The level populations of
        an ion are solved at every temperature at once, in a matrix as wide
        as the ion and the ion it recombines from have levels, which for an
        ion with hundreds of levels on a fine grid of temperatures is tens
        of gigabytes. Each temperature is solved on its own, so solving a
        few at a time changes nothing in the result.
    kwargs
        Additional arguments passed to :class:`fiasco.Ion` for every ion,
        and to :func:`fiasco.proton_electron_ratio`. They are passed
        unchanged to every chunk of ``size_chunk`` temperatures, so none of
        them may vary with the temperature, as an ``ionization_fraction``
        given as an array would.

    Examples
    --------
    The contribution functions of the line ESIS was built to observe and of
    the line of :math:`\mathrm{Mg\,X}` nearest it.

    .. jupyter-execute::

        import astropy.units as u
        import matplotlib.pyplot as plt
        import named_arrays as na
        import numpy as np
        import utu

        axis = "temperature"
        temperature = na.geomspace(1e4, 1e7, axis=axis, num=61) * u.K

        ion = na.ScalarArray(np.array(["O 5", "Mg 10"]), axes="line")
        wavelength = na.ScalarArray([629.732, 624.941] * u.AA, axes="line")

        result = utu.spectrum.contribution_function_line(
            ion=ion,
            wavelength=wavelength,
            temperature=temperature,
            density=1e15 * u.K / u.cm ** 3 / temperature,
            axis_temperature=axis,
        )

        fig, ax = plt.subplots(constrained_layout=True)
        na.plt.plot(
            temperature,
            result.outputs,
            ax=ax,
            axis=axis,
            label=utu.spectrum.spectroscopic(ion),
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(1e-26, 1e-21)
        ax.set_xlabel(f"temperature ({temperature.unit:latex_inline})")
        ax.set_ylabel(f"$G(T)$ ({result.outputs.unit:latex_inline})")
        ax.legend()
    """
    temperature = na.as_named_array(temperature)
    if tuple(temperature.shape) != (axis_temperature,):
        raise ValueError(
            f"The temperature must be along {axis_temperature!r} and no other "
            f"axis, got {temperature.shape}."
        )
    num_temperature = temperature.shape[axis_temperature]
    if num_temperature == 0:
        raise ValueError("The temperature is empty.")

    if size_chunk is None:
        size_chunk = num_temperature
    if size_chunk < 1:
        raise ValueError(f"size_chunk must be at least 1, got {size_chunk}.")

    ion = na.as_named_array(ion)
    wavelength = na.as_named_array(wavelength)
    shape = na.shape_broadcasted(ion, wavelength)
    if axis_temperature in shape:
        raise ValueError(
            f"The lines must not be along the axis of the temperature, "
            f"{axis_temperature!r}."
        )
    ion = ion.broadcast_to(shape)

    # a copy, so that the inputs of the result are its own and can be written
    wavelength = wavelength.broadcast_to(shape).copy()

    # A scalar density is the same density at every temperature, and is
    # computed as such, so that it adds no axis to the result.
    density = na.as_named_array(density)
    axes_density = tuple(density.shape)
    if not axes_density:
        density = density.broadcast_to(temperature.shape)
        axes_density = (axis_temperature,)
    if len(axes_density) > 1 or set(axes_density) & set(shape):
        raise ValueError(
            f"The density must be a scalar, along {axis_temperature!r}, or "
            f"along one axis of its own which the lines are not along, got "
            f"{density.shape}."
        )
    coupled = axes_density == (axis_temperature,)

    shape_result = {axis_temperature: num_temperature}
    if not coupled:
        shape_result |= density.shape
    shape_result |= shape

    result = na.ScalarArray.zeros(shape_result) << (u.erg * u.cm**3 / u.s)

    # Every line is matched to its transitions before any ion is solved, so
    # that a wavelength with no transition fails at once rather than after
    # the ions before it. The lines are grouped by the ion rather than by its
    # name, so that an ion named two ways is still solved once. The
    # wavelengths do not depend on the ratio of protons to electrons, which
    # is given only so that fiasco does not compute it.
    axis_transition = "_transition"
    wavelength_ion: dict[tuple[int, int], u.Quantity] = {}
    selections: dict[tuple[int, int], list[tuple[dict, na.ScalarArray]]] = {}
    for index in na.ndindex(shape):
        name = str(ion[index].ndarray)
        key = fiasco.util.parse_ion_name(name)

        if key not in selections:
            transitions = fiasco.Ion(
                key,
                temperature.ndarray,
                proton_electron_ratio=1,
                **kwargs,
            ).transitions
            wavelength_ion[key] = transitions.wavelength[transitions.is_bound_bound]
            selections[key] = []

        w = wavelength[index].ndarray
        near = np.abs(wavelength_ion[key] - w) < tolerance
        if not np.any(near):
            raise ValueError(
                f"CHIANTI lists no transition of {name} within {tolerance} of {w}."
            )
        transition = na.ScalarArray(np.flatnonzero(near), axes=axis_transition)
        selections[key].append((index, transition))

    # The ratio of protons to electrons depends on the temperature and on
    # nothing else, and computing it walks the entire database, so it is
    # computed once and sliced along with the temperature.
    if proton_electron_ratio is None:
        proton_electron_ratio = fiasco.proton_electron_ratio(
            temperature.ndarray,
            **kwargs,
        )
    if not isinstance(proton_electron_ratio, na.AbstractArray):
        proton_electron_ratio = na.ScalarArray(
            ndarray=proton_electron_ratio,
            axes=(axis_temperature,),
        )

    for key, selection in selections.items():
        for start in range(0, num_temperature, size_chunk):
            chunk = {axis_temperature: slice(start, start + size_chunk)}

            g = contribution_function(
                ion=fiasco.Ion(
                    key,
                    temperature[chunk].ndarray,
                    proton_electron_ratio=proton_electron_ratio[chunk].ndarray,
                    **kwargs,
                ),
                density=density[chunk],
                axis_temperature=axis_temperature,
                axis=axis_transition,
            )

            # the axis of a density of its own, by the name it was given
            outputs = g.outputs
            if not coupled:
                axes = [axes_density[0] if a == "_density" else a for a in outputs.axes]
                outputs = outputs.replace(axes=tuple(axes))

            for index, transition in selection:
                line = outputs[{axis_transition: transition}]
                result[index | chunk] = line.sum(axis_transition)

    return na.FunctionArray(
        inputs=wavelength,
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
    r"""
    Compute the emission lines of an optically thin plasma, brightest first.

    Every line of every ion abundant enough to contribute, with the intensity
    it would have from a plasma with the given emission measure.

    The wavelength and the ion of a line are components of the inputs of the
    result, and its intensity is the output, so that sorting or selecting
    lines carries all three together.

    If the intensity is uncertain, from an uncertain emission measure for
    instance, the lines are in order of their nominal intensity. A list of
    lines needs a single order to keep the wavelength and the ion of each
    line certain, so a sample in which an uncertainty that differs between
    temperatures has changed which line is brighter keeps the nominal order.

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
        How much plasma there is at each temperature,
        :math:`\int n_e n_H \, dh` through the column in each bin of
        temperature. It is the product of the densities of electrons and of
        hydrogen rather than the square of the first, since :mod:`fiasco`
        leaves the ratio of the two out of the contribution functions, see
        :func:`contribution_function`.
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
        to every ion as the ion is built. Pass it to compute more than one
        spectrum over one grid of temperatures without paying for it again.
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
    # once here and given to every ion as it is built, which passes it on to
    # the ion it recombines from.
    if proton_electron_ratio is None:
        proton_electron_ratio = fiasco.proton_electron_ratio(t, **kwargs)

    # Broadcast here, so that a ratio which does not fit the temperature
    # fails here, rather than as each ion is built below, where a failure is
    # taken to mean that the ion cannot be computed and the ion is skipped.
    proton_electron_ratio = _quantity(proton_electron_ratio) * np.ones(t.shape)

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
                ion=fiasco.Ion(
                    name,
                    t,
                    proton_electron_ratio=proton_electron_ratio,
                    **kwargs,
                ),
                density=density,
                axis_temperature=axis_temperature,
                axis=axis,
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
    # name, which is what `__getitem__` takes. The order is that of the
    # nominal intensity, so that an uncertain intensity, from an uncertain
    # emission measure say, leaves each line in the same place in every
    # sample, and its wavelength and ion certain.
    order = np.argsort(na.nominal(result.outputs), axis=axis)
    result = result[order]

    return result[{axis: slice(None, None, -1)}]
