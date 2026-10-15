import astropy.constants
import astropy.units as u
import fiasco
import named_arrays as na
import numpy as np
import pytest

import utu

from .test_lines import needs_database

# Two lines of two ions, with contribution functions made up rather than read
# from the database, so that these tests run wherever the package does. The
# shape of a contribution function has nothing to do with how it is spread
# into a profile.
temperature = na.geomspace(1e5, 10**6.5, axis="temperature", num=6) * u.K

contribution = na.FunctionArray(
    inputs=na.ScalarArray([465.221, 499.406] * u.AA, axes="line"),
    outputs=na.ScalarArray(
        ndarray=np.array(
            [
                [1, 3, 2, 1, 0.5, 0.1],
                [0.1, 0.2, 1, 4, 3, 1],
            ]
        )
        * 1e-24
        * u.erg
        * u.cm**3
        / u.s,
        axes=("line", "temperature"),
    ),
)

mass = na.ScalarArray([20.18, 28.09] * u.u, axes="line")

emission_measure = na.ScalarArray(
    ndarray=np.array([1, 2, 4, 2, 1, 0.5]) * 1e26 / u.cm**5,
    axes="temperature",
)

# wide enough to hold eleven standard deviations of the broadest line here,
# Ne VII at the hottest temperature, so that nothing is lost off the ends
velocity = na.linspace(-400, 400, axis="velocity", num=801) * u.km / u.s


def _total(result: na.FunctionArray) -> na.AbstractScalar:
    """The radiance of each line, summed over the bins of velocity."""
    width = result.inputs.wavelength
    width = width[{"velocity": slice(1, None)}] - width[{"velocity": slice(None, -1)}]
    return (result.outputs * width).sum("velocity")


def _moments(result: na.FunctionArray) -> tuple[na.AbstractScalar, na.AbstractScalar]:
    """The mean and the variance of the velocity of each line."""
    v = result.inputs.velocity
    v = (v[{"velocity": slice(1, None)}] + v[{"velocity": slice(None, -1)}]) / 2
    weight = result.outputs / result.outputs.sum("velocity")
    mean = (weight * v).sum("velocity")
    variance = (weight * np.square(v - mean)).sum("velocity")
    return mean, variance


def test_synthesize():
    result = utu.spectrum.synthesize(
        contribution=contribution,
        temperature=temperature,
        mass=mass,
        emission_measure=emission_measure,
        velocity=velocity,
    )

    assert isinstance(result, na.FunctionArray)
    assert isinstance(result.inputs, na.DopplerVectorArray)
    assert result.outputs.unit.is_equivalent(u.erg / u.cm**2 / u.s / u.sr / u.AA)
    assert np.all(result.outputs >= 0)

    # every line has its own bins, at the velocities asked for
    assert na.shape(result.outputs) == dict(line=2, velocity=800)
    assert np.allclose(result.inputs.velocity, velocity)
    assert np.all(result.inputs.wavelength_rest == contribution.inputs)


def test_synthesize_conserves():
    """The bins of a line hold all of its radiance, however coarse they are."""
    expected = (contribution.outputs * emission_measure).sum("temperature")
    expected = expected / (4 * np.pi * u.sr)

    for num in [801, 9]:
        result = utu.spectrum.synthesize(
            contribution=contribution,
            temperature=temperature,
            mass=mass,
            emission_measure=emission_measure,
            velocity=na.linspace(-400, 400, axis="velocity", num=num) * u.km / u.s,
        )
        assert np.allclose(_total(result), expected, rtol=1e-10)


def test_synthesize_symmetric():
    """A line at rest is the same on either side of its rest wavelength."""
    result = utu.spectrum.synthesize(
        contribution=contribution,
        temperature=temperature,
        mass=mass,
        emission_measure=emission_measure,
        velocity=velocity,
    )

    reverse = result.outputs[{"velocity": slice(None, None, -1)}]
    assert np.allclose(reverse, result.outputs, rtol=1e-10)


@pytest.mark.parametrize(
    argnames="shift",
    argvalues=[-30, 0, 45] * u.km / u.s,
)
def test_synthesize_shift(shift: u.Quantity):
    result = utu.spectrum.synthesize(
        contribution=contribution,
        temperature=temperature,
        mass=mass,
        emission_measure=emission_measure,
        velocity=velocity,
        shift=shift,
    )

    mean, _ = _moments(result)
    assert np.allclose(mean, shift, atol=1e-6 * u.km / u.s)


@pytest.mark.parametrize(
    argnames="width_nonthermal",
    argvalues=[0, 20] * u.km / u.s,
)
def test_synthesize_width(width_nonthermal: u.Quantity):
    """Plasma at one temperature has the width of its ions' thermal motion."""
    index = {"temperature": slice(3, 4)}

    result = utu.spectrum.synthesize(
        contribution=contribution[index],
        temperature=temperature[index],
        mass=mass,
        emission_measure=emission_measure[index],
        velocity=velocity,
        width_nonthermal=width_nonthermal,
    )

    _, variance = _moments(result)

    expected = astropy.constants.k_B * temperature[index] / mass
    expected = expected + np.square(width_nonthermal)

    # Sheppard's correction: binning a Gaussian adds the variance of a uniform
    # distribution across one bin
    step = velocity[{"velocity": 1}] - velocity[{"velocity": 0}]
    expected = expected + np.square(step) / 12

    assert np.allclose(variance, expected, rtol=1e-6)


def test_synthesize_single():
    """A line on its own, without an axis of lines, is the same line."""
    result = utu.spectrum.synthesize(
        contribution=contribution,
        temperature=temperature,
        mass=mass,
        emission_measure=emission_measure,
        velocity=velocity,
    )

    for i in range(2):
        index = {"line": i}
        single = utu.spectrum.synthesize(
            contribution=contribution[index],
            temperature=temperature,
            mass=mass[index],
            emission_measure=emission_measure,
            velocity=velocity,
        )
        assert "line" not in na.shape(single.outputs)
        assert np.allclose(single.outputs, result.outputs[index], rtol=1e-12)


def test_synthesize_map():
    """
    The axes of the emission measure are carried through, and the shift can
    vary along them and with temperature, each pixel on its own.
    """
    em = emission_measure * na.ScalarArray(np.array([1, 2, 3]), axes="x")
    em = em * na.ScalarArray(np.array([1, 0.5]), axes="y")

    shift = na.ScalarArray([-20, 0, 30] * u.km / u.s, axes="x")
    shift = shift + na.linspace(0, 10, axis="temperature", num=6) * u.km / u.s

    result = utu.spectrum.synthesize(
        contribution=contribution,
        temperature=temperature,
        mass=mass,
        emission_measure=em,
        velocity=velocity,
        shift=shift,
    )

    assert na.shape(result.outputs) == dict(line=2, velocity=800, x=3, y=2)

    for i in range(3):
        for j in range(2):
            index = {"x": i, "y": j}
            pixel = utu.spectrum.synthesize(
                contribution=contribution,
                temperature=temperature,
                mass=mass,
                emission_measure=em[index],
                velocity=velocity,
                shift=shift[index],
            )
            assert np.allclose(pixel.outputs, result.outputs[index], rtol=1e-12)


@needs_database
def test_synthesize_lines():
    """The radiance of the lines is that :func:`utu.spectrum.lines` gives."""
    t = na.geomspace(1e5, 1e6, axis="temperature", num=6) * u.K
    density = 1e15 * u.K / u.cm**3 / t
    em = 1e27 / u.cm**5 * na.ScalarArray(np.ones(6), axes="temperature")

    ratio = fiasco.proton_electron_ratio(t.ndarray)

    ion = fiasco.Ion("O 5", t.ndarray)

    g = utu.spectrum.contribution_function(
        ion=ion,
        density=density,
        axis_temperature="temperature",
        proton_electron_ratio=ratio,
    )

    result = utu.spectrum.synthesize(
        contribution=g,
        temperature=t,
        mass=ion.mass,
        emission_measure=em,
        velocity=velocity,
    )

    expected = utu.spectrum.lines(
        temperature=t,
        density=density,
        emission_measure=em,
        ions=["O 5"],
        proton_electron_ratio=ratio,
    )

    total = _total(result).sum("line")
    expected = expected.outputs.sum("line") / (4 * np.pi * u.sr)
    assert np.allclose(total, expected, rtol=1e-10)
