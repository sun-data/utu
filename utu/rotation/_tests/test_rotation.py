import astropy.time
import astropy.units as u
import named_arrays as na
import numpy as np
import pytest

import utu

_time = astropy.time.Time("2026-09-02T04:45")


def _position(num_x: int = 5, num_y: int = 4) -> na.Cartesian2dVectorArray:
    """A patch of on-disk coordinates, well inside the limb."""
    return na.Cartesian2dVectorArray(
        x=na.linspace(-400, 400, axis="x", num=num_x) * u.arcsec,
        y=na.linspace(-300, 300, axis="y", num=num_y) * u.arcsec,
    )


@pytest.mark.parametrize("num", [1, 2, 3, 5])
def test_rotate_shape(num: int):
    """The result has the shape and the axes of the inputs, broadcast."""
    position = _position()

    result = utu.rotation.rotate(
        position=position,
        time=_time,
        time_out=_time + 1 * u.hour,
        num=num,
    )

    assert isinstance(result, na.Cartesian2dVectorArray)
    assert na.shape(result) == na.shape(position)
    assert result.x.unit.is_equivalent(u.arcsec)
    assert np.all(np.isfinite(result.x.ndarray))


@pytest.mark.parametrize("num", [1, 3])
def test_rotate_identity(num: int):
    """
    Rotating to the time already observed changes nothing.

    Not to the last bit: the coordinates go out to the solar surface and back
    through several frames, which costs a few times ten to the minus eight
    arcseconds. That is a ten-millionth of an IRIS pixel, so the tolerance
    here is set by floating point rather than by anything physical.
    """
    position = _position()

    result = utu.rotation.rotate(
        position=position,
        time=_time,
        time_out=_time,
        num=num,
    )

    shape = na.shape(position)
    for component in ("x", "y"):
        found = na.broadcast_to(getattr(result, component), shape)
        expected = na.broadcast_to(getattr(position, component), shape)
        difference = (found - expected).ndarray.to_value(u.arcsec)
        assert np.allclose(difference, 0, atol=1e-6)


def test_rotate_direction():
    """The Sun turns west, and faster at the equator than at the poles."""
    position = na.Cartesian2dVectorArray(
        x=0 * u.arcsec,
        y=na.ScalarArray(np.array([0.0, 800.0]) * u.arcsec, axes="y"),
    )

    result = utu.rotation.rotate(
        position=position,
        time=_time,
        time_out=_time + 1 * u.day,
    )

    shift = (result.x - position.x).ndarray.to_value(u.arcsec)

    assert np.all(shift > 0), "the Sun rotates towards solar west"
    assert shift[0] > shift[1], "the equator outruns the mid latitudes"


def test_rotate_time_array():
    """
    A time per raster step is honoured, not quietly replaced by one time.

    :mod:`sunpy` returns NaN for a multidimensional ``obstime``, so a wrapper
    which passes the times straight through would fail here.
    """
    axis = "x"
    num = 6

    position = na.Cartesian2dVectorArray(
        x=na.linspace(-400, 400, axis=axis, num=num) * u.arcsec,
        y=0 * u.arcsec,
    )
    time = na.ScalarArray(_time + np.arange(num) * u.hour, axes=(axis,))

    # Every point is rotated to the time of the first one, so the first moves
    # not at all and the rest move further the later they were observed.
    result = utu.rotation.rotate(
        position=position,
        time=time,
        time_out=_time,
    )

    shift = (result.x - position.x).ndarray.to_value(u.arcsec)

    assert np.all(np.isfinite(shift))
    assert np.isclose(shift[0], 0, atol=1e-6)
    assert np.all(np.diff(shift) < 0), "later steps rotate further back"


def _across_the_limb() -> na.Cartesian2dVectorArray:
    """One point at disk centre and one well beyond the limb."""
    return na.Cartesian2dVectorArray(
        x=na.ScalarArray(np.array([0.0, 1200.0]) * u.arcsec, axes="x"),
        y=0 * u.arcsec,
    )


def test_rotate_off_disk_nan():
    """Points beyond the limb have no surface to move along, and come back NaN."""
    position = _across_the_limb()

    result = utu.rotation.rotate(
        position=position,
        time=_time,
        time_out=_time + 1 * u.hour,
        off_disk="nan",
    )

    finite = np.isfinite(result.x.ndarray)

    assert finite[0], "a point at disk centre rotates"
    assert not finite[1], "a point beyond the limb does not"


def test_rotate_off_disk_static():
    """
    Points beyond the limb can be left where they were seen instead.

    The on-disk point must be unaffected by the choice, so that asking for the
    off-disk points does not quietly change the rest of the answer.
    """
    position = _across_the_limb()
    time_out = _time + 1 * u.hour

    kwargs = dict(position=position, time=_time, time_out=time_out)
    result = utu.rotation.rotate(**kwargs, off_disk="static")
    expected = utu.rotation.rotate(**kwargs, off_disk="nan")

    x = result.x.ndarray.to_value(u.arcsec)
    y = result.y.ndarray.to_value(u.arcsec)

    assert np.all(np.isfinite(x)), "nothing is dropped"

    # The point beyond the limb came back exactly where it was given.
    assert x[1] == 1200
    assert y[1] == 0

    # The point on the disk moved, and moved to where it would have anyway.
    assert x[0] != 0
    assert np.isclose(x[0], expected.x.ndarray.to_value(u.arcsec)[0], atol=1e-10)


def test_rotate_interpolation_accuracy():
    """
    Interpolating between a few anchors matches rotating every step exactly.

    The comparison is against this same function with one anchor per step,
    which does no interpolation at all, over a raster the length of one IRIS
    tile. More anchors must not do worse, and even two must land well inside
    a tenth of an IRIS pixel, which is 0.33 arcsec.
    """
    axis = "x"
    num_step = 40

    position = na.Cartesian2dVectorArray(
        x=na.linspace(-400, 400, axis=axis, num=num_step) * u.arcsec,
        y=na.linspace(-300, 300, axis="y", num=3) * u.arcsec,
    )
    time = na.ScalarArray(
        ndarray=_time + np.linspace(0, 1, num_step) * u.hour,
        axes=(axis,),
    )
    time_out = _time + 8 * u.hour

    kwargs = dict(position=position, time=time, time_out=time_out)

    exact = utu.rotation.rotate(**kwargs, num=num_step)

    error = {}
    for num in (2, 3, 5):
        result = utu.rotation.rotate(**kwargs, num=num)
        difference = np.hypot(
            (result.x - exact.x).ndarray.to_value(u.arcsec),
            (result.y - exact.y).ndarray.to_value(u.arcsec),
        )
        error[num] = np.nanmax(difference)

    assert error[2] < 0.033, "two anchors are within a tenth of an IRIS pixel"
    assert error[3] <= error[2]
    assert error[5] <= error[3]


@pytest.mark.parametrize(
    argnames="span,expected",
    argvalues=[
        (0 * u.hour, 2),
        (1 * u.hour, 2),
        (6 * u.hour, 7),
        (24 * u.hour, 25),
    ],
)
def test_rotate_num_from_span(span: u.Quantity, expected: int):
    """
    Left to itself, the rotation is computed at times an hour apart.

    There is no way to ask the result how many were used, so it is compared
    against asking for that many, which must give the same answer to the last
    bit, and against asking for one fewer, which must not.
    """
    axis = "x"
    num_point = 8

    position = na.Cartesian2dVectorArray(
        x=na.linspace(-300, 300, axis=axis, num=num_point) * u.arcsec,
        y=0 * u.arcsec,
    )
    time = na.ScalarArray(
        ndarray=_time + np.linspace(0, span.to_value(u.hour), num_point) * u.hour,
        axes=(axis,),
    )

    kwargs = dict(position=position, time=time, time_out=_time + 1 * u.day)

    automatic = utu.rotation.rotate(**kwargs).x.ndarray.to_value(u.arcsec)
    asked = utu.rotation.rotate(**kwargs, num=expected).x.ndarray.to_value(u.arcsec)

    assert np.allclose(automatic, asked, rtol=0, atol=1e-12)

    if expected > 2:
        fewer = utu.rotation.rotate(**kwargs, num=expected - 1)
        assert not np.allclose(
            automatic,
            fewer.x.ndarray.to_value(u.arcsec),
            rtol=0,
            atol=1e-12,
        )


def test_rotate_num_from_span_long():
    """
    A long observation stays accurate without being asked to.

    Three times spanning the range, which used to be the default, are worth
    a quarter of an arcsecond over a day. Spacing them an hour apart instead
    has to do far better than that, and better than a tenth of an IRIS pixel.
    """
    axis = "x"
    num_step = 25

    position = na.Cartesian2dVectorArray(
        x=na.linspace(-300, 300, axis=axis, num=num_step) * u.arcsec,
        y=na.linspace(-200, 200, axis="y", num=3) * u.arcsec,
    )
    time = na.ScalarArray(
        ndarray=_time + np.linspace(0, 24, num_step) * u.hour,
        axes=(axis,),
    )

    kwargs = dict(position=position, time=time, time_out=_time + 12 * u.hour)

    exact = utu.rotation.rotate(**kwargs, num=num_step)

    def error(result):
        return np.nanmax(
            np.hypot(
                (result.x - exact.x).ndarray.to_value(u.arcsec),
                (result.y - exact.y).ndarray.to_value(u.arcsec),
            )
        )

    automatic = error(utu.rotation.rotate(**kwargs))
    three = error(utu.rotation.rotate(**kwargs, num=3))

    assert automatic < 0.033, "inside a tenth of an IRIS pixel"
    assert automatic < three / 10, "and far better than a fixed three"


def test_rotate_num_one():
    """
    One anchor uses the middle of the time range and does not interpolate.

    It is the cheapest thing the function can do, and is what should be asked
    for when every point shares a time.
    """
    axis = "x"
    num = 4

    position = na.Cartesian2dVectorArray(
        x=na.linspace(-400, 400, axis=axis, num=num) * u.arcsec,
        y=0 * u.arcsec,
    )
    time = na.ScalarArray(_time + np.arange(num) * u.hour, axes=(axis,))

    result = utu.rotation.rotate(
        position=position,
        time=time,
        time_out=_time,
        num=1,
    )

    assert np.all(np.isfinite(result.x.ndarray))

    # Every point is treated as though it were observed at the middle of the
    # range, so the answer is the one a single time in the middle gives.
    middle = astropy.time.Time(
        (time.ndarray.jd.min() + time.ndarray.jd.max()) / 2, format="jd"
    )
    expected = utu.rotation.rotate(
        position=position,
        time=middle,
        time_out=_time,
    )

    difference = (result.x - expected.x).ndarray.to_value(u.arcsec)
    assert np.allclose(difference, 0, atol=1e-8)
