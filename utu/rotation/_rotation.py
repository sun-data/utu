"""Differential rotation of helioprojective coordinates."""

from typing import Literal

import astropy.coordinates
import astropy.time
import astropy.units as u
import named_arrays as na
import numpy as np
import sunpy.coordinates

__all__ = [
    "rotate",
]


def _observer(time: astropy.time.Time) -> astropy.coordinates.SkyCoord:
    """Earth at a single time."""
    return sunpy.coordinates.get_earth(time)


def _transform(
    x: u.Quantity,
    y: u.Quantity,
    time: astropy.time.Time,
    observer: astropy.coordinates.SkyCoord,
    time_out: astropy.time.Time,
    observer_out: astropy.coordinates.SkyCoord,
) -> tuple[u.Quantity, u.Quantity]:
    """
    Rotate flat arrays of coordinates, observed at one time, to a new time.

    `time` must be a scalar. :mod:`sunpy` builds one rotation matrix per point
    in a Python loop whenever it is given an array, and evaluates the
    ephemeris once per point rather than once per distinct time, which is
    hundreds of times slower for no gain in accuracy. Everything in this
    module is arranged to call this function with a scalar time.
    """
    frame = sunpy.coordinates.Helioprojective(obstime=time, observer=observer)
    frame_out = sunpy.coordinates.Helioprojective(
        obstime=time_out,
        observer=observer_out,
    )

    coordinate = astropy.coordinates.SkyCoord(x, y, frame=frame)

    with sunpy.coordinates.propagate_with_solar_surface():
        result = coordinate.transform_to(frame_out)

    return result.Tx, result.Ty


def rotate(
    position: na.AbstractCartesian2dVectorArray,
    time: astropy.time.Time | na.AbstractScalar,
    time_out: astropy.time.Time,
    num: int = 3,
    off_disk: Literal["nan", "static"] = "nan",
    observer: None | astropy.coordinates.SkyCoord = None,
    observer_out: None | astropy.coordinates.SkyCoord = None,
) -> na.Cartesian2dVectorArray:
    r"""
    Differentially rotate helioprojective coordinates to a new time.

    Each point is carried along the solar surface, which rotates faster at the
    equator than at the poles, and is then viewed from where the observer has
    moved to. This is :func:`sunpy.coordinates.propagate_with_solar_surface`
    expressed in named arrays.

    Parameters
    ----------
    position
        The helioprojective coordinates to rotate.
    time
        The time at which each coordinate was observed.
        Broadcast against `position`, so a raster may give one time per step.
    time_out
        The time to rotate the coordinates to.
    num
        The number of times at which to evaluate the rotation exactly.
        The rest is interpolated. See the notes below.
    off_disk
        What to do with points which miss the solar disk, and so have no
        surface to be carried along.
        If ``"nan"`` (the default), they come back as NaN.
        If ``"static"``, they come back where they were given, which keeps
        material above the limb in a mosaic instead of discarding it.
    observer
        The observer at `time`. If :obj:`None`, the Earth is used.
    observer_out
        The observer at `time_out`. If :obj:`None`, the Earth is used.

    Notes
    -----
    Points beyond the limb have no answer. Differential rotation moves a point
    along the solar surface, and a line of sight which misses the Sun never
    reaches that surface. :mod:`sunpy` returns NaN for these, which is what
    ``off_disk="nan"`` passes on.

    ``off_disk="static"`` returns them unmoved instead. That is the useful
    thing to do when assembling a mosaic, where the alternative is to throw
    away every spicule and prominence above the limb, but the two kinds of
    point then mean different things: on-disk ones have been carried to
    `time_out` and off-disk ones are still where they were seen. Nothing marks
    which is which in the result, so prefer the default unless the off-limb
    material is wanted.

    The remaining option, :class:`sunpy.coordinates.SphericalScreen`, is not
    offered here. It answers a different question, putting every point on a
    sphere through the observer, and it moves on-disk results by tens of
    arcseconds rather than leaving them alone.

    The cost of a rotation is set by the number of distinct times, not by the
    number of points: :mod:`sunpy` spends about 20 ms on a transform however
    many points it is given, but repeats that work for every time. Rotating a
    400-step raster honestly, one transform per step, therefore takes a few
    seconds, nearly all of it overhead.

    Since the rotation of any one point is smooth and nearly linear in time,
    this function instead rotates every point at `num` times spanning the
    range of `time`, and interpolates each point between the two surrounding
    ones. Against an exact per-step calculation for a 400-step raster over an
    hour, two of these give a worst-case error of 0.010 arcsec, three give
    0.004 arcsec, and five give 0.002 arcsec, which for reference is a
    hundredth of an IRIS pixel. Three is the default because the error is
    already far below where anything else in a pointing is uncertain. The
    error grows with the span of `time`, so raise `num` when rotating
    observations taken hours apart, and pass ``num=1`` to skip the
    interpolation when every point shares a time.

    Examples
    --------
    Rotate a point at disk centre forward by a day. It moves west by about a
    fifth of the solar radius.

    .. jupyter-execute::

        import astropy.time
        import astropy.units as u
        import named_arrays as na
        import utu

        position = na.Cartesian2dVectorArray(
            x=0 * u.arcsec,
            y=0 * u.arcsec,
        )

        utu.rotation.rotate(
            position=position,
            time=astropy.time.Time("2026-09-02T00:00"),
            time_out=astropy.time.Time("2026-09-03T00:00"),
        )

    Rotation is faster at the equator than at the poles, so a line of points
    at the same longitude does not stay straight.

    .. jupyter-execute::

        position = na.Cartesian2dVectorArray(
            x=0 * u.arcsec,
            y=na.linspace(-800, 800, axis="y", num=5) * u.arcsec,
        )

        result = utu.rotation.rotate(
            position=position,
            time=astropy.time.Time("2026-09-02T00:00"),
            time_out=astropy.time.Time("2026-09-03T00:00"),
        )

        result.x
    """
    if num < 1:  # pragma: nocover
        raise ValueError(f"{num=} must be at least one")

    if off_disk not in ("nan", "static"):  # pragma: nocover
        raise ValueError(f"{off_disk=} must be 'nan' or 'static'")

    time = na.as_named_array(time)

    shape = na.broadcast_shapes(position.shape, time.shape)
    axes = tuple(shape)
    shape_flat = tuple(shape.values())

    def flat(value: na.ArrayLike) -> np.ndarray | u.Quantity:
        """One component, broadcast to the common shape and flattened."""
        array = na.broadcast_to(na.as_named_array(value), shape)
        ndarray = array.ndarray_aligned(axes)
        if isinstance(ndarray, u.Quantity):
            # `numpy.broadcast_to` drops the unit from a quantity.
            value = np.broadcast_to(ndarray.value, shape_flat).ravel()
            return u.Quantity(value, ndarray.unit)
        return np.broadcast_to(ndarray, shape_flat).ravel()

    x = flat(position.x)
    y = flat(position.y)
    jd = astropy.time.Time(flat(time)).jd

    if observer_out is None:
        observer_out = _observer(time_out)

    # The times at which the rotation is computed exactly. A single one is
    # enough if every point shares a time, and then there is nothing to
    # interpolate between.
    if num == 1 or jd.min() == jd.max():
        anchor = np.array([(jd.min() + jd.max()) / 2])
    else:
        anchor = np.linspace(jd.min(), jd.max(), num)

    x_anchor = []
    y_anchor = []
    for a in anchor:
        time_anchor = astropy.time.Time(a, format="jd")
        observer_anchor = _observer(time_anchor) if observer is None else observer
        x_a, y_a = _transform(
            x=x,
            y=y,
            time=time_anchor,
            observer=observer_anchor,
            time_out=time_out,
            observer_out=observer_out,
        )
        x_anchor.append(x_a)
        y_anchor.append(y_a)

    if len(anchor) == 1:
        x_out, y_out = x_anchor[0], y_anchor[0]

    else:
        # Each point is interpolated between the two anchors surrounding its
        # own time, which is why the anchors are computed for every point
        # rather than only for the points that share their time.
        lower = np.clip(np.searchsorted(anchor, jd) - 1, 0, len(anchor) - 2)
        upper = lower + 1
        weight = (jd - anchor[lower]) / (anchor[upper] - anchor[lower])

        index = np.arange(jd.size)
        x_anchor = u.Quantity(x_anchor)
        y_anchor = u.Quantity(y_anchor)

        x_out = x_anchor[lower, index] * (1 - weight) + x_anchor[upper, index] * weight
        y_out = y_anchor[lower, index] * (1 - weight) + y_anchor[upper, index] * weight

    if off_disk == "static":
        # A point the rotation could not place is one which missed the solar
        # surface, so it is returned where it was given. A point which was
        # NaN to begin with stays NaN.
        missing = ~np.isfinite(x_out.value) | ~np.isfinite(y_out.value)
        x_out = u.Quantity(
            np.where(missing, x.to_value(x_out.unit), x_out.value), x_out.unit
        )
        y_out = u.Quantity(
            np.where(missing, y.to_value(y_out.unit), y_out.value), y_out.unit
        )

    return na.Cartesian2dVectorArray(
        x=na.ScalarArray(x_out.reshape(shape_flat), axes=axes),
        y=na.ScalarArray(y_out.reshape(shape_flat), axes=axes),
    )
