"""Differential rotation of helioprojective coordinates."""

import math
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

_spacing = 1 * u.hour
"""
How far apart to put the times at which the rotation is computed exactly,
when the caller does not say how many of them there should be.

The interpolation error grows as the square of this, so it is the spacing
and not the number of times which has to be held fixed as the span of an
observation grows. An hour is worth about 0.002 arcsec over a day and 0.010
arcsec over a whole disk transit, which is inside a pixel of any instrument
likely to ask.
"""


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
    num: None | int = None,
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
        If :obj:`None` (the default), enough of them are used to space them
        an hour apart, which is one for a raster and a few hundred for an
        observation which follows a feature across the disk.
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
    ones.

    What that costs in accuracy is set by how far apart those times are, not
    by how many of them there are, and it grows as the square of the spacing.
    A fixed `num` is therefore only ever right for one length of observation.
    Against an exact calculation at every step, three times spanning the
    range are worth 0.004 arcsec over an hour but 0.22 arcsec over a day and
    25 arcsec over a week, which is a way to be badly and quietly wrong about
    a long observation. Holding the spacing at an hour instead costs 0.002
    arcsec over a day and 0.010 arcsec over a disk transit.

    So `num` is chosen from the span of `time` unless it is given, and what
    it is chosen to be is ``_spacing`` apart. A raster wants two, a mosaic
    taking a day wants twenty-five, and following a feature from one limb to
    the other wants a few hundred, which is the right price for the answer.
    Pass ``num=1`` to skip the interpolation when every point shares a time.

    Examples
    --------
    Take a grid of points across the disk and rotate it forward by a day,
    drawing a line from where each point was to where the rotation puts it.
    The lines are shortest near the limb, where the motion is mostly toward
    the observer and hardly changes where a point appears to be, and shorter
    toward the poles, which turn more slowly than the equator.

    .. jupyter-execute::

        import astropy.time
        import astropy.units as u
        import matplotlib.pyplot as plt
        import named_arrays as na
        import sunpy.coordinates.sun
        import utu

        time = astropy.time.Time("2026-09-02T00:00")
        radius = sunpy.coordinates.sun.angular_radius(time)

        start = na.Cartesian2dVectorArray(
            x=na.linspace(-600, 600, axis="x", num=7) * u.arcsec,
            y=na.linspace(-600, 600, axis="y", num=7) * u.arcsec,
        )

        end = utu.rotation.rotate(
            position=start,
            time=time,
            time_out=time + 1 * u.day,
        )

        # the two ends of each line, along an axis of their own
        track = na.Cartesian2dVectorArray(
            x=na.stack([na.broadcast_to(start.x, end.shape), end.x], axis="end"),
            y=na.stack([na.broadcast_to(start.y, end.shape), end.y], axis="end"),
        )

        fig, ax = plt.subplots(figsize=(5, 5), constrained_layout=True)
        ax.add_patch(plt.Circle((0, 0), radius.to_value(u.arcsec), color="0.95"))
        na.plt.plot(track.x, track.y, ax=ax, axis="end", color="tab:blue")
        na.plt.scatter(start.x, start.y, ax=ax, s=10, color="black")
        ax.set_aspect("equal")
        ax.set_xlabel(f"helioprojective $x$ ({u.arcsec:latex_inline})")
        ax.set_ylabel(f"helioprojective $y$ ({u.arcsec:latex_inline})");

    Which is what makes the rotation differential: a meridian, whose points
    all start at the same longitude, does not stay straight.

    .. jupyter-execute::

        meridian = na.Cartesian2dVectorArray(
            x=0 * u.arcsec,
            y=na.linspace(-850, 850, axis="y", num=25) * u.arcsec,
        )

        fig, ax = plt.subplots(figsize=(5, 5), constrained_layout=True)
        ax.add_patch(plt.Circle((0, 0), radius.to_value(u.arcsec), color="0.95"))

        for day in range(5):
            rotated = utu.rotation.rotate(
                position=meridian,
                time=time,
                time_out=time + day * u.day,
            )
            na.plt.plot(rotated.x, rotated.y, ax=ax, axis="y", label=f"{day} d")

        ax.legend()
        ax.set_aspect("equal")
        ax.set_xlabel(f"helioprojective $x$ ({u.arcsec:latex_inline})")
        ax.set_ylabel(f"helioprojective $y$ ({u.arcsec:latex_inline})");
    """
    if num is not None and num < 1:  # pragma: nocover
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

    # The times at which the rotation is computed exactly. Enough of them to
    # be `_spacing` apart, unless the caller said how many to use. A single
    # one is enough if every point shares a time, and then there is nothing
    # to interpolate between.
    span = (jd.max() - jd.min()) * u.day

    if num is None:
        # Rounded before the ceiling, so that a span which is a whole number
        # of `_spacing` does not gain a time it does not need. A Julian date
        # is a number near two and a half million, so the difference of two
        # of them is only good to about forty microseconds, and an hour comes
        # out of it as a shade over an hour.
        ratio = float((span / _spacing).to_value(u.dimensionless_unscaled))
        num = max(math.ceil(round(ratio, 6)) + 1, 2)

    if num == 1 or span == 0:
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
