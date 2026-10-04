"""
The per-pixel iteration of :func:`utu.dem.plowman`, compiled with :mod:`numba`.

A transliteration of ``simple_reg_dem`` from the appendix of Plowman & Caspi
(2020) and from EMToolKit, which reproduces the reference to rounding error.
The arithmetic is kept in the order of the reference wherever the order can
matter, so that the comparison in the tests is a comparison of two
implementations of one algorithm rather than of two algorithms.

Nothing here allocates inside the iteration. Each thread allocates its work
arrays once, for a chunk of pixels, and reuses them for every pixel and every
step of that chunk.
"""

import numba
import numpy as np

__all__ = [
    "plowman",
]

_flags = dict(
    error_model="numpy",
    fastmath={"contract", "reassoc"},
)
"""
Compiler options shared by every function in this module.

``error_model="numpy"`` makes a division by zero return an infinity or a NaN
instead of raising, which the reference relies on: a step whose two trial
values of :math:`\\chi^2` are equal divides by zero, and the result is then
clipped like any other.

``contract`` and ``reassoc`` let the compiler fuse multiply-adds and reorder
the sums in the dot products, which is a third faster and moves the result
by about :math:`10^{-12}` relative, below the rounding error the reference
itself accumulates. ``reassoc`` also lets it simplify ``exp(log(x))`` to
``x``, which loses the NaN of the logarithm of a negative number, so the one
place which takes a logarithm, the initial guess, checks for that itself.
The remaining fast-math flags are left off, because ``nnan`` and ``ninf``
would allow the compiler to delete the checks which decide that a pixel has
failed.
"""


@numba.njit(inline="always", **_flags)
def _clip(x: float, lower: float, upper: float) -> float:
    """Clip like :func:`numpy.clip`, which passes NaN through."""
    if x < lower:
        return lower
    if x > upper:
        return upper
    return x


@numba.njit(**_flags)
def _chi2(
    data: np.ndarray,
    inverse_error: np.ndarray,
    rmat: np.ndarray,
    s: np.ndarray,
    exp_s: np.ndarray,
) -> float:
    """
    The reduced :math:`\\chi^2` of the coefficients ``exp(s)``.

    ``exp(s)`` is left in ``exp_s``, so that a caller about to need it does
    not compute it twice.
    """
    num_channel, num_temperature = rmat.shape
    for j in range(num_temperature):
        exp_s[j] = np.exp(s[j])
    total = 0.0
    for i in range(num_channel):
        model = 0.0
        for j in range(num_temperature):
            model += rmat[i, j] * exp_s[j]
        residual = (data[i] - model) * inverse_error[i]
        total += residual * residual
    return total / num_channel


@numba.njit(**_flags)
def _cholesky(a: np.ndarray) -> bool:
    """
    Factor the lower triangle of ``a`` in place into a Cholesky factor.

    Returns :obj:`False` if ``a`` is not positive definite or not finite,
    which is where the reference catches the exception raised by
    :func:`scipy.linalg.cho_factor`.
    """
    n = a.shape[0]
    for j in range(n):
        d = a[j, j]
        for k in range(j):
            d -= a[j, k] * a[j, k]
        if not (d > 0.0) or not np.isfinite(d):
            return False
        d = np.sqrt(d)
        a[j, j] = d
        for i in range(j + 1, n):
            x = a[i, j]
            for k in range(j):
                x -= a[i, k] * a[j, k]
            a[i, j] = x / d
    return True


@numba.njit(**_flags)
def _cholesky_solve(factor: np.ndarray, b: np.ndarray, x: np.ndarray) -> None:
    """Solve ``A x = b`` given the lower Cholesky factor of ``A``."""
    n = factor.shape[0]
    for i in range(n):
        v = b[i]
        for k in range(i):
            v -= factor[i, k] * x[k]
        x[i] = v / factor[i, i]
    for i in range(n - 1, -1, -1):
        v = x[i]
        for k in range(i + 1, n):
            v -= factor[k, i] * x[k]
        x[i] = v / factor[i, i]


@numba.njit(**_flags)
def _pixel(
    data_raw: np.ndarray,
    error: np.ndarray,
    rmat: np.ndarray,
    regmat: np.ndarray,
    rvec: np.ndarray,
    iterations_max: int,
    iterations_min: int,
    step_small: float,
    step_large: float,
    chi2_target: float,
    tolerance: float,
    floor: float,
    dem: np.ndarray,
    s: np.ndarray,
    exp_s: np.ndarray,
    s_trial: np.ndarray,
    exp_trial: np.ndarray,
    delta: np.ndarray,
    rhs: np.ndarray,
    solution: np.ndarray,
    gram: np.ndarray,
    a: np.ndarray,
    data: np.ndarray,
    inverse_error: np.ndarray,
    data_linear: np.ndarray,
) -> float:
    """
    The DEM of one pixel, written into ``dem``; returns its :math:`\\chi^2`.

    The arrays after ``dem`` are work space, overwritten here.
    """
    num_channel, num_temperature = rmat.shape

    # An uncertainty which is not positive fails the pixel. The reference
    # fails on NaN and zero the same way, but takes a negative one as positive.
    for i in range(num_channel):
        if not error[i] > 0.0:
            for j in range(num_temperature):
                dem[j] = np.nan
            return -1.0

    # The initial guess is the flat DEM which best fits the data, floored
    numerator = 0.0
    denominator = 0.0
    for i in range(num_channel):
        x = data_raw[i]
        data[i] = x if (x != x or x > 0.0) else 0.0
        inverse_error[i] = 1.0 / error[i]
        floored = data[i] if (data[i] != data[i] or data[i] > floor) else floor
        numerator += rvec[i] * (floored / error[i] ** 2)
        denominator += (rvec[i] / error[i]) ** 2
    ratio = numerator / denominator
    # exp(log(x)) compiles to x, which would turn the NaN of the logarithm of
    # a negative ratio back into a number, and the pixel would iterate on it
    if not ratio >= 0.0:
        ratio = np.nan
    s_initial = np.log(ratio)
    for j in range(num_temperature):
        s[j] = s_initial
        exp_s[j] = np.exp(s_initial)

    # R^T W R, which does not change from one step to the next
    for j in range(num_temperature):
        for k in range(j + 1):
            x = 0.0
            for i in range(num_channel):
                x += rmat[i, j] * inverse_error[i] * (rmat[i, k] * inverse_error[i])
            gram[j, k] = x

    chi2 = -1.0
    for iteration in range(iterations_max):
        # Linearize exp(s) about the current s, and add the regularization
        for j in range(num_temperature):
            for k in range(j + 1):
                a[j, k] = exp_s[j] * gram[j, k] * exp_s[k] + regmat[j, k]
        if not _cholesky(a):
            break

        chi2_current = 0.0
        for i in range(num_channel):
            model = 0.0
            model_linear = 0.0
            for j in range(num_temperature):
                model += rmat[i, j] * exp_s[j]
                model_linear += rmat[i, j] * ((1.0 - s[j]) * exp_s[j])
            residual = (data[i] - model) * inverse_error[i]
            chi2_current += residual * residual
            data_linear[i] = (data[i] - model_linear) / error[i]
        chi2_current /= num_channel

        for j in range(num_temperature):
            x = 0.0
            for i in range(num_channel):
                x += rmat[i, j] * inverse_error[i] * data_linear[i]
            rhs[j] = x * exp_s[j]
        _cholesky_solve(a, rhs, solution)

        # Limit the largest change of s to 0.5 at the small step. A numpy
        # float, so that with numba disabled a division by a zero `largest`
        # gives a NaN, as it does compiled, rather than raising.
        largest = np.float64(0.0)
        for j in range(num_temperature):
            delta[j] = solution[j] - s[j]
            d = abs(delta[j])
            if d > largest or d != d:
                largest = d
        limit = 0.5 / step_small
        scale = (largest if largest < limit or largest != largest else limit) / largest
        for j in range(num_temperature):
            delta[j] *= scale

        # Step toward the solution if chi2 is too large, away if too small,
        # by the fraction which interpolates to the target
        sign = -1.0 if chi2_current < chi2_target else 1.0
        for j in range(num_temperature):
            s_trial[j] = s[j] + delta[j] * sign * step_small
        chi2_small = _chi2(data, inverse_error, rmat, s_trial, exp_trial)
        for j in range(num_temperature):
            s_trial[j] = s[j] + delta[j] * sign * step_large
        chi2_large = _chi2(data, inverse_error, rmat, s_trial, exp_trial)
        step = (
            step_small * (chi2_large - chi2_target)
            + step_large * (chi2_target - chi2_small)
        ) / (chi2_large - chi2_small)
        step = _clip(step, step_small, step_large)
        for j in range(num_temperature):
            s[j] += delta[j] * sign * step
        chi2 = _chi2(data, inverse_error, rmat, s, exp_s)

        stalled = sign * (chi2_current - chi2_small) / step_small < tolerance
        if (stalled and iteration > iterations_min) or abs(
            chi2 - chi2_target
        ) < tolerance:
            break

    for j in range(num_temperature):
        dem[j] = np.exp(s[j])
    return chi2


@numba.njit(parallel=True, cache=True, **_flags)
def plowman(
    data: np.ndarray,
    errors: np.ndarray,
    rmat: np.ndarray,
    regmat: np.ndarray,
    rvec: np.ndarray,
    iterations_max: int,
    iterations_min: int,
    step_small: float,
    step_large: float,
    chi2_target: float,
    tolerance: float,
    floor: float,
    num_chunks: int,
    dems: np.ndarray,
    chi2: np.ndarray,
) -> None:
    """
    The DEM of every pixel, written into ``dems`` and ``chi2``.

    ``data`` and ``errors`` are ``(pixel, channel)``, ``rmat`` is
    ``(channel, temperature)``, ``regmat`` is ``(temperature, temperature)``,
    ``dems`` is ``(pixel, temperature)``, and ``chi2`` is ``(pixel,)``.
    ``floor`` is in the units of ``data``.

    The pixels are dealt into ``num_chunks`` chunks, every ``num_chunks``-th
    pixel to the same chunk, and each chunk allocates its work arrays once.
    Dealing them, rather than cutting the image into blocks, matters because
    :mod:`numba` hands each thread a contiguous block of chunks: in blocks, the
    threads which drew an active region would do most of the work while the
    rest sat idle, where dealt, every thread draws pixels from the whole
    image. ``num_chunks`` is passed in rather than computed here because
    asking :mod:`numba` for the number of threads from inside a compiled
    function makes it impossible to cache.
    """
    num_pixel, num_channel = data.shape
    num_temperature = rmat.shape[1]
    for c in numba.prange(num_chunks):
        s = np.empty(num_temperature)
        exp_s = np.empty(num_temperature)
        s_trial = np.empty(num_temperature)
        exp_trial = np.empty(num_temperature)
        delta = np.empty(num_temperature)
        rhs = np.empty(num_temperature)
        solution = np.empty(num_temperature)
        gram = np.empty((num_temperature, num_temperature))
        a = np.empty((num_temperature, num_temperature))
        data_clipped = np.empty(num_channel)
        inverse_error = np.empty(num_channel)
        data_linear = np.empty(num_channel)
        for p in range(c, num_pixel, num_chunks):
            chi2[p] = _pixel(
                data[p],
                errors[p],
                rmat,
                regmat,
                rvec,
                iterations_max,
                iterations_min,
                step_small,
                step_large,
                chi2_target,
                tolerance,
                floor,
                dems[p],
                s,
                exp_s,
                s_trial,
                exp_trial,
                delta,
                rhs,
                solution,
                gram,
                a,
                data_clipped,
                inverse_error,
                data_linear,
            )
