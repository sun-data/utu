import itertools

import astropy.units as u
import matplotlib
import matplotlib.pyplot as plt
import named_arrays as na
import numpy as np
import pytest

import utu

matplotlib.use("agg")

# A spectrum written down rather than computed, so that these tests say
# nothing about the atomic database and do not need it to run.
spectrum = na.FunctionArray(
    inputs=na.CartesianNdVectorArray(
        components={
            "wavelength": na.ScalarArray(
                ndarray=np.array([584.3, 609.8, 629.7]) * u.AA,
                axes=("line",),
            ),
            "ion": na.ScalarArray(
                ndarray=np.array(["He 1", "Mg 10", "O 5"]),
                axes=("line",),
            ),
        },
    ),
    outputs=na.ScalarArray(
        ndarray=np.array([10.0, 30.0, 100.0]) * u.erg / u.s / u.cm**2,
        axes=("line",),
    ),
)


@pytest.mark.parametrize("num_label", [None, 1, 2])
def test_stem(num_label: None | int):
    fig, ax = plt.subplots()

    result = utu.spectrum.stem(spectrum, ax=ax, num_label=num_label)

    expected = num_label if num_label is not None else spectrum.shape["line"]
    assert len(result) == expected

    # the brightest line is labelled first, whichever order it was given in
    assert result[0].get_text().startswith("O V")

    # every line is drawn, whether or not it is labelled
    segments = [s for c in ax.collections for s in c.get_segments()]
    assert len(segments) == spectrum.shape["line"]

    # and there is room above the tallest of them for the labels
    assert ax.get_ylim()[1] > 100

    plt.close(fig)


def test_stem_latex():
    """The name of an ion, as it would be set in a journal."""
    fig, ax = plt.subplots()

    result = utu.spectrum.stem(spectrum, ax=ax, num_label=1, latex=True)

    assert result[0].get_text().startswith(r"O\,\textsc{v}")

    plt.close(fig)


def test_stem_kwargs():
    """Every default the caller might disagree with can be replaced."""
    fig, ax = plt.subplots()

    result = utu.spectrum.stem(
        spectrum,
        ax=ax,
        num_label=1,
        kwargs_line={"color": "red"},
        kwargs_text={"fontsize": 5},
        kwargs_adjust={"time_lim": 1},
    )

    assert result[0].get_fontsize() == 5
    assert np.all(ax.collections[0].get_color() == np.array([[1, 0, 0, 1]]))

    plt.close(fig)


# The eight brightest lines of the quiet Sun in the passband of ESIS, which is
# the arrangement `stem` was written for and the one which found every way it
# had of going wrong. Two of them are a hundredth of an angstrom apart.
passband = na.FunctionArray(
    inputs=na.CartesianNdVectorArray(
        components={
            "wavelength": na.ScalarArray(
                ndarray=np.array(
                    [562.80, 584.33, 599.59, 608.40, 609.79, 609.83, 624.94, 629.73]
                )
                * u.AA,
                axes=("line",),
            ),
            "ion": na.ScalarArray(
                ndarray=np.array(
                    ["Ne 6", "He 1", "O 3", "O 4", "Mg 10", "O 4", "Mg 10", "O 5"]
                ),
                axes=("line",),
            ),
        },
    ),
    outputs=na.ScalarArray(
        ndarray=np.array([15.8, 148.5, 26.8, 12.6, 54.6, 23.4, 26.8, 219.2])
        * u.erg
        / u.s
        / u.cm**2
        / u.sr,
        axes=("line",),
    ),
)


def _boxes(texts, renderer) -> dict:
    """
    What each label covers on the page.

    The box which is drawn rather than the glyphs inside it, since it is the
    box which hides whatever is under it, and it is larger than the glyphs by
    its padding.
    """
    result = {}
    for text in texts:
        patch = text.get_bbox_patch()
        extent = patch.get_window_extent(renderer) if patch else None
        result[text] = extent or text.get_window_extent(renderer)
    return result


def _touches(box, a, b) -> bool:
    """Whether the segment from ``a`` to ``b`` passes through ``box``."""
    num = max(int(np.hypot(*(b - a))), 8)
    for s in np.linspace(0, 1, num):
        x, y = a + s * (b - a)
        if box.x0 <= x <= box.x1 and box.y0 <= y <= box.y1:
            return True
    return False


def test_stem_collisions():
    """
    Nothing a label covers is anything a reader needs.

    A label lying along the axis, a label over the leader of another, and a
    label over a line have each been drawn by this function, and each was
    found by looking at the figure rather than by anything here. The four of
    them are what this measures.

    At the width this is drawn at, which is the width of the text of a
    journal page. Eight labels do not fit into much less than that without
    one of them covering something: at four inches this same spectrum still
    hides a leader and crosses a line, and no arrangement of the solver
    tried here avoids it. What is asserted is therefore what is achievable,
    not what would be ideal.
    """
    fig, ax = plt.subplots(figsize=(7.1, 2.4), constrained_layout=True)
    texts = utu.spectrum.stem(passband, ax=ax, kwargs_text={"fontsize": 6})
    fig.canvas.draw()

    renderer = fig.canvas.get_renderer()
    box = _boxes(texts, renderer)

    # no label lying along the axis the lines stand on
    floor = ax.transData.transform([[0, 0]])[0][1]
    for text in texts:
        assert box[text].y0 >= floor

    # no label over another
    for a, b in itertools.combinations(texts, 2):
        overlap = matplotlib.transforms.Bbox.intersection(box[a], box[b])
        assert overlap is None or overlap.width <= 0 or overlap.height <= 0

    # no label over a leader which is not its own. The leader is clipped to
    # start at the label it belongs to, so it can only ever strike another.
    leaders = [
        (p.patchA, p.get_path().transformed(p.get_transform()).vertices)
        for p in ax.patches
        if isinstance(p, matplotlib.patches.FancyArrowPatch)
    ]
    assert len(leaders) == len(texts)
    for text in texts:
        for owner, vertices in leaders:
            if owner is text:
                continue
            for a, b in zip(vertices[:-1], vertices[1:]):
                assert not _touches(box[text], a, b)

    # no label over a line
    for collection in ax.collections:
        for segment in collection.get_segments():
            a, b = ax.transData.transform(segment)[[0, -1]]
            for text in texts:
                assert not _touches(box[text], a, b)

    plt.close(fig)
