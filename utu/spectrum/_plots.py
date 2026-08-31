"""Drawing the emission lines of a spectrum."""

import adjustText
import matplotlib.axes
import matplotlib.pyplot as plt
import matplotlib.text
import named_arrays as na
import numpy as np

from ._names import spectroscopic

__all__ = [
    "stem",
]


def stem(
    spectrum: na.FunctionArray,
    ax: None | matplotlib.axes.Axes = None,
    num_label: None | int = None,
    latex: bool = False,
    headroom: float = 1.45,
    axis: str = "line",
    kwargs_line: None | dict = None,
    kwargs_text: None | dict = None,
    kwargs_adjust: None | dict = None,
) -> list[matplotlib.text.Text]:
    """
    Draw a spectrum as a stem from zero for each line, and label the
    brightest of them with the ion which emitted them.

    A spectrum of lines is mostly empty, and what matters about it is which
    line is where and how much brighter it is than its neighbours. Drawn
    this way it is the picture an instrument is designed against.

    The labels are moved apart so that none covers another or crosses a line
    it does not belong to. Each line is handed to the solver as points along
    its length rather than as a point at its tip, more of them for a
    brighter line, which is what keeps a label from being pushed across one.

    Parameters
    ----------
    spectrum
        The lines to draw, as :func:`~utu.spectrum.lines` returns them: a
        wavelength and an ion for each line, and its intensity. Every line
        given is drawn, so slice it first to draw fewer.
    ax
        The axes to draw on. If :obj:`None` (the default), the current axes.
    num_label
        How many of the brightest lines to label.
        If :obj:`None` (the default), all of them are labelled.
    latex
        Whether to set the name of each ion in LaTeX, which needs the axes
        to be rendering text through LaTeX to come out right.
    headroom
        How much taller than the brightest line to make the axes, so that
        the labels have somewhere to be pushed into.
    axis
        The name of the axis along the lines of ``spectrum``.
    kwargs_line
        Additional arguments passed to :meth:`matplotlib.axes.Axes.vlines`.
    kwargs_text
        Additional arguments passed to :meth:`matplotlib.axes.Axes.text`.
    kwargs_adjust
        Additional arguments passed to :func:`adjustText.adjust_text`.

    Returns
    -------
    The label of each line that was labelled, in the order they were drawn.

    Examples
    --------
    The brightest lines of two ions, from a plasma spread evenly over a
    decade of temperature.

    .. jupyter-execute::

        import astropy.units as u
        import matplotlib.pyplot as plt
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

        fig, ax = plt.subplots(figsize=(6, 3), constrained_layout=True)
        utu.spectrum.stem(result[{"line": slice(6)}], ax=ax)
        ax.set_xlabel(f"wavelength ({u.AA:latex_inline})")
        ax.set_ylabel(f"intensity ({result.outputs.unit:latex_inline})");
    """
    if ax is None:  # pragma: nocover
        ax = plt.gca()

    kwargs_line = kwargs_line if kwargs_line is not None else {}
    kwargs_text = kwargs_text if kwargs_text is not None else {}
    kwargs_adjust = kwargs_adjust if kwargs_adjust is not None else {}

    wavelength = na.value(spectrum.inputs.wavelength).ndarray
    intensity = na.value(spectrum.outputs).ndarray

    ax.vlines(
        x=wavelength,
        ymin=0,
        ymax=intensity,
        **{"color": "black", "linewidth": 1} | kwargs_line,
    )

    # room above the tallest line for the labels to be pushed into
    ax.set_ylim(0, intensity.max() * headroom)

    # Every line is given to the solver as points along its length, more of
    # them for a brighter line, so that a label is pushed away from a line
    # it would otherwise cross rather than only from the tip of it.
    x_static = []
    y_static = []
    for i in range(intensity.size):
        num = max(int(100 * intensity[i] / intensity.max()), 3)
        y = np.linspace(0, intensity[i], num=num)
        y_static.append(y)
        x_static.append(np.broadcast_to(wavelength[i], y.shape))

    # brightest first, so that taking the first few takes the brightest few
    order = np.argsort(spectrum.outputs, axis=axis)
    brightest = spectrum[order][{axis: slice(None, None, -1)}]
    brightest = brightest[{axis: slice(num_label)}]

    x_label = na.value(brightest.inputs.wavelength).ndarray
    y_label = na.value(brightest.outputs).ndarray
    name = spectroscopic(brightest.inputs.ion, latex=latex).ndarray

    text = [
        ax.text(
            x=x_label[i],
            y=y_label[i],
            s=f"{name[i]} {x_label[i]:.1f}",
            **{
                "ha": "center",
                "va": "bottom",
                "bbox": {
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.75,
                    "pad": 0.8,
                },
            }
            | kwargs_text,
        )
        for i in range(x_label.size)
    ]

    adjustText.adjust_text(
        texts=text,
        x=np.concatenate(x_static),
        y=np.concatenate(y_static),
        ax=ax,
        **{
            "arrowprops": {
                "arrowstyle": "-",
                "connectionstyle": "arc3",
                "alpha": 0.5,
                "linewidth": 0.5,
            },
            "force_static": (0.4, 0.6),
            "force_text": (0.4, 0.6),
            "expand": (1.15, 1.4),
            "max_move": (30, 30),
            "time_lim": 10,
        }
        | kwargs_adjust,
    )

    return text
