utu
===

:mod:`utu` is a library of solar physics utilities built on
:mod:`named_arrays`, named for the Sumerian god of the sun.

Installation
============

:mod:`utu` is available on PyPI and can be installed using pip::

    pip install utu

Tutorials
=========

.. toctree::
    :maxdepth: 1

    tutorials/spectrum


API Reference
=============

.. autosummary::
    :toctree: _autosummary
    :template: module_custom.rst
    :recursive:

    utu

Citation
========

If you use :mod:`utu` in your research, please cite it.
The citation metadata is kept in
`CITATION.cff <https://github.com/sun-data/utu/blob/main/CITATION.cff>`_,
which the "Cite this repository" button on the
`GitHub page <https://github.com/sun-data/utu>`_
can export as BibTeX or APA.
Please include the version of :mod:`utu` that you used,
which is given by ``importlib.metadata.version("utu")``.

.. code-block:: bibtex

    @software{utu,
      author = {Smart, Roy T.},
      title = {utu},
      version = {X.Y.Z},
      url = {https://github.com/sun-data/utu},
    }

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
