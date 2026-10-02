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

Every release of :mod:`utu` is archived on Zenodo with its own DOI.
The concept DOI,
`10.5281/zenodo.23107787 <https://doi.org/10.5281/zenodo.23107787>`_,
always resolves to the latest version,
and the Zenodo page lists the DOI of every version.
Please include the version of :mod:`utu` that you used,
which is given by ``importlib.metadata.version("utu")``.
The BibTeX entry below uses the concept DOI.
To cite a specific version instead,
replace ``doi`` with the DOI of that version.

.. code-block:: bibtex

    @software{utu,
      author = {Smart, Roy T.},
      title = {utu},
      version = {X.Y.Z},
      doi = {10.5281/zenodo.23107787},
      url = {https://github.com/sun-data/utu},
    }

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
