"""Configuration for the Sphinx documentation builder."""

import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path("..").resolve()))

project = "utu"
copyright = "2026, Roy T. Smart"
author = "Roy T. Smart"

extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "jupyter_sphinx",
    "nbsphinx",
    "sphinx_codeautolink",
    "sphinx_favicon",
]

autosummary_generate = True
autosummary_imported_members = True
autosummary_ignore_module_all = False
autodoc_typehints = "description"

templates_path = ["_templates"]
# `jupyter_sphinx` leaves a notebook behind for every page it runs code
# on, and `nbsphinx` would otherwise pick those up as pages of their own
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "jupyter_execute",
    "**.ipynb_checkpoints",
]

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/sun-data/utu",
            "icon": "fa-brands fa-github",
            "type": "fontawesome",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/utu/",
            "icon": "fa-brands fa-python",
        },
    ],
}

# https://github.com/readthedocs/readthedocs.org/issues/2569
master_doc = "index"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "astropy": ("https://docs.astropy.org/en/stable", None),
    "named_arrays": ("https://named-arrays.readthedocs.io/en/stable", None),
    "fiasco": ("https://fiasco.readthedocs.io/en/stable", None),
}

codeautolink_custom_blocks = {"jupyter-execute": None}

nbsphinx_execute = "always"


# -- The atomic database ----------------------------------------------------

ions = [
    "he_1",
    "o_3",
    "o_4",
    "o_5",
    "ne_7",
    "mg_10",
    "si_11",
]
"""
The ions the examples on these pages compute with.

Every page which reads the CHIANTI database is written against these ions,
and the database built below describes only these, so that a page can be
built in a minute rather than in twenty. The whole database is 1.6 GB of
text, almost all of it the collision strengths of iron, and turning it into
the form :mod:`fiasco` reads takes about as long as Read the Docs allows a
build to run in total.
"""

extension = [
    "auto",
    "cilvl",
    "diparams",
    "drparams",
    "easplom",
    "easplups",
    "elvlc",
    "fblvl",
    "psplups",
    "reclvl",
    "rrlvl",
    "rrparams",
    "scups",
    "trparams",
    "wgfa",
]
"""
The kinds of file an ion can have.

Not every ion has every one of them; the ones an ion does not have are
skipped as the database is built.
"""

if os.environ.get("READTHEDOCS") == "True":  # pragma: no cover
    import fiasco
    from fiasco.util import get_chianti_catalog
    from fiasco.util.setup_db import (
        _get_chianti_dbase_url,
        build_hdf5_dbase,
        download_dbase,
    )

    ascii_dbase_root = pathlib.Path(fiasco.defaults["ascii_dbase_root"])
    hdf5_dbase_root = pathlib.Path(fiasco.defaults["hdf5_dbase_root"])

    if not (ascii_dbase_root / "VERSION").is_file():
        # which version to fetch is asked of `fiasco` rather than written
        # down here, since the two are not the same question: the URL of a
        # version is not something one can guess from its number
        download_dbase(_get_chianti_dbase_url(), ascii_dbase_root)

    if not hdf5_dbase_root.is_file():
        catalog = get_chianti_catalog(ascii_dbase_root)
        files = [f"{ion}.{ext}" for ion in ions for ext in extension]
        files += catalog["abundance_files"]
        files += catalog["ioneq_files"]
        files += catalog["ip_files"]
        build_hdf5_dbase(
            ascii_dbase_root,
            hdf5_dbase_root,
            files=files,
            show_progress=False,
        )
