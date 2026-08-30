"""Configuration for the Sphinx documentation builder."""

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
    "sphinx_codeautolink",
    "sphinx_favicon",
]

autosummary_generate = True
autosummary_imported_members = True
autosummary_ignore_module_all = False
autodoc_typehints = "description"

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

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
