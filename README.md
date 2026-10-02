# utu

[![tests](https://github.com/sun-data/utu/actions/workflows/tests.yml/badge.svg)](https://github.com/sun-data/utu/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/sun-data/utu/graph/badge.svg)](https://codecov.io/gh/sun-data/utu)
[![Black](https://github.com/sun-data/utu/actions/workflows/black.yml/badge.svg)](https://github.com/sun-data/utu/actions/workflows/black.yml)
[![Ruff](https://github.com/sun-data/utu/actions/workflows/ruff.yml/badge.svg)](https://github.com/sun-data/utu/actions/workflows/ruff.yml)
[![Documentation Status](https://readthedocs.org/projects/utu/badge/?version=latest)](https://utu.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/utu.svg)](https://badge.fury.io/py/utu)

A Python library of solar physics utilities built on
[named arrays](https://github.com/sun-data/named-arrays).

Named for the Sumerian god of the sun.

## Documentation

The documentation is at [utu.readthedocs.io](https://utu.readthedocs.io/en/latest/).

## `utu.spectrum`

The emission lines of an optically thin plasma, computed from the CHIANTI
atomic database through [fiasco](https://github.com/wtbarnes/fiasco), and
returned as named arrays.

## Citation

If you use utu in your research, please cite it.
The citation metadata is kept in [`CITATION.cff`](https://github.com/sun-data/utu/blob/main/CITATION.cff),
which the "Cite this repository" button on GitHub can export as BibTeX or APA.
Please include the version of utu that you used,
which is given by `importlib.metadata.version("utu")`.

```bibtex
@software{utu,
  author = {Smart, Roy T.},
  title = {utu},
  version = {X.Y.Z},
  url = {https://github.com/sun-data/utu},
}
```
