import utu


def test_version():
    """The package reports the version setuptools-scm gave it."""
    assert isinstance(utu.__version__, str)
    assert utu.__version__
