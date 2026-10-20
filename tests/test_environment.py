import importlib.util


def test_finance360_package_is_available():
    assert importlib.util.find_spec("finance360") is not None