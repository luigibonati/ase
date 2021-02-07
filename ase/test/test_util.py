import pytest
from ase.utils import deprecated, devnull


def test_deprecated_decorator():

    class MyWarning(UserWarning):
        pass

    @deprecated('hello', MyWarning)
    def add(a, b):
        return a + b


    with pytest.warns(MyWarning, match='hello'):
        assert add(2, 2) == 4


def test_deprecated_devnull():
    with pytest.warns(DeprecationWarning):
        devnull.tell()
