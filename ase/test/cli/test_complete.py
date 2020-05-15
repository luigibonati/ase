"""Check that our tab-completion script has been updated."""
from ase.cli.completion import update_complete_dot_py


def test_complete():
    update_complete_dot_py(test=True)
