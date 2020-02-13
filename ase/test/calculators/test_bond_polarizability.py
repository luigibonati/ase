import numpy as np
import pytest

from ase import Atoms
from ase.calculators.bond_polarizability import BondPolarizability


def test_2to3():
    """Compare polarizabilties of one and two bonds"""
    Si2 = Atoms('Si2', positions=[[0, 0, 0], [0, 0, 2.5]])
    Si3 = Atoms('Si3', positions=[[0, 2.5, 0], [0, 0, 0], [0, 0, 2.5]])
    bp = BondPolarizability()
    assert bp(Si3) == 2 * bp(Si2)


def main():
    test_2to3()

if __name__ == '__main__':
    main()
