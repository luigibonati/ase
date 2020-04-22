from ase import Atoms
from ase.calculators.bond_polarizability import BondPolarizability
from ase.calculators.bond_polarizability import Linearized


def test_CC_bond():
    """Test polarizabilties of a single CC bond"""
    C2 = Atoms('C2', positions=[[0, 0, 0], [0, 0, 1.69]])
    bp = BondPolarizability()
    print(bp(C2))
    bp = BondPolarizability(Linearized())
    print(bp(C2))


def test_2to3():
    """Compare polarizabilties of one and two bonds"""
    Si2 = Atoms('Si2', positions=[[0, 0, 0], [0, 0, 2.5]])
    Si3 = Atoms('Si3', positions=[[0, 0, -2.5], [0, 0, 0], [0, 0, 2.5]])
    bp = BondPolarizability()
    bp2 = bp(Si2)
    # polarizability is a tensor
    assert bp2.shape == (3, 3)
    # check sum of equal bonds
    assert (bp(Si3) == 2 * bp2).all()
