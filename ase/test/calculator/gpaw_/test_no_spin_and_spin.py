import pytest


@pytest.mark.calculator('gpaw')
def test_no_spin_and_spin(factory):
    from ase.build import molecule
    from ase import io

    txt = 'out.txt'

    calculator = factory.calc(h=0.3, txt=txt)
    atoms = molecule('H2', calculator=calculator)
    atoms.center(vacuum=3)
    atoms.get_potential_energy()
    atoms.set_initial_magnetic_moments([0.5, 0.5])
    calculator.set(charge=1)
    atoms.get_potential_energy()

    # read again
    t = io.read(txt, index=':')
    assert isinstance(t, list)
    M = t[1].get_magnetic_moments()
    assert abs(M - 0.2).max() < 0.1
