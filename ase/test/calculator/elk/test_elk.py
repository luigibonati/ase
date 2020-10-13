import pytest

from ase.build import bulk, molecule
from ase.units import Hartree


@pytest.mark.calculator('elk', tasks=0, ngridk=(3, 3, 3))
def test_elk_bulk(factory):
    atoms = bulk('Si')
    calc = factory.calc()
    atoms.calc = calc
    props = atoms.get_properties(['energy', 'forces'])
    # XXX crashes due to forces even if forces not requested
    assert props


    keys = list(props)
    print(keys)

    expected_props = {
        'energy', 'free_energy', 'forces', 'ibz_kpoints',
        'eigenvalues', 'occupations'
    }

    assert expected_props < set(props)
    #print(props['energy'])
    #print(props['forces'])
    #forces = atoms.get_forces()
    #energy = atoms.get_potential_energy()
    #assert energy == pytest.approx(-15729.719246, abs=0.1)
    #assert forces.shape == (len(atoms), 3)
    # XXX Needs more test.

    #atoms.calculate_properties()
