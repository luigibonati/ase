import pytest

from ase.build import bulk, molecule
from ase.units import Hartree


@pytest.mark.calculator('elk', tasks=0, ngridk=(3, 3, 3))
def test_elk_bulk(factory):
    atoms = bulk('Si')
    calc = factory.calc()
    atoms.calc = calc
    props = atoms.get_properties(['energy', 'forces'])
    energy = props['energy']

    # Need more thorough tests.
    assert energy == pytest.approx(-15729.719246, abs=0.1)

    expected_props = {
        'energy', 'free_energy', 'forces', 'ibz_kpoints',
        'eigenvalues', 'occupations'
    }

    assert expected_props < set(props)
