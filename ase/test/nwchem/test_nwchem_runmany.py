import pytest
from ase.build import molecule
from ase.calculators.nwchem import NWChem
from numpy.testing import assert_allclose


@pytest.fixture()
def atoms():
    return molecule('H2O')


@pytest.mark.parametrize(
    'theory,eref,forces,kwargs',
    [
        ['dft', -2051.9802410863354, True, dict(basis='3-21G')],
        ['scf', -2056.7877421222634, True, dict(basis='3-21G')],
        ['mp2', -2060.1413846247333, True, dict(basis='3-21G')],
        ['ccsd', -2060.3418911515882, False, dict(basis='3-21G')],
        ['tce', -2060.319141863451, False, dict(
            basis='3-21G',
            tce={'ccd': None}
        )],
        ['tddft', -2044.3908422254976, True, dict(
            basis='3-21G',
            tddft=dict(
                nroots=2,
                algorithm=1,
                notriplet=None,
                target=1,
                civecs=None,
                grad={'root': 1},
            )
        )]
    ]
)
def test_nwchem(atoms, theory, eref, forces, kwargs):
    calc = NWChem(label=theory, theory=theory, **kwargs)
    atoms.calc = calc
    print(atoms.get_potential_energy())
    assert_allclose(atoms.get_potential_energy(), eref, atol=1e-4, rtol=1e-4)
    if forces:
        assert_allclose(atoms.get_forces(),
                        calc.calculate_numerical_forces(atoms),
                        atol=1e-4, rtol=1e-4)
