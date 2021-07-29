import pytest
import numpy as np

from ase.outputs import Properties, all_outputs


@pytest.fixture
def rng():
    return np.random.RandomState(17)


@pytest.fixture
def props(rng):
    nspins, nkpts, nbands = 2, 3, 5
    natoms = 4

    results = dict(
        natoms=natoms,
        energy=rng.rand(),
        free_energy=rng.rand(),
        energies=rng.rand(natoms),
        forces=rng.rand(natoms, 3),
        stress=rng.rand(6),
        stresses=rng.rand(natoms, 6),
        nspins=nspins,
        nkpts=nkpts,
        nbands=nbands,
        eigenvalues=rng.rand(nspins, nkpts, nbands),
        occupations=rng.rand(nspins, nkpts, nbands),
        fermi_level=rng.rand(),
        ibz_kpoints=rng.rand(nkpts, 3),
        kpoint_weights=rng.rand(nkpts),
        dipole=rng.rand(3),
        charges=rng.rand(natoms),
        magmom=rng.rand(),
        magmoms=rng.rand(natoms),
    )
    return Properties(results)


def test_properties_big(props):
    for name in all_outputs:
        assert name in props, name
        obj = props[name]
        print(name, obj)


def test_singlepoint_roundtrip(props):
    from ase.build import bulk
    from ase.calculators.singlepoint import (SinglePointDFTCalculator,
                                             arrays_to_kpoints)

    atoms = bulk('Au') * (1, 1, props['natoms'])

    kpts = arrays_to_kpoints(props['eigenvalues'], props['occupations'],
                             props['kpoint_weights'])
    calc = SinglePointDFTCalculator(atoms=atoms, kpts=kpts,
                                    forces=props['forces'])

    props1 = calc.properties()
    print(props1)

    for prop in ['eigenvalues', 'occupations', 'kpoint_weights']:
        assert props[prop] == pytest.approx(props1[prop])
