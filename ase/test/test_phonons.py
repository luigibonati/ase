import ase.build
from ase.phonons import Phonons
from ase.calculators.emt import EMT


def test_set_atoms_indices(testdir):
    atoms = ase.build.molecule('CO2')
    atoms.calc = EMT()

    phonons = Phonons(atoms, EMT())
    phonons.set_atoms([0, 1])
    # Check that atom 2 was skipped.
    # TODO: This check is incomplete.
    #       Once there is a public API to iterate over/inspect displacements, we can
    #       get rid of the .run() call and look directly at the atom indices instead.
    phonons.run()
    assert len(phonons.cache) == 2 * 6 + 1


def test_set_atoms_symbol(testdir):
    atoms = ase.build.molecule('CO2')
    atoms.calc = EMT()

    phonons = Phonons(atoms, EMT())
    phonons.set_atoms(['O'])
    # Check that atom 0 was skipped.
    # TODO: This check is incomplete.
    #       Once there is a public API to iterate over/inspect displacements, we can
    #       get rid of the .run() call and look directly at the atom indices instead.
    phonons.run()
    assert len(phonons.cache) == 2 * 6 + 1


def test_check_eq_forces(testdir):
    atoms = ase.build.bulk('C')
    atoms.calc = EMT()

    phonons = Phonons(atoms, EMT(), supercell=(1, 2, 1))
    phonons.run()
    fmin, fmax, _i_min, _i_max = phonons.check_eq_forces()
    assert fmin < fmax


# Regression test for #953;  data stored for eq should resemble data for displacements
def test_check_consistent_format(testdir):
    atoms = ase.build.molecule('H2')
    atoms.calc = EMT()

    phonons = Phonons(atoms, EMT())
    phonons.run()

    # Check that the data stored for `eq` is shaped like the data stored for displacements.
    eq_data = phonons.cache['eq']
    disp_data = phonons.cache['0x-']
    assert isinstance(eq_data, dict) and isinstance(disp_data, dict)
    assert set(eq_data) == set(disp_data), "dict keys mismatch"
    for array_key in eq_data:
        assert eq_data[array_key].shape == disp_data[array_key].shape, array_key
