import pytest
from unittest import mock

import numpy as np
from ase.calculators.vasp.create_input import GenerateVaspInput
from ase.calculators.vasp.create_input import _args_without_comment
from ase.calculators.vasp.create_input import _to_vasp_bool, _from_vasp_bool

from ase.build import bulk


@pytest.fixture
def rng():
    return np.random.RandomState(seed=42)


@pytest.fixture
def nacl(rng):
    atoms = bulk('NaCl', crystalstructure='rocksalt', a=4.1,
                 cubic=True) * (3, 3, 3)
    rng.shuffle(atoms.symbols)  # Ensure symbols are mixed
    return atoms


@pytest.fixture
def vaspinput_factory(nacl):
    """Factory for GenerateVaspInput class, which mocks the generation of
    pseudopotentials."""
    def _vaspinput_factory(atoms, **kwargs) -> GenerateVaspInput:
        mocker = mock.Mock()
        inputs = GenerateVaspInput()
        inputs.set(**kwargs)
        inputs._build_pp_list = mocker(return_value=None)  # type: ignore
        inputs.initialize(atoms)
        return inputs

    return _vaspinput_factory


def test_sorting(nacl, vaspinput_factory):
    """Test that the sorting/resorting scheme works"""
    vaspinput = vaspinput_factory(nacl)
    srt = vaspinput.sort
    resrt = vaspinput.resort
    atoms = nacl.copy()
    assert atoms[srt] != nacl
    assert atoms[resrt] != nacl
    assert atoms[srt][resrt] == nacl

    # Check the first and second half of the sorted atoms have the same symbols
    assert len(atoms) % 2 == 0  # We should have an even number of atoms
    atoms_sorted = atoms[srt]
    N = len(atoms) // 2
    seq1 = set(atoms_sorted.symbols[:N])
    seq2 = set(atoms_sorted.symbols[N:])
    assert len(seq1) == 1
    assert len(seq2) == 1
    # Check that we have two different symbols
    assert len(seq1.intersection(seq2)) == 0


@pytest.fixture(params=['random', 'ones', 'binaries'])
def magmoms_factory(rng, request):
    """Factory for generating various kinds of magnetic moments"""
    kind = request.param
    if kind == 'random':
        # Array of random
        func = rng.rand
    elif kind == 'ones':
        # Array of just 1's
        func = np.ones
    elif kind == 'binaries':
        # Array of 0's and 1's
        def rand_binary(x):
            return rng.randint(2, size=x)

        func = rand_binary
    else:
        raise ValueError(f'Unknown kind: {kind}')

    def _magmoms_factory(atoms):
        magmoms = func(len(atoms))
        assert len(magmoms) == len(atoms)
        return magmoms

    return _magmoms_factory


def read_magmom_from_file(filename) -> np.ndarray:
    """Helper function to parse the magnetic moments from an INCAR file"""
    found = False
    with open(filename) as file:
        for line in file:
            # format "MAGMOM = n1*val1 n2*val2 ..."
            if 'MAGMOM = ' in line:
                found = True
                parts = line.strip().split()[2:]
                new_magmom = []
                for part in parts:
                    n, val = part.split('*')
                    # Add "val" to magmom "n" times
                    new_magmom += int(n) * [float(val)]
                break
    assert found
    return np.array(new_magmom)


@pytest.fixture
def assert_magmom_equal_to_incar_value():
    """Fixture to compare a pre-made magmom array to the value
    a GenerateVaspInput.write_incar object writes to a file"""
    def _assert_magmom_equal_to_incar_value(atoms, expected_magmom, vaspinput):
        assert len(atoms) == len(expected_magmom)
        vaspinput.write_incar(atoms)
        new_magmom = read_magmom_from_file('INCAR')
        assert len(new_magmom) == len(expected_magmom)
        srt = vaspinput.sort
        resort = vaspinput.resort
        # We round to 4 digits
        assert np.allclose(expected_magmom, new_magmom[resort], atol=1e-3)
        assert np.allclose(np.array(expected_magmom)[srt],
                           new_magmom,
                           atol=1e-3)

    return _assert_magmom_equal_to_incar_value


@pytest.mark.parametrize('list_func', [list, tuple, np.array])
def test_write_magmom(magmoms_factory, list_func, nacl, vaspinput_factory,
                      assert_magmom_equal_to_incar_value):
    """Test writing magnetic moments to INCAR, and ensure we can do it
    passing different types of sequences"""
    magmom = magmoms_factory(nacl)

    vaspinput = vaspinput_factory(nacl, magmom=magmom, ispin=2)
    assert vaspinput.spinpol
    assert_magmom_equal_to_incar_value(nacl, magmom, vaspinput)


def test_atoms_with_initial_magmoms(magmoms_factory, nacl, vaspinput_factory,
                                    assert_magmom_equal_to_incar_value):
    """Test passing atoms with initial magnetic moments"""
    magmom = magmoms_factory(nacl)
    assert len(magmom) == len(nacl)
    nacl.set_initial_magnetic_moments(magmom)
    vaspinput = vaspinput_factory(nacl)
    assert vaspinput.spinpol
    assert_magmom_equal_to_incar_value(nacl, magmom, vaspinput)


def test_vasp_from_bool():
    for s in ('T', '.true.'):
        assert _from_vasp_bool(s) is True
    for s in ('f', '.False.'):
        assert _from_vasp_bool(s) is False
    with pytest.raises(ValueError):
        _from_vasp_bool('yes')
    with pytest.raises(AssertionError):
        _from_vasp_bool(True)


def test_vasp_to_bool():
    for x in ('T', '.true.', True):
        assert _to_vasp_bool(x) == '.TRUE.'
    for x in ('f', '.FALSE.', False):
        assert _to_vasp_bool(x) == '.FALSE.'

    with pytest.raises(ValueError):
        _to_vasp_bool('yes')
    with pytest.raises(AssertionError):
        _to_vasp_bool(1)


@pytest.mark.parametrize('args, expected_len',
                         [(['a', 'b', '#', 'c'], 2),
                          (['a', 'b', '!', 'c', '#', 'd'], 2),
                          (['#', 'a', 'b', '!', 'c', '#', 'd'], 0)])
def test_vasp_args_without_comment(args, expected_len):
    """Test comment splitting logic"""
    clean_args = _args_without_comment(args)
    assert len(clean_args) == expected_len
