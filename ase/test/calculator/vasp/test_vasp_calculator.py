"""Test module for explicitly unittesting parts of the VASP calculator"""

import pytest

from ase.build import molecule
from ase.calculators.calculator import CalculatorSetupError, get_calculator_class
from ase.calculators.vasp import Vasp
from ase.calculators.vasp.vasp import check_atoms, check_pbc, check_cell, check_atoms_type


@pytest.fixture
def atoms():
    return molecule('H2', vacuum=5, pbc=True)


def test_check_atoms(atoms, mock_vasp_calculate):
    """Test checking atoms passes for a good atoms object"""
    check_atoms(atoms)
    check_pbc(atoms)
    check_cell(atoms)


@pytest.mark.parametrize(
    'bad_atoms',
    [
        None,
        'a_string',
        # We cannot handle lists of atoms either
        [molecule('H2', vacuum=5)],
    ])
def test_not_atoms(bad_atoms, mock_vasp_calculate):
    """Check that passing in objects which are not
    actually Atoms objects raises a setup error """

    with pytest.raises(CalculatorSetupError):
        check_atoms_type(bad_atoms)
    with pytest.raises(CalculatorSetupError):
        check_atoms(bad_atoms)

    # Test that error is also raised properly when launching
    # from calculator
    calc = Vasp()
    with pytest.raises(CalculatorSetupError):
        calc.calculate(atoms=bad_atoms)


@pytest.mark.parametrize('pbc', [
    3 * [False],
    [True, False, True],
    [False, True, False],
])
def test_bad_pbc(atoms, pbc, mock_vasp_calculate):
    """Test handling of PBC"""
    atoms.pbc = pbc

    check_cell(atoms)  # We have a cell, so this should not raise

    # Check that our helper functions raises the expected error
    with pytest.raises(CalculatorSetupError):
        check_pbc(atoms)
    with pytest.raises(CalculatorSetupError):
        check_atoms(atoms)

    # Check we also raise in the calculator when launching
    # a calculation, but before VASP is actually executed
    calc = Vasp()
    atoms.calc = calc
    with pytest.raises(CalculatorSetupError):
        atoms.get_potential_energy()


def test_vasp_no_cell(mock_vasp_calculate):
    """Check missing cell handling."""
    # Molecules come with no unit cell
    atoms = molecule('CH4')
    # We should not have a cell
    assert atoms.cell.rank == 0

    with pytest.raises(CalculatorSetupError):
        check_cell(atoms)
    with pytest.raises(CalculatorSetupError):
        check_atoms(atoms)

    with pytest.raises(RuntimeError):
        atoms.write('POSCAR')

    calc = Vasp()
    atoms.calc = calc
    with pytest.raises(CalculatorSetupError):
        atoms.get_total_energy()


def test_vasp_name(mock_vasp_calculate):
    """Test the calculator class has the expected name"""
    expected = 'vasp'
    assert Vasp.name == expected  # Test class attribute
    assert Vasp().name == expected  # Ensure instance attribute hasn't changed


def test_vasp_get_calculator(mock_vasp_calculate):
    cls_ = get_calculator_class('vasp')

    assert cls_ == Vasp

    # Test we get the correct calculator when loading from name
    assert get_calculator_class(Vasp.name) == cls_
