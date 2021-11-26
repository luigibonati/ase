# flake8: noqa
import numpy as np
from ase.io import read
from ase.io.aims import AimsOutChunk, AimsOutHeaderChunk, AimsOutCalcChunk
from ase.stress import full_3x3_to_voigt_6_stress

from numpy.linalg import norm

import pytest


@pytest.fixture
def default_chunk():
    lines = ["TEST", "A", "TEST", "| Number of atoms: 200 atoms"]
    return AimsOutChunk(lines)


def test_reverse_search_for(default_chunk):
    assert default_chunk.reverse_search_for(["TEST"]) == 2
    assert default_chunk.reverse_search_for(["TEST"], 1) == 2

    assert default_chunk.reverse_search_for(["TEST A"]) == 4

    assert default_chunk.reverse_search_for(["A"]) == 1
    assert default_chunk.reverse_search_for(["A"], 2) == 4


def test_search_for_all(default_chunk):
    assert default_chunk.search_for_all("TEST") == [0, 2]
    assert default_chunk.search_for_all("TEST", 0, 1) == [0]
    assert default_chunk.search_for_all("TEST", 1, -1) == [2]


def test_search_parse_scalar(default_chunk):
    assert default_chunk.parse_scalar("n_atoms") == 200
    assert default_chunk.parse_scalar("n_electrons") is None


@pytest.fixture
def empty_header_chunk():
    return AimsOutHeaderChunk([])


def test_default_header_n_atoms_raises_error(empty_header_chunk):
    with pytest.raises(IOError) as excinfo:
        n_atoms = empty_header_chunk.n_atoms

    assert "No information about the number of atoms in the header." in str(excinfo)


def test_default_header_n_bands_raises_error(empty_header_chunk):
    with pytest.raises(IOError) as excinfo:
        n_atoms = empty_header_chunk.n_bands

    assert "No information about the number of Kohn-Sham states in the header." in str(
        excinfo
    )


def test_default_header_n_electrons_raises_error(empty_header_chunk):
    with pytest.raises(IOError) as excinfo:
        n_atoms = empty_header_chunk.n_electrons

    assert "No information about the number of electrons in the header." in str(excinfo)


def test_default_header_n_spins_raises_error(empty_header_chunk):
    with pytest.raises(IOError) as excinfo:
        n_atoms = empty_header_chunk.n_spins

    assert "No information about the number of spin channels in the header." in str(
        excinfo
    )


def test_default_header_atoms_raises_error(empty_header_chunk):
    with pytest.raises(IOError) as excinfo:
        atoms = empty_header_chunk.initial_atoms

    assert "No structure information is inside the chunk." in str(excinfo)


def test_default_header_electronic_temperature(empty_header_chunk):
    assert empty_header_chunk.electronic_temperature == 0.1


def test_default_header_constraints(empty_header_chunk):
    assert empty_header_chunk.constraints is None


def test_default_header_initial_cell(empty_header_chunk):
    assert empty_header_chunk.initial_cell is None


def test_default_header_is_md(empty_header_chunk):
    assert not empty_header_chunk.is_md


def test_default_header_is_relaxation(empty_header_chunk):
    assert not empty_header_chunk.is_relaxation


def test_default_header_n_k_points(empty_header_chunk):
    assert empty_header_chunk.n_k_points is None


def test_default_header_k_points(empty_header_chunk):
    assert empty_header_chunk.k_points is None


def test_default_header_k_point_weights(empty_header_chunk):
    assert empty_header_chunk.k_point_weights is None


@pytest.fixture
def header_chunk():
    lines = [
        "| Number of atoms                   :        2",
        "| Number of Kohn-Sham states (occupied + empty):       20",
        "The structure contains        2 atoms,  and a total of         28.000 electrons.",
        "| Number of spin channels           :        2",
        "Occupation type: Gaussian broadening, width =   0.500000E-01 eV.",
        "Found relaxation constraint for atom      1: x coordinate fixed.",
        "Found relaxation constraint for atom      1: y coordinate fixed.",
        "Found relaxation constraint for atom      2: All coordinates fixed.",
        "Input geometry:",
        "| Unit cell:",
        "|        0.00000000        2.70300000        2.70300000",
        "|        2.70300000        0.00000000        2.70300000",
        "|        2.70300000        2.70300000        0.00000000",
        "| Atomic structure:",
        "|       Atom                x [A]            y [A]            z [A]",
        "|    1: Species Na            0.00000000        0.00000000        0.00000000",
        "|    2: Species Cl            2.70300000        2.70300000        2.70300000",
        "Initializing the k-points",
        "Using symmetry for reducing the k-points",
        "| k-points reduced from:        8 to        8",
        "| Number of k-points                             :         8",
        "The eigenvectors in the calculations are REAL.",
        "| K-points in task   0:         2",
        "| K-points in task   1:         2",
        "| K-points in task   2:         2",
        "| K-points in task   3:         2",
        "| k-point: 1 at     0.000000    0.000000    0.000000  , weight:   0.12500000",
        "| k-point: 2 at     0.000000    0.000000    0.500000  , weight:   0.12500000",
        "| k-point: 3 at     0.000000    0.500000    0.000000  , weight:   0.12500000",
        "| k-point: 4 at     0.000000    0.500000    0.500000  , weight:   0.12500000",
        "| k-point: 5 at     0.500000    0.000000    0.000000  , weight:   0.12500000",
        "| k-point: 6 at     0.500000    0.000000    0.500000  , weight:   0.12500000",
        "| k-point: 7 at     0.500000    0.500000    0.000000  , weight:   0.12500000",
        "| k-point: 8 at     0.500000    0.500000    0.500000  , weight:   0.12500000",
        'Geometry relaxation: A file "geometry.in.next_step" is written out by default after each step.',
        "Complete information for previous time-step:",
    ]
    return AimsOutHeaderChunk(lines)


def test_header_n_atoms(header_chunk):
    assert header_chunk.n_atoms == 2


def test_header_n_bands(header_chunk):
    assert header_chunk.n_bands == 20


def test_header_n_electrons(header_chunk):
    assert header_chunk.n_electrons == 28


def test_header_n_spins(header_chunk):
    assert header_chunk.n_spins == 2


def test_header_constraints(header_chunk):
    assert len(header_chunk.constraints) == 2
    assert header_chunk.constraints[0].index == 0
    assert header_chunk.constraints[1].index == 1
    assert np.all(header_chunk.constraints[0].mask == [False, False, True])


def test_header_initial_atoms(header_chunk):
    assert len(header_chunk.initial_atoms) == 2
    assert np.allclose(
        header_chunk.initial_atoms.cell,
        np.array([[0.000, 2.703, 2.703], [2.703, 0.000, 2.703], [2.703, 2.703, 0.000]]),
    )
    assert np.allclose(
        header_chunk.initial_atoms.positions,
        np.array([[0.000, 0.000, 0.000], [2.703, 2.703, 2.703]]),
    )
    assert np.all(["Na", "Cl"] == header_chunk.initial_atoms.symbols)
    assert all(
        [
            str(const_1) == str(const_2)
            for const_1, const_2 in zip(
                header_chunk.constraints, header_chunk.initial_atoms.constraints
            )
        ]
    )


def test_header_initial_cell(header_chunk):
    assert np.allclose(
        header_chunk.initial_cell,
        np.array([[0.000, 2.703, 2.703], [2.703, 0.000, 2.703], [2.703, 2.703, 0.000]]),
    )


def test_header_electronic_temperature(header_chunk):
    assert header_chunk.electronic_temperature == 0.05


def test_header_is_md(header_chunk):
    assert header_chunk.is_md


def test_header_is_relaxation(header_chunk):
    assert header_chunk.is_relaxation


def test_header_n_k_points(header_chunk):
    assert header_chunk.n_k_points == 8


@pytest.fixture
def k_points():
    return np.array(
        [
            [0.000, 0.000, 0.000],
            [0.000, 0.000, 0.500],
            [0.000, 0.500, 0.000],
            [0.000, 0.500, 0.500],
            [0.500, 0.000, 0.000],
            [0.500, 0.000, 0.500],
            [0.500, 0.500, 0.000],
            [0.500, 0.500, 0.500],
        ]
    )


@pytest.fixture
def k_point_weights():
    return np.full((8), 0.125)


def test_header_k_point_weights(header_chunk, k_point_weights):
    assert np.allclose(header_chunk.k_point_weights, k_point_weights)


def test_header_k_points(header_chunk, k_points):
    assert np.allclose(header_chunk.k_points, k_points)


def test_header_header_summary(header_chunk, k_points, k_point_weights):
    header_summary = {
        "initial_atoms": header_chunk.initial_atoms,
        "initial_cell": header_chunk.initial_cell,
        "constraints": header_chunk.constraints,
        "is_relaxation": True,
        "is_md": True,
        "n_atoms": 2,
        "n_bands": 20,
        "n_electrons": 28,
        "n_spins": 2,
        "electronic_temperature": 0.05,
        "n_k_points": 8,
        "k_points": k_points,
        "k_point_weights": k_point_weights,
    }
    for key, val in header_chunk.header_summary.items():
        if isinstance(val, np.ndarray):
            assert np.allclose(val, header_summary[key])
        else:
            assert val == header_summary[key]


@pytest.fixture
def empty_calc_chunk(header_chunk):
    return AimsOutCalcChunk([], header_chunk)


def test_header_transfer_n_atoms(empty_calc_chunk):
    assert empty_calc_chunk.n_atoms == 2


def test_header_transfer_n_bands(empty_calc_chunk):
    assert empty_calc_chunk.n_bands == 20


def test_header_transfer_n_electrons(empty_calc_chunk):
    assert empty_calc_chunk.n_electrons == 28


def test_header_transfer_n_spins(empty_calc_chunk):
    assert empty_calc_chunk.n_spins == 2


def test_header_transfer_constraints(empty_calc_chunk):
    assert len(empty_calc_chunk.constraints) == 2
    assert empty_calc_chunk.constraints[0].index == 0
    assert empty_calc_chunk.constraints[1].index == 1
    assert np.all(empty_calc_chunk.constraints[0].mask == [False, False, True])


def test_header_transfer_initial_cell(empty_calc_chunk):
    assert np.allclose(
        empty_calc_chunk.initial_cell,
        np.array([[0.000, 2.703, 2.703], [2.703, 0.000, 2.703], [2.703, 2.703, 0.000]]),
    )


def test_header_transfer_initial_atoms(empty_calc_chunk):
    assert np.allclose(
        empty_calc_chunk.initial_atoms.cell, empty_calc_chunk.initial_cell
    )
    assert len(empty_calc_chunk.initial_atoms) == 2
    assert np.allclose(
        empty_calc_chunk.initial_atoms.positions,
        np.array([[0.000, 0.000, 0.000], [2.703, 2.703, 2.703]]),
    )
    assert np.all(["Na", "Cl"] == empty_calc_chunk.initial_atoms.symbols)
    assert all(
        [
            str(const_1) == str(const_2)
            for const_1, const_2 in zip(
                empty_calc_chunk.constraints, empty_calc_chunk.initial_atoms.constraints
            )
        ]
    )


def test_header_transfer_electronic_temperature(empty_calc_chunk):
    assert empty_calc_chunk.electronic_temperature == 0.05


def test_header_transfer_n_k_points(empty_calc_chunk):
    assert empty_calc_chunk.n_k_points == 8


def test_header_transfer_k_point_weights(empty_calc_chunk, k_point_weights):
    assert np.allclose(empty_calc_chunk.k_point_weights, k_point_weights)


def test_header_transfer_k_points(empty_calc_chunk, k_points):
    assert np.allclose(empty_calc_chunk.k_points, k_points)


def test_default_calc_energy_raises_error(empty_calc_chunk):
    with pytest.raises(IOError) as excinfo:
        energy = empty_calc_chunk.energy

    assert "No energy is associated with the structure." in str(excinfo)


def test_default_calc_forces(empty_calc_chunk):
    assert empty_calc_chunk.forces is None


def test_default_calc_stresses(empty_calc_chunk):
    assert empty_calc_chunk.stresses is None


def test_default_calc_stress(empty_calc_chunk):
    assert empty_calc_chunk.stress is None


def test_default_calc_free_energy(empty_calc_chunk):
    assert empty_calc_chunk.free_energy is None


def test_default_calc_n_iter(empty_calc_chunk):
    assert empty_calc_chunk.n_iter is None


def test_default_calc_magmom(empty_calc_chunk):
    assert empty_calc_chunk.magmom is None


def test_default_calc_E_f(empty_calc_chunk):
    assert empty_calc_chunk.E_f is None


def test_default_calc_dipole(empty_calc_chunk):
    assert empty_calc_chunk.dipole is None


def test_default_calc_is_metallic(empty_calc_chunk):
    assert not empty_calc_chunk.is_metallic


def test_default_calc_converged(empty_calc_chunk):
    assert not empty_calc_chunk.converged


def test_default_calc_hirshfeld_charges(empty_calc_chunk):
    assert empty_calc_chunk.hirshfeld_charges is None


def test_default_calc_hirshfeld_volumes(empty_calc_chunk):
    assert empty_calc_chunk.hirshfeld_volumes is None


def test_default_calc_hirshfeld_atomic_dipoles(empty_calc_chunk):
    assert empty_calc_chunk.hirshfeld_atomic_dipoles is None


def test_default_calc_hirshfeld_dipole(empty_calc_chunk):
    assert empty_calc_chunk.hirshfeld_dipole is None


def test_default_calc_eigenvalues(empty_calc_chunk):
    assert empty_calc_chunk.eigenvalues is None


def test_default_calc_occupancies(empty_calc_chunk):
    assert empty_calc_chunk.occupancies is None


@pytest.fixture
def calc_chunk(header_chunk):
    lines = [
        "| Number of self-consistency cycles          :           58",
        "| N = N_up - N_down (sum over all k points):         0.00000",
        "| Chemical potential (Fermi level):    -8.24271207 eV",
        "Total atomic forces (unitary forces were cleaned, then relaxation constraints were applied) [eV/Ang]:",
        "|    1          1.000000000000000E+00          2.000000000000000E+00          3.000000000000000E+00",
        "|    2          6.000000000000000E+00          5.000000000000000E+00          4.000000000000000E+00",
        "- Per atom stress (eV) used for heat flux calculation:",
        "Atom   | Stress components (1,1), (2,2), (3,3), (1,2), (1,3), (2,3)",
        "-------------------------------------------------------------------",
        "1 |    -0.6112417237E+01   -0.6112387168E+01   -0.6112387170E+01    0.3126499410E-07    0.3125308439E-07   -0.7229688499E-07",
        "2 |     0.5613699864E+01    0.5613658189E+01    0.5613658190E+01   -0.1032586958E-06   -0.1037261077E-06    0.2151906191E-06",
        "-------------------------------------------------------------------",
        "+-------------------------------------------------------------------+",
        "|              Analytical stress tensor - Symmetrized               |",
        "|                  Cartesian components [eV/A**3]                   |",
        "+-------------------------------------------------------------------+",
        "|                x                y                z                |",
        "|                                                                   |",
        "|  x        -0.00383816      -0.00000001      -0.00000001           |",
        "|  y        -0.00000001      -0.00383829       0.00000002           |",
        "|  z        -0.00000001       0.00000002      -0.00383829           |",
        "|                                                                   |",
        "|  Pressure:       0.00383825   [eV/A**3]                           |",
        "|                                                                   |",
        "+-------------------------------------------------------------------+",
        "Energy and forces in a compact form:",
        "| Total energy uncorrected      :         -0.169503986610555E+05 eV",
        "| Total energy corrected        :         -0.169503986610555E+05 eV  <-- do not rely on this value for anything but (periodic) metals",
        "| Electronic free energy        :         -0.169503986610555E+05 eV",
        "Performing Hirshfeld analysis of fragment charges and moments.",
        "----------------------------------------------------------------------",
        "| Atom     1: Na",
        "|   Hirshfeld charge        :      0.20898543",
        "|   Free atom volume        :     82.26734086",
        "|   Hirshfeld volume        :     73.39467444",
        "|   Hirshfeld spin moment   :      0.00000000",
        "|   Hirshfeld dipole vector :      0.00000000       0.00000000      -0.00000000",
        "|   Hirshfeld dipole moment :      0.00000000",
        "|   Hirshfeld second moments:      0.19955428       0.00000002       0.00000002",
        "|                                  0.00000002       0.19955473      -0.00000005",
        "|                                  0.00000002      -0.00000005       0.19955473",
        "----------------------------------------------------------------------",
        "| Atom     2: Cl",
        "|   Hirshfeld charge        :     -0.20840994",
        "|   Free atom volume        :     65.62593663",
        "|   Hirshfeld volume        :     62.86011074",
        "|   Hirshfeld spin moment   :      0.00000000",
        "|   Hirshfeld dipole vector :      0.00000000       0.00000000       0.00000000",
        "|   Hirshfeld dipole moment :      0.00000000",
        "|   Hirshfeld second moments:      0.01482616      -0.00000001      -0.00000001",
        "|                                 -0.00000001       0.01482641       0.00000001",
        "|                                 -0.00000001       0.00000001       0.01482641",
        "----------------------------------------------------------------------",
        "Writing Kohn-Sham eigenvalues.",
        "",
        "Spin-up eigenvalues:",
        "K-point:       1 at    0.000000    0.000000    0.000000 (in units of recip. lattice)",
        "",
        "State    Occupation    Eigenvalue [Ha]    Eigenvalue [eV]",
        "1       1.00000        -101.052346        -2749.77423",
        "2       1.00000         -37.846018        -1029.84255",
        "3       1.00000          -9.207298         -250.54332",
        "4       1.00000          -6.996674         -190.38919",
        "5       1.00000          -6.996674         -190.38919",
        "6       1.00000          -6.996674         -190.38919",
        "7       1.00000          -2.086619          -56.77979",
        "8       1.00000          -1.075694          -29.27112",
        "9       1.00000          -1.075694          -29.27112",
        "10       1.00000          -1.075694          -29.27112",
        "11       1.00000          -0.764262          -20.79662",
        "12       1.00000          -0.301104           -8.19346",
        "13       1.00000          -0.301104           -8.19346",
        "14       1.00000          -0.301104           -8.19346",
        "15       0.00000          -0.133232           -3.62543",
        "16       0.00000           0.122284            3.32753",
        "17       0.00000           0.122284            3.32753",
        "18       0.00000           0.122285            3.32754",
        "19       0.00000           0.188362            5.12558",
        "20       0.00000           0.188362            5.12559",
        "",
        "Spin-down eigenvalues:",
        "K-point:       1 at    0.000000    0.000000    0.000000 (in units of recip. lattice)",
        "",
        "State    Occupation    Eigenvalue [Ha]    Eigenvalue [eV]",
        "1       1.00000        -101.052346        -2749.77423",
        "2       1.00000         -37.846018        -1029.84255",
        "3       1.00000          -9.207298         -250.54332",
        "4       1.00000          -6.996674         -190.38919",
        "5       1.00000          -6.996674         -190.38919",
        "6       1.00000          -6.996674         -190.38919",
        "7       1.00000          -2.086619          -56.77979",
        "8       1.00000          -1.075694          -29.27112",
        "9       1.00000          -1.075694          -29.27112",
        "10       1.00000          -1.075694          -29.27112",
        "11       1.00000          -0.764262          -20.79662",
        "12       1.00000          -0.301104           -8.19346",
        "13       1.00000          -0.301104           -8.19346",
        "14       1.00000          -0.301104           -8.19346",
        "15       0.00000          -0.133232           -3.62543",
        "16       0.00000           0.122284            3.32753",
        "17       0.00000           0.122284            3.32753",
        "18       0.00000           0.122285            3.32754",
        "19       0.00000           0.188362            5.12558",
        "20       0.00000           0.188362            5.12559",
        "",
        "",
        "Spin-up eigenvalues:",
        "K-point:       2 at    0.000000    0.000000    0.500000 (in units of recip. lattice)",
        "",
        "State    Occupation    Eigenvalue [Ha]    Eigenvalue [eV]",
        "1       1.00000        -101.052346        -2749.77423",
        "2       1.00000         -37.846018        -1029.84255",
        "3       1.00000          -9.207298         -250.54332",
        "4       1.00000          -6.996674         -190.38920",
        "5       1.00000          -6.996674         -190.38919",
        "6       1.00000          -6.996674         -190.38919",
        "7       1.00000          -2.086586          -56.77890",
        "8       1.00000          -1.076824          -29.30187",
        "9       1.00000          -1.075778          -29.27340",
        "10       1.00000          -1.075778          -29.27340",
        "11       1.00000          -0.750762          -20.42926",
        "12       1.00000          -0.378240          -10.29242",
        "13       1.00000          -0.311747           -8.48307",
        "14       1.00000          -0.311747           -8.48306",
        "15       0.00000          -0.045316           -1.23310",
        "16       0.00000           0.075636            2.05815",
        "17       0.00000           0.104554            2.84507",
        "18       0.00000           0.104555            2.84508",
        "19       0.00000           0.259589            7.06379",
        "20       0.00000           0.320229            8.71387",
        "",
        "Spin-down eigenvalues:",
        "K-point:       2 at    0.000000    0.000000    0.500000 (in units of recip. lattice)",
        "",
        "State    Occupation    Eigenvalue [Ha]    Eigenvalue [eV]",
        "1       1.00000        -101.052346        -2749.77423",
        "2       1.00000         -37.846018        -1029.84255",
        "3       1.00000          -9.207298         -250.54332",
        "4       1.00000          -6.996674         -190.38920",
        "5       1.00000          -6.996674         -190.38919",
        "6       1.00000          -6.996674         -190.38919",
        "7       1.00000          -2.086586          -56.77890",
        "8       1.00000          -1.076824          -29.30187",
        "9       1.00000          -1.075778          -29.27340",
        "10       1.00000          -1.075778          -29.27340",
        "11       1.00000          -0.750762          -20.42926",
        "12       1.00000          -0.378240          -10.29242",
        "13       1.00000          -0.311747           -8.48307",
        "14       1.00000          -0.311747           -8.48306",
        "15       0.00000          -0.045316           -1.23310",
        "16       0.00000           0.075636            2.05815",
        "17       0.00000           0.104554            2.84507",
        "18       0.00000           0.104555            2.84508",
        "19       0.00000           0.259589            7.06379",
        "20       0.00000           0.320229            8.71387",
        "",
        "",
        "Spin-up eigenvalues:",
        "K-point:       3 at    0.000000    0.500000    0.000000 (in units of recip. lattice)",
        "",
        "State    Occupation    Eigenvalue [Ha]    Eigenvalue [eV]",
        "1       1.00000        -101.052346        -2749.77423",
        "2       1.00000         -37.846018        -1029.84255",
        "3       1.00000          -9.207298         -250.54332",
        "4       1.00000          -6.996674         -190.38920",
        "5       1.00000          -6.996674         -190.38919",
        "6       1.00000          -6.996674         -190.38919",
        "7       1.00000          -2.086586          -56.77890",
        "8       1.00000          -1.076824          -29.30187",
        "9       1.00000          -1.075778          -29.27340",
        "10       1.00000          -1.075778          -29.27340",
        "11       1.00000          -0.750762          -20.42926",
        "12       1.00000          -0.378240          -10.29242",
        "13       1.00000          -0.311747           -8.48307",
        "14       1.00000          -0.311747           -8.48306",
        "15       0.00000          -0.045316           -1.23310",
        "16       0.00000           0.075636            2.05815",
        "17       0.00000           0.104554            2.84507",
        "18       0.00000           0.104555            2.84508",
        "19       0.00000           0.259589            7.06379",
        "20       0.00000           0.320229            8.71387",
        "",
        "Spin-down eigenvalues:",
        "K-point:       3 at    0.000000    0.500000    0.000000 (in units of recip. lattice)",
        "",
        "State    Occupation    Eigenvalue [Ha]    Eigenvalue [eV]",
        "1       1.00000        -101.052346        -2749.77423",
        "2       1.00000         -37.846018        -1029.84255",
        "3       1.00000          -9.207298         -250.54332",
        "4       1.00000          -6.996674         -190.38920",
        "5       1.00000          -6.996674         -190.38919",
        "6       1.00000          -6.996674         -190.38919",
        "7       1.00000          -2.086586          -56.77890",
        "8       1.00000          -1.076824          -29.30187",
        "9       1.00000          -1.075778          -29.27340",
        "10       1.00000          -1.075778          -29.27340",
        "11       1.00000          -0.750762          -20.42926",
        "12       1.00000          -0.378240          -10.29242",
        "13       1.00000          -0.311747           -8.48307",
        "14       1.00000          -0.311747           -8.48306",
        "15       0.00000          -0.045316           -1.23310",
        "16       0.00000           0.075636            2.05815",
        "17       0.00000           0.104554            2.84507",
        "18       0.00000           0.104555            2.84508",
        "19       0.00000           0.259589            7.06379",
        "20       0.00000           0.320229            8.71387",
        "",
        "",
        "Spin-up eigenvalues:",
        "K-point:       4 at    0.000000    0.500000    0.500000 (in units of recip. lattice)",
        "",
        "State    Occupation    Eigenvalue [Ha]    Eigenvalue [eV]",
        "1       1.00000        -101.052346        -2749.77423",
        "2       1.00000         -37.846018        -1029.84255",
        "3       1.00000          -9.207298         -250.54332",
        "4       1.00000          -6.996674         -190.38919",
        "5       1.00000          -6.996674         -190.38919",
        "6       1.00000          -6.996674         -190.38919",
        "7       1.00000          -2.086575          -56.77860",
        "8       1.00000          -1.076491          -29.29280",
        "9       1.00000          -1.076015          -29.27984",
        "10       1.00000          -1.076014          -29.27984",
        "11       1.00000          -0.748853          -20.37731",
        "12       1.00000          -0.367549          -10.00153",
        "13       1.00000          -0.324951           -8.84236",
        "14       1.00000          -0.324951           -8.84236",
        "15       0.00000          -0.043001           -1.17011",
        "16       0.00000           0.010911            0.29690",
        "17       0.00000           0.110273            3.00069",
        "18       0.00000           0.239812            6.52562",
        "19       0.00000           0.239812            6.52562",
        "20       0.00000           0.252921            6.88232",
        "",
        "Spin-down eigenvalues:",
        "K-point:       4 at    0.000000    0.500000    0.500000 (in units of recip. lattice)",
        "",
        "State    Occupation    Eigenvalue [Ha]    Eigenvalue [eV]",
        "1       1.00000        -101.052346        -2749.77423",
        "2       1.00000         -37.846018        -1029.84255",
        "3       1.00000          -9.207298         -250.54332",
        "4       1.00000          -6.996674         -190.38919",
        "5       1.00000          -6.996674         -190.38919",
        "6       1.00000          -6.996674         -190.38919",
        "7       1.00000          -2.086575          -56.77860",
        "8       1.00000          -1.076491          -29.29280",
        "9       1.00000          -1.076015          -29.27984",
        "10       1.00000          -1.076014          -29.27984",
        "11       1.00000          -0.748853          -20.37731",
        "12       1.00000          -0.367549          -10.00153",
        "13       1.00000          -0.324951           -8.84236",
        "14       1.00000          -0.324951           -8.84236",
        "15       0.00000          -0.043001           -1.17011",
        "16       0.00000           0.010911            0.29690",
        "17       0.00000           0.110273            3.00069",
        "18       0.00000           0.239812            6.52562",
        "19       0.00000           0.239812            6.52562",
        "20       0.00000           0.252921            6.88232",
        "",
        "",
        "Spin-up eigenvalues:",
        "K-point:       5 at    0.500000    0.000000    0.000000 (in units of recip. lattice)",
        "",
        "State    Occupation    Eigenvalue [Ha]    Eigenvalue [eV]",
        "1       1.00000        -101.052346        -2749.77423",
        "2       1.00000         -37.846018        -1029.84255",
        "3       1.00000          -9.207298         -250.54332",
        "4       1.00000          -6.996674         -190.38920",
        "5       1.00000          -6.996674         -190.38919",
        "6       1.00000          -6.996674         -190.38919",
        "7       1.00000          -2.086586          -56.77890",
        "8       1.00000          -1.076824          -29.30187",
        "9       1.00000          -1.075778          -29.27340",
        "10       1.00000          -1.075778          -29.27340",
        "11       1.00000          -0.750762          -20.42926",
        "12       1.00000          -0.378240          -10.29242",
        "13       1.00000          -0.311747           -8.48306",
        "14       1.00000          -0.311747           -8.48306",
        "15       0.00000          -0.045316           -1.23310",
        "16       0.00000           0.075636            2.05815",
        "17       0.00000           0.104554            2.84507",
        "18       0.00000           0.104555            2.84508",
        "19       0.00000           0.259590            7.06379",
        "20       0.00000           0.320229            8.71387",
        "",
        "Spin-down eigenvalues:",
        "K-point:       5 at    0.500000    0.000000    0.000000 (in units of recip. lattice)",
        "",
        "State    Occupation    Eigenvalue [Ha]    Eigenvalue [eV]",
        "1       1.00000        -101.052346        -2749.77423",
        "2       1.00000         -37.846018        -1029.84255",
        "3       1.00000          -9.207298         -250.54332",
        "4       1.00000          -6.996674         -190.38920",
        "5       1.00000          -6.996674         -190.38919",
        "6       1.00000          -6.996674         -190.38919",
        "7       1.00000          -2.086586          -56.77890",
        "8       1.00000          -1.076824          -29.30187",
        "9       1.00000          -1.075778          -29.27340",
        "10       1.00000          -1.075778          -29.27340",
        "11       1.00000          -0.750762          -20.42926",
        "12       1.00000          -0.378240          -10.29242",
        "13       1.00000          -0.311747           -8.48306",
        "14       1.00000          -0.311747           -8.48306",
        "15       0.00000          -0.045316           -1.23310",
        "16       0.00000           0.075636            2.05815",
        "17       0.00000           0.104554            2.84507",
        "18       0.00000           0.104555            2.84508",
        "19       0.00000           0.259590            7.06379",
        "20       0.00000           0.320229            8.71387",
        "",
        "",
        "Spin-up eigenvalues:",
        "K-point:       6 at    0.500000    0.000000    0.500000 (in units of recip. lattice)",
        "",
        "State    Occupation    Eigenvalue [Ha]    Eigenvalue [eV]",
        "1       1.00000        -101.052346        -2749.77423",
        "2       1.00000         -37.846018        -1029.84255",
        "3       1.00000          -9.207298         -250.54332",
        "4       1.00000          -6.996674         -190.38919",
        "5       1.00000          -6.996674         -190.38919",
        "6       1.00000          -6.996674         -190.38919",
        "7       1.00000          -2.086575          -56.77860",
        "8       1.00000          -1.076491          -29.29280",
        "9       1.00000          -1.076014          -29.27984",
        "10       1.00000          -1.076014          -29.27984",
        "11       1.00000          -0.748853          -20.37731",
        "12       1.00000          -0.367549          -10.00153",
        "13       1.00000          -0.324951           -8.84236",
        "14       1.00000          -0.324951           -8.84235",
        "15       0.00000          -0.043001           -1.17011",
        "16       0.00000           0.010911            0.29690",
        "17       0.00000           0.110273            3.00068",
        "18       0.00000           0.239812            6.52562",
        "19       0.00000           0.239813            6.52564",
        "20       0.00000           0.252921            6.88232",
        "",
        "Spin-down eigenvalues:",
        "K-point:       6 at    0.500000    0.000000    0.500000 (in units of recip. lattice)",
        "",
        "State    Occupation    Eigenvalue [Ha]    Eigenvalue [eV]",
        "1       1.00000        -101.052346        -2749.77423",
        "2       1.00000         -37.846018        -1029.84255",
        "3       1.00000          -9.207298         -250.54332",
        "4       1.00000          -6.996674         -190.38919",
        "5       1.00000          -6.996674         -190.38919",
        "6       1.00000          -6.996674         -190.38919",
        "7       1.00000          -2.086575          -56.77860",
        "8       1.00000          -1.076491          -29.29280",
        "9       1.00000          -1.076014          -29.27984",
        "10       1.00000          -1.076014          -29.27984",
        "11       1.00000          -0.748853          -20.37731",
        "12       1.00000          -0.367549          -10.00153",
        "13       1.00000          -0.324951           -8.84236",
        "14       1.00000          -0.324951           -8.84235",
        "15       0.00000          -0.043001           -1.17011",
        "16       0.00000           0.010911            0.29690",
        "17       0.00000           0.110273            3.00068",
        "18       0.00000           0.239812            6.52562",
        "19       0.00000           0.239813            6.52564",
        "20       0.00000           0.252921            6.88232",
        "",
        "",
        "Spin-up eigenvalues:",
        "K-point:       7 at    0.500000    0.500000    0.000000 (in units of recip. lattice)",
        "",
        "State    Occupation    Eigenvalue [Ha]    Eigenvalue [eV]",
        "1       1.00000        -101.052346        -2749.77423",
        "2       1.00000         -37.846018        -1029.84255",
        "3       1.00000          -9.207298         -250.54332",
        "4       1.00000          -6.996674         -190.38919",
        "5       1.00000          -6.996674         -190.38919",
        "6       1.00000          -6.996674         -190.38919",
        "7       1.00000          -2.086575          -56.77860",
        "8       1.00000          -1.076491          -29.29280",
        "9       1.00000          -1.076014          -29.27984",
        "10       1.00000          -1.076014          -29.27984",
        "11       1.00000          -0.748853          -20.37731",
        "12       1.00000          -0.367549          -10.00153",
        "13       1.00000          -0.324951           -8.84236",
        "14       1.00000          -0.324951           -8.84235",
        "15       0.00000          -0.043001           -1.17011",
        "16       0.00000           0.010911            0.29690",
        "17       0.00000           0.110273            3.00068",
        "18       0.00000           0.239812            6.52562",
        "19       0.00000           0.239813            6.52564",
        "20       0.00000           0.252921            6.88232",
        "",
        "Spin-down eigenvalues:",
        "K-point:       7 at    0.500000    0.500000    0.000000 (in units of recip. lattice)",
        "",
        "State    Occupation    Eigenvalue [Ha]    Eigenvalue [eV]",
        "1       1.00000        -101.052346        -2749.77423",
        "2       1.00000         -37.846018        -1029.84255",
        "3       1.00000          -9.207298         -250.54332",
        "4       1.00000          -6.996674         -190.38919",
        "5       1.00000          -6.996674         -190.38919",
        "6       1.00000          -6.996674         -190.38919",
        "7       1.00000          -2.086575          -56.77860",
        "8       1.00000          -1.076491          -29.29280",
        "9       1.00000          -1.076014          -29.27984",
        "10       1.00000          -1.076014          -29.27984",
        "11       1.00000          -0.748853          -20.37731",
        "12       1.00000          -0.367549          -10.00153",
        "13       1.00000          -0.324951           -8.84236",
        "14       1.00000          -0.324951           -8.84235",
        "15       0.00000          -0.043001           -1.17011",
        "16       0.00000           0.010911            0.29690",
        "17       0.00000           0.110273            3.00068",
        "18       0.00000           0.239812            6.52562",
        "19       0.00000           0.239813            6.52564",
        "20       0.00000           0.252921            6.88232",
        "",
        "",
        "Spin-up eigenvalues:",
        "K-point:       8 at    0.500000    0.500000    0.500000 (in units of recip. lattice)",
        "",
        "State    Occupation    Eigenvalue [Ha]    Eigenvalue [eV]",
        "1       1.00000        -101.052346        -2749.77423",
        "2       1.00000         -37.846018        -1029.84255",
        "3       1.00000          -9.207298         -250.54332",
        "4       1.00000          -6.996674         -190.38920",
        "5       1.00000          -6.996674         -190.38919",
        "6       1.00000          -6.996674         -190.38919",
        "7       1.00000          -2.086586          -56.77890",
        "8       1.00000          -1.076824          -29.30187",
        "9       1.00000          -1.075778          -29.27340",
        "10       1.00000          -1.075778          -29.27340",
        "11       1.00000          -0.750762          -20.42926",
        "12       1.00000          -0.378240          -10.29242",
        "13       1.00000          -0.311747           -8.48306",
        "14       1.00000          -0.311747           -8.48306",
        "15       0.00000          -0.045316           -1.23310",
        "16       0.00000           0.075636            2.05815",
        "17       0.00000           0.104554            2.84507",
        "18       0.00000           0.104555            2.84508",
        "19       0.00000           0.259590            7.06379",
        "20       0.00000           0.320229            8.71387",
        "",
        "Spin-down eigenvalues:",
        "K-point:       8 at    0.500000    0.500000    0.500000 (in units of recip. lattice)",
        "",
        "State    Occupation    Eigenvalue [Ha]    Eigenvalue [eV]",
        "1       1.00000        -101.052346        -2749.77423",
        "2       1.00000         -37.846018        -1029.84255",
        "3       1.00000          -9.207298         -250.54332",
        "4       1.00000          -6.996674         -190.38920",
        "5       1.00000          -6.996674         -190.38919",
        "6       1.00000          -6.996674         -190.38919",
        "7       1.00000          -2.086586          -56.77890",
        "8       1.00000          -1.076824          -29.30187",
        "9       1.00000          -1.075778          -29.27340",
        "10       1.00000          -1.075778          -29.27340",
        "11       1.00000          -0.750762          -20.42926",
        "12       1.00000          -0.378240          -10.29242",
        "13       1.00000          -0.311747           -8.48306",
        "14       1.00000          -0.311747           -8.48306",
        "15       0.00000          -0.045316           -1.23310",
        "16       0.00000           0.075636            2.05815",
        "17       0.00000           0.104554            2.84507",
        "18       0.00000           0.104555            2.84508",
        "19       0.00000           0.259590            7.06379",
        "20       0.00000           0.320229            8.71387",
        "",
        "Current spin moment of the entire structure :",
        "| N = N_up - N_down (sum over all k points):         0.00000",
        "| S (sum over all k points)                :         0.00000",
        "",
        "What follows are estimated values for band gap, HOMO, LUMO, etc.",
        "| They are estimated on a discrete k-point grid and not necessarily exact.",
        "| For converged numbers, create a DOS and/or band structure plot on a denser k-grid.",
        "",
        "Highest occupied state (VBM) at     -8.19345940 eV (relative to internal zero)",
        "| Occupation number:      1.00000000",
        "| K-point:       1 at    0.000000    0.000000    0.000000 (in units of recip. lattice)",
        "| Spin channel:        1",
        "",
        "Lowest unoccupied state (CBM) at    -3.62542909 eV (relative to internal zero)",
        "| Occupation number:      0.00000000",
        "| K-point:       1 at    0.000000    0.000000    0.000000 (in units of recip. lattice)",
        "| Spin channel:        1",
        "",
        "ESTIMATED overall HOMO-LUMO gap:      4.56803031 eV between HOMO at k-point 1 and LUMO at k-point 1",
        "| This appears to be a direct band gap.",
        "The gap value is above 0.2 eV. Unless you are using a very sparse k-point grid",
        "this system is most likely an insulator or a semiconductor.",
        "| Chemical Potential                          :    -7.44914181 eV",
        "",
        "Self-consistency cycle converged.",
        "material is metallic within the approximate finite broadening function (occupation_type)",
        "Have a nice day.",
        "------------------------------------------------------------",
        "",
    ]
    return AimsOutCalcChunk(lines, header_chunk)


@pytest.fixture
def eigenvalues_occupancies():
    eigenvalues_occupancies = np.zeros((8, 20, 2, 2))

    eigenvalues_occupancies[0, :, 0, :] = [
        [1.00000, -2749.77423],
        [1.00000, -1029.84255],
        [1.00000, -250.54332],
        [1.00000, -190.38919],
        [1.00000, -190.38919],
        [1.00000, -190.38919],
        [1.00000, -56.77979],
        [1.00000, -29.27112],
        [1.00000, -29.27112],
        [1.00000, -29.27112],
        [1.00000, -20.79662],
        [1.00000, -8.19346],
        [1.00000, -8.19346],
        [1.00000, -8.19346],
        [0.00000, -3.62543],
        [0.00000, 3.32753],
        [0.00000, 3.32753],
        [0.00000, 3.32754],
        [0.00000, 5.12558],
        [0.00000, 5.12559],
    ]

    eigenvalues_occupancies[0, :, 1, :] = [
        [1.00000, -2749.77423],
        [1.00000, -1029.84255],
        [1.00000, -250.54332],
        [1.00000, -190.38919],
        [1.00000, -190.38919],
        [1.00000, -190.38919],
        [1.00000, -56.77979],
        [1.00000, -29.27112],
        [1.00000, -29.27112],
        [1.00000, -29.27112],
        [1.00000, -20.79662],
        [1.00000, -8.19346],
        [1.00000, -8.19346],
        [1.00000, -8.19346],
        [0.00000, -3.62543],
        [0.00000, 3.32753],
        [0.00000, 3.32753],
        [0.00000, 3.32754],
        [0.00000, 5.12558],
        [0.00000, 5.12559],
    ]

    eigenvalues_occupancies[1, :, 0, :] = [
        [1.00000, -2749.77423],
        [1.00000, -1029.84255],
        [1.00000, -250.54332],
        [1.00000, -190.38920],
        [1.00000, -190.38919],
        [1.00000, -190.38919],
        [1.00000, -56.77890],
        [1.00000, -29.30187],
        [1.00000, -29.27340],
        [1.00000, -29.27340],
        [1.00000, -20.42926],
        [1.00000, -10.29242],
        [1.00000, -8.48307],
        [1.00000, -8.48306],
        [0.00000, -1.23310],
        [0.00000, 2.05815],
        [0.00000, 2.84507],
        [0.00000, 2.84508],
        [0.00000, 7.06379],
        [0.00000, 8.71387],
    ]

    eigenvalues_occupancies[1, :, 1, :] = [
        [1.00000, -2749.77423],
        [1.00000, -1029.84255],
        [1.00000, -250.54332],
        [1.00000, -190.38920],
        [1.00000, -190.38919],
        [1.00000, -190.38919],
        [1.00000, -56.77890],
        [1.00000, -29.30187],
        [1.00000, -29.27340],
        [1.00000, -29.27340],
        [1.00000, -20.42926],
        [1.00000, -10.29242],
        [1.00000, -8.48307],
        [1.00000, -8.48306],
        [0.00000, -1.23310],
        [0.00000, 2.05815],
        [0.00000, 2.84507],
        [0.00000, 2.84508],
        [0.00000, 7.06379],
        [0.00000, 8.71387],
    ]

    eigenvalues_occupancies[2, :, 0, :] = [
        [1.00000, -2749.77423],
        [1.00000, -1029.84255],
        [1.00000, -250.54332],
        [1.00000, -190.38920],
        [1.00000, -190.38919],
        [1.00000, -190.38919],
        [1.00000, -56.77890],
        [1.00000, -29.30187],
        [1.00000, -29.27340],
        [1.00000, -29.27340],
        [1.00000, -20.42926],
        [1.00000, -10.29242],
        [1.00000, -8.48307],
        [1.00000, -8.48306],
        [0.00000, -1.23310],
        [0.00000, 2.05815],
        [0.00000, 2.84507],
        [0.00000, 2.84508],
        [0.00000, 7.06379],
        [0.00000, 8.71387],
    ]

    eigenvalues_occupancies[2, :, 1, :] = [
        [1.00000, -2749.77423],
        [1.00000, -1029.84255],
        [1.00000, -250.54332],
        [1.00000, -190.38920],
        [1.00000, -190.38919],
        [1.00000, -190.38919],
        [1.00000, -56.77890],
        [1.00000, -29.30187],
        [1.00000, -29.27340],
        [1.00000, -29.27340],
        [1.00000, -20.42926],
        [1.00000, -10.29242],
        [1.00000, -8.48307],
        [1.00000, -8.48306],
        [0.00000, -1.23310],
        [0.00000, 2.05815],
        [0.00000, 2.84507],
        [0.00000, 2.84508],
        [0.00000, 7.06379],
        [0.00000, 8.71387],
    ]

    eigenvalues_occupancies[3, :, 0, :] = [
        [1.00000, -2749.77423],
        [1.00000, -1029.84255],
        [1.00000, -250.54332],
        [1.00000, -190.38919],
        [1.00000, -190.38919],
        [1.00000, -190.38919],
        [1.00000, -56.77860],
        [1.00000, -29.29280],
        [1.00000, -29.27984],
        [1.00000, -29.27984],
        [1.00000, -20.37731],
        [1.00000, -10.00153],
        [1.00000, -8.84236],
        [1.00000, -8.84236],
        [0.00000, -1.17011],
        [0.00000, 0.29690],
        [0.00000, 3.00069],
        [0.00000, 6.52562],
        [0.00000, 6.52562],
        [0.00000, 6.88232],
    ]

    eigenvalues_occupancies[3, :, 1, :] = [
        [1.00000, -2749.77423],
        [1.00000, -1029.84255],
        [1.00000, -250.54332],
        [1.00000, -190.38919],
        [1.00000, -190.38919],
        [1.00000, -190.38919],
        [1.00000, -56.77860],
        [1.00000, -29.29280],
        [1.00000, -29.27984],
        [1.00000, -29.27984],
        [1.00000, -20.37731],
        [1.00000, -10.00153],
        [1.00000, -8.84236],
        [1.00000, -8.84236],
        [0.00000, -1.17011],
        [0.00000, 0.29690],
        [0.00000, 3.00069],
        [0.00000, 6.52562],
        [0.00000, 6.52562],
        [0.00000, 6.88232],
    ]

    eigenvalues_occupancies[4, :, 0, :] = [
        [1.00000, -2749.77423],
        [1.00000, -1029.84255],
        [1.00000, -250.54332],
        [1.00000, -190.38920],
        [1.00000, -190.38919],
        [1.00000, -190.38919],
        [1.00000, -56.77890],
        [1.00000, -29.30187],
        [1.00000, -29.27340],
        [1.00000, -29.27340],
        [1.00000, -20.42926],
        [1.00000, -10.29242],
        [1.00000, -8.48306],
        [1.00000, -8.48306],
        [0.00000, -1.23310],
        [0.00000, 2.05815],
        [0.00000, 2.84507],
        [0.00000, 2.84508],
        [0.00000, 7.06379],
        [0.00000, 8.71387],
    ]

    eigenvalues_occupancies[4, :, 1, :] = [
        [1.00000, -2749.77423],
        [1.00000, -1029.84255],
        [1.00000, -250.54332],
        [1.00000, -190.38920],
        [1.00000, -190.38919],
        [1.00000, -190.38919],
        [1.00000, -56.77890],
        [1.00000, -29.30187],
        [1.00000, -29.27340],
        [1.00000, -29.27340],
        [1.00000, -20.42926],
        [1.00000, -10.29242],
        [1.00000, -8.48306],
        [1.00000, -8.48306],
        [0.00000, -1.23310],
        [0.00000, 2.05815],
        [0.00000, 2.84507],
        [0.00000, 2.84508],
        [0.00000, 7.06379],
        [0.00000, 8.71387],
    ]

    eigenvalues_occupancies[5, :, 0, :] = [
        [1.00000, -2749.77423],
        [1.00000, -1029.84255],
        [1.00000, -250.54332],
        [1.00000, -190.38919],
        [1.00000, -190.38919],
        [1.00000, -190.38919],
        [1.00000, -56.77860],
        [1.00000, -29.29280],
        [1.00000, -29.27984],
        [1.00000, -29.27984],
        [1.00000, -20.37731],
        [1.00000, -10.00153],
        [1.00000, -8.84236],
        [1.00000, -8.84235],
        [0.00000, -1.17011],
        [0.00000, 0.29690],
        [0.00000, 3.00068],
        [0.00000, 6.52562],
        [0.00000, 6.52564],
        [0.00000, 6.88232],
    ]

    eigenvalues_occupancies[5, :, 1, :] = [
        [1.00000, -2749.77423],
        [1.00000, -1029.84255],
        [1.00000, -250.54332],
        [1.00000, -190.38919],
        [1.00000, -190.38919],
        [1.00000, -190.38919],
        [1.00000, -56.77860],
        [1.00000, -29.29280],
        [1.00000, -29.27984],
        [1.00000, -29.27984],
        [1.00000, -20.37731],
        [1.00000, -10.00153],
        [1.00000, -8.84236],
        [1.00000, -8.84235],
        [0.00000, -1.17011],
        [0.00000, 0.29690],
        [0.00000, 3.00068],
        [0.00000, 6.52562],
        [0.00000, 6.52564],
        [0.00000, 6.88232],
    ]

    eigenvalues_occupancies[6, :, 0, :] = [
        [1.00000, -2749.77423],
        [1.00000, -1029.84255],
        [1.00000, -250.54332],
        [1.00000, -190.38919],
        [1.00000, -190.38919],
        [1.00000, -190.38919],
        [1.00000, -56.77860],
        [1.00000, -29.29280],
        [1.00000, -29.27984],
        [1.00000, -29.27984],
        [1.00000, -20.37731],
        [1.00000, -10.00153],
        [1.00000, -8.84236],
        [1.00000, -8.84235],
        [0.00000, -1.17011],
        [0.00000, 0.29690],
        [0.00000, 3.00068],
        [0.00000, 6.52562],
        [0.00000, 6.52564],
        [0.00000, 6.88232],
    ]

    eigenvalues_occupancies[6, :, 1, :] = [
        [1.00000, -2749.77423],
        [1.00000, -1029.84255],
        [1.00000, -250.54332],
        [1.00000, -190.38919],
        [1.00000, -190.38919],
        [1.00000, -190.38919],
        [1.00000, -56.77860],
        [1.00000, -29.29280],
        [1.00000, -29.27984],
        [1.00000, -29.27984],
        [1.00000, -20.37731],
        [1.00000, -10.00153],
        [1.00000, -8.84236],
        [1.00000, -8.84235],
        [0.00000, -1.17011],
        [0.00000, 0.29690],
        [0.00000, 3.00068],
        [0.00000, 6.52562],
        [0.00000, 6.52564],
        [0.00000, 6.88232],
    ]

    eigenvalues_occupancies[7, :, 0, :] = [
        [1.00000, -2749.77423],
        [1.00000, -1029.84255],
        [1.00000, -250.54332],
        [1.00000, -190.38920],
        [1.00000, -190.38919],
        [1.00000, -190.38919],
        [1.00000, -56.77890],
        [1.00000, -29.30187],
        [1.00000, -29.27340],
        [1.00000, -29.27340],
        [1.00000, -20.42926],
        [1.00000, -10.29242],
        [1.00000, -8.48306],
        [1.00000, -8.48306],
        [0.00000, -1.23310],
        [0.00000, 2.05815],
        [0.00000, 2.84507],
        [0.00000, 2.84508],
        [0.00000, 7.06379],
        [0.00000, 8.71387],
    ]

    eigenvalues_occupancies[7, :, 1, :] = [
        [1.00000, -2749.77423],
        [1.00000, -1029.84255],
        [1.00000, -250.54332],
        [1.00000, -190.38920],
        [1.00000, -190.38919],
        [1.00000, -190.38919],
        [1.00000, -56.77890],
        [1.00000, -29.30187],
        [1.00000, -29.27340],
        [1.00000, -29.27340],
        [1.00000, -20.42926],
        [1.00000, -10.29242],
        [1.00000, -8.48306],
        [1.00000, -8.48306],
        [0.00000, -1.23310],
        [0.00000, 2.05815],
        [0.00000, 2.84507],
        [0.00000, 2.84508],
        [0.00000, 7.06379],
        [0.00000, 8.71387],
    ]
    return eigenvalues_occupancies


def test_calc_atoms(calc_chunk):
    assert len(calc_chunk.atoms) == 2
    assert np.allclose(
        calc_chunk.atoms.cell,
        np.array([[0.000, 2.703, 2.703], [2.703, 0.000, 2.703], [2.703, 2.703, 0.000]]),
    )
    assert np.allclose(
        calc_chunk.atoms.positions,
        np.array([[0.000, 0.000, 0.000], [2.703, 2.703, 2.703]]),
    )
    assert np.all(["Na", "Cl"] == calc_chunk.atoms.symbols)
    assert all(
        [
            str(const_1) == str(const_2)
            for const_1, const_2 in zip(
                calc_chunk.constraints, calc_chunk.atoms.constraints
            )
        ]
    )


def test_calc_forces(calc_chunk):
    assert np.allclose(calc_chunk.forces, np.array([[1.0, 2.0, 3.0], [6.0, 5.0, 4.0]]))

    # Different because of the constraints
    assert np.allclose(
        calc_chunk.atoms.get_forces(), np.array([[0.0, 0.0, 3.0], [0.0, 0.0, 0.0]])
    )
    assert np.allclose(
        calc_chunk.atoms.get_forces(apply_constraint=False),
        np.array([[1.0, 2.0, 3.0], [6.0, 5.0, 4.0]]),
    )
    assert np.allclose(
        calc_chunk.results["forces"], np.array([[1.0, 2.0, 3.0], [6.0, 5.0, 4.0]])
    )


def test_calc_stresses(calc_chunk):
    stresses = np.array(
        [
            [
                -0.6112417237e01,
                -0.6112387168e01,
                -0.6112387170e01,
                -0.7229688499e-07,
                0.3125308439e-07,
                0.3126499410e-07,
            ],
            [
                0.5613699864e01,
                0.5613658189e01,
                0.5613658190e01,
                0.2151906191e-06,
                -0.1037261077e-06,
                -0.1032586958e-06,
            ],
        ]
    )
    assert np.allclose(calc_chunk.stresses, stresses)
    assert np.allclose(calc_chunk.atoms.get_stresses(), stresses)
    assert np.allclose(calc_chunk.results["stresses"], stresses)


def test_calc_stress(calc_chunk):
    stress = full_3x3_to_voigt_6_stress(
        np.array(
            [
                [-0.00383816, -0.00000001, -0.00000001],
                [-0.00000001, -0.00383829, 0.00000002],
                [-0.00000001, 0.00000002, -0.00383829],
            ]
        )
    )
    assert np.allclose(calc_chunk.stress, stress)
    assert np.allclose(calc_chunk.atoms.get_stress(), stress)
    assert np.allclose(calc_chunk.results["stress"], stress)


def test_calc_free_energy(calc_chunk):
    assert np.abs(calc_chunk.free_energy + 0.169503986610555e05) < 1e-15
    assert np.abs(calc_chunk.results["free_energy"] + 0.169503986610555e05) < 1e-15


def test_calc_energy(calc_chunk):
    assert np.abs(calc_chunk.energy + 0.169503986610555e05) < 1e-15
    assert (
        np.abs(calc_chunk.atoms.get_potential_energy() + 0.169503986610555e05) < 1e-15
    )
    assert np.abs(calc_chunk.results["energy"] + 0.169503986610555e05) < 1e-15


def test_calc_magnetic_moment(calc_chunk):
    assert calc_chunk.magmom == 0
    assert calc_chunk.atoms.get_magnetic_moment() == 0
    assert calc_chunk.results["magmom"] == 0


def test_calc_n_iter(calc_chunk):
    assert calc_chunk.n_iter == 58
    assert calc_chunk.results["n_iter"] == 58


def test_calc_fermi_energy(calc_chunk):
    assert np.abs(calc_chunk.E_f + 8.24271207) < 1e-7
    assert np.abs(calc_chunk.results["fermi_energy"] + 8.24271207) < 1e-7


def test_calc_dipole(calc_chunk):
    assert calc_chunk.dipole is None


def test_calc_is_metallic(calc_chunk):
    assert calc_chunk.is_metallic


def test_calc_converged(calc_chunk):
    assert calc_chunk.converged


def test_calc_hirshfeld_charges(calc_chunk):
    assert np.allclose(calc_chunk.hirshfeld_charges, [0.20898543, -0.20840994])
    assert np.allclose(
        calc_chunk.results["hirshfeld_charges"], [0.20898543, -0.20840994]
    )


def test_calc_hirshfeld_volumes(calc_chunk):
    assert np.allclose(calc_chunk.hirshfeld_volumes, [73.39467444, 62.86011074])
    assert np.allclose(
        calc_chunk.results["hirshfeld_volumes"], [73.39467444, 62.86011074]
    )


def test_calc_hirshfeld_atomic_dipoles(calc_chunk):
    assert np.allclose(calc_chunk.hirshfeld_atomic_dipoles, np.zeros((2, 3)))
    assert np.allclose(calc_chunk.results["hirshfeld_atomic_dipoles"], np.zeros((2, 3)))


def test_calc_hirshfeld_dipole(calc_chunk):
    assert calc_chunk.hirshfeld_dipole is None


def test_calc_eigenvalues(calc_chunk, eigenvalues_occupancies):
    assert np.allclose(calc_chunk.eigenvalues, eigenvalues_occupancies[:, :, :, 1])
    assert np.allclose(
        calc_chunk.results["eigenvalues"], eigenvalues_occupancies[:, :, :, 1]
    )


def test_calc_occupancies(calc_chunk, eigenvalues_occupancies):
    assert np.allclose(calc_chunk.occupancies, eigenvalues_occupancies[:, :, :, 0])
    assert np.allclose(
        calc_chunk.results["occupancies"], eigenvalues_occupancies[:, :, :, 0]
    )


@pytest.fixture
def molecular_header_chunk():
    lines = [
        "| Number of atoms                   :        3",
        "| Number of spin channels           :        1",
        "The structure contains        3 atoms,  and a total of         10.000 electrons.",
        "Input geometry:",
        "| Atomic structure:",
        "|       Atom                x [A]            y [A]            z [A]",
        "|    1: Species O             0.00000000        0.00000000        0.00000000",
        "|    2: Species H             0.95840000        0.00000000        0.00000000",
        "|    3: Species H            -0.24000000        0.92790000        0.00000000",
        'Geometry relaxation: A file "geometry.in.next_step" is written out by default after each step.',
        "| Maximum number of basis functions            :        7",
        "| Number of Kohn-Sham states (occupied + empty):       11",
        "Reducing total number of  Kohn-Sham states to        7.",
    ]
    return AimsOutHeaderChunk(lines)


def test_molecular_header_k_points(molecular_header_chunk):
    assert molecular_header_chunk.k_points is None


def test_molecular_header_k_point_weights(molecular_header_chunk):
    assert molecular_header_chunk.k_point_weights is None


def test_molecular_header_constraints(molecular_header_chunk):
    assert molecular_header_chunk.constraints is None


def test_molecular_header_initial_cell(molecular_header_chunk):
    assert molecular_header_chunk.initial_cell is None


def test_molecular_header_n_k_points(molecular_header_chunk):
    assert molecular_header_chunk.n_k_points is None


def test_molecular_header_n_bands(molecular_header_chunk):
    assert molecular_header_chunk.n_bands == 7


def test_molecular_header_initial_cell(molecular_header_chunk):
    assert molecular_header_chunk.initial_cell is None


def test_molecular_header_initial_cell(molecular_header_chunk):
    assert molecular_header_chunk.initial_cell is None


def test_molecular_header_initial_atoms(molecular_header_chunk):
    assert len(molecular_header_chunk.initial_atoms) == 3
    assert np.all(["O", "H", "H"] == molecular_header_chunk.initial_atoms.symbols)
    assert np.allclose(
        molecular_header_chunk.initial_atoms.positions,
        np.array(
            [
                [0.00000000, 0.00000000, 0.00000000],
                [0.95840000, 0.00000000, 0.00000000],
                [-0.24000000, 0.92790000, 0.00000000],
            ]
        ),
    )


@pytest.fixture
def molecular_calc_chunk(molecular_header_chunk):
    lines = [
        "| Number of self-consistency cycles          :           7",
        "| Chemical Potential                          :    -0.61315483 eV",
        "Updated atomic structure:",
        "x [A]             y [A]             z [A]",
        "atom        -0.00191785       -0.00243279        0.00000000  O",
        "atom         0.97071531       -0.00756333        0.00000000  H",
        "atom        -0.25039746        0.93789612       -0.00000000  H",
        "| Total dipole moment [eAng]          :          0.260286493869765E+00         0.336152447755231E+00         0.470003778119121E-15",
        "Energy and forces in a compact form:",
        "| Total energy uncorrected      :         -0.206778551123339E+04 eV",
        "| Total energy corrected        :         -5.206778551123339E+04 eV  <-- do not rely on this value for anything but (periodic) metals",
        "| Electronic free energy        :         -0.206778551123339E+04 eV",
        "Total atomic forces (unitary forces cleaned) [eV/Ang]:",
        "|    1          0.502371357164392E-03          0.518627676606471E-03          0.000000000000000E+00",
        "|    2         -0.108826758257187E-03         -0.408128912334209E-03         -0.649037698626122E-27",
        "|    3         -0.393544598907207E-03         -0.110498764272267E-03         -0.973556547939183E-27",
        "Performing Hirshfeld analysis of fragment charges and moments.",
        "----------------------------------------------------------------------",
        "| Atom     1: O",
        "|   Hirshfeld charge        :     -0.32053200",
        "|   Free atom volume        :     23.59848617",
        "|   Hirshfeld volume        :     21.83060659",
        "|   Hirshfeld dipole vector :      0.04249319       0.05486053       0.00000000",
        "|   Hirshfeld dipole moment :      0.06939271",
        "|   Hirshfeld second moments:      0.04964380      -0.04453278      -0.00000000",
        "|                                 -0.04453278       0.02659295       0.00000000",
        "|                                 -0.00000000       0.00000000      -0.05608173",
        "----------------------------------------------------------------------",
        "| Atom     2: H",
        "|   Hirshfeld charge        :      0.16022630",
        "|   Free atom volume        :     10.48483941",
        "|   Hirshfeld volume        :      6.07674041",
        "|   Hirshfeld dipole vector :      0.13710134      -0.00105126       0.00000000",
        "|   Hirshfeld dipole moment :      0.13710537",
        "|   Hirshfeld second moments:      0.12058896      -0.01198026      -0.00000000",
        "|                                 -0.01198026       0.14550360       0.00000000",
        "|                                 -0.00000000       0.00000000       0.10836357",
        "----------------------------------------------------------------------",
        "| Atom     3: H",
        "|   Hirshfeld charge        :      0.16020375",
        "|   Free atom volume        :     10.48483941",
        "|   Hirshfeld volume        :      6.07684447",
        "|   Hirshfeld dipole vector :     -0.03534982       0.13248706       0.00000000",
        "|   Hirshfeld dipole moment :      0.13712195",
        "|   Hirshfeld second moments:      0.14974686      -0.00443579      -0.00000000",
        "|                                 -0.00443579       0.11633028      -0.00000000",
        "|                                 -0.00000000      -0.00000000       0.10836209",
        "----------------",
        "Writing Kohn-Sham eigenvalues.",
        "",
        "State    Occupation    Eigenvalue [Ha]    Eigenvalue [eV]",
        "1       2.00000         -18.640915         -507.24511",
        "2       2.00000          -0.918449          -24.99226",
        "3       2.00000          -0.482216          -13.12175",
        "4       2.00000          -0.338691           -9.21626",
        "5       2.00000          -0.264427           -7.19543",
        "6       0.00000          -0.000414           -0.01127",
        "7       0.00000           0.095040            2.58616",
        "",
        "Highest occupied state (VBM) at     -7.19542820 eV",
        "| Occupation number:      2.00000000",
        "",
        "Lowest unoccupied state (CBM) at    -0.01126981 eV",
        "| Occupation number:      0.00000000",
        "",
        "Overall HOMO-LUMO gap:      7.18415839 eV.",
        "| Chemical Potential                          :    -0.61315483 eV",
        "",
        "Self-consistency cycle converged.",
        "Have a nice day.",
        "------------------------------------------------------------",
        "",
    ]
    return AimsOutCalcChunk(lines, molecular_header_chunk)


@pytest.fixture
def molecular_positions():
    return np.array(
        [
            [-0.00191785, -0.00243279, 0.00000000],
            [0.97071531, -0.00756333, 0.00000000],
            [-0.25039746, 0.93789612, 0.00000000],
        ]
    )


def test_molecular_calc_atoms(molecular_calc_chunk, molecular_positions):
    assert len(molecular_calc_chunk.atoms) == 3
    assert np.allclose(molecular_calc_chunk.atoms.positions, molecular_positions)
    assert np.all(["O", "H", "H"] == molecular_calc_chunk.atoms.symbols)


def test_molecular_calc_forces(molecular_calc_chunk):
    forces = np.array(
        [
            [0.502371357164392e-03, 0.518627676606471e-03, 0.000000000000000e00],
            [-0.108826758257187e-03, -0.408128912334209e-03, -0.649037698626122e-27],
            [-0.393544598907207e-03, -0.110498764272267e-03, -0.973556547939183e-27],
        ]
    )
    assert np.allclose(molecular_calc_chunk.forces, forces)
    assert np.allclose(molecular_calc_chunk.atoms.get_forces(), forces)
    assert np.allclose(molecular_calc_chunk.results["forces"], forces)


def test_molecular_calc_stresses(molecular_calc_chunk):
    assert molecular_calc_chunk.stresses is None


def test_molecular_calc_stress(molecular_calc_chunk):
    assert molecular_calc_chunk.stress is None


def test_molecular_calc_free_energy(molecular_calc_chunk):
    assert np.abs(molecular_calc_chunk.free_energy + 0.206778551123339e04) < 1e-15
    assert (
        np.abs(molecular_calc_chunk.results["free_energy"] + 0.206778551123339e04)
        < 1e-15
    )


def test_molecular_calc_energy(molecular_calc_chunk):
    assert np.abs(molecular_calc_chunk.energy + 0.206778551123339e04) < 1e-15
    assert (
        np.abs(molecular_calc_chunk.atoms.get_potential_energy() + 0.206778551123339e04)
        < 1e-15
    )
    assert np.abs(molecular_calc_chunk.results["energy"] + 0.206778551123339e04) < 1e-15


def test_molecular_calc_magmom(molecular_calc_chunk):
    assert molecular_calc_chunk.magmom is None


def test_molecular_calc_n_iter(molecular_calc_chunk):
    assert molecular_calc_chunk.n_iter == 7
    assert molecular_calc_chunk.results["n_iter"] == 7


def test_molecular_calc_fermi_energy(molecular_calc_chunk):
    assert molecular_calc_chunk.E_f is None


def test_molecular_calc_dipole(molecular_calc_chunk):
    dipole = [0.260286493869765, 0.336152447755231, 0.470003778119121e-15]
    assert np.allclose(molecular_calc_chunk.dipole, dipole)
    assert np.allclose(molecular_calc_chunk.atoms.get_dipole_moment(), dipole)
    assert np.allclose(molecular_calc_chunk.results["dipole"], dipole)


def test_molecular_calc_is_metallic(molecular_calc_chunk):
    assert not molecular_calc_chunk.is_metallic


def test_molecular_calc_converged(molecular_calc_chunk):
    assert molecular_calc_chunk.converged


@pytest.fixture
def molecular_hirshfeld_charges():
    return np.array([-0.32053200, 0.16022630, 0.16020375])


def test_molecular_calc_hirshfeld_charges(
    molecular_calc_chunk, molecular_hirshfeld_charges
):
    assert np.allclose(
        molecular_calc_chunk.hirshfeld_charges, molecular_hirshfeld_charges
    )
    assert np.allclose(
        molecular_calc_chunk.results["hirshfeld_charges"], molecular_hirshfeld_charges
    )


def test_molecular_calc_hirshfeld_volumes(molecular_calc_chunk):
    hirshfeld_volumes = np.array([21.83060659, 6.07674041, 6.07684447])
    assert np.allclose(molecular_calc_chunk.hirshfeld_volumes, hirshfeld_volumes)
    assert np.allclose(
        molecular_calc_chunk.results["hirshfeld_volumes"], hirshfeld_volumes
    )


def test_molecular_calc_hirshfeld_atomic_dipoles(molecular_calc_chunk):
    hirshfeld_atomic_dipoles = np.array(
        [
            [0.04249319, 0.05486053, 0.00000000],
            [0.13710134, -0.00105126, 0.00000000],
            [-0.03534982, 0.13248706, 0.00000000],
        ]
    )
    assert np.allclose(
        molecular_calc_chunk.hirshfeld_atomic_dipoles, hirshfeld_atomic_dipoles
    )
    assert np.allclose(
        molecular_calc_chunk.results["hirshfeld_atomic_dipoles"],
        hirshfeld_atomic_dipoles,
    )


def test_molecular_calc_hirshfeld_dipole(
    molecular_calc_chunk, molecular_hirshfeld_charges, molecular_positions
):
    hirshfeld_dipole = np.sum(
        molecular_hirshfeld_charges.reshape((-1, 1)) * molecular_positions, axis=1
    )
    assert np.allclose(molecular_calc_chunk.hirshfeld_dipole, hirshfeld_dipole)
    assert np.allclose(
        molecular_calc_chunk.results["hirshfeld_dipole"], hirshfeld_dipole
    )


def test_molecular_calc_eigenvalues(molecular_calc_chunk):
    eigenvalues = [
        -507.24511,
        -24.99226,
        -13.12175,
        -9.21626,
        -7.19543,
        -0.01127,
        2.58616,
    ]
    assert np.allclose(molecular_calc_chunk.eigenvalues[0, :, 0], eigenvalues)
    assert np.allclose(
        molecular_calc_chunk.results["eigenvalues"][0, :, 0], eigenvalues
    )


def test_molecular_calc_occupancies(molecular_calc_chunk):
    occupancies = [
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        0.0,
        0.0,
    ]
    assert np.allclose(molecular_calc_chunk.occupancies[0, :, 0], occupancies)
    assert np.allclose(
        molecular_calc_chunk.results["occupancies"][0, :, 0], occupancies
    )
