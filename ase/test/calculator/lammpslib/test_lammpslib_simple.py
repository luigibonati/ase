import pytest
import numpy as np
from ase import Atom
from ase.build import bulk
import ase.io
from ase import units
from ase.md.verlet import VelocityVerlet

from .common_fixtures import calc_params_NiH


@pytest.fixture
def fcc_Ni_with_H_at_center():
    atoms = bulk("Ni", cubic=True)
    atoms += Atom("H", position=atoms.cell.diagonal() / 2)
    return atoms


@pytest.fixture
def calc_params_Fe():
    calc_params = {}
    calc_params["lammps_header"] = [
        "units           real",
        "atom_style      full",
        "boundary        p p p",
        "box tilt        large",
        "pair_style      lj/cut/coul/long 12.500",
        "bond_style      harmonic",
        "angle_style     harmonic",
        "kspace_style    ewald 0.0001",
        "kspace_modify   gewald 0.01",
        "read_data       lammps.data",
    ]
    calc_params["lmpcmds"] = []
    calc_params["atom_types"] = {"Fe": 1}
    calc_params["create_atoms"] = False
    calc_params["create_box"] = False
    calc_params["boundary"] = False
    calc_params["log_file"] = "test.log"
    calc_params["keep_alive"] = True
    return calc_params


@pytest.fixture
def lammps_data_file():
    lammps_data_file = """
    8 atoms
    1 atom types
    6 bonds
    1 bond types
    4 angles
    1 angle types

    -5.1188800000000001e+01 5.1188800000000001e+01 xlo xhi
    -5.1188800000000001e+01 5.1188800000000001e+01 ylo yhi
    -5.1188800000000001e+01 5.1188800000000001e+01 zlo zhi
    0.000000 0.000000 0.000000 xy xz yz

    Masses

    1 56

    Bond Coeffs

    1 646.680887 1.311940

    Angle Coeffs

    1 300.0 107.0

    Pair Coeffs

    1 0.105000 3.430851

    Atoms

    1 1 1 0.0 -7.0654012878945753e+00 -4.7737244253442213e-01 -5.1102452666801824e+01 2 -1 6
    2 1 1 0.0 -8.1237844371679362e+00 -1.3340695922796841e+00 4.7658302278206179e+01 2 -1 5
    3 1 1 0.0 -1.2090525219882498e+01 -3.2315354021627760e+00 4.7363437099502839e+01 2 -1 5
    4 1 1 0.0 -8.3272244953257601e+00 -4.8413162043515321e+00 4.5609055410298623e+01 2 -1 5
    5 2 1 0.0 -5.3879618209198750e+00 4.9524635221072280e+01 3.0054862714858366e+01 6 -7 -2
    6 2 1 0.0 -8.4950075933508273e+00 -4.9363297129348325e+01 3.2588925816534982e+01 6 -6 -2
    7 2 1 0.0 -9.7544282093133940e+00 4.9869755980935565e+01 3.6362287886934432e+01 6 -7 -2
    8 2 1 0.0 -5.5712437770663756e+00 4.7660225526197003e+01 3.8847235874270240e+01 6 -7 -2

    Velocities

    1 -1.2812627962466232e-02 -1.8102422526771818e-03 8.8697845357364469e-03
    2 7.7087896348612683e-03 -5.6149199730983867e-04 1.3646724560472424e-02
    3 -3.5128553734623657e-03 1.2368758037550581e-03 9.7460093657088121e-03
    4 1.1626059392751346e-02 -1.1942908859710665e-05 8.7505240354339674e-03
    5 1.0953500823880464e-02 -1.6710422557096375e-02 2.2322216388444985e-03
    6 3.7515599452757294e-03 1.4091708517087744e-02 7.2963916249300454e-03
    7 5.3953961772651359e-03 -8.2013715102925017e-03 2.0159609509813853e-02
    8 7.5074008407567160e-03 5.9398495239242483e-03 7.3144909044607909e-03

    Bonds

    1 1 1 2
    2 1 2 3
    3 1 3 4
    4 1 5 6
    5 1 6 7
    6 1 7 8

    Angles

    1 1 1 2 3
    2 1 2 3 4
    3 1 5 6 7
    4 1 6 7 8
    """
    return lammps_data_file


@pytest.mark.calculator("lammpslib")
def test_lammpslib_simple(
    factory, calc_params_NiH, fcc_Ni_with_H_at_center, calc_params_Fe, lammps_data_file
):
    """
    Get energy from a LAMMPS calculation of an uncharged system.
    This was written to run with the 30 Apr 2019 version of LAMMPS,
    for which uncharged systems require the use of 'kspace_modify gewald'.
    """
    NiH = fcc_Ni_with_H_at_center

    # Add a bit of distortion to the cell
    NiH.set_cell(
        NiH.cell + [[0.1, 0.2, 0.4], [0.3, 0.2, 0.0], [0.1, 0.1, 0.1]], scale_atoms=True
    )

    calc = factory.calc(**calc_params_NiH)
    NiH.calc = calc

    E = NiH.get_potential_energy()
    F = NiH.get_forces()
    S = NiH.get_stress()

    print("Energy: ", E)
    print("Forces:", F)
    print("Stress: ", S)
    print()

    E = NiH.get_potential_energy()
    F = NiH.get_forces()
    S = NiH.get_stress()

    calc = factory.calc(**calc_params_NiH)
    NiH.calc = calc

    E2 = NiH.get_potential_energy()
    F2 = NiH.get_forces()
    S2 = NiH.get_stress()

    assert E == pytest.approx(E2, rel=1e-4)
    assert F == pytest.approx(F2, rel=1e-4)
    assert S == pytest.approx(S2, rel=1e-4)

    NiH.rattle(stdev=0.2)
    E3 = NiH.get_potential_energy()
    F3 = NiH.get_forces()
    S3 = NiH.get_stress()

    print("rattled atoms")
    print("Energy: ", E3)
    print("Forces:", F3)
    print("Stress: ", S3)
    print()

    assert not np.allclose(E, E3)
    assert not np.allclose(F, F3)
    assert not np.allclose(S, S3)

    # Add another H
    NiH += Atom("H", position=NiH.cell.diagonal() / 4)
    E4 = NiH.get_potential_energy()
    F4 = NiH.get_forces()
    S4 = NiH.get_stress()

    assert not np.allclose(E4, E3)
    assert not np.allclose(F4[:-1, :], F3)
    assert not np.allclose(S4, S3)

    # the example from the docstring

    NiH = fcc_Ni_with_H_at_center
    calc = factory.calc(**calc_params_NiH)
    NiH.calc = calc
    print("Energy ", NiH.get_potential_energy())

    # a more complicated example, reading in a LAMMPS data file

    # then we run the actual test
    with open("lammps.data", "w") as fd:
        fd.write(lammps_data_file)

    at = ase.io.read(
        "lammps.data", format="lammps-data", Z_of_type={1: 26}, units="real"
    )

    calc = factory.calc(**calc_params_Fe)
    at.calc = calc
    dyn = VelocityVerlet(at, 1 * units.fs)

    energy = at.get_potential_energy()
    assert energy == pytest.approx(2041.411982950972, rel=1e-4)

    dyn.run(10)
    energy = at.get_potential_energy()
    assert energy == pytest.approx(312.4315854721744, rel=1e-4)
