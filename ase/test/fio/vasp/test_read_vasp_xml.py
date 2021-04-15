import pytest

import numpy as np
from ase.io import read
from io import StringIO


@pytest.fixture()
def vasprun():
    # "Hand-written" (reduced) vasprun.xml
    sample_vasprun = """\
<?xml version="1.0" encoding="ISO-8859-1"?>
<modeling>
 <structure name="primitive_cell" >
  <crystal>
   <varray name="basis" >
    <v>       3.16000000       0.00000000       0.00000000 </v>
    <v>       0.00000000       3.16000000       0.00000000 </v>
    <v>       0.00000000       0.00000000       3.16000000 </v>
   </varray>
  </crystal>
  <varray name="positions" >
   <v>       0.00000000       0.00000000       0.00000000 </v>
   <v>       0.50000000       0.50000000       0.50000000 </v>
  </varray>
 </structure>
 <kpoints>
  <varray name="kpointlist" >
   <v>       0.00000000       0.00000000       0.00000000 </v>
  </varray>
 </kpoints>
 <atominfo>
  <atoms>       2 </atoms>
  <types>       1 </types>
  <array name="atoms" >
   <set>
    <rc><c>W </c><c>   1</c></rc>
    <rc><c>W </c><c>   1</c></rc>
   </set>
  </array>
 </atominfo>
 <structure name="initialpos" >
  <crystal>
   <varray name="basis" >
    <v>       3.16000000       0.00000000       0.00000000 </v>
    <v>       0.00000000       3.16000000       0.00000000 </v>
    <v>       0.00000000       0.00000000       3.16000000 </v>
   </varray>
   <i name="volume">     31.55449600 </i>
  </crystal>
  <varray name="positions" >
   <v>       0.00000000       0.00000000       0.00000000 </v>
   <v>       0.50000000       0.50000000       0.50000000 </v>
  </varray>
 </structure>
"""
    return sample_vasprun


@pytest.fixture()
def calculation():
    # "Hand-written" calculation record
    sample_calculation ="""\
 <calculation>
  <scstep>
   <energy>
    <i name="e_fr_energy">     32.61376955 </i>
    <i name="e_wo_entrp">     32.60797165 </i>
    <i name="e_0_energy">     32.61183692 </i>
   </energy>
  </scstep>
  <scstep>
   <energy>
    <i name="e_fr_energy">    -29.67243317 </i>
    <i name="e_wo_entrp">    -29.68588381 </i>
    <i name="e_0_energy">    -29.67691672 </i>
   </energy>
  </scstep>
  <structure>
   <crystal>
    <varray name="basis" >
     <v>       3.16000000       0.00000000       0.00000000 </v>
     <v>       0.00000000       3.16000000       0.00000000 </v>
     <v>       0.00000000       0.00000000       3.16000000 </v>
    </varray>
    <i name="volume">     31.55449600 </i>
    <varray name="rec_basis" >
     <v>       0.31645570       0.00000000       0.00000000 </v>
     <v>       0.00000000       0.31645570       0.00000000 </v>
     <v>       0.00000000       0.00000000       0.31645570 </v>
    </varray>
   </crystal>
   <varray name="positions" >
    <v>       0.00000000       0.00000000       0.00000000 </v>
    <v>       0.50000000       0.50000000       0.50000000 </v>
   </varray>
  </structure>
  <varray name="forces" >
   <v>       7.58587457      -5.22590317       6.88227285 </v>
   <v>      -7.58587457       5.22590317      -6.88227285 </v>
  </varray>
  <varray name="stress" >
   <v>    4300.36902090    -284.50040544   -1468.20603140 </v>
   <v>    -284.50040595    4824.17435683   -1625.37541639 </v>
   <v>   -1468.20603158   -1625.37541697    5726.84189498 </v>
  </varray>
  <energy>
   <i name="e_fr_energy">    -29.67243317 </i>
   <i name="e_wo_entrp">    -29.68588381 </i>
   <i name="e_0_energy">    -29.67691672 </i>
  </energy>
"""
    return sample_calculation


def test_atoms(vasprun):

    atoms = read(StringIO(vasprun), index=-1, format='vasp-xml')

    # check number of atoms
    assert len(atoms) == 2

    # make sure it is still tungsten
    assert all(np.array(atoms.get_chemical_symbols()) == "W")

    # check scaled_positions
    expected_scaled_positions = np.array([[0.0, 0.0, 0.0],
                                         [0.5,  0.5, 0.5]])

    np.testing.assert_allclose(atoms.get_scaled_positions(),
                                  expected_scaled_positions)

    expected_cell = np.array([[3.16, 0.0, 0.0],
                              [0.0, 3.16, 0.0],
                              [0.0, 0.0, 3.16]])

    # check cell
    np.testing.assert_allclose(atoms.cell, expected_cell)

    # check real positions
    np.testing.assert_allclose(atoms.positions,
                                  expected_scaled_positions @
                                  atoms.cell.complete())


def test_calculation(vasprun, calculation):

    from ase.units import GPa

    atoms = read(StringIO(vasprun + calculation), index=-1, format='vasp-xml')

    expected_e_0_energy = -29.67691672
    assert atoms.get_potential_energy() == expected_e_0_energy

    expected_e_fr_energy = -29.67243317
    assert (atoms.get_potential_energy(force_consistent=True) ==
            expected_e_fr_energy)

    expected_forces = np.array([[7.58587457, -5.22590317, 6.88227285],
                                [-7.58587457, 5.22590317, -6.88227285]])
    np.testing.assert_allclose(atoms.get_forces(),
                                  expected_forces)

    expected_stress = np.array([[4300.36902090, -284.50040544, -1468.20603140],
                               [-284.50040595, 4824.17435683, -1625.37541639],
                               [-1468.20603158, -1625.37541697, 5726.84189498]])

    expected_stress *= -0.1 * GPa
    expected_stress = expected_stress.reshape(9)[[0, 4, 8, 5, 2, 1]]

    np.testing.assert_allclose(atoms.get_stress(), expected_stress)
