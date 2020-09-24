import numpy as np
from ase import Atoms
from ase.units import Ha, Bohr
# from ase.io import read
from ase.calculators.openmx.reader import read_openmx

openmx_out_sample = """
System.CurrentDirectory        ./
System.Name        ch4

Atoms.SpeciesAndCoordinates.Unit        Ang
<Definition.of.Atomic.Species
    C  C5.0-s1p1  C_PBE19
    H  H5.0-s1  H_PBE19
Definition.of.Atomic.Species>

Atoms.SpeciesAndCoordinates.Unit        Ang
<Atoms.SpeciesAndCoordinates
    1  C  0.0  0.0  0.1  2.0  2.0
    2  H  0.682793  0.682793  0.682793  0.5  0.5
    3  H  -0.682793  -0.682793  0.68279  0.5  0.5
    4  H  -0.682793  0.682793  -0.682793  0.5  0.5
    5  H  0.682793  -0.682793  -0.682793  0.5  0.5
Atoms.SpeciesAndCoordinates>

<Atoms.UnitVectors
    10.0  0.0  0.0
    0.0  10.0  0.0
    0.0  0.0  10.0
Atoms.UnitVectors>

scf.EigenvalueSolver        Band

  ...

  Utot.         -8.055096450113

  ...

  Chemical potential (Hartree)      -0.156250000000

  ...

*************************************************************************
*************************************************************************
            Decomposed energies in Hartree unit

   Utot = Utot(up) + Utot(dn)
        = Ukin(up) + Ukin(dn) + Uv(up) + Uv(dn)
        + Ucon(up)+ Ucon(dn) + Ucore+UH0 + Uvdw

   Uele = Ukin(up) + Ukin(dn) + Uv(up) + Uv(dn)
   Ucon arizes from a constant potential added in the formalism

           up: up spin state, dn: down spin state
*************************************************************************
*************************************************************************

  Total energy (Hartree) = -8.055096425922011

  Decomposed.energies.(Hartree).with.respect.to.atom

                 Utot
     1    C     -6.261242355014   ...
     2    H     -0.445956460556   ...
     3    H     -0.445956145906   ...
     4    H     -0.450970732231   ...
     5    H     -0.450970732215   ...

  ...

<coordinates.forces
  5
    1     C     0.00   0.00   0.10   0.00000  0.00000 -0.091659
    2     H     0.68   0.68   0.68   0.02700  0.02700  0.029454
    3     H    -0.68  -0.68   0.68  -0.02700 -0.02700  0.029455
    4     H    -0.68   0.68  -0.68   0.00894 -0.00894  0.016362
    5     H     0.68  -0.68  -0.68  -0.00894  0.00894  0.016362
coordinates.forces>

  ...

***********************************************************
***********************************************************
       Fractional coordinates of the final structure
***********************************************************
***********************************************************

     1      C     0.00000   0.00000   0.01000
     2      H     0.06827   0.06827   0.06827
     3      H     0.93172   0.93172   0.06827
     4      H     0.93172   0.06827   0.93172
     5      H     0.06827   0.93172   0.93172


"""


def test_openmx_out():
    with open('openmx_fio_test.out', 'w') as f:
        f.write(openmx_out_sample)
    atoms = read_openmx('openmx_fio_test')
    tol = 1e-2

    # Expected values
    energy = -8.0551
    energies = np.array([-6.2612, -0.4459, -0.4459, -0.4509, -0.4509])
    forces = np.array([[0.00000, 0.00000, -0.091659],
                       [0.02700, 0.02700, 0.029454],
                       [-0.02700, -0.02700, 0.029455],
                       [0.00894, -0.00894, 0.016362],
                       [-0.00894, 0.00894, 0.016362]])

    assert isinstance(atoms, Atoms)

    assert np.isclose(atoms.calc.results['energy'], energy * Ha, atol=tol)
    assert np.all(np.isclose(atoms.calc.results['energies'],
                  energies * Ha, atol=tol))
    assert np.all(np.isclose(atoms.calc.results['forces'],
                  forces * Ha / Bohr, atol=tol))
