from ase import Atoms
from ase.calculators.emt import EMT
from ase.calculators.idealgas import IdealGas
from ase.md.verlet import VelocityVerlet
from ase.calculators.plumed import Plumed
from ase.calculators.lj import LennardJones
import numpy as np
from ase.io.trajectory import Trajectory
from pytest import approx


def test_CVs(testdir):
    ''' This test calls plumed-ASE calculator for computing some CVs.
    Moreover, it computes those CVs directly from atoms.positions and
    compares them'''
    # plumed setting
    input = ["c1: COM ATOMS=1,2",
             "c2: CENTER ATOMS=1,2",
             "l: DISTANCE ATOMS=c1,c2",
             "d: DISTANCE ATOMS=1,2",
             "c: COORDINATION GROUPA=1 GROUPB=2 R_0=100 MM=0 NN=10",
             "PRINT ARG=d,c,l STRIDE=10 FILE=COLVAR_test1"]

    # execution
    atoms = Atoms('CO', positions=[[0, 0, 0], [0, 0, 5]])  # CO molecule
    timestep = 5
    with Plumed(EMT(), input, timestep, atoms) as calc:
        with VelocityVerlet(atoms, timestep) as dyn:
            dyn.run(100)

    # this compares the time calculated by ASE and plumed
    timeASE = np.arange(0., 501., 50)
    timePlumed = np.loadtxt('COLVAR_test1', usecols=0)
    assert timeASE == approx(timePlumed), "Error in time registered by plumed"
    
    # This compares the distance of atoms calculated by ASE and plumed
    distASE = np.array([5., 51.338332, 141.252854, 231.167376, 321.081899,
                        410.996421, 500.910943, 590.825465, 680.739987,
                        770.654509, 860.569031])
    distPlumed = np.loadtxt('COLVAR_test1', usecols=1)
    assert distPlumed == approx(distASE), "Error in distance "

    # this compares the coordination number calculated by ASE and plumed
    CASE = np.array([1.0000e+00, 9.9873e-01, 3.0655e-02, 2.2900e-04, 9.0000e-06,
                     1.0000e-06, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
                     0.0000e+00])
    CPlumed = np.loadtxt('COLVAR_test1', usecols=2)
    assert CASE == approx(CPlumed, abs=1E-5), "Error in coordination number"
    
    # this compares the distance between center of mass and geometrical center
    # calculated by ASE and plumed
    centersASE = np.array([0.355944, 3.654717, 10.05563, 16.456542, 22.857455,
                           29.258367, 35.65928, 42.060192, 48.461104, 54.862017,
                           61.262929])

    centersPlumed = np.loadtxt('COLVAR_test1', usecols=3)

    assert centersASE == approx(centersPlumed)
    

def test_metadyn(testdir):
    '''This test computes a Metadynamics calculation,
    This result is compared with the same calulation made externally'''
    input = ["d: DISTANCE ATOMS=1,2",
             "METAD ARG=d SIGMA=0.5 HEIGHT=2 PACE=20 FILE=HILLS_direct"]

    # Metadynamics simulation
    atoms = Atoms('CO', positions=[[0, 0, 0], [6.7, 0, 0]])
    timestep = 0.05
    with Plumed(LennardJones(epsilon=10, sigma=6), input, timestep, atoms) as calc:
        with VelocityVerlet(atoms, timestep, trajectory='test-direct.traj') as dyn:
            dyn.run(58)
       
    position1 = -0.0491871
    position2 = 6.73693
    forceWithBias = 0.28807

    assert (atoms.get_positions()[0][0] == approx(position1, abs=0.01) and
            atoms.get_positions()[1][0] == approx(position2, abs=0.01)), "Error in the metadynamics simulation"
    assert atoms.get_forces()[0][0] == approx(forceWithBias, abs=0.01), "Error in the computation of Bias-forces"


def test_restart(testdir):
    input = ["d: DISTANCE ATOMS=1,2",
             "METAD ARG=d SIGMA=0.5 HEIGHT=2 PACE=20 FILE=HILLS"]

    # Metadynamics simulation
    atoms = Atoms('CO', positions=[[0, 0, 0], [6.7, 0, 0]])
    timestep = 0.05

    # first steps
    with Plumed(LennardJones(epsilon=10, sigma=6), input, timestep, atoms) as calc:
        with VelocityVerlet(atoms, timestep, trajectory='test-restart.traj') as dyn:
            dyn.run(29)
    
    # rest of steps with restart
    atoms1 = Atoms('CO')
    with Plumed(LennardJones(epsilon=10, sigma=6), input, timestep, atoms1,
                prev_traj='test-restart.traj') as calc:
        with VelocityVerlet(atoms1, timestep,
                         trajectory='test-restart.traj',
                         append_trajectory=True) as dyn:
            dyn.run(29)
    
    # Values computed externally
    position1 = -0.0491871
    position2 = 6.73693
    forceWithBias = 0.28807

    assert (atoms1.get_positions()[0][0] == approx(position1, abs=0.01) and
            atoms1.get_positions()[1][0] == approx(position2, abs=0.01)), "Error in the metadynamics simulation"
    assert atoms1.get_forces()[0][0] == approx(forceWithBias, abs=0.01), "Error in the computation of Bias-forces"


def test_postpro(testdir):
    # Metadynamics simulation
    input = ["d: DISTANCE ATOMS=1,2",
             "METAD ARG=d SIGMA=0.5 HEIGHT=2 PACE=20 FILE=HILLS_direct"]

    atoms = Atoms('CO', positions=[[0, 0, 0], [6.7, 0, 0]])
    timestep = 0.05
    with Plumed(LennardJones(epsilon=10, sigma=6), input, timestep, atoms) as calc:
        with VelocityVerlet(atoms, timestep, trajectory='test-direct.traj') as dyn:
            dyn.run(58)
    
    # Postpro resconstruction
    input = ["d: DISTANCE ATOMS=1,2",
             "METAD ARG=d SIGMA=0.5 HEIGHT=2 PACE=20 FILE=HILLS_postpro"]
    atoms = Atoms('CO', positions=[[0, 0, 0], [6.7, 0, 0]])
    timestep = 0.05
    with Plumed(IdealGas(), input, timestep, atoms) as calc:
        traj = Trajectory('test-direct.traj')
        calc.analysis(traj)
        traj.close()

    direct = np.loadtxt("HILLS_direct")
    postpr = np.loadtxt("HILLS_postpro")

    assert postpr == approx(direct)
