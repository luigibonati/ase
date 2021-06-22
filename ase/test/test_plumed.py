from ase import Atoms
from ase.calculators.emt import EMT
from ase.md.verlet import VelocityVerlet
from ase.calculators.plumed import Plumed
from ase.calculators.lj import LennardJones
import numpy as np
from ase.io.trajectory import Trajectory
import subprocess
from time import time

t = time()


''' This test calls plumed-ASE calculator for computing some CVs.
Moreover, it computes those CVs directly from atoms.positions and
compares them'''

def test_CVs():
    # plumed setting
    input = ["c1: COM ATOMS=1,2",
             "c2: CENTER ATOMS=1,2",
             "l: DISTANCE ATOMS=c1,c2",
             "d: DISTANCE ATOMS=1,2",
             "c: COORDINATION GROUPA=1 GROUPB=2 R_0=100 MM=0 NN=10",
             "PRINT ARG=d,c,l STRIDE=10 FILE=COLVAR_test1{}".format(t)]

    # execution
    atoms = Atoms('CO', positions=[[0, 0, 0], [0, 0, 5]])  # CO molecule
    timestep = 5
    calc = Plumed(EMT(), input, timestep, atoms, log='/dev/null')
    dyn = VelocityVerlet(atoms, timestep)
    for i in range(10):
        dyn.run(10)
    calc.close()

    # this compares the time calculated by ASE and plumed
    timeASE = np.arange(0., 501., 50)
    timePlumed = np.loadtxt('COLVAR_test1{}'.format(t), usecols=0)
    a = abs(timeASE - timePlumed)
    assert np.all(a < 1E-5), "Error in time registered by plumed"

    # This compares the distance of atoms calculated by ASE and plumed
    distASE = np.array([5., 51.338332, 141.252854, 231.167376, 321.081899,
                        410.996421, 500.910943, 590.825465, 680.739987,
                        770.654509, 860.569031])
    distPlumed = np.loadtxt('COLVAR_test1{}'.format(t), usecols=1)
    b = abs(distASE - distPlumed)
    assert np.all(b < 1E-5), "Error in distance"

    # this compares the coordination number calculated by ASE and plumed
    CASE = np.array([1.0000e+00, 9.9873e-01, 3.0655e-02, 2.2900e-04, 9.0000e-06,
                     1.0000e-06, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
                     0.0000e+00])
    CPlumed = np.loadtxt('COLVAR_test1{}'.format(t), usecols=2)
    c = abs(CASE - CPlumed)
    assert np.all(c <= 1E-5), "Error in coordination number"

    # this compares the distance between center of mass and geometrical center
    # calculated by ASE and plumed
    centersASE = np.array([0.355944, 3.654717, 10.05563, 16.456542, 22.857455,
                           29.258367, 35.65928, 42.060192, 48.461104, 54.862017,
                           61.262929])

    centersPlumed = np.loadtxt('COLVAR_test1{}'.format(t), usecols=3)
    d = abs(centersASE - centersPlumed)
    assert np.all(d < 1E-5), "Error in distance between centers of mass"


'''This test computes a Metadynamics calculation,
This result is compared with the same calulation made externally'''


def test_metadyn():
    input = ["d: DISTANCE ATOMS=1,2",
             "METAD ARG=d SIGMA=0.5 HEIGHT=2 PACE=20 FILE=HILLS_direct{}".format(t)]

    # Metadynamics simulation
    atoms = Atoms('CO', positions=[[0, 0, 0], [6.7, 0, 0]])
    timestep = 0.05
    calc = Plumed(LennardJones(epsilon=10, sigma=6), input, timestep, atoms,
                  log='/dev/null')
    dyn = VelocityVerlet(atoms, timestep, trajectory='test-direct{}.traj'.format(t))
    dyn.run(58)

    # Values computed externally
    position1 = -0.0491871
    position2 = 6.73693
    forceWithBias = 0.28807

    deltaPos1 = abs(atoms.get_positions()[0][0] - position1)
    deltaPos2 = abs(atoms.get_positions()[1][0] - position2)
    deltaForce = abs(atoms.get_forces()[0][0] - forceWithBias)
    calc.close()
    
    assert (deltaPos1 < 0.01 and
            deltaPos2 < 0.01), "Error in the metadynamics simulation"
    assert (deltaForce < 0.01), "Error in the computation of Bias-forces"


def test_restart():
    input = ["d: DISTANCE ATOMS=1,2",
             "METAD ARG=d SIGMA=0.5 HEIGHT=2 PACE=20 FILE=HILLS{}".format(t)]

    # Metadynamics simulation
    atoms = Atoms('CO', positions=[[0, 0, 0], [6.7, 0, 0]])
    timestep = 0.05

    # first steps
    calc = Plumed(LennardJones(epsilon=10, sigma=6), input, timestep, atoms,
                  log='/dev/null')
    traj = Trajectory('test-restart{}.traj'.format(t), 'w', atoms)
    dyn = VelocityVerlet(atoms, timestep)
    dyn.attach(traj.write, interval=1)
    dyn.run(29)
    calc.close()
    traj.close()

    # rest of steps with restart
    atoms1 = Atoms('CO')
    calc = Plumed(LennardJones(epsilon=10, sigma=6), input, timestep, atoms1,
                  prev_traj='test-restart{}.traj'.format(t),
                  log='/dev/null')
    dyn = VelocityVerlet(atoms1, timestep,
                         trajectory='test-restart{}.traj'.format(t),
                         append_trajectory=True)
    dyn.run(29)
    # Values computed externally
    position1 = -0.0491871
    position2 = 6.73693
    forceWithBias = 0.28807

    deltaPos1 = abs(atoms1.get_positions()[0][0] - position1)
    deltaPos2 = abs(atoms1.get_positions()[1][0] - position2)
    deltaForce = abs(atoms1.get_forces()[0][0] - forceWithBias)
    calc.close()

    assert (deltaPos1 < 0.01 and
            deltaPos2 < 0.01), "Error in the metadynamics simulation"
    assert (deltaForce < 0.01), "Error in the computation of Bias-forces"

def test_postpro():
    input = ["d: DISTANCE ATOMS=1,2",
             "METAD ARG=d SIGMA=0.5 HEIGHT=2 PACE=20 FILE=HILLS_postpro{}".format(t)]

    # Metadynamics simulation
    atoms = Atoms('CO', positions=[[0, 0, 0], [6.7, 0, 0]])
    timestep = 0.05
    calc = Plumed('Dummy', input, timestep, atoms,
                  log='/dev/null')
    traj = Trajectory('test-direct{}.traj'.format(t))
    calc.analysis(traj)
    calc.close()

    direct = np.loadtxt("HILLS_direct{}".format(t))
    postpr = np.loadtxt("HILLS_postpro{}".format(t))
    diff = direct-postpr
    
    s=0
    for i in diff.flatten():
        s += i**2
    assert (s == 0), "Error in postprocessing"

def test_remove():
    # actually, it is not a test. It just removes the files creared in
    # the other tests
    subprocess.Popen("rm -f bck.*", shell=True)
    subprocess.Popen("rm -f COLVAR*", shell=True)
    subprocess.Popen("rm -f *HILLS*", shell=True)
    subprocess.Popen("rm -f *.log", shell=True)
    subprocess.Popen("rm -rf __pycache__", shell=True)
    subprocess.Popen("rm -r *.traj", shell=True)
    assert True


if __name__ == '__main__':
    test_CVs()
    test_metadyn()
    test_restart()
    test_postpro()
    test_remove()
    
