'''These tests ensure that the computed PEC curvature matche the actual 
geometries using a somewhat agressive angle_limit for each stepsize.'''
import pytest
from ase import Atoms
from ase.md.contour_exploration import ContourExploration
import numpy as np
from ase.calculators.emt import EMT
from ase.constraints import FixAtoms

def Al_atom_pair(pair_distance):
    atoms = Atoms('AlAl', positions = [
                        [-pair_distance/2, 0, 0],
                        [ pair_distance/2, 0, 0]])
    atoms.center(vacuum = 10)
    atoms.calc = EMT()
    return atoms

def test_kappa1():
    '''This basic test has an atom spinning counter-clockwise around a fixed
    atom. The radius (1/kappa) must therefore be very close the pair_distance.'''
    name = 'test_kappa1'
    
    radius = pair_distance = 2.6
    atoms = Al_atom_pair(pair_distance)
    
    atoms.set_constraint(FixAtoms(indices = [0]))
    atoms.set_velocities( [[0, 0, 0], [0, 1, 0]] )

    dyn = ContourExploration(atoms,
                    maxstep = 1.5,
                    parallel_drift = 0.0,
                    angle_limit = 30,
                    trajectory = name + '.traj',
                    logfile    = name + '.log',
                    ) 

    print("Target Radius (1/kappa) {: .6f} Ang".format( radius))
    for i in range(5):
        dyn.run(30)
        print('Radius (1/kappa) {: .6f} Ang'.format( 1/dyn.kappa))
        assert 0 == pytest.approx(radius - 1/dyn.kappa, abs=2e-3)

def test_kappa2():
    '''This test has two atoms spinning counter-clockwise around eachother. the
    The radius (1/kappa) is less obviously pair_distance*sqrt(2)/2. This is the 
    simplest multi-body analytic curvature test.'''
    name = 'test_kappa2'
    
    pair_distance = 2.5
    radius = pair_distance*np.sqrt(2)/2
    atoms = Al_atom_pair(pair_distance)
    
    atoms.set_velocities( [[0, -1, 0], [0, 1, 0]])

    dyn = ContourExploration(atoms,
                    maxstep = 1.0,
                    parallel_drift = 0.0,
                    angle_limit = 30,
                    trajectory = name + '.traj',
                    logfile    = name + '.log',
                    )
    
    print("Target Radius (1/kappa) {: .6f} Ang".format( radius))
    for i in range(5):
        dyn.run(30)
        print('Radius (1/kappa) {: .6f} Ang'.format( 1/dyn.kappa))
        assert 0 == pytest.approx(radius - 1/dyn.kappa, abs=2e-3)



