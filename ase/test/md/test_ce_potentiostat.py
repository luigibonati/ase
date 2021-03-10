'''These tests ensure that the potentiostat can keep a sysytem near the PEC'''

import pytest
from ase.lattice.cubic import FaceCenteredCubic
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.contour_exploration import ContourExploration
import numpy as np
from ase.calculators.emt import EMT
from ase import Atoms


from test_ce_kappa import Al_atom_pair

def test_potentiostat():
    '''This is very realistic and stringent test of the potentiostatic accuracy
     with 32 atoms at ~235 meV/atom above the ground state.'''
    name = 'test_potentiostat'

    size = 2
    seed = 19460926

    atoms = FaceCenteredCubic(directions=[[1, 0, 0],
                                          [0, 1, 0],
                                          [0, 0, 1]],
                              symbol='Al',
                              size=(size, size, size),
                              pbc=True)
    atoms.calc = EMT()

    E0 = atoms.get_potential_energy()

    atoms.rattle(stdev=0.18, seed=seed)
    initial_energy = atoms.get_potential_energy()

    rng = np.random.RandomState(seed)
    MaxwellBoltzmannDistribution(atoms, temperature_K=300, rng=rng)
    dyn = ContourExploration(atoms,
                             maxstep=1.0,
                             parallel_drift=0.05,
                             remove_translation=True,
                             potentiostat_step_scale=None,
                             energy_target=initial_energy,
                             use_frenet_serret=True,
                             angle_limit=20,
                             rng=rng,
                             trajectory=name + '.traj',
                             logfile=name + '.log',
                             )

    print("Energy Above Ground State: {: .4f} eV/atom".format(
        (initial_energy - E0) / len(atoms)))
    for i in range(5):
        dyn.run(5)
        energy_error = (atoms.get_potential_energy() -
                        initial_energy) / len(atoms)
        print('Potentiostat Error {: .4f} eV/atom'.format(energy_error))
        assert 0 == pytest.approx(energy_error, abs=0.01)


def test_potentiostat_no_fs():
    '''This test ensures that the potentiostat is working even when curvature
    extrapolation (use_fs) is turned off.'''
    name = 'test_potentiostat_no_fs'
#    radius = 2.5
#    atoms = Atoms('AlAl', positions=[[-radius / 2, 0, 0], [radius / 2, 0, 0]])
#    atoms.center(vacuum=10)
#    atoms.calc = EMT()
    atoms = Al_atom_pair()

    atoms.set_momenta([[0, -1, 0], [0, 1, 0]])

    initial_energy = atoms.get_potential_energy()
    dyn = ContourExploration(atoms,
                             maxstep=0.2,
                             parallel_drift=0.0,
                             remove_translation=False,
                             energy_target=initial_energy,
                             potentiostat_step_scale=None,
                             use_frenet_serret=False,
                             trajectory=name + '.traj',
                             logfile=name + '.log',
                             )

    for i in range(5):
        dyn.run(10)
        energy_error = (atoms.get_potential_energy() -
                        initial_energy) / len(atoms)
        print('Potentiostat Error {: .4f} eV/atom'.format(energy_error))
        assert 0 == pytest.approx(energy_error, abs=0.01)
