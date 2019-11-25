import numpy as np
from ase.calculators.datadriven import new_openmx
from ase.test.datadriven.main import test_singlepoint
from ase.build import molecule
from ase.utils import workdir

atoms = molecule('H2O')
atoms.center(vacuum=2.5)

calc = new_openmx(
    scf_energycutoff=50,
    kpts=1,
    scf_mixing_type='Rmm-Diis',
    scf_criterion=1e-3)

atoms.calc = calc
test_singlepoint('openmx', atoms)


e = atoms.get_potential_energy()
f = atoms.get_forces()


energy = -474.11500268758516
force = np.array([[-3.08532403e-10, -1.95403855e-09, -1.01401451e-01],
       [-5.09078464e-09,  7.42704323e-02, -2.33793323e-02],
       [ 5.60500531e-09, -7.42704305e-02, -2.33793365e-02]])

eerr = abs(e - energy)
ferr = np.abs(f - force).max()
assert eerr  < 1e-6
assert ferr < 1e-6
