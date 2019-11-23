import numpy as np
from ase.calculators.openmx import OpenMX
from ase.calculators.datadriven import new_openmx
from ase.build import molecule
from ase.utils import workdir

atoms = molecule('H2O')
atoms.center(vacuum=2.5)

#calc = OpenMX()
calc = new_openmx(
    scf_energycutoff=150,
    scf_mixing_type='Rmm-Diis',
    scf_criterion=1e-6)

atoms.calc = calc
with workdir('omx-work', mkdir=True):
    e = atoms.get_potential_energy()
    f = atoms.get_forces()

print(e)
print(f)
eref = -474.2232383806571
fref = np.array([[-1.25984064e-08,  0.00000000e+00,  1.54738047e-01],
                 [ 6.37633632e-09,  9.07735945e-02, -4.88082632e-02],
                 [ 6.37633632e-09, -9.07735945e-02, -4.88082632e-02]])

eerr = abs(e - eref)
ferr = np.abs(f - fref).max()
assert eerr  < 1e-6
assert ferr < 1e-6
