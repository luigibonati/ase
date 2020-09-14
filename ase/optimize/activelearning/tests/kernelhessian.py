from ase.build import fcc100
from ase.optimize.activelearning.gpfp.fingerprint import RadialAngularFP
import numpy as np
import time

'''
Test that the Hessian of the squared exponential kernel,
using the fingerprint, matches with the numerical one.
'''


def new_fp():
    return RadialAngularFP(limit=10.0,
                           Rlimit=4.0,
                           delta=0.5,
                           ascale=0.2)


# Create slab:
slab = fcc100('Ag', size=(2, 2, 2))
slab[-4].symbol = 'Au'
slab[-2].symbol = 'Au'
slab.center(axis=2, vacuum=4.0)
slab.rattle(0.05)

fp = new_fp()
fp.set_atoms(slab)

index1 = 1
index2 = 6

dx = 0.0001

# Numerical:
t0 = time.time()
hessian = np.zeros([3, 3])
for cc1 in range(3):
    for cc2 in range(3):

        change1, change2 = np.zeros(3), np.zeros(3)
        change1[cc1] = dx
        change2[cc2] = dx

        atoms1 = slab.copy()
        atoms1.rattle(0.01, seed=1)
        fp1 = new_fp()
        fp1.set_atoms(atoms1)

        atoms2 = atoms1.copy()
        atoms2[index1].position += change1
        fp2 = new_fp()
        fp2.set_atoms(atoms2)

        atoms3 = atoms1.copy()
        atoms3.rattle(0.02, seed=2)
        fp3 = new_fp()
        fp3.set_atoms(atoms3)

        atoms4 = atoms3.copy()
        atoms4[index2].position += change2
        fp4 = new_fp()
        fp4.set_atoms(atoms4)

        hessian[cc1, cc2] = (((fp1.kernel(fp2, fp4) - fp1.kernel(fp2, fp3)) -
                             (fp1.kernel(fp1, fp4) - fp1.kernel(fp1, fp3)))
                             / dx**2)
print('\nNumerical:\n', hessian, '\nTime consumed:',
      (time.time() - t0), 'seconds\n')

# Analytical:
t0 = time.time()
atoms1 = slab.copy()
atoms1.rattle(0.01, seed=1)
fp1 = new_fp()
fp1.set_atoms(atoms1)
atoms3 = atoms1.copy()
atoms3.rattle(0.02, seed=2)
fp3 = new_fp()
fp3.set_atoms(atoms3)
analytical = fp1.kernel_hessian(fp3, index1, index2)
print('\nAnalytical:\n', analytical, '\nTime consumed:',
      (time.time() - t0), 'seconds\n')
