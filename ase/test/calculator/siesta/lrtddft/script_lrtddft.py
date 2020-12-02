from ase.calculators.siesta.siesta_lrtddft import siestaLRTDDFT
from ase.build import molecule
import numpy as np
import matplotlib.pyplot as plt

# Define the systems
CH4 = molecule('CH4')

lr = siestaLRTDDFT(label="siesta", jcutoff=7, iter_broadening=0.15,
                    xc_code='LDA,PZ', tol_loc=1e-6, tol_biloc=1e-7)

# run siesta
lr.get_ground_state(CH4)

freq=np.arange(0.0, 25.0, 0.05)
pmat = lr.get_polarizability(freq)

# plot polarizability
fig = plt.figure(1)
ax1 = fig.add_subplot(111)
ax1.plot(freq, pmat[0, 0, :].imag)

ax1.set_xlabel(r"$\omega$ (eV)")
ax1.set_ylabel(r"Im($P_{xx}$) (au)")
ax1.set_title(r"Non interacting")

fig.tight_layout()
plt.show()
