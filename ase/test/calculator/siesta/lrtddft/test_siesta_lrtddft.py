from ase.calculators.siesta.siesta_lrtddft import siestaLRTDDFT
from ase.build import molecule
import numpy as np

def test_siesta_lrtddft():
    # Define the systems
    CH4 = molecule('CH4')

    lr = siestaLRTDDFT(label="siesta", xc_code='LDA,PZ')

    # run siesta
    lr.get_ground_state(CH4)

    freq = np.arange(0.0, 25.0, 0.5)
    pmat = lr.get_polarizability(freq)
