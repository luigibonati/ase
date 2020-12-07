from ase.calculators.siesta.siesta_lrtddft import siestaLRTDDFT
from ase.build import molecule
import numpy as np

def test_siesta_lrtddft(siesta_factory):

    try:
        import pynao
        performTest = True
        print("pynao version: ", pynao.__version__)
    except ModuleNotFoundError as err:
        print("Siesta lrtddft tests requires pynao: ", err)
        performTest = False

    if performTest:
        # Define the systems
        ch4 = molecule('CH4')

        LRTDDFT = siestaLRTDDFT(label="siesta", xc_code='LDA,PZ')

        # run siesta
        LRTDDFT.get_ground_state(ch4)

        freq = np.arange(0.0, 25.0, 0.5)
        pmat = LRTDDFT.get_polarizability(freq)
        assert pmat.size == 3*3*freq.size
