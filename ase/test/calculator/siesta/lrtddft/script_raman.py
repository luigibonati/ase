from pytest import approx, fixture

from ase.units import Ry, eV, Ha
from ase.calculators.siesta import Siesta
from ase.calculators.siesta.siesta_lrtddft import siesta_raman
from ase.vibrations.raman import StaticRamanCalculator
from ase.vibrations.placzek import PlaczekStatic
#from ase.calculators.siesta.siesta_raman import SiestaRaman
from ase import Atoms
from ase.build import bulk
import numpy as np


def test_CO2():
    """
    Define the systems
    example of Raman calculation for CO2 molecule,
    comparison with QE calculation can be done from
    https://github.com/maxhutch/quantum-espresso/blob/master/PHonon/examples/example15/README
    """

    CO2 = Atoms('CO2',
                positions=[[-0.009026, -0.020241, 0.026760],
                           [1.167544, 0.012723, 0.071808],
                           [-1.185592, -0.053316, -0.017945]],
                cell=[20, 20, 20])

    # enter siesta input
    siesta = Siesta(
        mesh_cutoff=150 * Ry,
        basis_set='DZP',
        pseudo_qualifier='',
        energy_shift=(10 * 10**-3) * eV,
        fdf_arguments={
            'SCFMustConverge': False,
            'COOP.Write': True,
            'WriteDenchar': True,
            'PAO.BasisType': 'split',
            'DM.Tolerance': 1e-4,
            'DM.MixingWeight': 0.01,
            'MaxSCFIterations': 300,
            'DM.NumberPulay': 4,
            'XML.Write': True,
            'DM.UseSaveDM': True})

    CO2.calc = siesta
    delta = 0.02
    
    name = 'bp'
    rm = StaticRamanCalculator(CO2, siesta_raman, name=name,
                               delta=delta, exkwargs=dict(label="siesta",
                                   jcutoff=7, iter_broadening=0.15,
                                   xc_code='LDA,PZ', tol_loc=1e-6, tol_biloc=1e-7)
                               )
    rm.run()

    pz = PlaczekStatic(CO2, name=name)
    e_vib = pz.get_energies()
    i_vib = pz.get_absolute_intensities()
    assert len(e_vib) == 6
    pz.summary()


    #ram = SiestaRaman(CO2, siesta, label="siesta", jcutoff=7, iter_broadening=0.15/Ha,
    #                  xc_code='LDA,PZ', tol_loc=1e-6, tol_biloc=1e-7,
    #                  freq=np.arange(0.0, 5.0, 0.05))
    #ram.run()
    #ram.summary(intensity_unit_ram='A^4 amu^-1')
    #
    #ram.write_spectra(start=200)


#@fixture(scope='module')
def Cbulk():
    Cbulk = bulk('C', crystalstructure='fcc', a=2 * 1.221791471)
    Cbulk = Cbulk.repeat([2, 1, 1])

    siesta = Siesta(
        mesh_cutoff=150 * Ry,
        basis_set='DZP',
        pseudo_qualifier='',
        energy_shift=(10 * 10**-3) * eV,
        fdf_arguments={
            'SCFMustConverge': False,
            'COOP.Write': True,
            'WriteDenchar': True,
            'PAO.BasisType': 'split',
            'DM.Tolerance': 1e-4,
            'DM.MixingWeight': 0.01,
            'MaxSCFIterations': 300,
            'DM.NumberPulay': 4,
            'XML.Write': True,
            'DM.UseSaveDM': True})

    Cbulk.calc = siesta

    return Cbulk

def test_bulk(Cbulk):
    """
    Bulk FCC carbon (for Siesta) self consistency
    """
    delta = 0.02
    
    name = 'bp'
    rm = StaticRamanCalculator(Cbulk, siesta_lrtddft, name=name,
                               delta=delta)
    #, label="siesta", jcutoff=7,
    #                           iter_broadening=0.15, xc_code='LDA,PZ',
    #                           tol_loc=1e-6, tol_biloc=1e-7)
    rm.run()

    pz = PlaczekStatic(Cbulk, name=name)
    e_vib = pz.get_energies()
    i_vib = pz.get_absolute_intensities()
    assert len(e_vib) == 6
    pz.summary()

#    name = 'phbp'
#    rm = StaticRamanPhononsCalculator(Cbulk, BondPolarizability,
#                                      calc=EMT(),
#                                      name=name,
#                                      delta=delta, supercell=(1, 1, 1))
#    rm.run()
#
#    pz = PlaczekStaticPhonons(Cbulk, name=name)
#    e_phonons = pz.get_energies()
#    assert len(e_vib) == len(e_phonons)
#    assert e_vib[3:] == approx(e_phonons[3:], 1)
#    i_phonons = pz.get_absolute_intensities()
#    assert i_vib[3:] == approx(i_phonons[3:], 1)
#    pz.summary()

test_CO2()
