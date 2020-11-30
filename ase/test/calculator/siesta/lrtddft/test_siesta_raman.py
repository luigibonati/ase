from ase.units import Ry, eV
from ase.calculators.siesta import Siesta
from ase.calculators.siesta.siesta_lrtddft import siestaRaman
from ase.vibrations.raman import StaticRamanCalculator
from ase.vibrations.placzek import PlaczekStatic
from ase import Atoms

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
    CO2.calc = Siesta(
        mesh_cutoff=150 * Ry,
        basis_set='DZP',
        energy_shift=(10 * 10**-3) * eV,
        fdf_arguments={
            'COOP.Write': True,
            'WriteDenchar': True,
            'PAO.BasisType': 'split',
            'DM.Tolerance': 1e-4,
            'XML.Write': True})

    name = 'co2'
    rm = StaticRamanCalculator(CO2, siestaRaman, name=name, delta=0.011,
                               exkwargs=dict(label="siesta",
                                   jcutoff=7, iter_broadening=0.15,
                                   xc_code='LDA,PZ', tol_loc=1e-6, tol_biloc=1e-7)
                               )
    rm.run()

    pz = PlaczekStatic(CO2, name=name)
    e_vib = pz.get_energies()
    assert len(e_vib) == 9
    pz.summary()

test_CO2()
