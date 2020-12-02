import pytest
from ase.units import Ry, eV
from ase.calculators.siesta import Siesta
from ase.calculators.siesta.siesta_lrtddft import siestaRaman
from ase.vibrations.raman import StaticRamanCalculator
from ase.vibrations.placzek import PlaczekStatic
from ase.build import molecule

@pytest.mark.calculator('siesta')
def test_N2():

    n2 = molecule('N2')

    # enter siesta input
    n2.calc = Siesta(
        basis_set='DZP',
        fdf_arguments={
            'COOP.Write': True,
            'WriteDenchar': True,
            'XML.Write': True})

    name = 'n2'
    pynao_args = dict(label="siesta", jcutoff=7, iter_broadening=0.15,
                      xc_code='LDA,PZ', tol_loc=1e-6, tol_biloc=1e-7)
    Rm = StaticRamanCalculator(n2, siestaRaman, name=name, delta=0.011,
                               exkwargs=pynao_args)
    Rm.run()

    Pz = PlaczekStatic(n2, name=name)
    e_vib = Pz.get_energies()
    assert len(e_vib) == 6
    Pz.summary()
