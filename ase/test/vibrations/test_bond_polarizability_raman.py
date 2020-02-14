from ase import Atoms
from ase.vibrations.raman import RamanStaticCalculator
from ase.vibrations.placzek import PlaczekStatic
from ase.calculators.bond_polarizability import BondPolarizability
from ase.calculators.emt import EMT


def test_calculation():
    atoms = Atoms('Si2', positions=[[0, 0, 0], [0, 0, 2.5]])
    atoms.set_calculator(EMT())
    name = 'bp'
    rm = RamanStaticCalculator(atoms, BondPolarizability,
                               gsname=name, exname=name, txt='-')
    rm.run()
    pz = PlaczekStatic(atoms, gsname=name)
    pz.summary()
    

def main():
    test_calculation()


if __name__ == '__main__':
    main()
