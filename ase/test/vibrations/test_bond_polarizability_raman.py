from ase import Atoms
from ase.build import bulk
from ase.vibrations.raman import RamanStaticCalculator
from ase.vibrations.placzek import PlaczekStatic
from ase.calculators.bond_polarizability import BondPolarizability
from ase.calculators.emt import EMT
from ase.optimize import FIRE


def test_silicon():
    """Bulk silicon has a single Raman peak"""
    if 0:
        si4 = bulk('Si', orthorhombic=True)
    else:
        si4 = bulk('Si')
    si4.set_calculator(EMT())
    
    name = 'si4'
    rm = RamanStaticCalculator(si4, BondPolarizability, gsname=name,
                               delta=0.05)
    rm.run()
    pz = PlaczekStatic(si4, gsname=name)
    pz.summary()        

    if 0:
        si32 = si4.repeat([2, 2, 2])
        si32.set_calculator(EMT())
        name = 'si32'
        rm = RamanStaticCalculator(si32, BondPolarizability, gsname=name)
        rm.run()
        pz = PlaczekStatic(si32, gsname=name)
        pz.summary()        

def test_si3():
    y, z = 0.30646191, 1.14411339  # emt relaxed
    # XXX change this to C
    atoms = Atoms('Si3', positions=[[0, 0, 0], [0, y, z], [0, z, y]])
    atoms.set_calculator(EMT())
    if 0:
        import numpy as np
        opt = FIRE(atoms)
        opt.run(fmax=0.001)
        print(atoms.positions - atoms[0].position)
        print(np.sqrt(((atoms.positions - atoms[0].position)**2).sum(axis=1)))
    
    name = 'bp'
    rm = RamanStaticCalculator(atoms, BondPolarizability,
                               gsname=name, exname=name, txt='-')
    rm.run()
    pz = PlaczekStatic(atoms, gsname=name)
    pz.summary()
    

def main():
    test_si3()


if __name__ == '__main__':
    main()
