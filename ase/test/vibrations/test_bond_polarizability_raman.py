from ase import Atoms
from ase.build import bulk
from ase.vibrations.raman import RamanStaticCalculator
from ase.vibrations.placzek import PlaczekStatic
from ase.calculators.bond_polarizability import BondPolarizability
from ase.calculators.emt import EMT
from ase.optimize import FIRE


def test_bulk():
    """Bulk FCC carbon (for EMT) self consistency"""
    if 0:
        si4 = bulk('C', orthorhombic=True)
    else:
        # EMT relaxed value
        Cbulk = bulk('C', crystalstructure='fcc', a=2*1.221791471)
    #help(bulk)
    Cbulk = Cbulk.repeat([2, 1, 1])
    Cbulk.set_calculator(EMT())
    if 0:
        from ase.constraints import FixAtoms, UnitCellFilter
        
        Cbulk.set_constraint(FixAtoms(mask=[True for atom in Cbulk]))
        ucf = UnitCellFilter(Cbulk)
        opt = FIRE(ucf)
        opt.run(fmax=0.001)
        print(Cbulk)
        print(Cbulk.cell)
        
    
    name = 'bp'
    rm = RamanStaticCalculator(Cbulk, BondPolarizability, gsname=name,
                               delta=0.05)
    rm.run()
    pz = PlaczekStatic(Cbulk, gsname=name)
    print(pz.get_energies())
    pz.summary()        

    if 0:
        si32 = si4.repeat([2, 2, 2])
        si32.set_calculator(EMT())
        name = 'si32'
        rm = RamanStaticCalculator(si32, BondPolarizability, gsname=name)
        rm.run()
        pz = PlaczekStatic(si32, gsname=name)
        pz.summary()        

def test_c3():
    """Can we calculate triangular (EMT groundstate) C3?"""
    y, z = 0.30646191, 1.14411339  # emt relaxed
    atoms = Atoms('C3', positions=[[0, 0, 0], [0, y, z], [0, z, y]])
    atoms.set_calculator(EMT())
    
    name = 'bp'
    rm = RamanStaticCalculator(atoms, BondPolarizability,
                               gsname=name, exname=name, txt='-')
    rm.run()
    pz = PlaczekStatic(atoms, gsname=name)
    pz.summary()
    

def main():
    test_bulk()


if __name__ == '__main__':
    main()
