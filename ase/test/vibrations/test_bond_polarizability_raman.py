from pytest import fixture

from ase import Atoms
from ase.build import bulk
from ase.vibrations.raman import StaticRamanCalculator
from ase.vibrations.raman import StaticRamanPhononCalculator
from ase.vibrations.placzek import PlaczekStatic
from ase.calculators.bond_polarizability import BondPolarizability
from ase.calculators.emt import EMT
from ase.optimize import FIRE


def relaxC4():
    """Code used to relax C4 with EMT"""
    from ase.constraints import FixAtoms, UnitCellFilter
    Cbulk = bulk('C', orthorhombic=True)
    Cbulk.set_constraint(FixAtoms(mask=[True for atom in Cbulk]))

    ucf = UnitCellFilter(Cbulk)
    opt = FIRE(ucf)
    opt.run(fmax=0.001)

    print(Cbulk)
    print(Cbulk.cell)


@fixture(scope='module')
def Cbulk():
    # EMT relaxed, see relaxC4
    Cbulk = bulk('C', crystalstructure='fcc', a=2 * 1.221791471)
    Cbulk = Cbulk.repeat([2, 1, 1])
    Cbulk.calc = EMT()
    return Cbulk


def test_bulk(Cbulk, tmp_path):
    """Bulk FCC carbon (for EMT) self consistency"""
    
    name = str(tmp_path / 'bp')
    rm = StaticRamanCalculator(Cbulk, BondPolarizability, name=name,
                               delta=0.05)
    rm.run()

    pz = PlaczekStatic(Cbulk, name=name)
    print(pz.get_energies(), pz.get_absolute_intensities())
    pz.summary()

    


def test_bulk_phonons(tmp_path):
    """Bulk FCC carbon (for EMT) self consistency for phonons"""
    # EMT relaxed value, see relaxC4
    Cbulk = bulk('C', crystalstructure='fcc', a=2 * 1.221791471)
    Cbulk = Cbulk.repeat([2, 1, 1])
    Cbulk.calc = EMT()

    name = str(tmp_path / 'phbp')
    rm = StaticRamanPhononCalculator(Cbulk, BondPolarizability,
                                     calc=EMT(),
                                     name=name,
                                     delta=0.05, supercell=(2, 1, 1))
    rm.run()

    pz = PlaczekStatic(Cbulk, name=name)
    print(pz.get_energies())
    pz.summary(kpts=(2, 1, 1))


def test_c3():
    """Can we calculate triangular (EMT groundstate) C3?"""
    y, z = 0.30646191, 1.14411339  # emt relaxed
    atoms = Atoms('C3', positions=[[0, 0, 0], [0, y, z], [0, z, y]])
    atoms.calc = EMT()
    
    name = 'bp'
    rm = StaticRamanCalculator(atoms, BondPolarizability,
                               name=name, exname=name, txt='-')
    rm.run()
    pz = PlaczekStatic(atoms, name=name)
    pz.summary()
    

def main():
    test_bulk()


if __name__ == '__main__':
    main()
