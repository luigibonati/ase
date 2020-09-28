from pytest import approx, fixture

from ase import Atoms
from ase.build import bulk
from ase.vibrations.raman import StaticRamanCalculator
from ase.vibrations.raman import StaticRamanPhononsCalculator
from ase.vibrations.placzek import PlaczekStatic
from ase.vibrations.placzek import PlaczekStaticPhonons
from ase.calculators.bond_polarizability import BondPolarizability
from ase.calculators.emt import EMT
from ase.optimize import FIRE


def relaxC():
    """Code used to relax C with EMT"""
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
    # EMT relaxed, see relaxC
    Cbulk = bulk('C', crystalstructure='fcc', a=2 * 1.221791471)
    Cbulk = Cbulk.repeat([2, 1, 1])
    Cbulk.calc = EMT()
    return Cbulk


def test_bulk(Cbulk, tmp_path):
    """Bulk FCC carbon (for EMT) self consistency"""
    delta = 0.02
    
    name = str(tmp_path / 'bp')
    rm = StaticRamanCalculator(Cbulk, BondPolarizability, name=name,
                               delta=delta)
    rm.run()

    pz = PlaczekStatic(Cbulk, name=name)
    e_vib = pz.get_energies()
    print('#vib# modes=', pz.modes_Qq)
    i_vib = pz.get_absolute_intensities()
    assert len(e_vib) == 6
    pz.summary()

    name = str(tmp_path / 'phbp')
    rm = StaticRamanPhononsCalculator(Cbulk, BondPolarizability,
                                      calc=EMT(),
                                      name=name,
                                      delta=delta, supercell=(1, 1, 1))
    rm.run()

    pz = PlaczekStaticPhonons(Cbulk, name=name)
    e_phonons = pz.get_energies()
    assert len(e_vib) == len(e_phonons)
    # pz.read(symmetrize=0)
    # e_ph = pz.get_energies()
    m = pz.modes_Qq.real
    # print('#ph# modes=', np.dot(m, m.T))
    print('#ph# modes=', m)
    i_ph = pz.get_absolute_intensities()
    pz.summary()
    print(i_vib / i_ph)


def test_bulk_kpts(Cbulk, tmp_path):
    """Bulk FCC carbon (for EMT) for phonons"""

    name = str(tmp_path / 'phbp')
    rm = StaticRamanPhononsCalculator(Cbulk, BondPolarizability,
                                      calc=EMT(), name=name,
                                      delta=0.05, supercell=(2, 1, 1))
    rm.run()

    pz = PlaczekStaticPhonons(Cbulk, name=name, supercell=(2, 1, 1))
    energies_1kpt = pz.get_energies()
    pz.kpts = (2, 1, 1)
    energies_2kpts = pz.get_energies()
    assert len(energies_2kpts) == 2 * len(energies_1kpt)

    pz.kpts = (2, 1, 2)
    pz.summary()


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
    i_vib = pz.get_absolute_intensities()
    assert i_vib[-3:] == approx([5.36301901, 5.36680555, 35.7323934], 1e-6)
    pz.summary()
