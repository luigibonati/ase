from ase.calculators.datadriven import new_emt
from ase.build import bulk
from ase.utils import workdir

def test_singlepoint(name, calc):
    atoms = bulk('Au') * (2, 2, 2)
    atoms.rattle(stdev=0.2)

    atoms.calc = calc
    with workdir('files-{}'.format(name), mkdir=True):
        e = atoms.get_potential_energy()
        f = atoms.get_forces()

    print(e)
    print(f)
