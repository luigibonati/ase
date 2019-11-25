from ase.calculators.datadriven import new_emt
from ase.calculators.calculator import get_calculator_class
from ase.build import bulk
from ase.utils import workdir


def skip_if_not_enabled(name):
    cls = get_calculator_class(name)
    cls()  # the test suite monkeypatches these classes so this raises SkipTest


def test_singlepoint(name, atoms):
    skip_if_not_enabled(name)

    with workdir('files-{}'.format(name), mkdir=True):
        e = atoms.get_potential_energy()
        f = atoms.get_forces()

    print(e)
    print(f)
