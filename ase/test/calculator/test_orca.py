import pytest
import re
from ase.atoms import Atoms
from ase.optimize import BFGS

calc = pytest.mark.calculator


@pytest.fixture
def txt1():
    return '               Program Version 4.1.2  - RELEASE  -'


@pytest.fixture
def ref1():
    return '4.1.2'


def test_orca_version_from_string(txt1, ref1):
    from ase.calculators.orca import get_version_from_orca_header

    version = get_version_from_orca_header(txt1)
    assert version == ref1


@calc('orca')
def test_orca_version_from_executable(factory):
    # only check the format to be compatible with future versions
    version_regexp = re.compile(r'\d+.\d+.\d+')

    version = factory.version()
    assert version_regexp.match(version)


@calc('orca')
def test_ohh(factory):
    atoms = Atoms('OHH',
                  positions=[(0, 0, 0), (1, 0, 0), (0, 1, 0)])

    atoms.calc = factory.calc(orcasimpleinput='BLYP def2-SVP')

    with BFGS(atoms) as opt:
        opt.run(fmax=0.05)

    final_energy = atoms.get_potential_energy()
    print(final_energy)

    assert abs(final_energy + 2077.24420) < 1.0
