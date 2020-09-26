import numpy as np
import pytest

from ase.cluster.icosahedron import Icosahedron
from ase.cluster.octahedron import Octahedron
from ase.neighborlist import neighbor_list

sym = 'Au'
a0 = 4.05
ico_cubocta_sizes = [0, 1, 13, 55, 147, 309, 561, 923, 1415]
ico_corner_coordination = 6
ico_corners = 12
fcc_maxcoordination = 12


def coordination_numbers(atoms):
    return np.bincount(neighbor_list('i', atoms, 0.80 * a0))


@pytest.mark.parametrize('shells', range(1, 7))
def test_icosa(shells):
    atoms = Icosahedron(sym, shells)
    assert len(atoms) == ico_cubocta_sizes[shells]

    coordination = coordination_numbers(atoms)
    if shells == 1:
        return

    assert min(coordination) == ico_corner_coordination
    ncorners = sum(coordination == ico_corner_coordination)
    assert ncorners == ico_corners



octa_sizes = [0, 1, 6, 19, 44, 85, 146, 231, 344]
@pytest.mark.parametrize('shells', range(1, 8))
def test_regular_octahedron(shells):
    octa = Octahedron(sym, length=shells, cutoff=0)
    coordination = coordination_numbers(octa)
    assert len(octa) == octa_sizes[shells]
    if shells == 1:
        return

    assert min(coordination) == 4  # corner atoms
    assert sum(coordination == 4) == 6  # number of corners

    # All internal atoms must have coordination as if in bulk crystal:
    expected_internal_atoms = octa_sizes[shells - 2]
    assert sum(coordination == fcc_maxcoordination) == expected_internal_atoms


@pytest.mark.parametrize('shells', range(1, 7))
def test_cuboctahedron(shells):
    cutoff = shells - 1
    length = 2 * cutoff + 1
    cubocta = Octahedron(sym, length=length, cutoff=cutoff)
    print(cubocta)
    assert len(cubocta) == ico_cubocta_sizes[shells]

    coordination = coordination_numbers(cubocta)
    expected_internal_atoms = ico_cubocta_sizes[shells - 1]
    assert sum(coordination == fcc_maxcoordination) == expected_internal_atoms
