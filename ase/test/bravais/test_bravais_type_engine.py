import pytest
import numpy as np
from ase.geometry.bravais_type_engine import (
    generate_niggli_op_table, niggli_op_table)


# We disable the three lattices that have infinite reductions.
# Maybe we can test those, but not today.
#n3d_lattices = 14 - 3
#n2d_lattices = 5
#n1d_lattices = 1
#assert len(ref_info) == n3d_lattices + n2d_lattices + n1d_lattices

ignore_lattices = {'MCL', 'MCLC', 'TRI'}


def ref_info_iter():
    for latname, ops in niggli_op_table.items():
        if latname in ignore_lattices:
            continue
        yield latname, ops


@pytest.mark.parametrize('lattice_name,ref_ops', ref_info_iter())
def test_generate_niggli_table(lattice_name, ref_ops):
    length_grid = np.logspace(-1, 1, 41)
    angle_grid = np.linspace(30, 170, 51)
    table = generate_niggli_op_table(lattices=[lattice_name],
                                     angle_grid=angle_grid,
                                     length_grid=length_grid)
    for key in table:
        print('{}: {}'.format(key, len(table[key])))

    mappings = table[lattice_name]
    assert set(mappings) == set(ref_ops)
