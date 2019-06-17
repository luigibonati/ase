import numpy as np
from collections import namedtuple
from scipy.spatial.distance import cdist

#from ase import Atoms
#from ase.geometry.rmsd.alignment import get_shift_vectors
#from ase.geometry.rmsd.alignment import get_neighboring_cells
#from ase.geometry.rmsd.lattice_subgroups import get_basis
#from ase.geometry.rmsd.lattice_subgroups import get_group_elements
#from ase.geometry.rmsd.lattice_reducer import LatticeReducer
#from ase.geometry.rmsd.cell_projection import minkowski_reduce
#from ase.geometry.dimensionality.disjoint_set import DisjointSet


from ase.geometry.rmsd.cell_projection import intermediate_representation
from ase.geometry.rmsd.standard_form import standardize_atoms
from ase.geometry.rmsd.alignment import LatticeComparator


def find_line_groups(atoms, keep_all=False):

    Reduced = namedtuple('LineGroup', 'rmsd factor atoms')  #TODO: add line group

    n = len(atoms)
    a = atoms.copy()
    b = atoms.copy()
    res = standardize_atoms(a, b, False)
    atomic_perms, axis_perm = res
    res = intermediate_representation(a, b, 'central', False)
    pa, pb, _, _, _, _ = res

    lc = LatticeComparator(pa, pb)

    k = 2
    dist = np.zeros((k, n))

    for i in range(n):
        for j in range(k):
            print(i, j)
            res = lc.cherry_pick_line_group((j, k, i))
            rmsd, perm = res
            dist[j, i] = rmsd

    import matplotlib.pyplot as plt
    plt.imshow(dist, interpolation='nearest')
    plt.colorbar()
    plt.show()
