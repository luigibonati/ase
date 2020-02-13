import numpy as np

from ase.data import covalent_radii
from ase.neighborlist import NeighborList


class BondPolarizability:
    """Evaluate polarizability from bonds after
    Marinov and Zotov Phys. Rev. B 55 (1997) 2938-2944
    """
    # atomic polarizability values from
    # Lippincott and Stutman J. Phys. Chem. 68 (1964) 2926-2940
    # DOI:10.1021/j100792a033
    # unit: Angstrom^3
    atomic_polarizability = {
        'O': 0.592,
        'Si': 2.988
    }
     
    def __call__(self, atoms):
        return self.calculate(atoms)

    def calculate(self, atoms):
        radii = np.array([covalent_radii[z]
                          for z in atoms.numbers])
        nl = NeighborList(radii * 1.5, skin=0,
                          self_interaction=False)
        nl.update(atoms)
        alpha_a = [self.atomic_polarizability[atom.symbol]
                   for atom in atoms]
        pos_ac = atoms.get_positions()

        alpha = 0
        for ia, atom in enumerate(atoms):
            indices, offsets = nl.get_neighbors(ia)
            pos_ac = atoms.get_positions() - atoms.get_positions()[ia]
            for ib, offset in zip(indices, offsets):
                r_c = pos_ac[ib] + np.dot(offset, atoms.get_cell())
                r2 = np.dot(r_c, r_c)
                alpha += np.outer(r_c, r_c) * (r2 /
                    (4**4 * alpha_a[ia] * alpha_a[ib])**(1. / 6))
        return alpha
