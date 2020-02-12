import numpy as np

from ase.data import covalent_radii
from ase.neighborlist import NeighborList


class BondPolarizability:
    def __init__(self, atoms):
        self.atoms = atoms
        return self()
    
    def __call__(self):
        radii = np.array([covalent_radii[z]
                          for z in self.atoms.numbers])
        nl = NeighborList(radii * 1.5, skin=0,
                          self_interaction=False)
        nl.update(self.atoms)

        alpha = 0
        for ia, atom in enumerate(self.atoms):
            indices, offsets = nl.get_neighbors(ia)
            for ib in indices:
                r = self.atoms.get_distance(ia, ib)
                alpha += r**4
        return alpha
