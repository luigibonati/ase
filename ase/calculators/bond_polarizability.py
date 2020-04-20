import re
import numpy as np

from ase.data import covalent_radii
from ase.neighborlist import NeighborList

class LippincottStuttman:
    # atomic polarizability values from
    # Lippincott and Stutman J. Phys. Chem. 68 (1964) 2926-2940
    # DOI:10.1021/j100792a033
    # unit: Angstrom^3
    atomic_polarizability = {
        'C': 0.978,
        'N': 0.743,
        'O': 0.592,
        'Si': 2.988
    }

    def __init__(self):
        self._data = {}
    def __getitem__(self, key):
        if key not in self.data:
            el1, el2 = re.findall('[A-Z][^A-Z]*', key)
            self._data[key] = self.calculate(el1, el2)
        return self._data[key]


class LippincottStuttmanPerp(LippincottStuttman):
    def calculate(self, el1, el2):
        return 0


class LippincottStuttmanPara(LippincottStuttman):
    def calculate(self, el1, el2):
        return 0

    

alphaperp = {
    'CC': 0.71
}
alphapara = {
    'CC': 1.69
}

class BondPolarizability:
    """Evaluate polarizability from bonds after
    Marinov and Zotov Phys. Rev. B 55 (1997) 2938-2944
    """

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
                weight = 1
                if offset.any():  # this comes from a periodic image
                    weight = 0.5  # count half the bond only
                r_c = pos_ac[ib] + np.dot(offset, atoms.get_cell())
                r2 = np.dot(r_c, r_c)
                alpha += (weight * r2 /
                          (4**4 * alpha_a[ia] * alpha_a[ib])**(1. / 6) *
                          np.outer(r_c, r_c))
        return alpha
