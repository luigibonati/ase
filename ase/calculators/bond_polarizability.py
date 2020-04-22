import re
import numpy as np

from ase.data import covalent_radii
from ase.neighborlist import NeighborList


class LippincottStuttman:
    # atomic polarizability values from:
    #   Lippincott and Stutman J. Phys. Chem. 68 (1964) 2926-2940
    #   DOI: 10.1021/j100792a033
    # see also:
    #   Marinov and Zotov Phys. Rev. B 55 (1997) 2938-2944
    #   DOI: 10.1103/PhysRevB.55.2938
    # unit: Angstrom^3
    atomic_polarizability = {
        'C': 0.978,
        'N': 0.743,
        'O': 0.592,
        'Si': 2.988
    }

    def __call__(self, bond, length):
        el1, el2 = re.findall('[A-Z][^A-Z]*', bond)
        alpha1 = self.atomic_polarizability[el1]
        alpha2 = self.atomic_polarizability[el2]

        return length**4 / (4**4 * alpha1 * alpha2)**(1. / 6), 0


class Linearized():
    def __init__(self):
        self._data = {
            # L. Wirtz, M. Lazzeri, F. Mauri, A. Rubio,
            # Phys. Rev. B 2005, 71, 241402.
            'CC': (1.69, 1.53, 7.43, 0.71, 0.37),
        }

    def __call__(self, bond, length):
        assert bond in self._data
        length0, al, ald, ap, apd = self._data[bond]

        return al + ald * (length - length0), ap + apd * (length - length0)
        

class BondPolarizability:
    def __init__(self, model=LippincottStuttman()):
        self.model = model
    
    def __call__(self, atoms):
        return self.calculate(atoms)

    def calculate(self, atoms):
        """Sum up the bond polarizability from all bonds"""
        radii = np.array([covalent_radii[z]
                          for z in atoms.numbers])
        nl = NeighborList(radii * 1.5, skin=0,
                          self_interaction=False)
        nl.update(atoms)
        pos_ac = atoms.get_positions()

        alpha = 0
        for ia, atom in enumerate(atoms):
            indices, offsets = nl.get_neighbors(ia)
            pos_ac = atoms.get_positions() - atoms.get_positions()[ia]

            for ib, offset in zip(indices, offsets):
                weight = 1
                if offset.any():  # this comes from a periodic image
                    weight = 0.5  # count half the bond only

                dist_c = pos_ac[ib] + np.dot(offset, atoms.get_cell())
                dist = np.linalg.norm(dist_c)
                al, ap = self.model(atom.symbol + atoms[ib].symbol, dist)

                eye3 = np.eye(3) / 3
                alpha += weight * (al + 2 * ap) * eye3
                alpha += weight * (al - ap) * (
                    np.outer(dist_c, dist_c) / dist**2 - eye3)
        return alpha
