"""Helper utilities for creating defect structures."""
import numpy as np
from scipy.spatial import Voronoi
from ase import Atoms

def get_middle_point(p1, p2):
    coords = np.zeros(3)
    for i in range(2):
        coords[i] = (p1[i] + p2[i]) / 2.

    return coords


class DefectBuilder():
    """
    Builder for setting up defect structures.
    """
    from .defects import get_middle_point

    def __init__(self, atoms):
        self.atoms = atoms
        self.dim = np.sum(atoms.get_pbc())


    def return_atoms(self):
        return self.atoms


    def get_voronoi_object(self):
        # TODO: choose reasonable out of plane distance
        dist = 3
        if self.dim == 3:
            points = self.atoms.get_positions()
        elif self.dim == 2:
            points = self.atoms.get_positions()
            i_max = len(points)
            for i, point in enumerate(points):
                if i > i_max:
                    break
                elif i <= i_max:
                    points = np.append(points,
                                       [point + [0, 0, dist],
                                        point + [0, 0, -dist]],
                                       axis=0)
        elif self.dim == 1:
            raise NotImplementedError("Not implemented for 1D structures.")

        return Voronoi(points)


    def get_voronoi_points(self, voronoi):
        return vor.vortices


    def get_voronoi_middle(self, vertices, ridge_vertices):
        middle = np.zeros((len(ridge_vertices), 3))
        remove = []
        for i, ridge in enumerate(ridge_vertices):
            if not ridge[0] == -1:
                p1 = [vertices[ridge[0]][0],
                      vertices[ridge[0]][1],
                      vertices[ridge[0]][2]]
                p2 = [vertices[ridge[1]][0],
                      vertices[ridge[1]][1],
                      vertices[ridge[1]][2]]
                middle[i] = get_middle_point(p1, p2)
            else:
                remove.append(i)
        middle = np.delete(middle, remove, axis=0)

        return middle





