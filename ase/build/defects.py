"""Helper utilities for creating defect structures."""

import numpy as np
import scipy
from ase import Atoms


class DefectBuilder():
    """
    Builder for setting up defect structures.
    """
    def __init__(self, atoms):
        self.atoms = atoms


    def return_atoms(self):
        return self.atoms


    def get_voronoi_points(self):
        from scipy.spatial import Voronoi
        points = self.atoms.get_positions()
        vor = Voronoi(points)

        return vor.vertices
