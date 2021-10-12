"""Helper utilities for creating defect structures."""
import numpy as np
from ase.visualize import view
from scipy.spatial import Voronoi
from ase import Atoms, Atom
import spglib as spg

def get_middle_point(p1, p2):
    coords = np.zeros(3)
    for i in range(3):
        coords[i] = (p1[i] + p2[i]) / 2.

    return coords

def centeroidnp(arr):
    length = arr.shape[0]
    sum_x = np.sum(arr[:, 0])
    sum_y = np.sum(arr[:, 1])
    sum_z = np.sum(arr[:, 2])

    return np.array((sum_x/length, sum_y/length, sum_z/length))


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


    def setup_spg_cell(self,
                       positions=None,
                       numbers=None):
        if positions is None:
            positions = self.atoms.get_scaled_positions()
        if numbers is None:
            numbers = self.atoms.numbers
        cell = self.atoms.cell.array

        return (cell, positions, numbers)


    def get_wyckoff_symbols(self, spg_cell):
        dataset = spg.get_symmetry_dataset(spg_cell)
        wyckoffs = dataset['wyckoffs']

        return wyckoffs


    def create_interstitials(self):
        # add elemental dependency
        vor = self.get_voronoi_object()
        vertices = self.get_voronoi_points(vor)
        ridges = self.get_voronoi_ridges(vor)
        regions = self.get_voronoi_regions(vor)
        middle = self.get_voronoi_middle(vertices, ridges)
        center = self.get_voronoi_center(vertices, regions)
        voronoi_positions = self.get_voronoi_all(vertices, middle, center)
        spg_host = self.setup_spg_cell()
        host_wyckoffs = self.get_wyckoff_symbols(spg_host)
        wyckoffs = []
        for position in voronoi_positions:
            spg_temp = self.setup_spg_cell([position], [10])
            wyckoff = self.get_wyckoff_symbols(spg_temp)
            if wyckoff[0] not in wyckoffs:
                print(position, wyckoff)
                def_atoms = self.atoms.copy()
                newatom = Atom(10, position)
                def_atoms.append(newatom)
                wyckoffs.append(wyckoff[0])
                view(def_atoms)


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
                    continue
                    # points = np.append(points,
                    #                    [point + [0, 0, dist],
                    #                     point - [0, 0, dist]],
                    #                    axis=0)
        elif self.dim == 1:
            raise NotImplementedError("Not implemented for 1D structures.")
        return Voronoi(points)


    def get_voronoi_all(self, vertices, middle, center):
        voronoi_positions = np.empty((len(vertices) + len(middle) + len(center), 3))
        for i, element in enumerate(vertices):
            voronoi_positions[i] = np.array((element[0],
                                             element[1],
                                             element[2]))
        for i, element in enumerate(middle):
            voronoi_positions[i + len(vertices)] = np.array((element[0],
                                                             element[1],
                                                             element[2]))
        for i, element in enumerate(center):
            voronoi_positions[i + len(vertices) + len(middle)] = np.array((element[0],
                                                                           element[1],
                                                                           element[2]))

        return voronoi_positions


    def get_voronoi_points(self, voronoi):
        return voronoi.vertices


    def get_voronoi_regions(self, voronoi):
        return voronoi.regions


    def get_voronoi_ridges(self, voronoi):
        return voronoi.ridge_vertices


    def get_voronoi_center(self, vertices, regions):
        centers = np.zeros((len(regions), 3))
        remove = []
        for i, region in enumerate(regions):
            if -1 not in region and len(region) > 1:
                array = np.zeros((len(region), 3))
                for j, element in enumerate(region):
                    array[j][0] = vertices[element][0]
                    array[j][1] = vertices[element][1]
                    array[j][2] = vertices[element][2]
                centers[i] = centeroidnp(array)
            else:
                remove.append(i)
        centers = np.delete(centers, remove, axis=0)

        return centers


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
                # print(middle[i])
            else:
                remove.append(i)
        # print(middle)
        middle = np.delete(middle, remove, axis=0)

        return middle
