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


    def setup_spg_cell(self,
                       atoms=None,
                       positions=None,
                       numbers=None):
        if atoms is None:
                atoms = self.get_input_structure()
        if positions is None:
            positions = atoms.get_scaled_positions()
        if numbers is None:
            numbers = atoms.numbers
        cell = atoms.cell.array

        return (cell, positions, numbers)


    def get_wyckoff_symbols(self, spg_cell):
        dataset = spg.get_symmetry_dataset(spg_cell)

        return dataset['wyckoffs']


    def get_equivalent_atoms(self, spg_cell):
        dataset = spg.get_symmetry_dataset(spg_cell)

        return dataset['equivalent_atoms']


    def get_input_structure(self):
        return self.atoms


    def get_dimension(self):
        return self.dim


    def is_planar(self, atoms):
        z = atoms.get_positions()[:, 2]

        return np.all(z==z[0])


    def create_vacancies(self):
        atoms = self.get_input_structure()
        spg_host = self.setup_spg_cell()
        eq_pos = self.get_equivalent_atoms(spg_host)
        finished_list = []
        vacancies = []
        for i in range(len(atoms)):
            if not eq_pos[i] in finished_list:
                vac = self.get_input_structure().copy()
                sitename = vac.get_chemical_symbols()[i]
                vac.pop(i)
                finished_list.append(eq_pos[i])
                vacancies.append(vac)

        return vacancies


    def get_kindlist(self, intrinsic=True, extrinsic=None):
        atoms = self.get_input_structure().copy
        defect_list = []
        if intrinsic:
            for i in range(len(atoms)):
                symbol = atoms[i].symbol
                if symbol not in defect_list:
                    defect_list.append(symbol)
        if extrinsic is not None:
            for i, element in enumerate(extrinsic):
                if element not in defect_list:
                    defect_list.append(extrinsic[i])

        return defect_list


    def create_antisites(self, intrinsic=True, extrinsic=None):
        atoms = self.get_input_structure().copy()
        spg_host = self.setup_spg_cell()
        eq_pos = self.get_equivalent_atoms(spg_host)
        defect_list = self.get_kindlist(intrinsic=intrinsic,
                                        extrinsic=extrinsic)
        antisites = []
        finished_list = []
        for i in range(len(atoms)):
            if not eq_pos[i] in finished_list:
                for element in defect_list:
                    if not atoms[i].symbol == element:
                        antisite = atoms.copy()
                        sitename = antisite.get_chemical_symbols()[i]
                        antisite[i].symbol = element
                        antisites.append(antisite)
                finished_list.append(eq_pos[i])

        return antisites


    def create_interstitials(self, intrinsic=True, extrinsic=None):
        # add elemental dependency
        atoms = self.get_input_structure()
        dim = self.get_dimension()
        vor = self.get_voronoi_object()
        vertices = self.get_voronoi_points(vor)
        ridges = self.get_voronoi_ridges(vor)
        regions = self.get_voronoi_regions(vor)
        middle = self.get_voronoi_middle(vertices, ridges)
        center = self.get_voronoi_center(vertices, regions)
        voronoi_positions = self.get_voronoi_all(vertices, middle, center)
        # voronoi_positions = vertices
        spg_host = self.setup_spg_cell()
        dataset = spg.get_symmetry_dataset(spg_host)
        # print(dataset['number'])
        host_wyckoffs = self.get_wyckoff_symbols(spg_host)
        wyckoffs = []
        interstitials = []
        defect_list = self.get_kindlist(intrinsic=intrinsic,
                                        extrinsic=extrinsic)
        for i, position in enumerate(voronoi_positions):
            cell = atoms.get_cell()
            # position = [0.25 * cell[0][0], 0.5 * (cell[1][0] + cell[1][1]), 0]
            # position = [1, 2, 3]
            for kind in defect_list:
                interstitial = atoms.copy()
                positions = interstitial.get_positions()
                positions = np.append(positions, [position], axis=0)
                symbols = interstitial.get_chemical_symbols()
                symbols.append(kind)
                interstitial = Atoms(symbols,
                                     positions,
                                     cell=cell)
                spg_temp = self.setup_spg_cell(interstitial)
                wyckoff = self.get_wyckoff_symbols(spg_temp)
                dataset = spg.get_symmetry_dataset(spg_temp)
                pointgroup = dataset['number']
                # view(interstitial)
                print(wyckoff, pointgroup)
                # if wyckoff[-1] not in wyckoffs:
                overlap = False
                for element in atoms.get_positions():
                    if np.sum(abs(element - position)) < 0.1:
                        overlap = True
                if not overlap:
                    interstitials.append(interstitial)
            wyckoffs.append(wyckoff[-1])

        return interstitials


    def get_voronoi_object(self):
        # TODO: choose reasonable out of plane distance
        atoms = self.get_input_structure()
        dist = 3
        points = atoms.get_positions()
        if self.dim == 2 and self.is_planar(atoms):
            print('INFO: planar 2D structure.')
            # i_max = len(points)
            # for i, point in enumerate(points):
            #     if i > i_max:
            #         break
            #     elif i <= i_max:
            #         points = np.append(points,
            #                            [point + [0, 0, dist],
            #                             point - [0, 0, dist]],
            #                            axis=0)
            points = points[:, :2]
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


    def get_z_position(self, atoms):
        assert self.is_planar(atoms), 'No planar structure.'

        return atoms.get_positions()[0, 2]


    def get_voronoi_points(self, voronoi):
        atoms = self.get_input_structure()
        dim = self.get_dimension()
        vertices = voronoi.vertices
        if dim == 2 and self.is_planar(atoms):
            z = self.get_z_position(atoms)
            a = np.empty((len(vertices), 3))
            for i, element in enumerate(vertices):
                a[i][0] = element[0]
                a[i][1] = element[1]
                a[i][2] = z
            vertices = a

        return vertices


    def get_voronoi_regions(self, voronoi):
        return voronoi.regions


    def get_voronoi_ridges(self, voronoi):
        return voronoi.ridge_vertices


    def get_voronoi_center(self, vertices, regions, dim=3):
        if True:
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
        # elif dim == 2:
        #     centers = np.zeros((len(regions), 2))
        #     remove = []
        #     for i, region in enumerate(regions):
        #         if -1 not in region and len(region) > 1:
        #             array = np.zeros((len(region), 2))
        #             for j, element in enumerate(region):
        #                 array[j][0] = vertices[element][0]
        #                 array[j][1] = vertices[element][1]
        #             centers[i] = centeroidnp(array)
        #         else:
        #             remove.append(i)
        #     centers = np.delete(centers, remove, axis=0)

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
