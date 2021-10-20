"""Helper utilities for creating defect structures."""
import numpy as np
from ase.visualize import view
from ase.spacegroup.wyckoff import Wyckoff
from scipy.spatial import Voronoi
from ase import Atoms, Atom
import spglib as spg


def split(word):
    return [char for char in word]


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
        self.dim = np.sum(atoms.get_pbc())
        self.primitive = atoms
        self.atoms = self._set_construction_cell(atoms)


    def _set_construction_cell(self, atoms):
        dim = self.get_dimension()
        if dim == 3:
            return atoms.repeat((3, 3, 3))
        elif dim == 2:
            return atoms.repeat((3, 3, 1))


    def get_primitive_structure(self):
        return self.primitive


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


    def get_wyckoff_object(self, sg):
        return Wyckoff(sg)


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


    def draw_voronoi(self, voronoi, pos):
        from scipy.spatial import voronoi_plot_2d
        import matplotlib.pyplot as plt
        assert self.is_planar(), 'Can only be plotted for planar 2D structures!'
        fig = voronoi_plot_2d(voronoi)
        plt.show()


    def get_voronoi_positions(self):
        atoms = self.get_input_structure()
        dim = self.get_dimension()
        vor = self.get_voronoi_object()
        v1 = self.get_voronoi_points(vor)
        v2 = self.get_voronoi_lines(vor, v1)
        v3 = self.get_voronoi_faces(vor, v1)
        v4 = self.get_voronoi_ridges(vor)
        positions = np.concatenate([v1, v2, v3, v4], axis=0)
        # positions = self.remove_duplicates(positions)

        return positions


    def get_interstitial_mock(self, view=False):
        atoms = self.get_input_structure()
        dim = self.get_dimension()
        voronoi_positions = self.get_voronoi_positions()
        voronoi_positions = voronoi_positions[voronoi_positions[:,2].argsort()]
        # print(voronoi_positions)
        cell = atoms.get_cell()
        interstitial = atoms.copy()
        for i, position in enumerate(voronoi_positions):
            positions = interstitial.get_positions()
            symbols = interstitial.get_chemical_symbols()
            flag = True
            # for element in positions:
            #     if abs(np.sum(position - element, axis=0)) < 0.01:
            #         flag = False
            if flag:
                positions = np.append(positions, [position], axis=0)
                symbols.append('X')
                interstitial = Atoms(symbols,
                                     positions,
                                     cell=cell)
        interstitial = self.cut_positions(interstitial)
        if view:
            view(interstitial)

        return interstitial


    def get_spg_cell(self, atoms):
        return (atoms.get_cell(),
                atoms.get_scaled_positions(),
                atoms.get_atomic_numbers())


    def get_host_symmetry(self):
        atoms = self.get_primitive_structure()
        spg_cell = self.get_spg_cell(atoms)
        dataset = spg.get_symmetry_dataset(spg_cell)

        return dataset


    def get_wyckoff_data(self, number):
        wyckoff = Wyckoff(number).wyckoff
        coordinates = {}
        # for element in wyckoff['letters']:
        for element in ['a', 'b']:
            coordinates[element] = wyckoff[element]['coordinates']

        return coordinates


    def allowed_position(self, scaled_position, coordinates, letter='a'):
        import numexpr
        import math

        x = scaled_position[0]
        y = scaled_position[1]
        z = scaled_position[2]

        fit = True
        for coordinate in coordinates[letter]:
            for i in range(3):
                string = coordinate.split(',')[i]
                try:
                    val = numexpr.evaluate(string)
                except SyntaxError:
                    N = len(string)
                    for j in range(N):
                        if string[j] == '-':
                            tmpstr = ''.join(string[:j + 2])
                            insert = j + 2
                        else:
                            tmpstr = ''.join(string[:j + 1])
                            insert = j + 1
                        try:
                            val = numexpr.evaluate(coordinate.split(',')[i])
                        except SyntaxError:
                            string = ''.join(string[:insert]) + '*' + ''.join(string[insert:])
                            break
                val = numexpr.evaluate(string)
                if math.isclose(val, scaled_position[i], abs_tol=1e-4):
                    continue
                else:
                    fit = False
            if fit:
                print(f'INFO: {scaled_position} matched {coordinate}!')
                break

        return fit


    def allowed_position_tmp(self, scaled_position, coordinate):
        import numexpr
        import math

        x = scaled_position[0]
        y = scaled_position[1]
        z = scaled_position[2]

        fit = True
        for i in range(3):
            string = coordinate.split(',')[i]
            try:
                val = numexpr.evaluate(string)
            except SyntaxError:
                N = len(string)
                for j in range(N):
                    if string[j] == '-':
                        tmpstr = ''.join(string[:j + 2])
                        insert = j + 2
                    else:
                        tmpstr = ''.join(string[:j + 1])
                        insert = j + 1
                    try:
                        val = numexpr.evaluate(coordinate.split(',')[i])
                    except SyntaxError:
                        string = ''.join(string[:insert]) + '*' + ''.join(string[insert:])
                        break
            val = numexpr.evaluate(string)
            if math.isclose(val, scaled_position[i], abs_tol=1e-6):
                continue
            else:
                fit = False

        return fit


    def map_positions(self, coordinates):
        from ase.geometry.geometry import get_distances
        interstitial = self.get_interstitial_mock()
        scaled_positions = interstitial.get_scaled_positions()
        abs_positions = interstitial.get_positions()

        map_dict = {}
        uni_dict = {}
        for element in coordinates:
            map_dict[f'{element}'] = []
            uni_dict[f'{element}'] = []
        for pos in scaled_positions:
            for element in coordinates:
                for wyck in coordinates[element]:
                    if self.allowed_position_tmp(pos, wyck):
                        previous = map_dict[element]
                        uni = uni_dict[element]
                        new_uni = self.get_unique(pos, previous, uni)
                        uni_dict[f'{element}'] = new_uni
                        all_list = self.get_all_pos(pos, coordinates[element])
                        all_list = self.return_new_values(all_list, previous)
                        map_dict[f'{element}'] = all_list
                        break
                # break


        # for pos in scaled_positions:
        #     for element in coordinates:
        #         for wyck in coordinates[element]:
        #             if self.allowed_position_tmp(pos, wyck):
        #                 all_list = map_dict[element]
        #                 uni = uni_dict[element]
        #                 new_uni = self.get_unique(pos, all_list, uni)
        #                 uni_dict[f'{element}'] = new_uni
        #                 break
        #         break

        for element in uni_dict:
            print(f'Unique elements: {element}')
            for coord in uni_dict[element]:
                print(coord)
            print('===========================================')
        for element in map_dict:
            print(f'All elements: {element}')
            for coord in map_dict[element]:
                print(coord)
            print('===========================================')


    def return_new_values(self, list1, list2):
        import math
        newlist = list2
        for i, el1 in enumerate(list1):
            if len(list2) == 0:
                newlist.append(el1)
            else:
                cond = np.any([math.isclose(el1[0], list2[j][0], abs_tol=1e-15) and
                               math.isclose(el1[1], list2[j][1], abs_tol=1e-15) and
                               math.isclose(el1[2], list2[j][2], abs_tol=1e-15) for j in range(len(list2))])
                if cond:
                    continue
                else:
                    newlist.append(el1)
        return newlist


    def get_unique(self, new, list1, list2):
        import math
        if len(list2) == 0:
            list2.append(new)
        else:
            cond = np.any([math.isclose(new[0], list1[j][0], abs_tol=1e-15) and
                           math.isclose(new[1], list1[j][1], abs_tol=1e-15) and
                           math.isclose(new[2], list1[j][2], abs_tol=1e-15) for j in range(len(list1))])
            if not cond:
                list2.append(new)
        return list2


    def get_all_pos(self, pos, coordinates):
        import numexpr
        import math
        x = pos[0]
        y = pos[1]
        z = pos[2]
        positions = []
        for coordinate in coordinates:
            value = np.zeros(3)
            for i in range(3):
                string = coordinate.split(',')[i]
                try:
                    value[i] = numexpr.evaluate(string)
                except SyntaxError:
                    string = self.reconstruct_string(string)
                value[i] = numexpr.evaluate(string)
            if value[0] < 1 and value[1] < 1 and value[2] < 1:
                positions.append(value)

        return positions


    def reconstruct_string(self, string):
        import numexpr
        N = len(string)
        for j in range(N):
            if string[j] == '-':
                tmpstr = ''.join(string[:j + 2])
                insert = j + 2
            else:
                tmpstr = ''.join(string[:j + 1])
                insert = j + 1
            try:
                val = numexpr.evaluate(string)
            except SyntaxError:
                string = ''.join(string[:insert]) + '*' + ''.join(string[insert:])
                break
        return string


        # tmp_atoms = self.get_primitive_structure()
        # mapped = np.zeros((len(scaled_positions), 4))
        # for element in coordinates:
        #     # print(f'Wyckoff position: {element}')
        #     for i, pos in enumerate(scaled_positions):
        #         if interstitial.get_chemical_symbols()[i] == 'X':
        #             if self.allowed_position(pos, coordinates, element):
        #                 oldpos = tmp_atoms.get_positions()
        #                 cell = tmp_atoms.get_cell()
        #                 symbols = tmp_atoms.get_chemical_symbols()
        #                 distances = get_distances(oldpos, abs_positions[i],
        #                                           cell=cell, pbc=True)
        #                 # print(min(distances[1]))
        #                 if min(distances[1]) > 0.5:
        #    #                  print(f'Position {pos} matched {element}!')
        #                     mapped[i][0] = pos[0]
        #                     mapped[i][1] = pos[1]
        #                     mapped[i][2] = pos[2]
        #                     mapped[i][3] = 0
        #                     positions = np.append(tmp_atoms.get_scaled_positions(),
        #                                           [pos], axis=0)
        #                     symbols.append('X')
        #                     tmp_atoms = Atoms(symbols,
        #                                       positions,
        #                                       cell=cell)
        #                     tmp_atoms.set_scaled_positions(positions)
        # # print(tmp_atoms)
        # # view(tmp_atoms)


    def cut_positions(self, interstitial):
        cell = interstitial.get_cell()
        newcell = self.get_newcell(cell)
        newpos = self.shift_positions(interstitial.get_positions(),
                                      newcell)
        interstitial.set_cell(newcell)
        interstitial.set_positions(newpos, newcell)

        positions = interstitial.get_scaled_positions()
        symbols = interstitial.get_chemical_symbols()
        indexlist = []
        for i, symbol in enumerate(symbols):
            if symbol == 'X':
                th = 0.01
            else:
                th = 0
            pos = positions[i]
            if (pos[0] >= (1 + th) or pos[0] < (0 - th) or
               pos[1] >= (1 + th) or pos[1] < (0 - th) or
               pos[2] >= (1 + th) or pos[2] < (0 - th)):
                indexlist.append(i)
        interstitial = self.remove_atoms(interstitial, indexlist)

        return interstitial


    def get_newcell(self, cell, N=3):
        a = np.empty((3, 3))
        dim = self.get_dimension()
        if dim == 3:
            for i in range(3):
                a[i] = 1. / N * np.array([cell[i][0],
                                          cell[i][1],
                                          cell[i][2]])
        elif dim == 2:
            for i in range(2):
                a[i] = 1. / N * np.array([cell[i][0],
                                          cell[i][1],
                                          cell[i][2]])
            a[2] = np.array([cell[2][0],
                             cell[2][1],
                             cell[2][2]])

        return a


    def shift_positions(self, positions, cell):
        a = positions.copy()
        dim = self.get_dimension()
        if dim == 3:
            for i, pos in enumerate(positions):
                for j in range(3):
                    a[i][j] = pos[j] - (cell[0][j] + cell[1][j] + cell[2][j])
        elif dim == 2:
            for i, pos in enumerate(positions):
                for j in range(2):
                    a[i][j] = pos[j] - (cell[0][j] + cell[1][j])

        return a


    def remove_atoms(self, atoms, indexlist):
        indices = np.array(indexlist)
        indices = np.sort(indices)[::-1]
        for element in indices:
            atoms.pop(element)

        return atoms


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


    def get_voronoi_object(self):
        atoms = self.get_input_structure()
        dist = 3
        points = atoms.get_positions()
        if self.dim == 2 and self.is_planar(atoms):
            points = points[:, :2]
        elif self.dim == 1:
            raise NotImplementedError("Not implemented for 1D structures.")

        return Voronoi(points)


    def get_voronoi_lines(self, voronoi, points):
        ridges = voronoi.ridge_vertices
        lines = np.zeros((len(ridges), 3))
        remove = []
        for i, ridge in enumerate(ridges):
            if not ridge[0] == -1:
                p1 = [points[ridge[0]][0],
                      points[ridge[0]][1],
                      points[ridge[0]][2]]
                p2 = [points[ridge[1]][0],
                      points[ridge[1]][1],
                      points[ridge[1]][2]]
                lines[i] = get_middle_point(p1, p2)
            else:
                remove.append(i)
        lines = np.delete(lines, remove, axis=0)

        return lines


    def get_voronoi_faces(self, voronoi, points):
        regions = voronoi.regions
        centers = np.zeros((len(regions), 3))
        remove = []
        for i, region in enumerate(regions):
            if -1 not in region and len(region) > 1:
                array = np.zeros((len(region), 3))
                for j, element in enumerate(region):
                    array[j][0] = points[element][0]
                    array[j][1] = points[element][1]
                    array[j][2] = points[element][2]
                centers[i] = centeroidnp(array)
            else:
                remove.append(i)
        centers = np.delete(centers, remove, axis=0)

        return centers


    def get_voronoi_ridges(self, voronoi):
        ridge_points = voronoi.ridge_points
        points = self.get_input_structure().get_positions()
        array = np.zeros((len(ridge_points), 3))
        for i, ridge in enumerate(ridge_points):
            p1 = [points[ridge[0]][0],
                  points[ridge[0]][1],
                  points[ridge[0]][2]]
            p2 = [points[ridge[1]][0],
                  points[ridge[1]][1],
                  points[ridge[1]][2]]
            array[i] = get_middle_point(p1, p2)

        return array
