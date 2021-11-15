"""Helper utilities for creating defect structures."""
import numpy as np
from ase.visualize import view
from ase.spacegroup.wyckoff import Wyckoff
from ase.geometry.geometry import get_distances
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


def get_top_bottom(atoms, dim=2):
    positions = atoms.get_scaled_positions()
    assert dim == 2, "Can only be used for 2D structures."

    zs = positions[:, 2]
    for z in zs:
        if z == max(zs):
            top = z
        if z == min(zs):
            bottom = z

    return top, bottom


class DefectBuilder():
    """
    Builder for setting up defect structures.
    """
    from .defects import get_middle_point

    def __init__(self, atoms, min_dist=0.5):
        self.dim = np.sum(atoms.get_pbc())
        self.primitive = atoms
        self.atoms = self._set_construction_cell(atoms)
        self.min_dist = min_dist


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


    def get_voronoi_positions(self, kind='all', atoms=None):
        if atoms is None:
            atoms = self.get_input_structure()
        else:
            atoms = self._set_construction_cell(atoms)
        dim = self.get_dimension()
        vor = self.get_voronoi_object(atoms)
        v1 = self.get_voronoi_points(vor, atoms)
        v2 = self.get_voronoi_lines(vor, v1)
        v3 = self.get_voronoi_faces(vor, v1)
        v4 = self.get_voronoi_ridges(vor, atoms)
        if kind == 'all':
            positions = np.concatenate([v1, v2, v3, v4], axis=0)
        elif kind == 'points':
            positions = np.concatenate([v4, v1], axis=0)
        elif kind == 'lines':
            positions = v2
        elif kind == 'faces':
            positions = v3

        scaled_pos = self.get_interstitial_mock(positions, atoms)

        return scaled_pos


    def sort_array(self, a):
        """First, x, then y, then z."""
        a = a.astype('f8')
        return a[np.lexsort((a[:,2], a[:,1], a[:,0]))]


    def get_interstitial_mock(self, positions, atoms=None):
        if atoms is None:
            atoms = self.get_input_structure()
        cell = atoms.get_cell()
        dim = self.get_dimension()
        positions = self.sort_array(positions)
        interstitial = atoms.copy()
        abs_positions = interstitial.get_positions()
        symbols = interstitial.get_chemical_symbols()
        for i, pos in enumerate(positions):
            distances = get_distances(pos, abs_positions, cell=cell, pbc=True)
            if min(distances[1][0]) > self.min_dist:
                abs_positions = np.append(abs_positions, [pos], axis=0)
                symbols.append('X')
        interstitial = Atoms(symbols,
                             abs_positions,
                             cell=cell)
        interstitial = self.cut_positions(interstitial)

        return interstitial.get_scaled_positions()


    def get_spg_cell(self, atoms):
        return (atoms.get_cell(),
                atoms.get_scaled_positions(),
                atoms.get_atomic_numbers())


    def get_host_symmetry(self):
        atoms = self.get_primitive_structure()
        spg_cell = self.get_spg_cell(atoms)
        dataset = spg.get_symmetry_dataset(spg_cell, symprec=1e-2)

        return dataset


    def get_wyckoff_data(self, number):
        wyckoff = Wyckoff(number).wyckoff
        coordinates = {}
        for element in wyckoff['letters']:
            coordinates[element] = wyckoff[element]['coordinates']

        return coordinates


    def is_mapped(self, scaled_position, coordinate):
        import numexpr
        import math
        prim = self.get_primitive_structure()
        cell = prim.get_cell()

        x = scaled_position[0]
        y = scaled_position[1]
        z = scaled_position[2]

        fit = True
        for i in range(3):
            string = coordinate.split(',')[i]
            try:
                val = numexpr.evaluate(string)
            except SyntaxError:
                string = self.reconstruct_string(string)
            val = numexpr.evaluate(string)
            if math.isclose(val, scaled_position[i], abs_tol=1e-10):
                continue
            else:
                fit = False

        return fit


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


    def create_copies(self, scaled_position, coordinates, tmp_struc):
        import numexpr
        import math
        new_struc = tmp_struc.copy()
        x = scaled_position[0]
        y = scaled_position[1]
        z = scaled_position[2]
        for coordinate in coordinates:
            copy = True
            value = np.zeros(3)
            for i in range(3):
                string = coordinate.split(',')[i]
                try:
                    value[i] = numexpr.evaluate(string)
                except SyntaxError:
                    string = self.reconstruct_string(string)
                value[i] = numexpr.evaluate(string)
                if value[i] < 0:
                    value[i] = value[i] + 1
                if value[i] > 1:
                    value[i] = value[i] - 1
            if self.is_planar(new_struc) and value[2] != z:
                copy = False
            if self.get_dimension() == 2 and not self.is_planar(new_struc):
                top, bottom = get_top_bottom(self.get_primitive_structure())
                if value[2] <= bottom or value[2] >= top:
                    copy = False
                # print(top, bottom, value[2], copy)
            if (value[0] <= 1 and value[0] >= 0
               and value[1] <= 1 and value[1] >= 0
               and value[2] <= 1 and value[2] >= 0 and copy):
                dist, new_struc = self.check_distances(tmp_struc, value)
                if dist:
                    tmp_struc = new_struc

        return new_struc


    def get_top_layer(self):
        atoms = self.get_primitive_structure()
        zs = []
        for i in range(len(atoms)):
            zs.append(atoms.get_positions()[i][2])
        tops = []
        for i in range(len(zs)):
            if zs[i] == max(zs):
                tops.append(i)
        positions = np.empty((len(tops), 3))
        symbols = []
        for j, i in enumerate(tops):
            symbol = atoms.get_chemical_symbols()[i]
            positions[j] = atoms.get_positions()[i]
            symbols.append(symbol)
        cell = atoms.get_cell()

        return Atoms(symbols=symbols,
                     cell=cell,
                     positions=positions,
                     pbc=atoms.get_pbc())


    def create_interstitials(self, atoms=None, adsorption=False):
        if atoms is None:
            atoms = self.get_primitive_structure()
        sym = self.get_host_symmetry()
        wyck = self.get_wyckoff_data(sym['number'])
        if self.get_dimension() == 2 and adsorption:
            un, struc = self.map_positions(wyck,
                                           structure=atoms,
                                           adsorption=adsorption)
        else:
            un, struc = self.map_positions(wyck,
                                           structure=atoms)

        return un, struc


    def create_adsorption_sites(self):
        assert self.get_dimension() == 2, "Adsorption site creation only for 2D materials"

        atoms = self.get_top_layer()
        un, struc = self.create_interstitials(atoms)
        prim = self.get_primitive_structure()
        positions = prim.get_positions()
        symbols = prim.get_chemical_symbols()
        occ_sites = []
        for i, pos in enumerate(atoms.get_positions()):
            kind = atoms.get_chemical_symbols()[i]
            if not kind in occ_sites:
                positions = np.append(positions, [pos + [0, 0, 2]], axis=0)
                symbols.append('X')
                occ_sites.append(kind)
        cell = prim.get_cell()
        distances = []
        for i, atom in enumerate(un):
            if un.get_chemical_symbols()[i] == 'X':
                pos = un.get_positions()[i]
                distance = min(get_distances(pos,
                                             prim.get_positions(),
                                             cell=cell,
                                             pbc=True)[1][0])
                distances.append(distance)
        for i, distance in enumerate(distances):
            if distance == max(distances):
                index = i
        j = 0
        for i, atom in enumerate(un):
            if un.get_chemical_symbols()[i] == 'X':
                pos = un.get_positions()[i]
                if j == index:
                    positions = np.append(positions, [pos + [0, 0, 2]], axis=0)
                    symbols.append('X')
                j += 1

        return Atoms(symbols=symbols,
                     positions=positions,
                     cell=cell,
                     pbc=prim.get_pbc())


    def get_intrinsic_types(self):
        atoms = self.get_primitive_structure()
        symbols = []
        for i in range(len(atoms)):
            symbol = atoms.get_chemical_symbols()[i]
            if not symbol in symbols:
                symbols.append(symbol)

        return symbols


    def get_adsorbate_structures(self, atoms=None, kindlist=None,
                                 sc=3, mechanism='chemisorption'):
        if atoms is None:
            atoms = self.create_adsorption_sites()
        if kindlist is None:
            kindlist = self.get_intrinsic_types()

        structures = []
        primitive = self.get_primitive_structure()
        for i in range(len(atoms)):
            if atoms.get_chemical_symbols()[i] == 'X':
                for kind in kindlist:
                    structure = primitive.repeat((sc, sc, 1))
                    positions = structure.get_positions()
                    symbols = structure.get_chemical_symbols()
                    cell = structure.get_cell()
                    for z in np.arange(-2, 6, 0.1):
                        pos = atoms.get_positions()[i] + [0, 0, z]
                        if self.check_distance(primitive, pos, kind, mechanism):
                            positions = np.append(positions, [pos], axis=0)
                            symbols.append(kind)
                            structures.append(Atoms(symbols=symbols,
                                                    positions=positions,
                                                    cell=cell,
                                                    pbc=structure.get_pbc()))
                            break

        return structures


    def check_distance(self, atoms, position, symbol, mechanism):
        for i in range(len(atoms)):
            host_position = atoms.get_positions()[i]
            host_symbol = atoms.get_chemical_symbols()[i]
            distance = get_distances(position,
                                     host_position,
                                     atoms.get_cell(),
                                     pbc=True)[1][0][0]
            threshold = self.get_minimum_distance(host_symbol,
                                                  symbol,
                                                  mechanism)
            if distance < threshold:
                return False

        return True


    def get_minimum_distance(self, el1, el2, mechanism):
        from ase.data import (vdw_radii,
                              covalent_radii,
                              atomic_numbers)
        N1 = atomic_numbers[el1]
        N2 = atomic_numbers[el2]
        if mechanism == 'physisorption':
            R1 = vdw_radii[N1]
            R2 = vdw_radii[N2]
            if str(R1) == 'nan':
                R1 = 2
            if str(R2) == 'nan':
                R2 = 2
        elif mechanism == 'chemisorption':
            R1 = covalent_radii[N1]
            R2 = covalent_radii[N2]
            if str(R1) == 'nan':
                R1 = 1.9
            if str(R2) == 'nan':
                R2 = 1.9
        else:
            raise ValueError(f'No adsorption mechanism "{mechanism}" known! '
                             'Choose "physisorption" or "chemisorption".')

        return R1 + R2


    def map_positions(self, coordinates, structure=None, adsorption=False):
        if structure is None:
            structure = self.get_primitive_structure()

        equivalent = structure.copy()
        unique = structure.copy()

        kinds = ['points', 'lines', 'faces']
        for kind in kinds:
            scaled_positions = self.get_voronoi_positions(kind=kind,
                                                          atoms=structure)
            for pos in scaled_positions:
                mapped = False
                for element in coordinates:
                    for wyck in coordinates[element]:
                        true_int = False
                        if self.is_mapped(pos, wyck):
                            tmp_eq = equivalent.copy()
                            tmp_un = unique.copy()
                            dist, tmp_eq = self.check_distances(tmp_eq, pos)
                            true_int = self.is_true_interstitial(pos, adsorption)
                            if dist and true_int:
                                unique = self.create_unique(pos, tmp_un)
                                equivalent = self.create_copies(pos, coordinates[element], tmp_eq)
                                mapped = True
                                break
                    if mapped:
                        break

        return unique, equivalent


    def is_true_interstitial(self, pos, adsorption):
        dim = self.get_dimension()
        atoms = self.get_primitive_structure()
        if dim == 3:
            return True
        elif dim == 2 and not adsorption:
            return True
        elif dim == 2 and adsorption:
            top, bottom = get_top_bottom(atoms)
            if pos[2] <= top and pos[2] >= bottom:
                return True
            else:
                return False
        else:
            return True


    def create_unique(self, pos, unique):
        positions = unique.get_scaled_positions()
        cell = unique.get_cell()
        symbols = unique.get_chemical_symbols()
        symbols.append('X')
        positions = np.append(positions, [pos], axis=0)
        unique = Atoms(symbols=symbols,
                       scaled_positions=positions,
                       cell=cell)

        return unique


    def check_distances(self, structure, pos):
        symbols = structure.get_chemical_symbols()
        symbols.append('X')
        tmp_pos = structure.get_scaled_positions()
        tmp_pos = np.append(tmp_pos, [pos], axis=0)
        tmp_struc = Atoms(symbols=symbols,
                          scaled_positions=tmp_pos,
                          cell=structure.get_cell())
        distances = get_distances(tmp_struc.get_positions()[-1],
                                  tmp_struc.get_positions()[:-1],
                                  cell=structure.get_cell(),
                                  pbc=True)
        min_dist = min(distances[1][0])
        if min_dist > self.min_dist:
            return True, tmp_struc
        else:
            return False, structure


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


    def get_voronoi_points(self, voronoi, atoms):
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


    def get_voronoi_object(self, atoms=None):
        if atoms is None:
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


    def get_voronoi_ridges(self, voronoi, atoms=None):
        if atoms is None:
            atoms = self.get_input_structure()
        ridge_points = voronoi.ridge_points
        points = atoms.get_positions()
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
