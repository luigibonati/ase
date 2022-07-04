"""Helper utilities for creating defect structures."""
import numpy as np
from ase.spacegroup.wyckoff import Wyckoff
from ase.geometry.geometry import get_distances
from scipy.spatial import Voronoi
from ase import Atoms
import spglib as spg


def setup_supercell(atoms, sc=(3, 3, 3), size=None):
    """Set up and return supercell with integer repititions.

    Parameters
    ----------
    atoms : ASE Atoms object
        input atomic structure
    sc : int
        integer value to repeat input structure (NxNxN for 3D,
        NxNx1 for 2D).
    size: float
        physical minimum size of the supercell in Angstrom.
    """
    dim = sum(atoms.pbc)
    if size is not None:
        assert size > 0, 'Choose size larger than zero!'
        sc = get_supercell_repetition(atoms, size, dim=dim)

    if dim == 2:
        assert sc[2] == 1, ('2D supercell should only have one '
                            'repitition out of plane!')
        supercell = atoms.repeat(sc)
    elif dim == 3:
        supercell = atoms.repeat(sc)
    elif dim == 1:
        raise ValueError('1D not implemented!')

    return supercell


def get_supercell_repetition(atoms, size, dim):
    """Return supercell repitition based on physical size criterion."""
    cell = atoms.get_cell()
    dim = sum(atoms.pbc)
    # if dim == 3:
    #     min_length = min(cell.lengths())
    # elif dim == 2:
    #     min_length = min(cell.lengths()[:2])
    # loop over integer repititions until minimum size criterion fulfilled
    sc_list = [1, 1, 1]
    for i in range(dim):
        for N in range(1, 50, 1):
            length = cell.lengths()[i]
            tmp = N * length
            if tmp > size:
                sc_list[i] = N
                break

    sc_tuple = tuple(sc_list)
    return sc_tuple

    raise ValueError('Only works for repetitions smaller '
                     'than 50! Input smaller physical '
                     'supercell size.')


def get_kindlist(atoms, intrinsic=True, extrinsic=None):
    """Return list of chemical symbols for doping of an atomic structure.

    Parameters
    ----------
    atoms : ASE Atoms object
        input atomic structure
    intrinsic : bool
        choose whether intrinsic elements should be included
    extrinsic : list
        give a list of chemical symbols
    """
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


def calculate_interstitial_distances(atoms_int, atoms_prim):
    cell = atoms_prim.get_cell()
    distances = []
    for i, atom in enumerate(atoms_int):
        if atoms_int.get_chemical_symbols()[i] == 'X':
            pos = atoms_int.get_positions()[i]
            distance = min(get_distances(pos,
                                         atoms_prim.get_positions(),
                                         cell=cell,
                                         pbc=True)[1][0])
            distances.append(distance)

    return distances


def get_wyckoff_data(number):
    wyckoff = Wyckoff(number).wyckoff
    coordinates = {}
    for element in wyckoff['letters']:
        coordinates[element] = wyckoff[element]['coordinates']

    return coordinates


def get_spg_cell(atoms):
    return (atoms.get_cell(),
            atoms.get_scaled_positions(),
            atoms.get_atomic_numbers())


def has_same_kind(struc_1, struc_2):
    element_list_1 = struc_1.get_chemical_symbols()
    element_list_1.sort()
    element_list_2 = struc_2.get_chemical_symbols()
    element_list_2.sort()
    if len(element_list_1) != len(element_list_2):
        return False
    for i, element in enumerate(element_list_1):
        if element_list_1[i] != element_list_2[i]:
            return False

    return True


def get_equivalent_atoms(spg_cell):
    dataset = spg.get_symmetry_dataset(spg_cell)

    return dataset['equivalent_atoms']


def sort_array(a):
    """First, x, then y, then z."""
    a = a.astype('f8')
    return a[np.lexsort((a[:, 2], a[:, 1], a[:, 0]))]


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


def get_top_bottom(atoms, interc=False, dim=2):
    positions = atoms.get_scaled_positions()
    assert dim == 2, "Can only be used for 2D structures."

    zs = positions[:, 2]
    if interc:
        top = zs[atoms.get_tags() == 1].min()
        bottom = zs[atoms.get_tags() == 0].max()
    else:
        for z in zs:
            if z == max(zs):
                top = z
            if z == min(zs):
                bottom = z

    return top, bottom


def get_layer_tags(atoms):
    """Automatically assign tag 1 to top layer and tag 0 to bottom layer"""
    atoms_tmp = atoms.copy()
    atoms_tmp.center()
    tags = [1 if atom.scaled_position[2] > 0.5 else 0 for atom in atoms_tmp]
    return tags


class DefectBuilder():
    """
    Builder for setting up defect structures.

    The most relevant methods are:
        - 'get_vacancy_structures' (return list of vacancy structures)
        - 'get_substitution_structures' (return list of substitutional structures)
        - 'get_interstitial_structures' (return list of interstitial structures')
        - 'get_adsorbate_structures' (return list of adsorbate structures)
        - 'create_interstitials' (return overview structures of interstitial sites
           i.e. a tuple of unique interstitials and equivalent interstitials)
        - 'create_adsorption_sites' (return overview structure of all adsorption sites)
        - 'get_voronoi_positions' (get positions resulting from a Voronoi tessalation,
          has modes 'all', 'points', 'lines', 'faces')
        - 'map_positions' to use Voronoi positions and map them onto Wyckoff positions
          of the host crystal

    Attributes
    ----------
    atoms : ASE Atoms object
        input atomic structure (will be changed to a construction cell
        with integer reptitions for the Voronoi construction, see
        '_set_construction_cell()' method).
    min_dist : float
        minimum distance between newly generated defect positions (
        for interstitial, adsorption site creation).
    dim : int
        dimension of the input structure
    primitive : ASE Atoms object
        primitive atomic structure = input atomic structure.
    """
    from .defects import get_middle_point

    def __init__(self, atoms, min_dist=1):
        """
        Parameters
        ----------
        atoms : ASE Atoms object
            input atomic structure (will be changed to a construction cell
            with integer reptitions for the Voronoi construction, see
            '_set_construction_cell()' method).
        min_dist : float
            minimum distance between newly generated defect positions (
            for interstitial, adsorption site creation). The smaller you
            chosse this parameter the more interstitial sites will be created.
        dim : int
            dimension of the input structure.
        primitive : ASE Atoms object
            primitive atomic structure = input atomic structure.
        """
        self.dim = sum(atoms.pbc)
        self.primitive = atoms
        self.atoms = self._set_construction_cell(atoms)
        self.min_dist = min_dist

    # small helper methods - start
    def _set_construction_cell(self, atoms):
        """Repeat atomic structure to mimic pbc for Voronoi construction."""
        repetitions = np.ones(3, dtype='int')
        assert (all(atoms.pbc == [True, True, False])
                or all(atoms.pbc == [True, True, True])), (
                'input atoms should either be periodic in all three directions '
                'or the non periodic axis should be the third one!')
        repetitions[atoms.pbc] = 3

        return atoms.repeat(repetitions)

        if self.dim == 3:
            return atoms.repeat((3, 3, 3))
        elif self.dim == 2:
            return atoms.repeat((3, 3, 1))

    def get_primitive_structure(self):
        """Return a copy of DefectBuilder's primitive structure attribute."""
        return self.primitive.copy()

    def get_host_symmetry(self):
        atoms = self.get_primitive_structure()
        spg_cell = get_spg_cell(atoms)
        dataset = spg.get_symmetry_dataset(spg_cell, symprec=1e-3)

        return dataset

    def get_input_structure(self):
        """Return a copy of DefectBuilder's input structure."""
        return self.atoms.copy()

    def setup_spg_cell(self, atoms=None, positions=None, numbers=None):
        """Return tuple of atomic cell, positions, and numbers in spglib format."""
        if atoms is None:
            atoms = self.get_input_structure()
        if positions is None:
            positions = atoms.get_scaled_positions()
        if numbers is None:
            numbers = atoms.numbers
        cell = atoms.cell.array

        return (cell, positions, numbers)

    def get_wyckoff_symbols(self, spg_cell):
        """Return spglib's symmetry dataset from a spglib cell input."""
        dataset = spg.get_symmetry_dataset(spg_cell)

        return dataset['wyckoffs']

    def get_wyckoff_object(self, sg):
        """Retrun Wyckoff object for a given spacegroup (ase.spacegroups.wyckoff)."""
        return Wyckoff(sg)

    def is_planar(self, atoms):
        """Check whether a given input structure is fully planar."""
        z = atoms.get_positions()[:, 2]

        return np.all(z == z[0])
    # small helper methods - end

    # defect creation methods (vacancies, subst., interstitial, adsorption) - start
    def get_vacancy_structures(self, sc=(3, 3, 3), size=None):
        """Create vacancy structures and set up desired supercell."""
        atoms = self.get_primitive_structure()
        spg_host = self.setup_spg_cell()
        eq_pos = get_equivalent_atoms(spg_host)
        finished_list = []
        vacancies = []
        structure = setup_supercell(atoms, sc, size)

        # create a vacancy structure inside supercell for each non-eq. position
        for i in range(len(atoms)):
            if not eq_pos[i] in finished_list:
                vac = structure.copy()
                vac.pop(i)
                finished_list.append(eq_pos[i])
                vacancies.append(vac)

        return vacancies

    def get_substitution_structures(self, intrinsic=True, extrinsic=None,
                                    sc=(3, 3, 3), size=None):
        """Create substutitional defect structures with desired supercell."""
        atoms = self.get_primitive_structure().copy()
        spg_host = self.setup_spg_cell()
        eq_pos = get_equivalent_atoms(spg_host)
        defect_list = get_kindlist(atoms, intrinsic=intrinsic, extrinsic=extrinsic)
        structure = setup_supercell(atoms, sc, size)

        substitutions = []
        finished_list = []
        for i in range(len(atoms)):
            if not eq_pos[i] in finished_list:
                for element in defect_list:
                    if not atoms[i].symbol == element:
                        substitution = structure.copy()
                        substitution[i].symbol = element
                        substitutions.append(substitution)
                finished_list.append(eq_pos[i])

        return substitutions

    def get_interstitial_structures(self, kindlist=None, sc=(3, 3, 3), size=None, Nsites=None):
        """Create interstitial structures and return in desired supercell.

        Interstitial structures are created with the following steps:
            1. find all interstitial positions (details in 'self.create_interstitials()')
            2. set up supercell
            3. loop over all interstitial positions and dope with kindlist elements
            4. return list of doped interstitial structures

        Parameters
        ----------
        kindlist : list(str)
            list of chemical elements to dope the interstitial structures with
            (if 'None', only intrisic elements will be considered)
        sc : tuple(int, int, int)
            integer repitition of the primitive supercell to create supercell
        size : float
            physical size criterion for supercell size in Angstrom
        mechanism : str
            adsorption mechanism, choose between 'chemisorption' and 'physisorption'
        Nsites : int
            choose number of adsorption sites to create (if 'Nsites' is None it will
            just use the 'min_dist' criterion in the constructor of the class)
        """
        atoms, _ = self.create_interstitials(Nsites=Nsites)
        if kindlist is None:
            kindlist = self.get_intrinsic_types()

        # set up pristine supercell
        prim = self.get_primitive_structure()
        supercell = setup_supercell(prim, sc, size)

        # loop over all interstitial sites (type 'X') and dope them wth kindlist elements
        structures = []
        for i in range(len(atoms)):
            if atoms.get_chemical_symbols()[i] == 'X':
                for kind in kindlist:
                    structure = supercell.copy()
                    positions = structure.get_positions()
                    symbols = structure.get_chemical_symbols()
                    cell = structure.get_cell()
                    pos = atoms.get_positions()[i]
                    positions = np.append(positions, [pos], axis=0)
                    symbols.append(kind)
                    structures.append(Atoms(symbols=symbols,
                                            positions=positions,
                                            cell=cell,
                                            pbc=structure.get_pbc()))

        return structures

    def get_adsorbate_structures(self, kindlist=None,
                                 sc=(3, 3, 3), mechanism='chemisorption',
                                 size=None, Nsites=None, tag=0):
        """Create adsorbate structures and return in desired supercell.

        Adsorption structures are created with the following steps:
            1. extract top and bottom atomic layers of the monolayer
            2. create adsorption sites for top and bottom layer from
               'self.create_adsorption_sites()' method
            3. check whether top and bottom atoms are the same -> only
               continue with the top layer if they are (equivalent)
            4. initialize z-positions and adjust height based on the sum
               of covalent/van-der-Waals radii of the included atomic species
            5. setup supercell
            6. create all absorbate structures with desired doping elements
            7. return list of adsorbate structures

        Parameters
        ----------
        kindlist : list
            list of chemical elements to dope the adsorbate structures with
            (if 'None', only intrisic elements will be considered)
        sc : int
            integer repitition of the primitive supercell to create supercell
        size : float
            physical size criterion for supercell size in Angstrom
        mechanism : str
            adsorption mechanism, choose between 'chemisorption' and 'physisorption'
        Nsites : int
            choose number of adsorption sites to create (if 'Nsites' is None it will
            just use the 'min_dist' criterion in the constructor of the class)
        tag : int
            tag of the adsorbed atoms
        """
        atoms_top = self.create_adsorption_sites('top', Nsites=Nsites)
        atoms_bottom = self.create_adsorption_sites('bottom', Nsites=Nsites)
        if not has_same_kind(self.get_layer('top'),
                             self.get_layer('bottom')):
            atoms_list = [atoms_top, atoms_bottom]
            z_ranges = [np.arange(-2, 6, 0.1),
                        np.arange(2, -6, -0.1)]
        else:
            atoms_list = [atoms_top]
            z_ranges = [np.arange(-2, 6, 0.1)]
        if kindlist is None:
            kindlist = self.get_intrinsic_types()

        # set up pristine supercell
        prim = self.get_primitive_structure()
        supercell = setup_supercell(prim, sc, size)

        # loop over all adsorption sites (type 'X') and dope them wth kindlist elements
        structures = []
        for j, atoms in enumerate(atoms_list):
            primitive = self.get_primitive_structure()
            for i in range(len(atoms)):
                if atoms.get_chemical_symbols()[i] == 'X':
                    for kind in kindlist:
                        structure = supercell.copy()
                        positions = structure.get_positions()
                        symbols = structure.get_chemical_symbols()
                        tags = list(structure.get_tags())
                        cell = structure.get_cell()
                        for z in z_ranges[j]:
                            pos = atoms.get_positions()[i] + [0, 0, z]
                            if self.check_distance(primitive, pos, kind, mechanism):
                                positions = np.append(positions, [pos], axis=0)
                                symbols.append(kind)
                                tags.append(tag)
                                structures.append(Atoms(symbols=symbols,
                                                        positions=positions,
                                                        cell=cell,
                                                        tags=tags,
                                                        pbc=structure.get_pbc()))
                                break

        return structures

    def get_intercalated_structures(self, kindlist=None, 
                                    Nsites=1, sc=2, 
                                    group_sites=True,
                                    randomize=False,
                                    symprec=1e-2,
                                    depth=0.15,
                                    dtol=2.0,
                                    return_labels=False):
        """Create intercalation structures in the desired supercell.

        Intercalation structures are created with the following steps:
            1. find all the hollow sites in the Van der Waals gap,
               and tag them according to their symmetry
            2. set up supercell and find the hollow site replicas inside it
            3. For each element provided in kindlist, 
               intercalate as many atoms as specified by nsites.
            4. return list of intercalation structures

        Parameters
        ----------
        kindlist : list
            list of chemical elements to dope the adsorbate structures with
            (if 'None', only intrisic elements will be considered)
        sc : int/list/numpy.ndarray
            if integer -> integer repitition of the primitive supercell to create supercell
            if 3x3 list or numpy array -> unit vector transformation matrix
        Nsites : int
            choose number of atoms to intercalate per element.
            If set to 0, it will simply return the supercell.
        randomize : bool
            if True, randomly choose the intercalation sites among the available ones.
            Overrides the default behaviour, i.e. spreading out the
            intercalation sites as much as possible according to their distances.
        group_sites : bool
            if False, don't group the hollow sites by symmetry 
            and fill them only according to distances. 
            This will reduce the total number of structures to one per kindlist element
        depth: float (0.0 < depth < 0.5)
            fraction of the interlayer gap width. Determines how deep in the 
            interlayer gap to start looking for hollow sites.
        dtol: float
            minimu distance between hollow sites and atoms that belong to the
            pristine structure.
        return_labels: bool
            ask to return, for each structure, a symmetry label of the form
            "[hollow_site_point_group]-[coordination_number]"
            The local symmetry analysis is performed on a local cluster 
            formed by the hollow site and its nearest neighbors.
        """
        from ase import Atom
        from ase.build import make_supercell

        tags = get_layer_tags(self.primitive)
        self.primitive.set_tags(tags)

        _, atoms = self.create_interstitials(atoms=self.primitive,
                                             interc=True,
                                             depth=depth)

        if kindlist is None:
            kindlist = self.get_intrinsic_types()

        # set up pristine and mock supercells
        prim = self.get_primitive_structure()
        if type(sc) == tuple:
            supercell_prim = setup_supercell(prim, sc)
            supercell_mock = setup_supercell(atoms, sc)
        elif type(sc) in [list, np.ndarray]:
            supercell_prim = make_supercell(prim, sc)
            supercell_mock = make_supercell(atoms, sc)

        if Nsites == 0:
            return [supercell_prim]

        site_positions = self.find_hollow_positions(supercell_mock,
                                                    dtol,
                                                    group_sites=group_sites)
        structures = []
        labels = []
        for kind in kindlist:
            for site in site_positions:
                order = self.get_filling_order(site_positions[site],
                                               supercell_mock.pbc,
                                               supercell_mock.get_cell(),
                                               randomize=randomize)
                supercell = supercell_prim.copy()
                for index in order[:Nsites]:
                    supercell += Atom(symbol=kind,
                                      position=site_positions[site][index],
                                      tag=2)
                structures.append(supercell)
                labels.append(site)

        if group_sites and return_labels and Nsites > 0:
            return structures, labels
        return structures
    # defect creation methods (vacancies, subst., interstitial, adsorption, intercalation) - end

    # interstitial and adsorbate overview methods - start
    def create_interstitials(self, atoms=None, Nsites=None, ads=False, interc=False, depth=0.1):
        """Create a 'mock' atomic structure to give an overview over all interstitials.

        Setps to create the interstitial position:
            1. analyse symmetry of the host material
            2. map positions (Voronoi tessalation and symmetry mapping,
               see details in 'self.map_positions')
            3. return atoms object with all unique interstitial positions
               and another atoms object with all interstitial positions (including
               equivalent positions based on symmetry)

        Parameters
        ----------
        Nsites : int
            number of desired interstitial positions (ordered by largest distance
            to next neighbor)
        ads : bool
            flag for creating adsorption sites
        interc : bool
            flag for creating intercalation sites
        depth:
            controls how deep in the slab to start searching for interstitials
        """
        if atoms is None:
            atoms = self.get_primitive_structure()
        sym = self.get_host_symmetry()
        wyck = get_wyckoff_data(sym['number'])

        un, struc = self.map_positions(wyck,
                                       structure=atoms,
                                       ads=ads,
                                       interc=interc,
                                       depth=depth)
        if Nsites is None:
            return un, struc

        else:
            # compute distances for all 2D interstitial (adsorption) sites
            prim = self.get_primitive_structure()
            cell = prim.get_cell()
            positions = prim.get_positions()
            symbols = prim.get_chemical_symbols()
            distances = calculate_interstitial_distances(un, prim)
            distances = np.array(distances)
            # get indices of distance array with the largest min. distance
            # (the Nsite largest elements, 1 by default)
            indices = np.argsort(distances)[::-1]
            indices = indices[:Nsites]
            j = 0
            for i, atom in enumerate(un):
                if un.get_chemical_symbols()[i] == 'X':
                    pos = un.get_positions()[i]
                    if j in indices:
                        positions = np.append(positions, [pos], axis=0)
                        symbols.append('X')
                    j += 1
        un = Atoms(symbols=symbols,
                   positions=positions,
                   cell=cell,
                   pbc=prim.get_pbc())

        return un, struc

    def create_adsorption_sites(self, layer='top', Nsites=None):
        """Create a 'mock' atomic structure to give an overview over all adsorption sites.

        Setps to create the adsorption sites:
            1. Extract top or bottom layer of the structure
            2. use interstitial algorithm to obtain absorption sites
            3. translate sites up or down to be above/below the layer
            4. return 'mock' structure with overview of all adsorption sites

        Parameters
        ----------
        Nsites : int
            number of desired interstitial positions (ordered by largest distance
            to next neighbor)
        layer : str - 'top'/'bottom'
            create adsorbates for 'top' or 'bottom' layer
        """
        assert self.dim == 2, "Adsorption site creation only for 2D materials"

        atoms = self.get_layer(layer)
        if layer == 'top':
            z = 2
        elif layer == 'bottom':
            z = -2
        un, struc = self.create_interstitials(atoms, Nsites, ads=True)
        prim = self.get_primitive_structure()
        cell = un.get_cell()
        positions = prim.get_positions()
        symbols = prim.get_chemical_symbols()
        occ_sites = []
        for i, pos in enumerate(atoms.get_positions()):
            kind = atoms.get_chemical_symbols()[i]
            if kind not in occ_sites:
                positions = np.append(positions, [pos + [0, 0, z]], axis=0)
                symbols.append('X')
                occ_sites.append(kind)

        for i, atom in enumerate(un):
            if un.get_chemical_symbols()[i] == 'X':
                pos = un.get_positions()[i]
                positions = np.append(positions, [pos + [0, 0, z]], axis=0)
                symbols.append('X')

        return Atoms(symbols=symbols,
                     positions=positions,
                     cell=cell,
                     pbc=prim.get_pbc())
    # interstitial and adsorbate overview methods - end

    # methods related to voronoi tessalation - start
    def draw_voronoi(self, voronoi, pos):
        from scipy.spatial import voronoi_plot_2d
        import matplotlib.pyplot as plt
        assert self.is_planar(), 'Can only be plotted for planar 2D structures!'
        voronoi_plot_2d(voronoi)
        plt.show()

    def get_voronoi_positions(self, kind='all', atoms=None):
        """Return positions resulting from a Voronoi tessalation with scipy.

        Parameters
        ----------
        kind : str ('all', 'points', 'lines', 'faces')
            choose kind of voronoi positions to be returned: Voronoi points,
            middle of Voronoi lines, centers of Voronoi faces, or all
        atoms : ASE Atoms object
            input atomic structure for Voronoi tessalation, if 'None' simply
            use the input atomic structure for the DefectBuilder
        """
        if atoms is None:
            atoms = self.get_input_structure()
        else:
            atoms = self._set_construction_cell(atoms)
        vor = self.get_voronoi_object(atoms)
        v1 = self.get_voronoi_points(vor, atoms)
        v2 = self.get_voronoi_lines(vor, v1)
        v3 = self.get_voronoi_faces(vor, v1)
        v4 = self.get_voronoi_ridges(vor, atoms)
        if kind == 'all':
            positions = np.concatenate([v1, v2, v3, v4], axis=0)
        elif kind == 'points':
            positions = v4
        elif kind == 'lines':
            positions = v2
        elif kind == 'faces':
            positions = np.concatenate([v3, v1], axis=0)

        scaled_pos = self.get_interstitial_mock(positions, atoms)

        return scaled_pos

    def get_voronoi_points(self, voronoi, atoms):
        dim = self.dim
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
            if ridge[0] != -1:
                p1 = self.get_ridgepoint(points, ridge, 0)
                p2 = self.get_ridgepoint(points, ridge, 1)
                # p1 = [rdgpnt[i] for i, rdgpnt in enumerate(points[ridge[0]])]
                # p2 = [rdgpnt[i] for i, rdgpnt in enumerate(points[ridge[1]])]
                # p1 = [points[ridge[0]][0],
                #       points[ridge[0]][1],
                #       points[ridge[0]][2]]
                # p2 = [points[ridge[1]][0],
                #       points[ridge[1]][1],
                #       points[ridge[1]][2]]
                lines[i] = get_middle_point(p1, p2)
            else:
                remove.append(i)
        lines = np.delete(lines, remove, axis=0)

        return lines

    def get_ridgepoint(self, points, ridge, j):
        return [rdgpnt for rdgpnt in points[ridge[j]]]

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
    # methods related to voronoi tessalation - end

    # mapping functionalities - start
    def map_positions(self, coordinates, structure=None, ads=False, interc=False, depth=0.1):
        """Get Voronoi positions and map them onto Wyckoff positions of the host crystal.

        Parameters
        ----------
        coordinates : dict
            dictionary of wyckoff coordinates coming from ase.spacegroup.wyckoff module
            (see more detals in 'get_wyckoff_data' function)
        structure : ASE Atoms object
            input structure for the mapping; will be set to primitive structure if 'None'
        ads : bool
            True if mapping should be for adsorption site generation, False otherwise
        interc : bool
            True if mapping should be for intercalation site generation, False otherwise
        """
        if structure is None:
            structure = self.get_primitive_structure()

        equivalent = structure.copy()
        unique = structure.copy()

        if interc:
            kinds = ['faces']
        else:
            kinds = ['points', 'lines', 'faces']

        for kind in kinds:
            scaled_positions = self.get_voronoi_positions(kind=kind,
                                                          atoms=structure)
            # mapped = False
            for element in coordinates:
                for wyck in coordinates[element]:
                    true_int = False
                    for indx, pos in enumerate(scaled_positions):
                        if self.is_mapped(pos, wyck):
                            tmp_eq = equivalent.copy()
                            tmp_un = unique.copy()
                            dist, tmp_eq = self.check_distances(tmp_eq, pos, indx)
                            true_int = self.is_true_interstitial(pos, ads, interc, depth)
                            if dist and true_int:
                                unique = self.create_unique(pos, tmp_un)
                                equivalent = self.create_copies(pos, 
                                                                coordinates[element], 
                                                                tmp_eq, 
                                                                indx,
                                                                interc)
                                # mapped = True
                                break

        return unique, equivalent

    def get_interstitial_mock(self, positions, atoms=None):
        if atoms is None:
            atoms = self.get_input_structure()
        cell = atoms.get_cell()
        positions = sort_array(positions)
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

    def is_mapped(self, scaled_position, coordinate):
        import numexpr
        import math

        x, y, z = [pos for pos in scaled_position]
        fit = True
        for i in range(3):
            string = coordinate.split(',')[i]
            try:
                val = numexpr.evaluate(string)
            except SyntaxError:
                string = self.reconstruct_string(string)
            val = numexpr.evaluate(string)
            if math.isclose(val, scaled_position[i], abs_tol=1e-2):
                continue
            else:
                fit = False

        return fit

    def get_all_pos(self, position, coordinates):
        import numexpr

        x, y, z = [pos for pos in position]
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

    def create_copies(self, scaled_position, coordinates, tmp_struc, index, interc=False):
        import numexpr

        new_struc = tmp_struc.copy()
        x, y, z = [pos for pos in scaled_position]
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
            if self.dim == 2 and not self.is_planar(new_struc):
                top, bottom = get_top_bottom(self.get_primitive_structure(), interc=interc)
                if value[2] <= bottom or value[2] >= top:
                    copy = False
            if (value[0] <= 1 and value[0] >= 0
               and value[1] <= 1 and value[1] >= 0
               and value[2] <= 1 and value[2] >= 0 and copy):
                dist, new_struc = self.check_distances(tmp_struc, value, index)
                if dist:
                    tmp_struc = new_struc

        return new_struc

    # mapping functionalities - start
    def get_layer(self, kind='top'):
        atoms = self.get_primitive_structure()
        zs = []
        for i in range(len(atoms)):
            zs.append(atoms.get_positions()[i][2])

        if kind == 'top':
            value = max(zs)
        elif kind == 'bottom':
            value = min(zs)

        layer = []
        for i in range(len(zs)):
            if np.isclose(zs[i], value):
                layer.append(i)
        positions = np.empty((len(layer), 3))
        symbols = []
        for j, i in enumerate(layer):
            symbol = atoms.get_chemical_symbols()[i]
            positions[j] = atoms.get_positions()[i]
            symbols.append(symbol)
        cell = atoms.get_cell()

        return Atoms(symbols=symbols,
                     cell=cell,
                     positions=positions,
                     pbc=atoms.get_pbc())

    def get_intrinsic_types(self):
        atoms = self.get_primitive_structure()
        symbols = []
        for i in range(len(atoms)):
            symbol = atoms.symbols[i]
            if symbol not in symbols:
                symbols.append(symbol)

        return symbols

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

    def is_true_interstitial(self, pos, ads=False, interc=False, depth=0.1):
        dim = self.dim
        atoms = self.get_primitive_structure()
        if dim == 3:
            return True
        elif dim == 2:
            assert 0.0 <= depth <= 0.5, 'Max allowed depth is 0.5!'
            top, bottom = get_top_bottom(atoms, interc=interc)
            delta = abs(top - bottom) * depth
            if ads:
                if pos[2] <= top and pos[2] >= bottom:
                    return True
                else:
                    return False
            else:
                if pos[2] <= top - delta and pos[2] >= bottom + delta:
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
                       cell=cell,
                       pbc=unique.pbc)

        return unique

    def check_distances(self, structure, pos, index):
        symbols = structure.get_chemical_symbols()
        symbols.append('X')
        tmp_pos = structure.get_scaled_positions()
        tmp_pos = np.append(tmp_pos, [pos], axis=0)
        tmp_tags = structure.get_tags()
        tmp_tags = np.append(tmp_tags, index+2)
        tmp_struc = Atoms(symbols=symbols,
                          scaled_positions=tmp_pos,
                          cell=structure.get_cell(),
                          pbc=structure.pbc,
                          tags=tmp_tags)
        distances = get_distances(tmp_struc.get_positions()[-1],
                                  tmp_struc.get_positions()[:-1],
                                  cell=structure.get_cell(),
                                  pbc=True)
        min_dist = min(distances[1][0])
        if min_dist > self.min_dist:
            return True, tmp_struc
        else:
            return False, structure
    # mapping functionalities - end

    # Intercalation-specific methods - start
    def find_hollow_positions(self, atoms, dtol=2.0, group_sites=True):
        """Find the positions and tags of all hollow sites.

        Requires the following tagging scheme:
            1:   atoms in the top layer
            0:   atoms in the bottom layer
            n>1: hollow sites
        i.e. as assigned by default when get_intercalated_structures is called.
        If group_sites is set to False, it will set the tags 
        of all hollow sites to 2.

        'dtol' controls the minimum distance of the hollow sites from lattice atoms.
        """
        if not group_sites:
            for atom in atoms:
                if atom.tag > 1:
                    atom.tag = 2

        uniq = np.unique([tag for tag in atoms.get_tags() if tag > 1])
        clean = self.clean_cell(atoms)
        site_pos_raw = {}
        for i, tag in enumerate(uniq):
            for atom in atoms:
                if atom.tag == tag:
                    position = atom.position
                    if group_sites:
                        sym = self.get_site_symmetry(position.copy(), clean)
                        label = 'h' + str(i) + '-' + sym
                    else:
                        label = 'positions'
                    if label in site_pos_raw.keys():
                        site_pos_raw[label].append(position)
                    else:
                        site_pos_raw[label] = [position]

        site_pos = self.filter_hollow_positions(
                site_pos_raw,
                clean,
                dtol)

        return site_pos

    def filter_hollow_positions(self, pos_dct, prim, dtol=2.0):
        """Filter out hollow site positions if they are within a distance
           dtol from any atom of the pristine structure"""
        positions = {}
        for sites in pos_dct:
            blacklist = []
            for i, pos in enumerate(pos_dct[sites]):
                dists = get_distances(
                        pos,
                        prim.get_positions(),
                        cell=prim.cell,
                        pbc=prim.pbc)
                if dists[1].min() < dtol:
                    blacklist.append(i)
            positions.update({sites: np.delete(pos_dct[sites], blacklist, axis=0)})
        return positions

    def clean_cell(self, atoms):
        """Remove ghost atoms 'X' from an Atoms object"""
        clean = atoms.copy()
        blacklist = []
        for i, atom in enumerate(clean):
            if atom.symbol == 'X':
                blacklist.append(i)
        del clean[blacklist]
        return clean

    def get_site_symmetry(self, pos, atoms, NN=1, symprec=1e-2):
        """Obtain a symmetry label for a specific defect site.

        The label consists in a number identifying the local symmetry around the defect
        (according to the international symmetry tables), followed by its coordination number.

        The hollow site local geometry is obtained by selecting neighbors
        up to the desired order (through the parameter NN). Then the point
        group symmetry of the resulting cluster is obtained.
        """

        import spglib as spg

        new = atoms.repeat((3, 3, 1))
        #center = 0.5 * (new.cell[0] + new.cell[1])
        T = atoms.cell[0] + atoms.cell[1]
        T[2] = 0.0
        pos += T
        new.translate(T)

        D = get_distances(
                pos, 
                new.get_positions(),
                cell=new.cell, 
                pbc=new.pbc
        )
        D_round = D[1].round(1).flatten()
        D_neighbors, counts = np.unique(D_round, return_counts=True)
        coordination = counts[0]

        cluster = np.argwhere(D_round <= D_neighbors[NN - 1]).flatten()
        atoms = new[cluster]
        spg_nr = spg.get_symmetry_dataset(atoms, symprec=symprec)
        spg_str = f"{spg_nr['number']}-{coordination}"
        return spg_str

    def get_filling_order(self, positions, pbc, cell, randomize):
        """Define order in which hollow positions will be filled.

        Criteria: as spread out as possible in terms of distances

        The algorithm is initialized by filling up the hollow position with 
        index 0 first, then the one at the largest distance.
        The following ones are chosen according to two criteria:
            1. maximize the average distance A from the previously filled ones
            2. keep the distances among intercalation sites as uniform as possible. 
               This is given by F = 1 - s, where s is the standard deviation
               of the distances between a given position and all the others.

        Each time a position is filled, all the remaining ones 
        are given a score S = A * F/F_max. The one with the highest S
        is chosen as the next one. All the filled sites are given S = -1

        If randomize is set to True, the method will just 
        return the indexes randomly shuffled.
        """
        if randomize:
            indexes = [i for i in range(len(positions))]
            np.random.shuffle(indexes)
            return indexes

        D = get_distances(positions, positions, 
                          pbc=pbc, 
                          cell=cell)[1]
        order = [0, np.argmax(D[0])]
        for i in range(2, len(D)):
            dists = np.asarray([D[indx] for indx in order])
            avgs = []
            stdevs = []
            for j in range(len(D)):
                if j in order:
                    avgs.append(-1)
                    stdevs.append(0)
                else:
                    avgs.append(np.average(dists[:, j]))
                    stdevs.append(np.std(dists[:, j]))
            if np.allclose(stdevs, 0.0):
                scores = avgs
            else:
                scores = avgs * (1 - stdevs / np.max(stdevs))
            order.append(np.argmax(scores))
        return order
    # Intercalation-specific methods - end

    # some minor helper methods - start
    def reconstruct_string(self, string):
        import numexpr
        N = len(string)
        for j in range(N):
            if string[j] == '-':
                # tmpstr = ''.join(string[:j + 2])
                insert = j + 2
            else:
                # tmpstr = ''.join(string[:j + 1])
                insert = j + 1
            try:
                # val = numexpr.evaluate(string)
                numexpr.evaluate(string)
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
        del interstitial[indexlist]

        return interstitial

    def get_newcell(self, cell, N=3):
        a = np.empty((3, 3))
        dim = self.dim
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
        dim = self.dim
        if dim == 3:
            for i, pos in enumerate(positions):
                for j in range(3):
                    a[i][j] = pos[j] - (cell[0][j] + cell[1][j] + cell[2][j])
        elif dim == 2:
            for i, pos in enumerate(positions):
                for j in range(2):
                    a[i][j] = pos[j] - (cell[0][j] + cell[1][j])

        return a

    def get_z_position(self, atoms):
        assert self.is_planar(atoms), 'No planar structure.'

        return atoms.get_positions()[0, 2]
    # some minor helper methods - end
