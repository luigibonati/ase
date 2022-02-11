from __future__ import print_function, division
# Copyright (C) 2018, JaeHwan Shim
# (see accompanying license files for details).
import os
import numpy as np
from ase.geometry import cellpar_to_cell
from itertools import chain
from ase.utils import basestring
from ase.spacegroup.spacegroup import SpacegroupValueError

"""
 Class for return spacegroup crystal based on Wyckoff positions
 See the physical details below

 Parameters:

 sg_name : dictionary sends spacegroup to Its name
 prim_lat_of_sg : Dictionary contains the functions that
                  send name to Primitivie lattice of spacegroup

 Avery, Patrick, and Eva Zurek.
 “RandSpg: An Open-Source Program for Generating Atomistic Crystal Structures
  with Specific Spacegroups.”
 Computer Physics Communications 213 (2017): 208–16.
 https://doi.org/10.1016/j.cpc.2016.12.005

"""


class TupleDict(dict):
    """
      Class to help construct the dictionary of primitive spacegroup lattice
    """
    def __getitem__(self, item):
        if type(item) != tuple:
            for key in self:
                if item in key:
                    return self[key]
        else:
            return super().__getitem__(item)


case = (lambda x, y: tuple(range(x, y+1)))
cases = (lambda *li: tuple(chain(*[case(x, y) for x, y in list(li)])))
# Name of spacegroup
sg_name = TupleDict({(1, 2): 'Tc', (3, 4, 6, 7, 10, 11, 13, 14): 'Mc',
                    (5, 8, 9, 12, 15): 'Mc2',
                    cases((16, 19), (25, 34), (47, 62)): 'Or',
                    cases((20, 21), (35, 41), (63, 68)): 'Or2',
                    cases((42, 43), (69, 70)): 'Or3', (22,): 'Or3',
                    cases((23, 24), (44, 46), (71, 74)): 'Or4',
                    cases((75, 78), (83, 86), (89, 96), (99, 106), (111, 118),
                          (123, 138)): 'Te', (81,): 'Te',
                    cases((79, 80), (87, 88), (97, 98), (107, 110), (119, 122),
                          (139, 142)): 'Te2', (82,): 'Te2',
                    cases((143, 145), (149, 154), (156, 159),
                          (162, 165)): 'Tr', (147,): 'Tr',
                         (146, 148, 155, 160, 161, 166, 167): 'Tr2',
                    case(168, 194): 'Hx',
                    cases((221, 224), (212, 213),
                          (207, 208), (200, 201)): 'Cb',
                         (195, 198, 205, 215, 218): 'Cb',
                    case(225, 228): 'Cb2',
                        (196, 202, 203, 209, 210, 216, 219): 'Cb2',
                    (197, 199, 204, 206, 211, 214, 217, 220, 229, 230): 'Cb3'
                     })
p = np.pi
px2 = np.pi*2
tet = 109.471220634491  # Angle between two edges of Tegtrahedron
ptet = p*tet/180
prim_lat_of_sg = {  # Primitive lattice of spacegroup
        'Tc': (lambda a, b, c, al, be, ga: (a, b, c, al, be, ga)),
        'Mc': (lambda a, b, c, al, be, ga: (a, b, c, p/2, p/2, ga)),
        'Mc2': (lambda a, b, c, al, be, ga: (a, a, c, al, al, ga)),
        'Or': (lambda a, b, c, al, be, ga: (a, b, c, p/2, p/2, p/2)),
        'Or2': (lambda a, b, c, al, be, ga: (a, a, c, p/2, p/2, p/2)),
        'Te': (lambda a, b, c, al, be, ga: (a, a, c, p/2, p/2, p/2)),
        'Tr': (lambda a, b, c, al, be, ga: (a, a, c, p/2, p/2, p*2/3)),
        'Tr2': (lambda a, b, c, al, be, ga: (a, a, a, al, al, al)),
        'Hx': (lambda a, b, c, al, be, ga: (a, a, c, p/2, p/2, p*2/3)),
        'Cb': (lambda a, b, c, al, be, ga: (a, a, a, p/2, p/2, p/2)),
        'Cb2': (lambda a, b, c, al, be, ga: (a, a, a, p/3, p/3, p/3)),
        'Cb3': (lambda a, b, c, al, be, ga: (a, a, a, ptet, ptet, ptet)),
        'Or3': (lambda a, b, c, al, be, ga: (a, b, c,
                                             np.arccos((b*b+c*c-a*a)/2/b/c),
                                             np.arccos((a*a+c*c-b*b)/2/a/c),
                                             np.arccos((a*a+b*b-c*c)/2/a/b))),
        'Or4': (lambda a, b, c, al, be, ga:
                (a, a, a, al, be, np.arccos(-1+np.cos(al)+np.cos(be)))
                if np.cos(al)+np.cos(be) > 0.
                else (a, a, a, al, be, np.arccos(-1-np.cos(al)-np.cos(be)))),
        'Te2': (lambda a, b, c, al, be, ga:
                (a, a, a, al, al, ga)
                if np.cos(al)+np.cos(be) > 0.
                else (a, a, a, al, al, np.arccos(-1-np.cos(al)-np.cos(be))))
         }


def spacegroup_lattice(volume=None, sgidx=None, cellpar=None):
    """ Returns a primitive lattice cellpar of specific spacegroup `sgidx`
        within given volume"""
    attempt = 0
    while attempt < 100:
        if cellpar is None:
            cellpar = np.random.rand(6)
            cellpar[3:6] *= np.pi/2
        sg_cellpar = prim_lat_of_sg[sg_name[sgidx]](*cellpar)
        sg_cellpar = np.array(sg_cellpar)
        sg_cellpar[3:6] *= 180/np.pi
        sg_volume = abs(np.linalg.det(cellpar_to_cell(sg_cellpar)))
        if volume is not None:
            sg_cellpar[0:3] *= (volume/sg_volume)**(1/3)
        if np.any(np.isnan(cellpar)):
            attempt += 1
            continue
        else:
            return sg_cellpar
    raise ValueError('Generating spacegroup keep failing')


class WyckoffCrystalNotFoundError(Exception):
    """Raise when given Wyckoff composition is Not found"""
    pass


class WyckoffValueError(Exception):
    """Raise when given wyckoff sites can not be parsed"""
    pass


class Letter:
    def __init__(self, dictionary):
        for k, v in dictionary.items():
            setattr(self, k, v)


class Wyckoff:
    def __init__(self, spacegroup=None, wyckoff=None,
                 setting=None, datafile=None):
        """A Wyckoff class.

        The instances of Wyckoff describes the Wyckoff positions for
        the given space group.

        Example:

        >>> from ase.spacegroup import Wyckoff
        >>>
        >>> wyck = Wyckoff(225)
        >>> print('Wyckoff positions', wyck.a.site_symmetry, wyck.a.multiplicity)
        Wyckoff positions ..2. 6
        >>> wyck.scaled_primitive_cell
        array([[ 0. ,  0.5,  0.5],
               [ 0.5,  0. ,  0.5],
               [ 0.5,  0.5,  0. ]])
        >>> wyck.a.coordinates
        array([[ 0. ,  0. ,  0. ],
               [ 0. ,  0.5,  0.5],
               [ 0.5,  0. ,  0.5],
               [ 0.5,  0.5,  0. ]])

        Or You can use Wyckoff class as dictionary object

        Example:


        >>> from ase.spacegroup import Wyckoff
        >>>
        >>> wyck = Wyckoff(225)
        >>> print('Wyckoff positions', wyck['a']['multiplicity'])
        Wyckoff positions 6
        >>> sg.scaled_primitive_cell
        array([[ 0. ,  0.5,  0.5],
               [ 0.5,  0. ,  0.5],
               [ 0.5,  0.5,  0. ]])
        >>> wyck['a']['coordinates']
        array([[ 0. ,  0. ,  0. ],
               [ 0. ,  0.5,  0.5],
               [ 0.5,  0. ,  0.5],
               [ 0.5,  0.5,  0. ]])


        wyckoff = {'spacegroup':45,
                   'letters':['a','b','c', ...],
                   'number_of_letters':6,
                   'a':{'multiplicity':3, 'site_symmetry':'-1',
                        'positions':['-x,y,0', '0,0,0']},
                   'b':{'multiplicity':4, 'site_symmetry':'-1',
                        'positions':[...]},
                    ...
                   }

        """
        if isinstance(wyckoff, Wyckoff):
            wyckoff = Wyckoff.wyckoff
        elif wyckoff is None:
            wyckoff = {}
        if not datafile:
            datafile = get_datafile()
        f = open(datafile, 'r')
        try:
            self.wyckoff = _read_datafile(wyckoff, spacegroup, setting, f)
            self.wyckoff['scaled_primitive_cell'] = spacegroup_lattice
        finally:
            f.close()
        for key in self.wyckoff.keys():
            if key in self.wyckoff['letters']:
                setattr(self, key, Letter(self.wyckoff[key]))
            else:
                setattr(self, key, self.wyckoff[key])

    def __getitem__(self, key):
        """Convenience method to retrieve a parameter as
        calculator[key] rather than calculator.parameters[key]

            Parameters:
                -key       : str, the name of the parameters to get.
        """
        return self.wyckoff[key]

    def get(self, key, default=None):
        return self.wyckoff.get(key, default)

    def __contains__(self, x):
        return x in self.wyckoff


def _read_datafile(wyc, spacegroup, setting, f):
    """Read the `wyckpos.dat` file of specific spacegroup and returns the
       dictionary `wyc`"""
    if isinstance(spacegroup, int):
        pass
    elif isinstance(spacegroup, basestring):
        spacegroup = ' '.join(spacegroup.strip().split())
    else:
        raise SpacegroupValueError('`spacegroup` must be of type int or str')
    line = _skip_to_spacegroup(f, spacegroup, setting)
    wyc['letters'] = []
    wyc['multiplicity'] = []
    wyc['number_of_letters'] = 0
    line_list = line.split()
    if line_list[0].isdigit():
        wyc['spacegroup'] = int(line_list[0])
    else:
        spacegroup, wyc['setting'] = line_list[0].split('-')
        wyc['spacegroup'] = int(spacegroup)
    if len(line.split()) > 1:
        eq_sites = line.split('(')[1:]
        wyc['equivalent_sites'] = ([eq[:-1] for eq in eq_sites])[1:]
        wyc['equivalent_sites'][-1] = wyc['equivalent_sites'][-1][:-1]

    while True:
        line = f.readline()
        if line == '\n':
            break
        letter, multiplicity = line.split()[:2]
        coordinates_raw = line.split()[-1].split('(')[1:]
        site_symmetry = ''.join(line.split()[2:-1])
        wyc['letters'].append(letter)
        wyc['number_of_letters'] += 1
        wyc['multiplicity'].append(int(multiplicity))
        coordinates = [coord[:-1] for coord in coordinates_raw]
        wyc[letter] = {'multiplicity': multiplicity,
                       'site_symmetry': site_symmetry,
                       'coordinates': coordinates,
                       }

    return wyc


def get_datafile():
    """Return default path to datafile."""
    return os.path.join(os.path.dirname(__file__), 'wyckpos.dat')


def _skip_to_spacegroup(f, spacegroup, setting=None):
    """Read lines from f until a blank line is encountered."""
    if setting is None:
        name = str(spacegroup)
    else:
        name = str(spacegroup)+'-'+setting
    while True:
        line = f.readline()
        if not line:
            raise RuntimeError(
                'invalid spacegroup `%s`, setting `%s` not found in data base' %
                (spacegroup, setting))
        if line.startswith(name):
            break
    return line
