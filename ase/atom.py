"""Definition of the Atom class.

This module defines the Atom object.
"""

import numpy as npy

from ase.data import atomic_numbers, chemical_symbols


#        singular,    plural,     type,  shape
data = {'symbol':   ('symbols',   str,   ()  ),
        'number':   ('numbers',   int,   ()  ),
        'position': ('positions', float, (3,)),
        'tag':      ('tags',      int,   ()  ),
        'momentum': ('momenta',   float, (3,)),
        'mass':     ('masses',    float, ()  ),
        'magmom':   ('magmoms',   float, ()  ),
        'charge':   ('charges',   float, ()  ),
        }

class Atom(object):
    """Class for representing a single atom."""
    def __init__(self, symbol='X', position=(0, 0, 0),
                 tag=None, momentum=None, mass=None,
                 magmom=None, charge=None,
                 atoms=None, index=None):
        """Construct Atom object.

        Parameters
        ----------
        symbol : str or int
            Can be a chemical symbol (str) or an atomic number (int).
        position : sequence of 3 floats
            Atomi position.
        tag : int
            Special purpose tag.
        momentum: sequence of 3 floats
            Momentum for atom.
        mass : float
            Atomic mass in atomic units.
        magmom: float
            Magnetic moment.
        charge : float
            Atomic charge.

        Examples
        --------
        The first two atoms are equivalent:

        >>> a = Atom('O', charge=-2)
        >>> b = Atom(8, charge=-2)
        >>> c = Atom('H', (1, 2, 3), magmom=1)
        >>> print a.charge, a.position
        -2 [ 0. 0. 0.]
        >>> c.x = 0.0
        >>> c.position
        array([ 0.,  2.,  3.])
        >>> b.symbol
        'O'
        >>> c.tag = 42
        >>> c.number
        1
        >>> c.symbol = 'Li'
        >>> c.number
        3
        
        """

        if atoms is None:
            if isinstance(symbol, str):
                self._number = atomic_numbers[symbol]
                self._symbol = symbol
            else:
                self._number = symbol
                self._symbol = chemical_symbols[symbol]
            self._position = npy.asarray(position, float)
            self._tag = tag
            self._momentum = npy.asarray(momentum, float)
            self._mass = mass
            self._magmom = magmom
            self._charge = charge

        self.index = index
        self.atoms = atoms

    def _get(self, name):
        if self.atoms is None:
            return getattr(self, '_' + name)
        elif name == 'symbol':
            return chemical_symbols[self.number]
        else:
            plural = data[name][0]
            if plural in self.atoms.arrays:
                return self.atoms.arrays[plural][self.index]
            else:
                return None

    def _set(self, name, value):
        if self.atoms is None:
            setattr(self, '_' + name, value)
            if name == 'symbol':
                self._number = atomic_numbers[value]
            elif name == 'number':
                self._symbol = chemical_symbols[value]
        elif name == 'symbol':
            self.number = atomic_numbers[value]
        else:
            plural, dtype, shape = data[name]
            if plural in self.atoms.arrays:
                self.atoms.arrays[plural][self.index] = value
            else:
                array = npy.zeros((len(self.atoms),) + shape, dtype)
                array[self.index] = value
                self.atoms.new_array(plural, array)

    def _getsymbol(self): return self._get('symbol')
    def _getnumber(self): return self._get('number')
    def _getposition(self): return self._get('position')
    def _gettag(self): return self._get('tag')
    def _getmomentum(self): return self._get('momentum')
    def _getmass(self): return self._get('mass')
    def _getmagmom(self): return self._get('magmom')
    def _getcharge(self): return self._get('charge')

    def _setsymbol(self, symbol): self._set('symbol', symbol)
    def _setnumber(self, number): self._set('number', number)
    def _setposition(self, position): self._set('position', position)
    def _settag(self, tag): self._set('tag', tag)
    def _setmomentum(self, momentum): self._set('momentum', momentum)
    def _setmass(self, mass): self._set('mass', mass)
    def _setmagmom(self, magmom): self._set('magmom', magmom)
    def _setcharge(self, charge): self._set('charge', charge)

    symbol = property(_getsymbol, _setsymbol, doc='Chemical symbol')
    number = property(_getnumber, _setnumber, doc='Atomic number')
    position = property(_getposition, _setposition, doc='XYZ-coordinates')
    tag = property(_gettag, _settag, doc='Integer tag')
    momentum = property(_getmomentum, _setmomentum, doc='XYZ-momentum')
    mass = property(_getmass, _setmass, doc='Atomic mass')
    magmom = property(_getmagmom, _setmagmom, doc='Magnetic moment')
    charge = property(_getcharge, _setcharge, doc='Atomic Charge')

    def _getx(self): return self.position[0]
    def _gety(self): return self.position[1]
    def _getz(self): return self.position[2]
    
    def _setx(self, x): self.position[0] = x
    def _sety(self, y): self.position[1] = y
    def _setz(self, z): self.position[2] = z

    x = property(_getx, _setx, doc='X-coordiante')
    y = property(_gety, _sety, doc='Y-coordiante')
    z = property(_getz, _setz, doc='Z-coordiante')

    def get_data(self):
        return (self.position, self.number,
                self.tag, self.momentum, self.mass,
                self.magmom, self.charge)
