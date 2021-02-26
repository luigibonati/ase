import ase
from copy import copy, deepcopy
from abc import abstractmethod
from collections.abc import Mapping

from ase.utils import lazyproperty
from ase.data import chemical_symbols


def bad_structure(name):
    return ValueError(f'Unexpected structure {name!r}')


def define_reference_state(Z, **data):
    spacegroup = data.get('sgs')

    if spacegroup is not None:
        return BulkReferenceState(Z, **data)

    symm = data.get('symmetry')

    if symm == 'diatom':
        return DiatomReferenceState(Z, **data)

    if symm == 'atom':
        return AtomReferenceState(Z, **data)

    raise bad_structure(symm)


class ReferenceState(Mapping):
    def __init__(self, Z, **data):
        self._Z = Z
        self._data = data

    def __len__(self):
        """Length of internal dictionary of reference state data."""
        return len(self._data)

    def __iter__(self):
        return iter(self._data)

    def __getitem__(self, key):
        return deepcopy(self._data[key])

    @property
    def Z(self):
        """Atomic number."""
        return self._Z

    @property
    def symbol(self):
        """Chemical symbol."""
        return chemical_symbols[self._Z]

    def copy(self):
        return copy(self)

    def __repr__(self):
        clsname = type(self).__name__
        tokens = [repr(self.symbol)]
        tokens += [f'{key}={value}' for key, value in self.items()]
        varlist = ', '.join(tokens)
        return f'{clsname}[{varlist}]'

class AtomReferenceState(ReferenceState):
    def toatoms(self):
        return ase.Atoms([self._Z])


class DiatomReferenceState(ReferenceState):
    def toatoms(self):
        # XXX magmoms
        d_half = 0.5 * self.bondlength
        atoms = ase.Atoms([self._Z, self._Z])
        atoms.positions[:, 2] = [-d_half, d_half]
        return atoms

    @property
    def bondlength(self):
        return self['d']


class BulkReferenceState(ReferenceState):
    @property
    def _crystal_data(self):
        # This is to avoid early import of spacegroup
        import ase.spacegroup.crystal_data as crystal_data
        return crystal_data

    @property
    def _spacegroup_int(self):
        return self._data['sgs']

    @property
    def crystal_family(self):
        return self._crystal_data._crystal_family[self._spacegroup_int]

    @property
    def lattice_centering(self):
        return self._crystal_data._lattice_centering[self._spacegroup_int]

    @property
    def _bravais_class(self):
        return self._crystal_data.get_bravais_class(self._spacegroup_int)

    @lazyproperty
    def spacegroup(self):
        from ase.spacegroup import Spacegroup
        return Spacegroup(self._spacegroup_int)
