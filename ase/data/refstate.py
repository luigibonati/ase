import ase
from copy import deepcopy
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


class BaseReferenceState(Mapping):
    def __init__(self, Z, **data):
        self._Z = Z
        self._data = data

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        return iter(self._data)

    def __getitem__(self, key):
        return deepcopy(self._data[key])

    @property
    def Z(self):
        return self._Z

    @property
    def symbol(self):
        return chemical_symbols[self._Z]


class AtomReferenceState(BaseReferenceState):
    def toatoms(self):
        return ase.Atoms([self._Z])


class DiatomReferenceState(BaseReferenceState):
    def toatoms(self):
        # XXX magmoms
        d_half = 0.5 * self.bondlength
        atoms = ase.Atoms([self._Z, self._Z])
        atoms.positions[:, 2] = [-d_half, d_half]
        return atoms

    @property
    def bondlength(self):
        return self['d']

    def __repr__(self):
        return f'Diatom[{self.symbol!r}, bondlength={self.bondlength}]'


class BulkReferenceState(BaseReferenceState):
    @property
    def _crystal_data(self):
        # Avoid early import of spacegroup
        import ase.spacegroup.crystal_data as crystal_data
        return crystal_data

    @property
    def _spacegroup_int(self):
        return self._data['sgs']

    @property
    def crystal_family(self):
        return self._crystal_data.crystal_family[self._spacegroup_int]

    @property
    def lattice_centering(self):
        return self._crystal_data.lattice_centering[self._spacegroup_int]

    @property
    def bravais_class(self):
        return self._crystal_data.get_bravais_class(self._spacegroup)

    @lazyproperty
    def spacegroup(self):
        from ase.spacegroup import Spacegroup
        return Spacegroup(self._spacegroup)
