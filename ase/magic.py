"""Magic arrays: Arraylike class which allows on-the-fly processing.

This can be used e.g. to emulate an array of zeros without allocating
an array of zeros."""

import numpy as np
import functools
from ase.utils.arraywrapper import inplace_methods, forward_methods


def arraymeth(meth):
    """Helper method for forwarding __add__, __mul__, etc. to ndarray."""
    @functools.wraps(meth)
    def newmeth(self, obj):
        return meth(self.array, obj)
    return newmeth


def inplace_arraymeth(meth):
    """Helper method for in-place array methods like __imul__."""
    @functools.wraps(meth)
    def newmeth(self, obj):
        meth(self.array, obj)
        return self
    return newmeth


def forward_method(cls, name, inplace):
    """Poke shortcut to ndarray method onto cls.

    Silently ignored if cls already has that method."""
    if hasattr(cls, name):
        return
    numpy_meth = getattr(np.ndarray, name)
    if inplace:
        wrapper = inplace_arraymeth(numpy_meth)
    else:
        wrapper = arraymeth(numpy_meth)
    setattr(cls, name, wrapper)


def magicarray(cls):
    """Decorator for adding array functions."""
    for name in inplace_methods:
        forward_method(cls, name, inplace=True)
    for name in forward_methods:
        forward_method(cls, name, inplace=False)
    return cls


@magicarray
class Magic:
    shape = tuple()
    dtype = int
    name = 'magic'  # Should be set to standard name of array, e.g., 'tags'

    def __init__(self, arrays, indices=None):
        # arrays should be the typical atoms.arrays dictionary.
        # We use at least the 'numbers' key.
        self.arrays = arrays

        # We have the full length-N arrays.  But if someone slices us,
        # we need to produce a view representing a slice of our array.
        # For this we use the indices variable.
        if indices is None:
            indices = range(self._global_length)
        self.indices = indices

    @property
    def _slice(self):
        i = self.indices
        return slice(i.start, i.stop, i.step)

    @property
    def _global_length(self):
        return len(self.arrays['numbers'])

    def new_array(self):
        shape = (self._global_length,) + self.shape
        array = np.zeros(shape, self.dtype)
        return array

    @property
    def array(self):
        if not self:
            self.arrays[self.name] = self.new_array()
        return self.arrays[self.name][self._slice]

    @array.setter
    def array(self, value):
        self.array[self.indices] = value

    def __bool__(self):
        """Whether the backend array is allocated."""
        return self.name in self.arrays

    def __iter__(self):
        if self:
            return iter(self.array)
        return (0 for _ in range(len(self)))

    def __getitem__(self, index):
        # Fancy array slicing not yet supported
        index = self.indices[index]

        if np.isscalar(index):
            if self:
                num = self.array[index]
                assert np.isscalar(num)
                return num
            else:
                return 0

        return self.__class__(self.arrays, index)

    def __setitem__(self, index, value):
        self.array[index] = value

    def __eq__(self, other):
        if self:
            return self.array == other
        return np.zeros(len(self), int) == other

    def __len__(self):
        return len(self.indices)

    def __repr__(self):
        if self:
            obj = self.array
        else:
            obj = '[{} zeros]'.format(len(self))
        return '{}({})'.format(type(self).__name__, obj)


class Tags(Magic):
    name = 'tags'
    shape = tuple()
    dtype = int


class Masses(Magic):
    name = 'masses'
    shape = tuple()
    dtype = float
    # Not finished
    # Provide some function to determine the defaults, etc.


class ThreeFloats(Magic):
    """Magic array for (natoms, 3) float quantities."""
    shape = (3,)
    dtype = float


class Momenta(ThreeFloats):
    name = 'momenta'
    # Not finished.  Might work though?


class Velocities(ThreeFloats):
    name = 'momenta'
    # Not finished
    # Override get/set


class ScaledPositions(ThreeFloats):
    name = 'positions'
    # Not finished
    # Override get/set.
    # Here we also need the cell
