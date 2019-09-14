import numpy as np


class Magic:
    shape = tuple()
    dtype = int
    name = 'magic'

    def __init__(self, arrays, indices=None):
        self.arrays = arrays
        if indices is None:
            indices = range(self.global_length)
        self.indices = indices

    @property
    def global_length(self):
        return len(self.arrays['numbers'])

    def new_array(self):
        assert not self.allocated
        print(self.global_length, self.shape)
        shape = (self.global_length,) + self.shape
        print(shape)
        array = np.zeros(shape, self.dtype)
        return array

    @property
    def array(self):
        return self.arrays[self.name][self.indices]

    @property
    def allocated(self):
        return self.name in self.arrays

    def __getitem__(self, index):
        # Fancy array slicing not yet supported
        index = self.indices[index]

        if np.isscalar(index):
            if self.allocated:
                num = self.array[index]
                assert np.isscalar(num)
                return num
            else:
                return 0

        return self.__class__(self.arrays, index)

    def __eq__(self, other):
        if self.allocated:
            return self.array == other
        return np.zeros(len(self), int) == other

    def __setitem__(self, index, value):
        # Fancy array slicing not yet supported
        index = self.indices[index]
        array = self.new_array()
        array[index] = value
        if any(array):
            self.arrays[self.name] = array
        else:
            self.arrays.pop(self.name, None)

    def __len__(self):
        return len(self.indices)

    def __repr__(self):
        if self.allocated:
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
    # Something about determining defaults?


class ThreeFloats(Magic):
    shape = (3,)
    dtype = float


class Momenta(ThreeFloats):
    name = 'momenta'


class Velocities(ThreeFloats):
    name = 'momenta'
    # Override get/set


class ScaledPositions(ThreeFloats):
    name = 'positions'
    # Override get/set.
    # Here we also need the cell
