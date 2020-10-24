class MagneticMoments:
    arrayname = 'initial_magmoms'
    components2type = {1: 'paired',
                       2: 'collinear',
                       3: 'noncollinear'}

    def __init__(self, atoms):
        self.atoms = atoms

    @property
    def _array(self):
        return self.atoms.arrays.get(self.arrayname)

    def __bool__(self):
        """False if unpolarized, else True."""
        return self._array is not None

    @property
    def spin_type(self):
        """Type of spin configuration as a string.

        One of 'unpolarized', 'collinear', or 'noncollinear'."""
        return self.components2type[self.spincomponents]

    @property
    def spincomponents(self):
        """Number of spin components.

        1 for unpolarized, 2 for collinear, or 3 for noncollinear."""
        array = self._array
        if array is None:
            return 1
        ndim = array.ndim
        if ndim == 1:
            return 2
        elif ndim == 2:
            return 3
        else:
            raise ValueError(f'Bad shape of magmoms: {array.shape}')

    @property
    def polarized(self) -> bool:
        return self._array is not None

    @property
    def collinear(self) -> bool:
        return self.spincomponents < 3

    def set(self, magmoms):
        self.atoms.set_initial_magnetic_moments(values)
