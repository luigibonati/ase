"""Vibrational modes."""

import collections
from math import sin, pi, sqrt, log
from numbers import Real, Integral
import os
import os.path as op
import pickle
import sys
from typing import Dict, List, NoReturn, Sequence, Tuple, Union, Any

import numpy as np

from ase.atoms import Atoms
import ase.units as units
import ase.io
from ase.parallel import world, paropen
from ase.utils import jsonable

from ase.utils import opencew, pickleload
from ase.calculators.singlepoint import SinglePointCalculator
from ase.spectrum.dosdata import RawDOSData
from ase.spectrum.doscollection import DOSCollection

RealSequence4D = Sequence[Sequence[Sequence[Sequence[Real]]]]


@jsonable('vibrationsdata')
class VibrationsData:
    """Class for storing and analyzing vibrational data (i.e. Atoms + Hessian)

    This class is not responsible for calculating Hessians; the Hessian should
    be computed by a Calculator or some other algorithm. Once the
    VibrationsData has been constructed, this class provides some common
    processing options; frequency calculation, mode animation, DOS etc.

    If the Atoms object is a periodic supercell, VibrationsData may be
    converted to a PhononData using the VibrationsData.to_phonondata() method.
    This provides access to q-point-dependent analyses such as phonon
    dispersion plotting.

    Args:
        atoms:
            Equilibrium geometry of vibrating system. This will be stored as a
            lightweight copy with just positions, masses, unit cell.

        hessian: Second-derivative in energy with respect to
            Cartesian nuclear movements as an (N, 3, N, 3) array.
        indices: indices of atoms which are included
            in Hessian.  Default value (None) includes all atoms.

    """
    def __init__(self,
                 atoms: Atoms,
                 hessian: Union[RealSequence4D, np.ndarray],
                 indices: Union[Sequence[int], np.ndarray] = None,
                 ) -> None:

        if indices is None:
            self._indices = None
        else:
            self._indices = list(indices)

        n_atoms = self._check_dimensions(atoms, np.asarray(hessian),
                                         indices=self._indices)
        masses = atoms.get_masses() if atoms.has('masses') else None

        self._atoms = Atoms(cell=atoms.cell, pbc=atoms.pbc,
                            numbers=atoms.numbers,
                            masses=masses,
                            positions=atoms.positions)

        self._hessian2d = (np.asarray(hessian)
                           .reshape(3 * n_atoms, 3 * n_atoms).copy())

        self._energies = None  # type: Union[np.ndarray, None]
        self._modes = None  # type: Union[np.ndarray, None]

    _setter_error = ("VibrationsData properties cannot be modified: construct "
                     "a new VibrationsData with consistent atoms, Hessian and "
                     "(optionally) indices/mask.")

    @classmethod
    def from_2d(cls, atoms: Atoms,
                hessian_2d: Union[Sequence[Sequence[Real]], np.ndarray],
                indices: Sequence[int] = None) -> 'VibrationsData':
        """Instantiate VibrationsData when the Hessian is in a 3Nx3N format

        Args:
            atoms: Equilibrium geometry of vibrating system

            hessian: Second-derivative in energy with respect to
                Cartesian nuclear movements as a (3N, 3N) array.

            indices: Indices of (non-frozen) atoms included in Hessian

        """
        hessian_2d_array = np.asarray(hessian_2d)
        n_atoms = cls._check_dimensions(atoms, hessian_2d_array,
                                        indices=indices, two_d=True)

        return cls(atoms, hessian_2d_array.reshape(n_atoms, 3, n_atoms, 3),
                   indices=indices)

    @staticmethod
    def indices_from_mask(mask: Union[Sequence[bool], np.ndarray]
                          ) -> List[int]:
        """Indices corresponding to boolean mask

        This is provided as a convenience for instantiating VibrationsData with
        a boolean mask. For example, if the Hessian data includes only the H
        atoms in a structure::

          h_mask = atoms.get_chemical_symbols() == 'H'
          vib_data = VibrationsData(atoms, hessian,
                                    VibrationsData.indices_from_mask(h_mask))

        Take care to ensure that the length of the mask corresponds to the full
        number of atoms; this function is only aware of the mask it has been
        given.

        Args:
            mask: a sequence of True, False values

        Returns:
            indices of True elements

        """

        return (np.arange(len(mask), dtype=int)[np.asarray(mask, dtype=bool)]
                ).tolist()

    @staticmethod
    def _check_dimensions(atoms: Atoms,
                          hessian: np.ndarray,
                          indices: Union[Sequence[int], None] = None,
                          two_d: bool = False) -> int:
        """Sanity check on array shapes from input data

        Args:
            atoms: Structure
            indices: Indices of atoms used in Hessian
            hessian: Proposed Hessian array

        Returns:
            Number of atoms contributing to Hessian

        Raises:
            ValueError if Hessian dimensions are not (N, 3, N, 3)

        """
        if indices is None:
            n_atoms = len(atoms)
        else:
            n_atoms = len(atoms[indices])

        if two_d:
            ref_shape = [n_atoms * 3, n_atoms * 3]
            ref_shape_txt = '{n:d}x{n:d}'.format(n=(n_atoms * 3))

        else:
            ref_shape = [n_atoms, 3, n_atoms, 3]
            ref_shape_txt = '{n:d}x3x{n:d}x3'.format(n=n_atoms)

        if (isinstance(hessian, np.ndarray)
            and hessian.shape == tuple(ref_shape)):
            return n_atoms
        else:
            raise ValueError("Hessian for these atoms should be a "
                             "{} numpy array.".format(ref_shape_txt))

    @property
    def atoms(self) -> Atoms:
        return self._atoms.copy()

    @atoms.setter
    def atoms(self, new_atoms) -> NoReturn:
        raise NotImplementedError(self._setter_error)

    @property
    def indices(self) -> Union[None, np.ndarray]:
        if self._indices is None:
            return None
        else:
            return np.array(self._indices, dtype=int)

    @indices.setter
    def indices(self, new_indices) -> NoReturn:
        raise NotImplementedError(self._setter_error)

    @property
    def mask(self) -> np.ndarray:
        """Boolean mask of atoms selected by indices"""
        return self._mask_from_indices(self.atoms, self.indices)

    @mask.setter
    def mask(self, values) -> NoReturn:
        raise NotImplementedError(self._setter_error)

    @staticmethod
    def _mask_from_indices(atoms: Atoms,
                           indices: Union[None, Sequence[int], np.ndarray]
                           ) -> np.ndarray:
        """Boolean mask of atoms selected by indices"""
        natoms = len(atoms)
        if indices is None:
            return np.full(natoms, True, dtype=bool)
        else:
            # Wrap indices to allow negative values
            indices = np.asarray(indices) % natoms
            mask = np.full(natoms, False, dtype=bool)
            mask[indices] = True
            return mask

    @property
    def hessian(self) -> np.ndarray:
        """The Hessian; second derivative of energy wrt positions

        This format is preferred for iteration over atoms and when
        addressing specific elements of the Hessian.

        Returns:
            array with shape (n_atoms, 3, n_atoms, 3) where
            - the first and third indices identify atoms in self.atoms
            - the second and fourth indices cover the corresponding Cartesian
              movements in x, y, z

            e.g. the element h[0, 2, 1, 0] gives a harmonic force exerted on
            atoms[1] in the x-direction in response to a movement in the
            z-direction of atoms[0]

        """
        n_atoms = int(self._hessian2d.shape[0] / 3)
        return self._hessian2d.reshape(n_atoms, 3, n_atoms, 3).copy()

    @hessian.setter
    def hessian(self, new_values) -> NoReturn:
        raise NotImplementedError(self._setter_error)

    def get_hessian_2d(self) -> np.ndarray:
        """Get the Hessian as a 2-D array

        This format may be preferred for use with standard linear algebra
        functions

        Returns:
            array with shape (n_atoms * 3, n_atoms * 3) where the elements are
            ordered by atom and Cartesian direction

            [[at1x_at1x, at1x_at1y, at1x_at1z, at1x_at2x, ...],
             [at1y_at1x, at1y_at1y, at1y_at1z, at1y_at2x, ...],
             [at1z_at1x, at1z_at1y, at1z_at1z, at1z_at2x, ...],
             [at2x_at1x, at2x_at1y, at2x_at1z, at2x_at2x, ...],
             ...]

            e.g. the element h[2, 3] gives a harmonic force exerted on
            atoms[1] in the x-direction in response to a movement in the
            z-direction of atoms[0]

        """
        return self._hessian2d.copy()

    def todict(self) -> Dict[str, Any]:
        return {'atoms': self.atoms,
                'hessian': self.hessian,
                'indices': self.indices}

    @classmethod
    def fromdict(cls, data: Dict[str, Any]) -> 'VibrationsData':
        # mypy is understandably suspicious of data coming from a dict that
        # holds mixed types, but it can see if we sanity-check with 'assert'
        assert isinstance(data['atoms'], Atoms)
        assert isinstance(data['hessian'], (collections.abc.Sequence,
                                            np.ndarray))
        if data['indices'] is not None:
            assert isinstance(data['indices'], (collections.abc.Sequence,
                                                np.ndarray))
            for index in data['indices']:
                assert isinstance(index, Integral)

        return cls(data['atoms'], data['hessian'], indices=data['indices'])

    def _calculate_energies_and_modes(self) -> Tuple[np.ndarray, np.ndarray]:
        """Diagonalise the Hessian to obtain harmonic modes

        This method is an internal implementation of get_energies_and_modes(),
        see the docstring of that method for more information.

        """
        active_atoms = self.atoms[self.mask]
        n_atoms = len(active_atoms)
        masses = active_atoms.get_masses()

        if not np.all(masses):
            raise ValueError('Zero mass encountered in one or more of '
                             'the vibrated atoms. Use Atoms.set_masses()'
                             ' to set all masses to non-zero values.')
        mass_weights = np.repeat(masses**-0.5, 3)

        omega2, vectors = np.linalg.eigh(mass_weights
                                         * self.get_hessian_2d()
                                         * mass_weights[:, np.newaxis])

        unit_conversion = units._hbar * units.m / sqrt(units._e * units._amu)
        energies = unit_conversion * omega2.astype(complex)**0.5

        modes = vectors.T.reshape(n_atoms * 3, n_atoms, 3)
        modes = modes * masses[np.newaxis, :, np.newaxis]**-0.5

        return (energies, modes)

    def get_energies_and_modes(self) -> Tuple[np.ndarray, np.ndarray]:
        """Diagonalise the Hessian to obtain harmonic modes

        Results are cached so diagonalization will only be performed once for
        this object instance.

        Returns:
            tuple (energies, modes)

            Energies are given in units of eV. (To convert these to frequencies
            in cm-1, divide by ase.units.invcm.)

            Modes are given in Cartesian coordinates as a (3N, N, 3) array
            where indices correspond to the (mode_index, atom, direction).

            Note that in this array only the moving atoms are included.

        """
        if self._energies is None or self._modes is None:
            self._energies, self._modes = self._calculate_energies_and_modes()
            return self.get_energies_and_modes()
        else:
            return (self._energies.copy(), self._modes.copy())

    def get_modes(self) -> np.ndarray:
        """Diagonalise the Hessian to obtain harmonic modes

        Results are cached so diagonalization will only be performed once for
        this object instance.

        Returns:
            Modes in Cartesian coordinates as a (3N, N, 3) array where indices
            correspond to the (mode_index, atom, direction).

        """
        return self.get_energies_and_modes()[1]

    def get_energies(self) -> np.ndarray:
        """Diagonalise the Hessian to obtain eigenvalues

        Results are cached so diagonalization will only be performed once for
        this object instance.

        Returns:
            Harmonic mode energies in units of eV

        """
        return self.get_energies_and_modes()[0]

    def get_frequencies(self) -> np.ndarray:
        """Diagonalise the Hessian to obtain frequencies in cm^-1

        Results are cached so diagonalization will only be performed once for
        this object instance.

        Returns:
            Harmonic mode frequencies in units of cm^-1

        """

        return self.get_energies() / units.invcm

    def get_zero_point_energy(self) -> float:
        """Diagonalise the Hessian and sum hw/2 to obtain zero-point energy

        Args:
            energies:
                Pre-computed energy eigenvalues. Use if available to avoid
                re-calculating these from the Hessian.

        Returns:
            zero-point energy in eV
        """
        energies, _ = self.get_energies_and_modes()

        return 0.5 * energies.real.sum()

    def summary(self,
                logfile: str = None,
                im_tol: float = 1e-8) -> None:
        """Print a summary of the vibrational frequencies.

        Args:
            energies:
                Pre-computed set of energies. Use if available to avoid
                re-calculation from the Hessian.
            logfile: if specified, write output to a different location
                than stdout. Can be an object with a write() method or the name
                of a file to create.
            im_tol:
                Tolerance for imaginary frequency in eV. If frequency has a
                larger imaginary component than im_tol, the imaginary component
                is shown int the summary table.
        """

        if logfile is None:
            log = sys.stdout
        elif isinstance(logfile, str):
            log = paropen(logfile, 'a')

        energies = self.get_energies()

        log.write('---------------------\n')
        log.write('  #    meV     cm^-1\n')
        log.write('---------------------\n')
        for n, e in enumerate(energies):
            if abs(e.imag) > im_tol:
                c = 'i'
                e = e.imag
            else:
                c = ''
                e = e.real
            log.write('{index:3d} {mev:6.1f}{im:1s}  {cm:7.1f}{im}\n'.format(
                index=n, mev=(e * 1e3), cm=(e / units.invcm), im=c))

        log.write('---------------------\n')
        log.write('Zero-point energy: {:.3f} eV\n'.format(
            self.get_zero_point_energy()))

    def show_as_force(self,
                      mode: int,
                      scale: float = 0.2,
                      show: bool = True) -> ase.Atoms:
        """Illustrate mode as "forces" on atoms

        Args:
            mode: mode index
            scale: scale factor
            show: if True, open the ASE GUI

        Returns:
            copy of input atoms with a SinglePointCalculator holding scaled
            forces corresponding to mode eigenvectors. This is the structure
            shown in the GUI when show=True.

        """

        atoms = self.atoms  # Spawns a copy to avoid mutating underlying data
        mode = self.get_modes()[mode] * len(atoms) * 3 * scale
        atoms.calc = SinglePointCalculator(atoms, forces=mode)
        if show:
            self.atoms.edit()

        return atoms

    def write_jmol(self,
                   filename: str = 'vib.xyz',
                   ir_intensities: Union[Sequence[float], np.ndarray] = None
                   ) -> None:
        """Writes file for viewing of the modes with jmol.

        This is an extended XYZ file with eigenvectors given as extra columns
        and metadata given in the label/comment line for each image. The format
        is not quite human-friendly, but has the advantage that it can be
        imported back into ASE with ase.io.read.

        Args:
            filename: Path for output file
            energies_and_modes: Use pre-computed eigenvalue/eigenvector data if
                available; otherwise it will be recalculated from the Hessian.
            ir_intensities: If available, IR intensities can be included in the
                header lines. This does not affect the visualisation, but may
                be convenient when comparing to experimental data.
        """

        energies_and_modes = self.get_energies_and_modes()

        all_images = []
        for i, (energy, mode) in enumerate(zip(*energies_and_modes)):
            # write imaginary frequencies as negative numbers
            if energy.imag > energy.real:
                energy = float(-energy.imag)
            else:
                energy = energy.real

            image = self.atoms.copy()
            image.info.update({'mode#': str(i),
                               'frequency_cm-1': energy / units.invcm,
                               })
            image.arrays['mode'] = np.zeros_like(image.positions)
            image.arrays['mode'][self.mask] = mode

            # Custom masses are quite useful in vibration analysis, but will
            # show up in the xyz file unless we remove them
            if image.has('masses'):
                del image.arrays['masses']

            if ir_intensities is not None:
                image.info['IR_intensity'] = float(ir_intensities[i])

            all_images.append(image)
        ase.io.write(filename, all_images, format='extxyz')

    def get_dos(self) -> RawDOSData:
        """Total phonon DOS"""
        energies = self.get_energies()
        return RawDOSData(energies, np.ones_like(energies))

    def get_pdos(self) -> DOSCollection:
        """Phonon DOS, including atomic contributions"""
        energies = self.get_energies()
        masses = self.atoms[self.mask].get_masses()

        # Get weights as N_moving_atoms x N_modes array
        vectors = self.get_modes() / masses[np.newaxis, :, np.newaxis]**-0.5
        all_weights = (np.linalg.norm(vectors, axis=-1)**2).T

        all_info = [{'index': i, 'symbol': a.symbol}
                    for i, a in enumerate(self.atoms) if self.mask[i]]

        return DOSCollection([RawDOSData(energies, weights, info=info)
                              for weights, info in zip(all_weights, all_info)])

class Vibrations:
    """Class for calculating vibrational modes using finite difference.

    The vibrational modes are calculated from a finite difference
    approximation of the Hessian matrix.

    The *summary()*, *get_energies()* and *get_frequencies()* methods all take
    an optional *method* keyword.  Use method='Frederiksen' to use the method
    described in:

      T. Frederiksen, M. Paulsson, M. Brandbyge, A. P. Jauho:
      "Inelastic transport theory from first-principles: methodology and
      applications for nanoscale devices", Phys. Rev. B 75, 205413 (2007)

    atoms: Atoms object
        The atoms to work on.
    indices: list of int
        List of indices of atoms to vibrate.  Default behavior is
        to vibrate all atoms.
    name: str
        Name to use for files.
    delta: float
        Magnitude of displacements.
    nfree: int
        Number of displacements per atom and cartesian coordinate, 2 and 4 are
        supported. Default is 2 which will displace each atom +delta and
        -delta for each cartesian coordinate.

    Example:

    >>> from ase import Atoms
    >>> from ase.calculators.emt import EMT
    >>> from ase.optimize import BFGS
    >>> from ase.vibrations import Vibrations
    >>> n2 = Atoms('N2', [(0, 0, 0), (0, 0, 1.1)],
    ...            calculator=EMT())
    >>> BFGS(n2).run(fmax=0.01)
    BFGS:   0  16:01:21        0.440339       3.2518
    BFGS:   1  16:01:21        0.271928       0.8211
    BFGS:   2  16:01:21        0.263278       0.1994
    BFGS:   3  16:01:21        0.262777       0.0088
    >>> vib = Vibrations(n2)
    >>> vib.run()
    Writing vib.eq.pckl
    Writing vib.0x-.pckl
    Writing vib.0x+.pckl
    Writing vib.0y-.pckl
    Writing vib.0y+.pckl
    Writing vib.0z-.pckl
    Writing vib.0z+.pckl
    Writing vib.1x-.pckl
    Writing vib.1x+.pckl
    Writing vib.1y-.pckl
    Writing vib.1y+.pckl
    Writing vib.1z-.pckl
    Writing vib.1z+.pckl
    >>> vib.summary()
    ---------------------
    #    meV     cm^-1
    ---------------------
    0    0.0       0.0
    1    0.0       0.0
    2    0.0       0.0
    3    2.5      20.4
    4    2.5      20.4
    5  152.6    1230.8
    ---------------------
    Zero-point energy: 0.079 eV
    >>> vib.write_mode(-1)  # write last mode to trajectory file

    """

    def __init__(self, atoms, indices=None, name='vib', delta=0.01, nfree=2):
        assert nfree in [2, 4]
        self.atoms = atoms
        self.calc = atoms.calc
        if indices is None:
            indices = range(len(atoms))
        self.indices = np.asarray(indices)
        self.name = name
        self.delta = delta
        self.nfree = nfree
        self.H = None
        self.ir = None
        self.ram = None

    def run(self):
        """Run the vibration calculations.

        This will calculate the forces for 6 displacements per atom +/-x,
        +/-y, +/-z. Only those calculations that are not already done will be
        started. Be aware that an interrupted calculation may produce an empty
        file (ending with .pckl), which must be deleted before restarting the
        job. Otherwise the forces will not be calculated for that
        displacement.

        Note that the calculations for the different displacements can be done
        simultaneously by several independent processes. This feature relies
        on the existence of files and the subsequent creation of the file in
        case it is not found.

        If the program you want to use does not have a calculator in ASE, use
        ``iterdisplace`` to get all displaced structures and calculate the
        forces on your own.
        """

        if op.isfile(self.name + '.all.pckl'):
            raise RuntimeError(
                'Cannot run calculation. ' +
                self.name + '.all.pckl must be removed or split in order ' +
                'to have only one sort of data structure at a time.')
        for dispName, atoms in self.iterdisplace(inplace=True):
            filename = dispName + '.pckl'
            fd = opencew(filename)
            if fd is not None:
                self.calculate(atoms, filename, fd)

    def iterdisplace(self, inplace=False):
        """Yield name and atoms object for initial and displaced structures.

        Use this to export the structures for each single-point calculation
        to an external program instead of using ``run()``. Then save the
        calculated gradients to <name>.pckl and continue using this instance.
        """
        atoms = self.atoms if inplace else self.atoms.copy()
        yield self.name + '.eq', atoms
        for dispName, a, i, disp in self.displacements():
            if not inplace:
                atoms = self.atoms.copy()
            pos0 = atoms.positions[a, i]
            atoms.positions[a, i] += disp
            yield dispName, atoms
            if inplace:
                atoms.positions[a, i] = pos0

    def iterimages(self):
        """Yield initial and displaced structures."""
        for name, atoms in self.iterdisplace():
            yield atoms

    def displacements(self):
        for a in self.indices:
            for i in range(3):
                for sign in [-1, 1]:
                    for ndis in range(1, self.nfree // 2 + 1):
                        disp_name = ('%s.%d%s%s' %
                                     (self.name, a, 'xyz'[i],
                                      ndis * ' +-'[sign]))
                        disp = ndis * sign * self.delta
                        yield disp_name, a, i, disp

    def calculate(self, atoms, filename, fd):
        forces = self.calc.get_forces(atoms)
        if self.ir:
            dipole = self.calc.get_dipole_moment(atoms)
        if self.ram:
            freq, noninPol, pol = self.get_polarizability()
        if world.rank == 0:
            if self.ir and self.ram:
                pickle.dump([forces, dipole, freq, noninPol, pol],
                            fd, protocol=2)
                sys.stdout.write(
                    'Writing %s, dipole moment = (%.6f %.6f %.6f)\n' %
                    (filename, dipole[0], dipole[1], dipole[2]))
            elif self.ir and not self.ram:
                pickle.dump([forces, dipole], fd, protocol=2)
                sys.stdout.write(
                    'Writing %s, dipole moment = (%.6f %.6f %.6f)\n' %
                    (filename, dipole[0], dipole[1], dipole[2]))
            else:
                pickle.dump(forces, fd, protocol=2)
                sys.stdout.write('Writing %s\n' % filename)
            fd.close()
        sys.stdout.flush()

    def clean(self, empty_files=False, combined=True):
        """Remove pickle-files.

        Use empty_files=True to remove only empty files and
        combined=False to not remove the combined file.

        """

        if world.rank != 0:
            return 0

        n = 0
        filenames = [self.name + '.eq.pckl']
        if combined:
            filenames.append(self.name + '.all.pckl')
        for dispName, a, i, disp in self.displacements():
            filename = dispName + '.pckl'
            filenames.append(filename)

        for name in filenames:
            if op.isfile(name):
                if not empty_files or op.getsize(name) == 0:
                    os.remove(name)
                    n += 1
        return n

    def combine(self):
        """Combine pickle-files to one file ending with '.all.pckl'.

        The other pickle-files will be removed in order to have only one sort
        of data structure at a time.

        """
        if world.rank != 0:
            return 0
        filenames = [self.name + '.eq.pckl']
        for dispName, a, i, disp in self.displacements():
            filename = dispName + '.pckl'
            filenames.append(filename)
        combined_data = {}
        for name in filenames:
            if not op.isfile(name) or op.getsize(name) == 0:
                raise RuntimeError('Calculation is not complete. ' +
                                   name + ' is missing or empty.')
            with open(name, 'rb') as fl:
                f = pickleload(fl)
            combined_data.update({op.basename(name): f})
        filename = self.name + '.all.pckl'
        fd = opencew(filename)
        if fd is None:
            raise RuntimeError(
                'Cannot write file ' + filename +
                '. Remove old file if it exists.')
        else:
            pickle.dump(combined_data, fd, protocol=2)
            fd.close()
        return self.clean(combined=False)

    def split(self):
        """Split combined pickle-file.

        The combined pickle-file will be removed in order to have only one
        sort of data structure at a time.

        """
        if world.rank != 0:
            return 0
        combined_name = self.name + '.all.pckl'
        if not op.isfile(combined_name):
            raise RuntimeError('Cannot find combined file: ' +
                               combined_name + '.')
        with open(combined_name, 'rb') as fl:
            combined_data = pickleload(fl)
        filenames = [self.name + '.eq.pckl']
        for dispName, a, i, disp in self.displacements():
            filename = dispName + '.pckl'
            filenames.append(filename)
            if op.isfile(filename):
                raise RuntimeError(
                    'Cannot split. File ' + filename + 'already exists.')
        for name in filenames:
            fd = opencew(name)
            try:
                pickle.dump(combined_data[op.basename(name)], fd, protocol=2)
            except KeyError:
                pickle.dump(combined_data[name], fd, protocol=2)  # Old version
            fd.close()
        os.remove(combined_name)
        return 1  # One file removed

    def read(self, method='standard', direction='central'):
        self.method = method.lower()
        self.direction = direction.lower()
        assert self.method in ['standard', 'frederiksen']
        assert self.direction in ['central', 'forward', 'backward']

        def load(fname, combined_data=None):
            if combined_data is None:
                with open(fname, 'rb') as fl:
                    f = pickleload(fl)
            else:
                try:
                    f = combined_data[op.basename(fname)]
                except KeyError:
                    f = combined_data[fname]  # Old version
            if not hasattr(f, 'shape') and not hasattr(f, 'keys'):
                # output from InfraRed
                return f[0]
            return f

        n = 3 * len(self.indices)
        H = np.empty((n, n))
        r = 0
        if op.isfile(self.name + '.all.pckl'):
            # Open the combined pickle-file
            combined_data = load(self.name + '.all.pckl')
        else:
            combined_data = None
        if direction != 'central':
            feq = load(self.name + '.eq.pckl', combined_data)
        for a in self.indices:
            for i in 'xyz':
                name = '%s.%d%s' % (self.name, a, i)
                fminus = load(name + '-.pckl', combined_data)
                fplus = load(name + '+.pckl', combined_data)
                if self.method == 'frederiksen':
                    fminus[a] -= fminus.sum(0)
                    fplus[a] -= fplus.sum(0)
                if self.nfree == 4:
                    fminusminus = load(name + '--.pckl', combined_data)
                    fplusplus = load(name + '++.pckl', combined_data)
                    if self.method == 'frederiksen':
                        fminusminus[a] -= fminusminus.sum(0)
                        fplusplus[a] -= fplusplus.sum(0)
                if self.direction == 'central':
                    if self.nfree == 2:
                        H[r] = .5 * (fminus - fplus)[self.indices].ravel()
                    else:
                        H[r] = H[r] = (-fminusminus +
                                       8 * fminus -
                                       8 * fplus +
                                       fplusplus)[self.indices].ravel() / 12.0
                elif self.direction == 'forward':
                    H[r] = (feq - fplus)[self.indices].ravel()
                else:
                    assert self.direction == 'backward'
                    H[r] = (fminus - feq)[self.indices].ravel()
                H[r] /= 2 * self.delta
                r += 1
        H += H.copy().T
        self.H = H
        m = self.atoms.get_masses()
        if 0 in [m[index] for index in self.indices]:
            raise RuntimeError('Zero mass encountered in one or more of '
                               'the vibrated atoms. Use Atoms.set_masses()'
                               ' to set all masses to non-zero values.')

        self.im = np.repeat(m[self.indices]**-0.5, 3)
        omega2, modes = np.linalg.eigh(self.im[:, None] * H * self.im)
        self.modes = modes.T.copy()

        # Conversion factor:
        s = units._hbar * 1e10 / sqrt(units._e * units._amu)
        self.hnu = s * omega2.astype(complex)**0.5

    def get_vibrations(self, method='standard', direction='central', **kw):
        """Get vibrations as VibrationsData object"""
        if (self.H is None or method.lower() != self.method or
            direction.lower() != self.direction):
            self.read(method, direction, **kw)

        return VibrationsData.from_2d(self.atoms, self.H, indices=self.indices)

    def get_energies(self, method='standard', direction='central', **kw):
        """Get vibration energies in eV."""

        if (self.H is None or method.lower() != self.method or
            direction.lower() != self.direction):
            self.read(method, direction, **kw)
        return self.hnu

    def get_frequencies(self, method='standard', direction='central'):
        """Get vibration frequencies in cm^-1."""

        s = 1. / units.invcm
        return s * self.get_energies(method, direction)

    def summary(self, method='standard', direction='central', freq=None,
                log=sys.stdout):
        """Print a summary of the vibrational frequencies.

        Parameters:

        method : string
            Can be 'standard'(default) or 'Frederiksen'.
        direction: string
            Direction for finite differences. Can be one of 'central'
            (default), 'forward', 'backward'.
        freq : numpy array
            Optional. Can be used to create a summary on a set of known
            frequencies.
        log : if specified, write output to a different location than
            stdout. Can be an object with a write() method or the name of a
            file to create.
        """

        if isinstance(log, str):
            log = paropen(log, 'a')
        write = log.write

        s = 0.01 * units._e / units._c / units._hplanck
        if freq is not None:
            hnu = freq / s
        else:
            hnu = self.get_energies(method, direction)
        write('---------------------\n')
        write('  #    meV     cm^-1\n')
        write('---------------------\n')
        for n, e in enumerate(hnu):
            if e.imag != 0:
                c = 'i'
                e = e.imag
            else:
                c = ''
                e = e.real
            write('%3d %6.1f%1s  %7.1f%s\n' % (n, 1000 * e, c, s * e, c))
        write('---------------------\n')
        write('Zero-point energy: %.3f eV\n' %
              self.get_zero_point_energy(freq=freq))

    def get_zero_point_energy(self, freq=None):
        if freq is None:
            return 0.5 * self.hnu.real.sum()
        else:
            s = 0.01 * units._e / units._c / units._hplanck
            return 0.5 * freq.real.sum() / s

    def get_mode(self, n):
        """Get mode number ."""
        mode = np.zeros((len(self.atoms), 3))
        mode[self.indices] = (self.modes[n] * self.im).reshape((-1, 3))

        return mode

    def write_mode(self, n=None, kT=units.kB * 300, nimages=30):
        """Write mode number n to trajectory file. If n is not specified,
        writes all non-zero modes."""
        if n is None:
            for index, energy in enumerate(self.get_energies()):
                if abs(energy) > 1e-5:
                    self.write_mode(n=index, kT=kT, nimages=nimages)
            return
        mode = self.get_mode(n) * sqrt(kT / abs(self.hnu[n]))
        p = self.atoms.positions.copy()
        n %= 3 * len(self.indices)

        traj = ase.io.Trajectory('%s.%d.traj' % (self.name, n), 'w')
        calc = self.atoms.calc
        self.atoms.calc = None

        for x in np.linspace(0, 2 * pi, nimages, endpoint=False):
            self.atoms.set_positions(p + sin(x) * mode)
            traj.write(self.atoms)
        self.atoms.set_positions(p)
        self.atoms.calc = calc
        traj.close()

    def show_as_force(self, n, scale=0.2, show=True):
        mode = self.get_mode(n) * len(self.hnu) * scale
        calc = SinglePointCalculator(self.atoms, forces=mode)

        self.atoms.calc = calc
        if show:
            self.atoms.edit()

    def write_jmol(self):
        """Writes file for viewing of the modes with jmol."""

        fd = open(self.name + '.xyz', 'w')
        symbols = self.atoms.get_chemical_symbols()
        f = self.get_frequencies()
        for n in range(3 * len(self.indices)):
            fd.write('%6d\n' % len(self.atoms))
            if f[n].imag != 0:
                c = 'i'
                f_n = float(f[n].imag)
            else:
                f_n = float(f[n].real)
                c = ' '
            fd.write('Mode #%d, f = %.1f%s cm^-1' % (n, f_n, c))
            if self.ir:
                fd.write(', I = %.4f (D/Å)^2 amu^-1.\n' % self.intensities[n])
            else:
                fd.write('.\n')
            mode = self.get_mode(n)
            for i, pos in enumerate(self.atoms.positions):
                fd.write('%2s %12.5f %12.5f %12.5f %12.5f %12.5f %12.5f\n' %
                         (symbols[i], pos[0], pos[1], pos[2],
                          mode[i, 0], mode[i, 1], mode[i, 2]))
        fd.close()

    def fold(self, frequencies, intensities,
             start=800.0, end=4000.0, npts=None, width=4.0,
             type='Gaussian', normalize=False):
        """Fold frequencies and intensities within the given range
        and folding method (Gaussian/Lorentzian).
        The energy unit is cm^-1.
        normalize=True ensures the integral over the peaks to give the
        intensity.
        """

        lctype = type.lower()
        assert lctype in ['gaussian', 'lorentzian']
        if not npts:
            npts = int((end - start) / width * 10 + 1)
        prefactor = 1
        if lctype == 'lorentzian':
            intensities = intensities * width * pi / 2.
            if normalize:
                prefactor = 2. / width / pi
        else:
            sigma = width / 2. / sqrt(2. * log(2.))
            if normalize:
                prefactor = 1. / sigma / sqrt(2 * pi)

        # Make array with spectrum data
        spectrum = np.empty(npts)
        energies = np.linspace(start, end, npts)
        for i, energy in enumerate(energies):
            energies[i] = energy
            if lctype == 'lorentzian':
                spectrum[i] = (intensities * 0.5 * width / pi /
                               ((frequencies - energy)**2 +
                                0.25 * width**2)).sum()
            else:
                spectrum[i] = (intensities *
                               np.exp(-(frequencies - energy)**2 /
                                      2. / sigma**2)).sum()
        return [energies, prefactor * spectrum]

    def write_dos(self, out='vib-dos.dat', start=800, end=4000,
                  npts=None, width=10,
                  type='Gaussian', method='standard', direction='central'):
        """Write out the vibrational density of states to file.

        First column is the wavenumber in cm^-1, the second column the
        folded vibrational density of states.
        Start and end points, and width of the Gaussian/Lorentzian
        should be given in cm^-1."""
        frequencies = self.get_frequencies(method, direction).real
        intensities = np.ones(len(frequencies))
        energies, spectrum = self.fold(frequencies, intensities,
                                       start, end, npts, width, type)

        # Write out spectrum in file.
        outdata = np.empty([len(energies), 2])
        outdata.T[0] = energies
        outdata.T[1] = spectrum
        fd = open(out, 'w')
        fd.write('# %s folded, width=%g cm^-1\n' % (type.title(), width))
        fd.write('# [cm^-1] arbitrary\n')
        for row in outdata:
            fd.write('%.3f  %15.5e\n' %
                     (row[0], row[1]))
        fd.close()
