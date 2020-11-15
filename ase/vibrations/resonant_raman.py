"""Resonant Raman intensities"""

import pickle
import os
import sys

import numpy as np

import ase.units as u
from ase.parallel import world, paropen, parprint
from ase.vibrations import Vibrations
from ase.vibrations.raman import Raman, RamanCalculatorBase


class ResonantRamanCalculator(RamanCalculatorBase, Vibrations):
    """Base class for resonant Raman calculators using finite differences.
    """
    def __init__(self, atoms, ExcitationsCalculator, *args,
                 exkwargs={}, exext='.ex.gz', overlap=False,
                 **kwargs):
        """
        Parameters
        ----------
        atoms: Atoms
            The Atoms object
        ExcitationsCalculator: object
            Calculator for excited states
        exkwargs: dict
            Arguments given to the ExcitationsCalculator object
        exext: string
            Extension for filenames of Excitation lists (results of
            the ExcitationsCalculator).
        overlap : function or False
            Function to calculate overlaps between excitation at
            equilibrium and at a displaced position. Calculators are
            given as first and second argument, respectively.

        Example
        -------

        >>> from ase.calculators.h2morse import (H2Morse,
        ...                                      H2MorseExcitedStatesCalculator)
        >>> from ase.vibrations.resonant_raman import ResonantRamanCalculator
        >>>
        >>> atoms = H2Morse()
        >>> rmc = ResonantRamanCalculator(atoms, H2MorseExcitedStatesCalculator)
        >>> rmc.run()

        This produces all necessary data for further analysis.
        """
        self.exobj = ExcitationsCalculator
        self.exkwargs = exkwargs
        self.overlap = overlap
        super().__init__(atoms, *args, exext=exext, **kwargs)

    def calculate(self, atoms, filename, fd):
        """Call ground and excited state calculation"""
        assert(atoms == self.atoms)  # XXX action required
        self.timer.start('Ground state')
        forces = self.atoms.get_forces()
        if world.rank == 0:
            pickle.dump(forces, fd, protocol=2)

        if self.overlap:
            """Overlap is determined as

            ov_ij = int dr displaced*_i(r) eqilibrium_j(r)
            """
            self.timer.start('Overlap')
            ov_nn = self.overlap(self.atoms.calc,
                                 self.eq_calculator)
            if world.rank == 0:
                np.save(filename + '.ov', ov_nn)
            self.timer.stop('Overlap')
        self.timer.stop('Ground state')

        self.timer.start('Excitations')
        basename, _ = os.path.splitext(filename)
        excalc = self.exobj(**self.exkwargs)
        exlist = excalc.calculate(self.atoms)
        exlist.write(basename + self.exext)
        self.timer.stop('Excitations')

    def run(self):
        if self.overlap:
            # XXXX stupid way to make a copy
            self.atoms.get_potential_energy()
            self.eq_calculator = self.atoms.calc
            fname = self.exname + '.eq.gpw'
            self.eq_calculator.write(fname, 'all')
            self.eq_calculator = self.eq_calculator.__class__.read(fname)
            try:
                # XXX GPAW specific
                self.eq_calculator.converge_wave_functions()
            except AttributeError:
                pass
        Vibrations.run(self)


class ResonantRaman(Raman):
    """Base Class for resonant Raman intensities using finite differences.
    """
    def __init__(self, atoms, Excitations, *args,
                 observation={'geometry': '-Z(XX)Z'},
                 form='v',         # form of the dipole operator
                 exkwargs={},      # kwargs to be passed to Excitations
                 exext='.ex.gz',   # extension for Excitation names
                 overlap=False,
                 minoverlap=0.02,
                 minrep=0.8,
                 comm=world,
                 **kwargs):
        """
        Parameters
        ----------
        atoms: ase Atoms object
        Excitations: class
            Type of the excitation list object. The class object is
            initialized as::

                Excitations(atoms.calc)

            or by reading form a file as::

                Excitations('filename', **exkwargs)

            The file is written by calling the method
            Excitations.write('filename').

            Excitations should work like a list of ex obejects, where:
                ex.get_dipole_me(form='v'):
                    gives the velocity form dipole matrix element in
                    units |e| * Angstrom
                ex.energy:
                    is the transition energy in Hartrees
        approximation: string
            Level of approximation used.
        observation: dict
            Polarization settings
        form: string
            Form of the dipole operator, 'v' for velocity form (default)
            and 'r' for length form.
        overlap: bool or function
            Use wavefunction overlaps.
        minoverlap: float ord dict
            Minimal absolute overlap to consider. Defaults to 0.02 to avoid
            numerical garbage.
        minrep: float
            Minimal representation to consider derivative, defaults to 0.8
        """
        kwargs['exext'] = exext
        Raman.__init__(self, atoms, *args, **kwargs)
        assert(self.vibrations.nfree == 2)

        self.exobj = Excitations
        self.exkwargs = exkwargs
        self.observation = observation
        self.dipole_form = form

        self.overlap = overlap
        if not isinstance(minoverlap, dict):
            # assume it's a number
            self.minoverlap = {'orbitals': minoverlap,
                               'excitations': minoverlap}
        else:
            self.minoverlap = minoverlap
        self.minrep = minrep

    def get_absolute_intensities(self, omega, gamma=0.1, delta=0, **kwargs):
        """Absolute Raman intensity or Raman scattering factor

        Parameter
        ---------
        omega: float
           incoming laser energy, unit eV
        gamma: float
           width (imaginary energy), unit eV
        delta: float
           pre-factor for asymmetric anisotropy, default 0

        References
        ----------
        Porezag and Pederson, PRB 54 (1996) 7830-7836 (delta=0)
        Baiardi and Barone, JCTC 11 (2015) 3267-3280 (delta=5)

        Returns
        -------
        raman intensity, unit Ang**4/amu
        """
        alpha2_r, gamma2_r, delta2_r = self._invariants(
            self.electronic_me_Qcc(omega, gamma, **kwargs))
        return 45 * alpha2_r + delta * delta2_r + 7 * gamma2_r

    @property
    def approximation(self):
        return self._approx

    @approximation.setter
    def approximation(self, value):
        self.set_approximation(value)

    def read_excitations(self):
        """Read all finite difference excitations and select matching."""
        if self.overlap:
            return self.read_excitations_overlap()

        self.timer.start('read excitations')
        self.timer.start('really read')
        self.log('reading ' + self.exname + '.eq' + self.exext)
        ex0_object = self.exobj.read(self.exname + '.eq' + self.exext,
                                     **self.exkwargs)
        eu = ex0_object.energy_to_eV_scale
        self.timer.stop('really read')
        self.timer.start('index')
        matching = frozenset(ex0_object)
        self.timer.stop('index')

        def append(lst, exname, matching):
            self.timer.start('really read')
            self.log('reading ' + exname, end=' ')
            exo = self.exobj.read(exname, **self.exkwargs)
            lst.append(exo)
            self.timer.stop('really read')
            self.timer.start('index')
            matching = matching.intersection(exo)
            self.log('len={0}, matching={1}'.format(len(exo),
                                                    len(matching)), pre='')
            self.timer.stop('index')
            return matching

        exm_object_list = []
        exp_object_list = []
        for a, i in zip(self.myindices, self.myxyz):
            name = '%s.%d%s' % (self.exname, a, i)
            matching = append(exm_object_list,
                              name + '-' + self.exext, matching)
            matching = append(exp_object_list,
                              name + '+' + self.exext, matching)
        self.ndof = 3 * len(self.indices)
        self.nex = len(matching)
        self.timer.stop('read excitations')

        self.timer.start('select')

        def select(exl, matching):
            mlst = [ex for ex in exl if ex in matching]
            assert(len(mlst) == len(matching))
            return mlst
        ex0 = select(ex0_object, matching)
        exm = []
        exp = []
        r = 0
        for a, i in zip(self.myindices, self.myxyz):
            exm.append(select(exm_object_list[r], matching))
            exp.append(select(exp_object_list[r], matching))
            r += 1
        self.timer.stop('select')

        self.timer.start('me and energy')

        self.ex0E_p = np.array([ex.energy * eu for ex in ex0])
        self.ex0m_pc = (np.array(
            [ex.get_dipole_me(form=self.dipole_form) for ex in ex0]) *
            u.Bohr)
        exmE_rp = []
        expE_rp = []
        exF_rp = []
        exmm_rpc = []
        expm_rpc = []
        r = 0
        for a, i in zip(self.myindices, self.myxyz):
            exmE_rp.append([em.energy for em in exm[r]])
            expE_rp.append([ep.energy for ep in exp[r]])
            exF_rp.append(
                [(em.energy - ep.energy)
                 for ep, em in zip(exp[r], exm[r])])
            exmm_rpc.append(
                [ex.get_dipole_me(form=self.dipole_form)
                 for ex in exm[r]])
            expm_rpc.append(
                [ex.get_dipole_me(form=self.dipole_form)
                 for ex in exp[r]])
            r += 1
        # indicees: r=coordinate, p=excitation
        # energies in eV
        self.exmE_rp = np.array(exmE_rp) * eu
        self.expE_rp = np.array(expE_rp) * eu
        # forces in eV / Angstrom
        self.exF_rp = np.array(exF_rp) * eu / 2 / self.delta
        # matrix elements in e * Angstrom
        self.exmm_rpc = np.array(exmm_rpc) * u.Bohr
        self.expm_rpc = np.array(expm_rpc) * u.Bohr

        self.timer.stop('me and energy')

    def read_excitations_overlap(self):
        """Read all finite difference excitations and wf overlaps.

        We assume that the wave function overlaps are determined as

        ov_ij = int dr displaced*_i(r) eqilibrium_j(r)
        """
        self.timer.start('read excitations')
        self.timer.start('read+rotate')
        self.log('reading ' + self.exname + '.eq' + self.exext)
        ex0 = self.exobj.read(self.exname + '.eq' + self.exext,
                              **self.exkwargs)
        eu = ex0.energy_to_eV_scale
        rep0_p = np.ones((len(ex0)), dtype=float)

        def load(name, pm, rep0_p):
            self.log('reading ' + name + pm + self.exext)
            ex_p = self.exobj.read(name + pm + self.exext, **self.exkwargs)
            self.log('reading ' + name + pm + '.pckl.ov.npy')
            ov_nn = np.load(name + pm + '.pckl.ov.npy')
            # remove numerical garbage
            ov_nn = np.where(np.abs(ov_nn) > self.minoverlap['orbitals'],
                             ov_nn, 0)
            self.timer.start('ex overlap')
            ov_pp = ex_p.overlap(ov_nn, ex0)
            # remove numerical garbage
            ov_pp = np.where(np.abs(ov_pp) > self.minoverlap['excitations'],
                             ov_pp, 0)
            rep0_p *= (ov_pp.real**2 + ov_pp.imag**2).sum(axis=0)
            self.timer.stop('ex overlap')
            return ex_p, ov_pp

        def rotate(ex_p, ov_pp):
            e_p = np.array([ex.energy for ex in ex_p])
            m_pc = np.array(
                [ex.get_dipole_me(form=self.dipole_form) for ex in ex_p])
            r_pp = ov_pp.T
            return ((r_pp.real**2 + r_pp.imag**2).dot(e_p),
                    r_pp.dot(m_pc))

        exmE_rp = []
        expE_rp = []
        exF_rp = []
        exmm_rpc = []
        expm_rpc = []
        exdmdr_rpc = []
        for a, i in zip(self.myindices, self.myxyz):
            name = '%s.%d%s' % (self.exname, a, i)
            ex, ov = load(name, '-', rep0_p)
            exmE_p, exmm_pc = rotate(ex, ov)
            ex, ov = load(name, '+', rep0_p)
            expE_p, expm_pc = rotate(ex, ov)
            exmE_rp.append(exmE_p)
            expE_rp.append(expE_p)
            exF_rp.append(exmE_p - expE_p)
            exmm_rpc.append(exmm_pc)
            expm_rpc.append(expm_pc)
            exdmdr_rpc.append(expm_pc - exmm_pc)
        self.timer.stop('read+rotate')

        self.timer.start('me and energy')

        # select only excitations that are sufficiently represented
        self.comm.product(rep0_p)
        select = np.where(rep0_p > self.minrep)[0]

        self.ex0E_p = np.array([ex.energy * eu for ex in ex0])[select]
        self.ex0m_pc = (np.array(
            [ex.get_dipole_me(form=self.dipole_form)
             for ex in ex0])[select] * u.Bohr)

        if len(self.myr):
            # indicees: r=coordinate, p=excitation
            # energies in eV
            self.exmE_rp = np.array(exmE_rp)[:, select] * eu
            self.expE_rp = np.array(expE_rp)[:, select] * eu
            # forces in eV / Angstrom
            self.exF_rp = np.array(exF_rp)[:, select] * eu / 2 / self.delta
            # matrix elements in e * Angstrom
            self.exmm_rpc = np.array(exmm_rpc)[:, select, :] * u.Bohr
            self.expm_rpc = np.array(expm_rpc)[:, select, :] * u.Bohr
            # matrix element derivatives in e
            self.exdmdr_rpc = (np.array(exdmdr_rpc)[:, select, :] *
                               u.Bohr / 2 / self.delta)
        else:
            # did not read
            self.exmE_rp = self.expE_rp = self.exF_rp = np.empty((0))
            self.exmm_rpc = self.expm_rpc = self.exdmdr_rpc = np.empty((0))

        self.timer.stop('me and energy')
        self.timer.stop('read excitations')

    def read(self, *args, **kwargs):
        """Read data from a pre-performed calculation."""
        self.timer.start('read')
        self.timer.start('vibrations')
        self.vibrations.read(*args, **kwargs)
        self.timer.stop('vibrations')

        self.timer.start('excitations')
        self.init_parallel_read()
        if not hasattr(self, 'ex0E_p'):
            if self.overlap:
                self.read_excitations_overlap()
            else:
                self.read_excitations()
        self.timer.stop('excitations')

        self._already_read = True
        self.timer.stop('read')

    def get_cross_sections(self, omega, gamma):
        """Returns Raman cross sections for each vibration."""
        I_v = self.intensity(omega, gamma)
        pre = 1. / 16 / np.pi**2 / u._eps0**2 / u._c**4
        # frequency of scattered light
        omS_v = omega - self.om_v
        return pre * omega * omS_v**3 * I_v

    def get_spectrum(self, omega, gamma=0.1,
                     start=None, end=None, npts=None, width=20,
                     type='Gaussian', method='standard', direction='central',
                     intensity_unit='????', normalize=False):
        """Get resonant Raman spectrum.

        The method returns wavenumbers in cm^-1 with corresponding
        Raman cross section.
        Start and end point, and width of the Gaussian/Lorentzian should
        be given in cm^-1.
        """

        self.type = type.lower()
        assert self.type in ['gaussian', 'lorentzian']

        frequencies = self.get_frequencies(method, direction).real
        intensities = self.get_cross_sections(omega, gamma)
        if width is None:
            return [frequencies, intensities]

        if start is None:
            start = min(self.om_v) / u.invcm - 3 * width
        if end is None:
            end = max(self.om_v) / u.invcm + 3 * width

        if not npts:
            npts = int((end - start) / width * 10 + 1)

        prefactor = 1
        if self.type == 'lorentzian':
            intensities = intensities * width * np.pi / 2.
            if normalize:
                prefactor = 2. / width / np.pi
        else:
            sigma = width / 2. / np.sqrt(2. * np.log(2.))
            if normalize:
                prefactor = 1. / sigma / np.sqrt(2 * np.pi)
        # Make array with spectrum data
        spectrum = np.empty(npts)
        energies = np.linspace(start, end, npts)
        for i, energy in enumerate(energies):
            energies[i] = energy
            if self.type == 'lorentzian':
                spectrum[i] = (intensities * 0.5 * width / np.pi /
                               ((frequencies - energy)**2 +
                                0.25 * width**2)).sum()
            else:
                spectrum[i] = (intensities *
                               np.exp(-(frequencies - energy)**2 /
                                      2. / sigma**2)).sum()
        return [energies, prefactor * spectrum]

    def write_spectrum(self, omega, gamma,
                       out='resonant-raman-spectra.dat',
                       start=200, end=4000,
                       npts=None, width=10,
                       type='Gaussian', method='standard',
                       direction='central'):
        """Write out spectrum to file.

        Start and end
        point, and width of the Gaussian/Lorentzian should be given
        in cm^-1."""
        energies, spectrum = self.get_spectrum(omega, gamma,
                                               start, end, npts, width,
                                               type, method, direction)

        # Write out spectrum in file. First column is absolute intensities.
        outdata = np.empty([len(energies), 3])
        outdata.T[0] = energies
        outdata.T[1] = spectrum

        with paropen(out, 'w') as fd:
            fd.write('# Resonant Raman spectrum\n')
            if hasattr(self, '_approx'):
                fd.write('# approximation: {0}\n'.format(self._approx))
            for key in self.observation:
                fd.write('# {0}: {1}\n'.format(key, self.observation[key]))
            fd.write('# omega={0:g} eV, gamma={1:g} eV\n'
                     .format(omega, gamma))
            if width is not None:
                fd.write('# %s folded, width=%g cm^-1\n'
                         % (type.title(), width))
            fd.write('# [cm^-1]  [a.u.]\n')

            for row in outdata:
                fd.write('%.3f  %15.5g\n' %
                         (row[0], row[1]))

    def __del__(self):
        self.timer.write(self.txt)

    def summary(self, omega, gamma=0.1,
                method='standard', direction='central',
                log=sys.stdout):
        """Print summary for given omega [eV]"""
        self.read(method, direction)
        hnu = self.get_energies()
        intensities = self.get_absolute_intensities(omega, gamma)
        te = int(np.log10(intensities.max())) - 2
        scale = 10**(-te)
        if not te:
            ts = ''
        elif te > -2 and te < 3:
            ts = str(10**te)
        else:
            ts = '10^{0}'.format(te)

        if isinstance(log, str):
            log = paropen(log, 'a')

        parprint('-------------------------------------', file=log)
        parprint(' excitation at ' + str(omega) + ' eV', file=log)
        parprint(' gamma ' + str(gamma) + ' eV', file=log)
        parprint(' method:', self.vibrations.method, file=log)
        parprint(' approximation:', self.approximation, file=log)
        parprint(' Mode    Frequency        Intensity', file=log)
        parprint('  #    meV     cm^-1      [{0}A^4/amu]'.format(ts), file=log)
        parprint('-------------------------------------', file=log)
        for n, e in enumerate(hnu):
            if e.imag != 0:
                c = 'i'
                e = e.imag
            else:
                c = ' '
                e = e.real
            parprint('%3d %6.1f%s  %7.1f%s  %9.2f' %
                     (n, 1000 * e, c, e / u.invcm, c, intensities[n] * scale),
                     file=log)
        parprint('-------------------------------------', file=log)
        parprint('Zero-point energy: %.3f eV' %
                 self.vibrations.get_zero_point_energy(),
                 file=log)


class LrResonantRaman(ResonantRaman):
    """Resonant Raman for linear response

    Quick and dirty approach to enable loading of LrTDDFT calculations
    """
    def read_excitations(self):
        self.timer.start('read excitations')
        self.timer.start('really read')
        self.log('reading ' + self.exname + '.eq' + self.exext)
        ex0_object = self.exobj(self.exname + '.eq' + self.exext,
                                **self.exkwargs)
        eu = ex0_object.energy_to_eV_scale
        self.timer.stop('really read')
        self.timer.start('index')
        matching = frozenset(ex0_object.kss)
        self.timer.stop('index')

        def append(lst, exname, matching):
            self.timer.start('really read')
            self.log('reading ' + exname, end=' ')
            exo = self.exobj(exname, **self.exkwargs)
            lst.append(exo)
            self.timer.stop('really read')
            self.timer.start('index')
            matching = matching.intersection(exo.kss)
            self.log('len={0}, matching={1}'.format(len(exo.kss),
                                                    len(matching)), pre='')
            self.timer.stop('index')
            return matching

        exm_object_list = []
        exp_object_list = []
        for a in self.indices:
            for i in 'xyz':
                name = '%s.%d%s' % (self.exname, a, i)
                matching = append(exm_object_list,
                                  name + '-' + self.exext, matching)
                matching = append(exp_object_list,
                                  name + '+' + self.exext, matching)
        self.ndof = 3 * len(self.indices)
        self.timer.stop('read excitations')

        self.timer.start('select')

        def select(exl, matching):
            exl.diagonalize(**self.exkwargs)
            mlst = [ex for ex in exl]
#            mlst = [ex for ex in exl if ex in matching]
#            assert(len(mlst) == len(matching))
            return mlst
        ex0 = select(ex0_object, matching)
        self.nex = len(ex0)
        exm = []
        exp = []
        r = 0
        for a in self.indices:
            for i in 'xyz':
                exm.append(select(exm_object_list[r], matching))
                exp.append(select(exp_object_list[r], matching))
                r += 1
        self.timer.stop('select')

        self.timer.start('me and energy')

        self.ex0E_p = np.array([ex.energy * eu for ex in ex0])
#        self.exmE_p = np.array([ex.energy * eu for ex in exm])
#        self.expE_p = np.array([ex.energy * eu for ex in exp])
        self.ex0m_pc = (np.array(
            [ex.get_dipole_me(form=self.dipole_form) for ex in ex0]) *
            u.Bohr)
        self.exF_rp = []
        exmE_rp = []
        expE_rp = []
        exmm_rpc = []
        expm_rpc = []
        r = 0
        for a in self.indices:
            for i in 'xyz':
                exmE_rp.append([em.energy for em in exm[r]])
                expE_rp.append([ep.energy for ep in exp[r]])
                self.exF_rp.append(
                    [(em.energy - ep.energy)
                     for ep, em in zip(exp[r], exm[r])])
                exmm_rpc.append(
                    [ex.get_dipole_me(form=self.dipole_form) for ex in exm[r]])
                expm_rpc.append(
                    [ex.get_dipole_me(form=self.dipole_form) for ex in exp[r]])
                r += 1
        self.exmE_rp = np.array(exmE_rp) * eu
        self.expE_rp = np.array(expE_rp) * eu
        self.exF_rp = np.array(self.exF_rp) * eu / 2 / self.delta
        self.exmm_rpc = np.array(exmm_rpc) * u.Bohr
        self.expm_rpc = np.array(expm_rpc) * u.Bohr

        self.timer.stop('me and energy')
