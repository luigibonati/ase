import os
import re
from pathlib import Path
import numpy as np

from ase.calculators.octopus import parse_input_file, kwargs2atoms
from ase.utils import basestring
from ase.calculators.octopus import generate_input, process_special_kwargs
from ase.calculators.singlepoint import (SinglePointDFTCalculator,
                                         arrays_to_kpoints)
from ase.units import eV, Hartree, Angstrom, Bohr



class OctopusReader:
    def read_with_atoms(self, atoms):
        results = {}
        with Path('static/info').open() as fd:
            results.update(read_static_info(fd))

        kpts = None
        eigpath = Path('static/eigenvalues')
        if eigpath.exists():
            # If the eigenvalues file exists, we get the eigs/occs
            # from that one.  This probably means someone ran Octopus
            # in 'unocc' mode to get eigenvalues (e.g. for band
            # structures), and the values in static/info will be the
            # old (selfconsistent) ones.
            with eigpath.open() as fd:
                ibz_kpts, eigs, occs = read_eigenvalues_file(fd)

            # This quantity might also have come from previous reading
            results.update(ibz_k_points=ibz_kpts)
            # XXX octopus has symmetry reduction now!!  It will definitely
            # be possible to have weights different from one.
            # We need to support that.
            kpt_weights = np.ones(len(kpts))  # XXX ?  Or 1 / len(kpts) ?
            kpts = arrays_to_kpoints(eigs, occs, weights)

        # Pop the stuff which SinglePointDFTCalculator won't like:
        for name in ['k_point_weights', 'nspins', 'nkpts',
                     'nbands', 'eigenvalues', 'occupations']:
            results.pop(name, None)

        results['ibzkpts'] = results.pop('ibz_k_points')

        calc = SinglePointDFTCalculator(atoms, **results)
        calc.kpts = kpts
        return calc


def write_octopus_in(fd, atoms, properties=None, parameters=None):

    if properties is None:
        properties = ['energy']
    if parameters is None:
        parameters = {}

    txt = generate_input(atoms,
                         process_special_kwargs(atoms, parameters),
                         normalized2pretty={})
    fd.write(txt)


def read_octopus_in(fileobj, get_kwargs=False):
    if isinstance(fileobj, basestring):  # This could be solved with decorators...
        fileobj = open(fileobj)

    kwargs = parse_input_file(fileobj)

    # input files may contain internal references to other files such
    # as xyz or xsf.  We need to know the directory where the file
    # resides in order to locate those.  If fileobj is a real file
    # object, it contains the path and we can use it.  Else assume
    # pwd.
    #
    # Maybe this is ugly; maybe it can lead to strange bugs if someone
    # wants a non-standard file-like type.  But it's probably better than
    # failing 'ase gui somedir/inp'
    try:
        fname = fileobj.name
    except AttributeError:
        directory = None
    else:
        directory = os.path.split(fname)[0]

    atoms, remaining_kwargs = kwargs2atoms(kwargs, directory=directory)
    if get_kwargs:
        return atoms, remaining_kwargs
    else:
        return atoms



def read_static_info_stress(fd):
    stress_cv = np.empty((3, 3))

    headers = next(fd)
    assert headers.strip().startswith('T_{ij}')
    for i in range(3):
        line = next(fd)
        tokens = line.split()
        vec = np.array(tokens[1:4]).astype(float)
        stress_cv[i] = vec
    return stress_cv

def read_static_info_kpoints(fd):
    for line in fd:
        if line.startswith('List of k-points'):
            break

    tokens = next(fd).split()
    assert tokens == ['ik', 'k_x', 'k_y', 'k_z', 'Weight']
    bar = next(fd)
    assert bar.startswith('---')

    kpts = []
    weights = []

    for line in fd:
        # Format:        index   kx      ky      kz     weight
        m = re.match(r'\s*\d+\s*(\S+)\s*(\S+)\s*(\S+)\s*(\S+)', line)
        if m is None:
            break
        kxyz = m.group(1, 2, 3)
        weight = m.group(4)
        kpts.append(kxyz)
        weights.append(weight)

    ibz_k_points = np.array(kpts, float)
    k_point_weights = np.array(weights, float)
    return dict(ibz_k_points=ibz_k_points, k_point_weights=k_point_weights)


def read_static_info_eigenvalues(fd, energy_unit):

    values_sknx = {}

    nbands = 0
    fermilevel = None
    for line in fd:
        line = line.strip()
        if line.startswith('#'):
            continue
        if not line[:1].isdigit():
            m = re.match(r'Fermi energy\s*=\s*(\S+)', line)
            if m is not None:
                fermilevel = float(m.group(1)) * energy_unit
            break

        tokens = line.split()
        nbands = max(nbands, int(tokens[0]))
        energy = float(tokens[2]) * energy_unit
        occupation = float(tokens[3])
        values_sknx.setdefault(tokens[1], []).append((energy, occupation))

    nspins = len(values_sknx)
    if nspins == 1:
        val = [values_sknx['--']]
    else:
        val = [values_sknx['up'], values_sknx['dn']]
    val = np.array(val, float)
    nkpts, remainder = divmod(len(val[0]), nbands)
    assert remainder == 0

    eps_skn = val[:, :, 0].reshape(nspins, nkpts, nbands)
    occ_skn = val[:, :, 1].reshape(nspins, nkpts, nbands)
    eps_skn = eps_skn.transpose(1, 0, 2).copy()
    occ_skn = occ_skn.transpose(1, 0, 2).copy()
    assert eps_skn.flags.contiguous
    d = dict(nspins=nspins,
             nkpts=nkpts,
             nbands=nbands,
             eigenvalues=eps_skn,
             occupations=occ_skn)
    if fermilevel is not None:
        d.update(efermi=fermilevel)
    return d

def read_static_info_energy(fd, energy_unit):
    def get(name):
        for line in fd:
            if line.strip().startswith(name):
                return float(line.split('=')[-1].strip()) * energy_unit
    return dict(energy=get('Total'), free_energy=get('Free'))


def read_static_info(fd):
    results = {}

    def get_energy_unit(line):  # Convert "title [unit]": ---> unit
        return {'[eV]': eV, '[H]': Hartree}[line.split()[1].rstrip(':')]

    for line in fd:
        if line.strip('*').strip().startswith('Brillouin zone'):
            results.update(read_static_info_kpoints(fd))
        elif line.startswith('Eigenvalues ['):
            unit = get_energy_unit(line)
            results.update(read_static_info_eigenvalues(fd, unit))
        elif line.startswith('Energy ['):
            unit = get_energy_unit(line)
            results.update(read_static_info_energy(fd, unit))
        elif line.startswith('Stress tensor'):
            assert line.split()[-1] == '[H/b^3]'
            stress = read_static_info_stress(fd)
            stress *= Hartree / Bohr**3
            results.update(stress=stress)
        elif line.startswith('Total Magnetic Moment'):
            if 0:
                line = next(fd)
                values = line.split()
                results['magmom'] = float(values[-1])

                line = next(fd)
                assert line.startswith('Local Magnetic Moments')
                line = next(fd)
                assert line.split() == ['Ion', 'mz']
                # Reading  Local Magnetic Moments
                mag_moment = []
                for line in fd:
                    if line == '\n':
                        break  # there is no more thing to search for
                    line = line.replace('\n', ' ')
                    values = line.split()
                    mag_moment.append(float(values[-1]))

                results['magmoms'] = np.array(mag_moment)
        elif line.startswith('Dipole'):
            assert line.split()[-1] == '[Debye]'
            dipole = [float(next(fd).split()[-1]) for i in range(3)]
            results['dipole'] = np.array(dipole) * Debye
        elif line.startswith('Forces'):
            forceunitspec = line.split()[-1]
            forceunit = {'[eV/A]': eV / Angstrom,
                         '[H/b]': Hartree / Bohr}[forceunitspec]
            forces = []
            line = next(fd)
            assert line.strip().startswith('Ion')
            for line in fd:
                if line.strip().startswith('---'):
                    break
                tokens = line.split()[-3:]
                forces.append([float(f) for f in tokens])
            results['forces'] = np.array(forces) * forceunit
        elif line.startswith('Fermi'):
            tokens = line.split()
            unit = {'eV': eV, 'H': Hartree}[tokens[-1]]
            eFermi = float(tokens[-2]) * unit
            results['efermi'] = eFermi

    if 'ibz_k_points' not in results:
        results['ibz_k_points'] = np.zeros((1, 3))
        results['k_point_weights'] = np.ones(1)
    if 0: #'efermi' not in results:
        # Find HOMO level.  Note: This could be a very bad
        # implementation with fractional occupations if the Fermi
        # level was not found otherwise.
        all_energies = results['eigenvalues'].ravel()
        all_occupations = results['occupations'].ravel()
        args = np.argsort(all_energies)
        for arg in args[::-1]:
            if all_occupations[arg] > 0.1:
                break
        eFermi = all_energies[arg]
        results['efermi'] = eFermi

    return results
