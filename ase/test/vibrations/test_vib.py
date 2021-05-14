import os
import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal

from ase import units, Atoms
import ase.io
from ase.calculators.emt import EMT
from ase.optimize import QuasiNewton
from ase.vibrations import Vibrations, VibrationsData
from ase.thermochemistry import IdealGasThermo


class TestVibrationsClassic():
    """Tests for the ase.vibrations.Vibrations object

    This object is to be phased out in favour of separate calculating/loading
    and results/data objects. It therefore requires especially thorough
    testing to ensure there are no unexpected losses/changes in the process.

    """
    def setup(self):
        self.logfile = 'vibrations-log.txt'

    @pytest.fixture(scope="class")
    def n2_optimized(self):
        n2 = Atoms('N2',
                   positions=[(0, 0, 0), (0, 0, 1.1)],
                   calculator=EMT())
        QuasiNewton(n2).run(fmax=0.01)
        return n2

    @pytest.fixture
    def n2_emt(self, n2_optimized):
        atoms = n2_optimized.copy()
        atoms.calc = EMT()
        return atoms

    def test_consistency_with_vibrationsdata(self, testdir, n2_emt):
        atoms = n2_emt
        vib = Vibrations(atoms)
        vib.run()
        vib_data = vib.get_vibrations()

        assert_array_almost_equal(vib.get_energies(),
                                  vib_data.get_energies())

        # Compare the last mode as the others may be re-ordered by negligible
        # energy changes
        assert_array_almost_equal(vib.get_mode(5), vib_data.get_modes()[5])

    def test_pickle_manipulation(self, testdir, n2_emt):
        atoms = n2_emt
        vib = Vibrations(atoms, name='interrupt')
        vib.run()

        disp_file = 'interrupt/cache.1x-.json'
        comb_file = 'interrupt/combined.json'
        assert os.path.isfile(disp_file)
        assert not os.path.isfile(comb_file)

        #with pytest.raises(RuntimeError):
        #    vib.split()

        # Build a combined file
        assert vib.combine() == 13

        # Individual displacements should be gone, combination should exist
        assert not os.path.isfile(disp_file)
        assert os.path.isfile(comb_file)

        # Not allowed to run after data has been combined
        with pytest.raises(RuntimeError):
            vib.run()
        # But reading is allowed
            vib.read()

        # Splitting should fail if any split file already exists
        with open(disp_file, 'w') as fd:
            fd.write("hello")
        #with pytest.raises(RuntimeError):
        #    vib.split()
        os.remove(disp_file)

        # Now split() for real: replace .all.json file with displacements
        vib.split()
        assert os.path.isfile(disp_file)
        assert not os.path.isfile(comb_file)

        # Not allowed to clobber existing combined file
        with open(comb_file, 'w') as fd:
            fd.write("Hello")
        #with pytest.raises(RuntimeError):
        #    vib.combine()
        os.remove(comb_file)

        # Combining data also fails if some data is missing
        os.remove('interrupt/cache.1x-.json')
        #with pytest.raises(RuntimeError):
        #    vib.combine()

        #vib.clean()

    @pytest.mark.xfail
    def test_vibrations(self, testdir, n2_emt, n2_optimized):
        atoms = n2_emt
        vib = Vibrations(atoms)
        vib.run()
        freqs = vib.get_frequencies()
        vib.write_mode(n=None, nimages=5)
        vib.write_jmol()
        vib_energies = vib.get_energies()

        for image in vib.iterimages():
            assert len(image) == 2

        thermo = IdealGasThermo(vib_energies=vib_energies, geometry='linear',
                                atoms=atoms, symmetrynumber=2, spin=0)
        thermo.get_gibbs_energy(temperature=298.15, pressure=2 * 101325.,
                                verbose=False)

        with open(self.logfile, 'w') as fd:
            vib.summary(log=fd)
        with open(self.logfile, 'rt') as fd:
            log_txt = fd.read()
            assert log_txt == vibrations_n2_log

        mode1 = vib.get_mode(-1)
        assert_array_almost_equal(mode1, [[0., 0., -0.188935],
                                          [0., 0., 0.188935]])

        assert_array_almost_equal(vib.show_as_force(-1, show=False)
                                  .get_forces(),
                                  [[0., 0., -2.26722e-1],
                                   [0., 0., 2.26722e-1]])

        for i in range(3):
            assert not os.path.isfile('vib.{}.traj'.format(i))
        mode_traj = ase.io.read('vib.3.traj', index=':')
        assert len(mode_traj) == 5
        assert_array_almost_equal(mode_traj[0].get_all_distances(),
                                  atoms.get_all_distances())
        with pytest.raises(AssertionError):
            assert_array_almost_equal(mode_traj[4].get_all_distances(),
                                      atoms.get_all_distances())

        with open('vib.xyz', 'rt') as f:
            jmol_txt = f.read()
            assert jmol_txt == jmol_txt_ref

        assert vib.clean(empty_files=True) == 0
        assert vib.clean() == 13
        assert len(list(vib.iterimages())) == 13

        d = dict(vib.iterdisplace(inplace=False))

        for name, image in vib.iterdisplace(inplace=True):
            assert d[name] == atoms

        atoms2 = n2_emt
        vib2 = Vibrations(atoms2)
        vib2.run()

        assert_array_almost_equal(freqs,
                                  vib.get_frequencies())

        # write/read the data from another working directory
        atoms3 = n2_optimized.copy()  # No calculator needed!

        workdir = os.path.abspath(os.path.curdir)
        try:
            os.mkdir('run_from_here')
            os.chdir('run_from_here')
            vib = Vibrations(atoms3, name=os.path.join(os.pardir, 'vib'))
            assert_array_almost_equal(freqs, vib.get_frequencies())
            assert vib.clean() == 13
        finally:
            os.chdir(workdir)
            if os.path.isdir('run_from_here'):
                os.rmdir('run_from_here')


class TestVibrationsDataStaticMethods():
    @pytest.mark.parametrize('mask,expected_indices',
                        [([True, True, False, True], [0, 1, 3]),
                         ([False, False], []),
                         ([], []),
                         (np.array([True, True]), [0, 1]),
                         (np.array([False, True, True]), [1, 2]),
                         (np.array([], dtype=bool), [])])
    def test_indices_from_mask(self, mask, expected_indices):
        assert VibrationsData.indices_from_mask(mask) == expected_indices


class TestVibrationsData():
    @pytest.fixture
    def n2_data(self):
        return{'atoms': Atoms('N2', positions=[[0., 0., 0.05095057],
                                               [0., 0., 1.04904943]]),
               'hessian': np.array([[[[4.67554672e-03, 0.0, 0.0],
                                      [-4.67554672e-03, 0.0, 0.0]],

                                     [[0.0, 4.67554672e-03, 0.0],
                                      [0.0, -4.67554672e-03, 0.0]],

                                     [[0.0, 0.0, 3.90392599e+01],
                                      [0.0, 0.0, -3.90392599e+01]]],

                                    [[[-4.67554672e-03, 0.0, 0.0],
                                      [4.67554672e-03, 0.0, 0.0]],

                                     [[0.0, -4.67554672e-03, 0.0],
                                      [0.0, 4.67554672e-03, 0.0]],

                                     [[0.0, 0.0, -3.90392599e+01],
                                      [0.0, 0.0, 3.90392599e+01]]]]),
               'ref_frequencies': [0.00000000e+00 + 0.j,
                                   6.06775530e-08 + 0.j,
                                   3.62010442e-06 + 0.j,
                                   1.34737571e+01 + 0.j,
                                   1.34737571e+01 + 0.j,
                                   1.23118496e+03 + 0.j],
               'ref_zpe': 0.07799427233401508,
               'ref_forces': np.array([[0., 0., -2.26722e-1],
                                       [0., 0., 2.26722e-1]])
               }

    @pytest.fixture
    def n2_unstable_data(self):
        return{'atoms': Atoms('N2', positions=[[0., 0., 0.45],
                                               [0., 0., -0.45]]),
               'hessian': np.array(
                   [-5.150829928323684, 0.0, -0.6867385017096544,
                    5.150829928323684, 0.0, 0.6867385017096544, 0.0,
                    -5.158454318599951, 0.0, 0.0, 5.158454318599951, 0.0,
                    -0.6867385017096544, 0.0, 56.65107699250456,
                    0.6867385017096544, 0.0, -56.65107699250456,
                    5.150829928323684, 0.0, 0.6867385017096544,
                    -5.150829928323684, 0.0, -0.6867385017096544, 0.0,
                    5.158454318599951, 0.0, 0.0, -5.158454318599951, 0.0,
                    0.6867385017096544, 0.0, -56.65107699250456,
                    -0.6867385017096544, 0.0, 56.65107699250456
                    ]).reshape((2, 3, 2, 3))
               }

    def setup(self):
        self.jmol_file = 'vib-data.xyz'

    def test_energies_and_modes(self, n2_data):
        vib_data = VibrationsData(n2_data['atoms'], n2_data['hessian'])
        energies, modes = vib_data.get_energies_and_modes()
        assert_array_almost_equal(n2_data['ref_frequencies'],
                                  energies / units.invcm,
                                  decimal=5)
        assert_array_almost_equal(n2_data['ref_frequencies'],
                                  vib_data.get_energies() / units.invcm,
                                  decimal=5)
        assert_array_almost_equal(n2_data['ref_frequencies'],
                                  vib_data.get_frequencies(),
                                  decimal=5)

        assert (vib_data.get_zero_point_energy()
                == pytest.approx(n2_data['ref_zpe']))

        assert vib_data.tabulate() == vibrations_n2_log

        atoms_with_forces = vib_data.show_as_force(-1, show=False)

        try:
            assert_array_almost_equal(atoms_with_forces.get_forces(),
                                      n2_data['ref_forces'])
        except AssertionError:
            # Eigenvectors may be off by a sign change, which is allowed
            assert_array_almost_equal(atoms_with_forces.get_forces(),
                                      -n2_data['ref_forces'])

    def test_imaginary_energies(self, n2_unstable_data):
        vib_data = VibrationsData(n2_unstable_data['atoms'],
                                  n2_unstable_data['hessian'])
        assert vib_data.tabulate() == unstable_n2_log

    def test_zero_mass(self, n2_data):
        atoms = n2_data['atoms']
        atoms.set_masses([0., 1.])
        vib_data = VibrationsData(atoms, n2_data['hessian'])
        with pytest.raises(ValueError):
            vib_data.get_energies_and_modes()

    def test_new_mass(self, n2_data):
        vib_data = VibrationsData(n2_data['atoms'], n2_data['hessian'])
        original_masses = vib_data.get_atoms().get_masses()
        new_masses = original_masses * 3
        new_vib_data = vib_data.with_new_masses(new_masses)
        assert_array_almost_equal(new_vib_data.get_atoms().get_masses(),
                                  new_masses)
        assert_array_almost_equal(vib_data.get_energies() / np.sqrt(3),
                                  new_vib_data.get_energies())

    def test_fixed_atoms(self, n2_data):
        vib_data = VibrationsData(n2_data['atoms'],
                                  n2_data['hessian'][1:, :, 1:, :],
                                  indices=[1, ])
        assert vib_data.get_indices() == [1, ]
        assert vib_data.get_mask().tolist() == [False, True]

    def test_dos(self, n2_data):
        vib_data = VibrationsData(n2_data['atoms'], n2_data['hessian'])
        with pytest.warns(np.ComplexWarning):
            dos = vib_data.get_dos()
        assert_array_almost_equal(dos.get_energies(),
                                  vib_data.get_energies())

    def test_pdos(self, n2_data):
        vib_data = VibrationsData(n2_data['atoms'], n2_data['hessian'])
        with pytest.warns(np.ComplexWarning):
            pdos = vib_data.get_pdos()
        assert_array_almost_equal(pdos[0].get_energies(),
                                  vib_data.get_energies())
        assert_array_almost_equal(pdos[1].get_energies(),
                                  vib_data.get_energies())
        assert sum(pdos[0].get_weights()) == pytest.approx(3.0)

    def test_todict(self, n2_data):
        vib_data = VibrationsData(n2_data['atoms'], n2_data['hessian'])
        vib_data_dict = vib_data.todict()

        assert vib_data_dict['indices'] is None
        assert_array_almost_equal(vib_data_dict['atoms'].positions,
                                  n2_data['atoms'].positions)
        assert_array_almost_equal(vib_data_dict['hessian'],
                                  n2_data['hessian'])

    def test_dict_roundtrip(self, n2_data):
        vib_data = VibrationsData(n2_data['atoms'], n2_data['hessian'])
        vib_data_dict = vib_data.todict()
        vib_data_roundtrip = VibrationsData.fromdict(vib_data_dict)

        for getter in ('get_atoms',):
            assert (getattr(vib_data, getter)()
                    == getattr(vib_data_roundtrip, getter)())
        for array_getter in ('get_hessian', 'get_hessian_2d',
                             'get_mask', 'get_indices'):
            assert_array_almost_equal(
                getattr(vib_data, array_getter)(),
                getattr(vib_data_roundtrip, array_getter)())

    @pytest.mark.parametrize('indices, expected_mask',
                             [([1], [False, True]),
                              (None, [True, True])])
    def test_dict_indices(self, n2_data, indices, expected_mask):
        vib_data = VibrationsData(n2_data['atoms'], n2_data['hessian'])
        vib_data_dict = vib_data.todict()
        vib_data_dict['indices'] = indices

        # Reduce size of Hessian if necessary
        if indices is not None:
            n_active = len(indices)
            vib_data_dict['hessian'] = (
                np.asarray(vib_data_dict['hessian'])[:n_active, :, :n_active, :]
                .tolist())

        vib_data_fromdict = VibrationsData.fromdict(vib_data_dict)
        assert_array_almost_equal(vib_data_fromdict.get_mask(), expected_mask)

    def test_jmol_roundtrip(self, testdir, n2_data):
        ir_intensities = np.random.random(6)

        vib_data = VibrationsData(n2_data['atoms'], n2_data['hessian'])
        vib_data.write_jmol(self.jmol_file, ir_intensities=ir_intensities)

        images = ase.io.read(self.jmol_file, index=':')
        for i, image in enumerate(images):
            assert_array_almost_equal(image.positions,
                                      vib_data.get_atoms().positions)
            assert (image.info['IR_intensity']
                    == pytest.approx(ir_intensities[i]))
            assert_array_almost_equal(image.arrays['mode'],
                                      vib_data.get_modes()[i])

    def test_bad_hessian(self, n2_data):
        bad_hessians = (None, 'fish', 1,
                        np.array([1, 2, 3]),
                        np.eye(6),
                        np.array([[[1, 0, 0]],
                                  [[0, 0, 1]]]))

        for bad_hessian in bad_hessians:
            with pytest.raises(ValueError):
                VibrationsData(n2_data['atoms'], bad_hessian)

    def test_bad_hessian2d(self, n2_data):
        bad_hessians = (None, 'fish', 1,
                        np.array([1, 2, 3]),
                        n2_data['hessian'],
                        np.array([[[1, 0, 0]],
                                  [[0, 0, 1]]]))

        for bad_hessian in bad_hessians:
            with pytest.raises(ValueError):
                VibrationsData.from_2d(n2_data['atoms'], bad_hessian)


class TestSlab():
    "Adsorption of N2 on Ag slab - vibration with frozen molecules"
    def setup(self):
        # To reproduce n2_on_ag_data:
        # from ase.build import fcc111, add_adsorbate
        # ag_slab = fcc111('Ag', (4, 4, 2), a=2.031776)
        # ag_slab.calc = EMT()
        # n2 = Atoms('N2',
        #            positions=[(0, 0, 0), (0, 0, 1.1)],
        #            calculator=EMT())
        # QuasiNewton(n2).run(fmax=0.01)
        # add_adsorbate(ag_slab, n2, height=0.1, position='fcc')
        # QuasiNewton(ag_slab).run(fmax=1e-3)

        self.n2_on_ag = Atoms(**n2_on_ag_data)

    def test_vibrations_on_surface(self, testdir):
        atoms = self.n2_on_ag.copy()
        atoms.calc = EMT()
        vibs = Vibrations(atoms, indices=[-2, -1])
        vibs.run()

        freqs = vibs.get_frequencies()
        assert len(freqs) == 6

        vib_data = vibs.get_vibrations()
        assert_array_almost_equal(freqs, vib_data.get_frequencies())

        # These should blow up if the vectors don't match number of atoms
        vibs.summary()
        vibs.write_jmol()

        for i in range(6):
            # Frozen atoms should have zero displacement
            assert_array_almost_equal(vibs.get_mode(i)[0], [0., 0., 0.])

            # At least one of the N atoms should have finite displacement
            # (It's a strange test system, the N aren't interacting.)

            assert np.any(vibs.get_mode(i)[-2:, :])


n2_on_ag_data = {"positions": np.array([
    0.11522782450469321, 1.1144065461066939, -7.372462459539312,
    2.988453472155979, 1.1101021162088538, -7.348097091369633,
    2.9889081250223644, 1.1097319309858253, -2.6771726550093766,
    4.42748679033195, 0.27953500864036246, -5.023424263568356,
    2.988647487319677, 2.7672422535369448, -9.643893082987166,
    1.5522857100833238, 3.5986073870805897, -2.6783496788096266,
    2.9888605955178464, 2.771359213321457, -5.023112326741649,
    4.426002266203988, 3.5977377263928743, -2.676511065984475,
    1.5434899579941024, 1.9689693193513231, 4.359003102544383,
    2.9884153514662484, 4.426580053510861, -0.33110100322886604,
    2.9889669670192083, 6.086834077920163, -2.677099904430725,
    3.024247641227268, 4.433177847668567, 4.2898771182878415,
    1.5502364630326282, 5.258366760370252, -9.643885588772582,
    4.428995536176684, 3.5964348929876357, -7.372516399428504,
    4.426684159759102, 5.258550258416217, -9.643980112734312,
    7.295816104506667, 5.258864438883064, 2.019516665372352,
    -1.3252650813821554, -1.3804744498930257, -7.3724702111318585,
    1.5500159875082098, 0.27916914728271, -5.022740848670942,
    2.9715785808019946, -0.5543782652837727, 4.358845144185516,
    5.7954830502790236, -2.0529908028768005, 6.678266958798145,
    0.1153238670863118, 2.7689332793398296, -5.025638884052345,
    1.5600467349762661, 0.289360698297394, 1.99591074689944,
    4.426218846810004, 1.937761330337365, -0.32974010002269494,
    5.852146736614804, 2.7651313433743065, 2.0295010676684493,
    1.5284008009881223, 0.3517577750219494, 6.6584232454693435,
    1.5507901579723018, 1.9426262048750098, -0.3324184072062275,
    5.863789878343788, 4.425475594763032, -0.33120515517967963,
    4.420403704816752, 1.916168307407643, 4.444908498098164,
    4.122554698411387, 0.7417811921529287, 7.875168164626224,
    2.9952638361634243, 2.7603477361272026, 2.015529715672994,
    5.816292331229005, 2.8333022021373018, 6.783605490492659,
    7.438679230360008, 1.8826240456931018, 8.876555284894582,
    -1.2449602230175023, -0.135081713547721, 5.9937992070297925,
    0.7183412937260268, 0.4147345393028897, 44.45984325166285]).reshape(34, 3),
    "numbers": [47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47,
                47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47,
                47, 47, 47, 47, 47, 47, 47, 47, 7, 7],
    "cell": [[5.746730349808315, 0.0, 0.0],
             [2.8733651749041575, 4.976814471633033, 0.0],
             [0.0, 0.0, 0.0]],
    "pbc": [True, True, False]}


vibrations_n2_log = """---------------------
  #    meV     cm^-1
---------------------
  0    0.0       0.0
  1    0.0       0.0
  2    0.0       0.0
  3    1.7      13.5
  4    1.7      13.5
  5  152.6    1231.2
---------------------
Zero-point energy: 0.078 eV
"""

unstable_n2_log = """---------------------
  #    meV     cm^-1
---------------------
  0   55.5i    447.5i
  1   55.5i    447.5i
  2    0.0       0.0
  3    0.0       0.0
  4    0.0       0.0
  5  183.9    1483.2
---------------------
Zero-point energy: 0.092 eV
"""

jmol_txt_ref = """     2
Mode #0, f = 0.0  cm^-1.
 N      0.00000      0.00000      0.05095     -0.00000     -0.00000     -0.18894
 N      0.00000      0.00000      1.04905      0.00000     -0.00000     -0.18894
     2
Mode #1, f = 0.0  cm^-1.
 N      0.00000      0.00000      0.05095      0.18893     -0.00021      0.00000
 N      0.00000      0.00000      1.04905      0.18893     -0.00021     -0.00000
     2
Mode #2, f = 0.0  cm^-1.
 N      0.00000      0.00000      0.05095     -0.00021     -0.18893      0.00000
 N      0.00000      0.00000      1.04905     -0.00021     -0.18893      0.00000
     2
Mode #3, f = 13.5  cm^-1.
 N      0.00000      0.00000      0.05095     -0.18893     -0.00021      0.00000
 N      0.00000      0.00000      1.04905      0.18893      0.00021      0.00000
     2
Mode #4, f = 13.5  cm^-1.
 N      0.00000      0.00000      0.05095     -0.00021      0.18893      0.00000
 N      0.00000      0.00000      1.04905      0.00021     -0.18893      0.00000
     2
Mode #5, f = 1231.2  cm^-1.
 N      0.00000      0.00000      0.05095      0.00000      0.00000     -0.18894
 N      0.00000      0.00000      1.04905      0.00000     -0.00000      0.18894
"""  # noqa: E501
