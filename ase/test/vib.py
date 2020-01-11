import glob
import os
import unittest
import numpy as np
from numpy.testing import assert_array_almost_equal

from ase import units, Atoms
from ase.calculators.emt import EMT
from ase.optimize import QuasiNewton
from ase.vibrations import Vibrations, VibrationsData
from ase.thermochemistry import IdealGasThermo


class TestVibrationsClassic(unittest.TestCase):
    def setUp(self):
        self.logfile = 'vibrations-log.txt'
        self.opt_logs = 'opt-logs.txt'

        self.n2 = Atoms('N2',
                        positions=[(0, 0, 0), (0, 0, 1.1)],
                        calculator=EMT())
        QuasiNewton(self.n2, logfile=self.opt_logs).run(fmax=0.01)

    def get_emt_n2(self):
        atoms = self.n2.copy()
        atoms.set_calculator(EMT())
        return atoms

    def tearDown(self):
        for pattern in 'vib.*.pckl', 'vib.*.traj':
            for outfile in glob.glob(pattern):
                os.remove(outfile)
        for filename in (self.logfile, self.opt_logs):
            if os.path.isfile(filename):
                os.remove(filename)

    def test_vibrations(self):
        atoms = self.get_emt_n2()
        vib = Vibrations(atoms)
        vib.run()
        freqs = vib.get_frequencies()
        vib.get_mode(-1)
        vib.write_mode(n=None, nimages=20)
        vib_energies = vib.get_energies()

        for image in vib.iterimages():
            self.assertEqual(len(image), 2)

        thermo = IdealGasThermo(vib_energies=vib_energies, geometry='linear',
                                atoms=atoms, symmetrynumber=2, spin=0)
        thermo.get_gibbs_energy(temperature=298.15, pressure=2 * 101325.,
                                verbose=False)

        vib.summary(log=self.logfile)
        with open(self.logfile, 'rt') as f:
            log_txt = f.read()
            self.assertEqual(log_txt, vibrations_n2_log)

        self.assertEqual(vib.clean(empty_files=True), 0)
        self.assertEqual(vib.clean(), 13)
        self.assertEqual(len(list(vib.iterimages())), 13)

        d = dict(vib.iterdisplace(inplace=False))

        for name, image in vib.iterdisplace(inplace=True):
            self.assertEqual(d[name], atoms)

        atoms2 = self.get_emt_n2()
        vib2 = Vibrations(atoms2)
        vib2.run()

        self.assertEqual(vib.combine(), 13)
        assert_array_almost_equal(freqs,
                                  vib.get_frequencies())

        # write/read the data from another working directory
        atoms3 = self.n2.copy()  # No calculator needed!

        workdir = os.path.abspath(os.path.curdir)
        try:
            os.mkdir('run_from_here')
            os.chdir('run_from_here')
            vib = Vibrations(atoms3, name=os.path.join(os.pardir, 'vib'))
            assert_array_almost_equal(freqs, vib.get_frequencies())
            self.assertEqual(vib.clean(), 1)
        finally:
            os.chdir(workdir)
            if os.path.isdir('run_from_here'):
                os.rmdir('run_from_here')


class TestVibrationsData(unittest.TestCase):
    def setUp(self):
        self.n2 = Atoms('N2', positions=[[0., 0., 0.05095057],
                                         [0., 0., 1.04904943]])
        self.h_n2 = np.array([[[[4.67554672e-03, 0.0, 0.0],
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
                               [0.0, 0.0, 3.90392599e+01]]]])

        # Frequencies in cm-1 from the Vibrations() test case
        self.ref_frequencies = [0.00000000e+00 + 0.j,
                                6.06775530e-08 + 0.j,
                                3.62010442e-06 + 0.j,
                                1.34737571e+01 + 0.j,
                                1.34737571e+01 + 0.j,
                                1.23118496e+03 + 0.j]

    def tearDown(self):
        pass

    def test_energies(self):
        vib_data = VibrationsData(self.n2.copy(), self.h_n2)
        energies, modes = vib_data.get_energies_and_modes()
        assert_array_almost_equal(self.ref_frequencies,
                                  energies / units.invcm,
                                  decimal=5)

        with self.assertRaises(ValueError):
            vib_data.atoms.set_masses([14, 0])
            vib_data.get_energies_and_modes()

    def test_fixed_atoms(self):
        vib_data = VibrationsData(self.n2.copy(), self.h_n2[1:, :, 1:, :],
                                  indices=[1, ])
        self.assertEqual(vib_data.indices, [1, ])
        self.assertEqual(vib_data.mask.tolist(), [False, True])

        vib_data = VibrationsData(self.n2.copy(), self.h_n2[:1, :, :1, :],
                                  mask=[True, False])
        self.assertEqual(vib_data.indices, [0, ])
        self.assertEqual(vib_data.mask.tolist(), [True, False])

        with self.assertRaises(ValueError):
            vib_data = VibrationsData(self.n2.copy(), self.h_n2[:1, :, :1, :],
                                      indices=[0, ], mask=[True, False])

        with self.assertRaises(ValueError):
            vib_data = VibrationsData(self.n2.copy(), self.h_n2[:1, :, :1, :],
                                      mask=[True, False, True])

    def test_edit_data(self):
        # Check that it is possible to mutate the data and recalculate.

        # --- Modify Hessian and mask to fix an atom ---
        vib_data = VibrationsData(self.n2.copy(), self.h_n2)

        vib_data.hessian = vib_data.hessian[:1, :, :1, :]
        vib_data.mask = [True, False]
        vib_data.get_energies_and_modes()

        with self.assertRaises(ValueError):
            vib_data.hessian = np.eye(4)

        # --- Modify atoms and Hessian ---
        vib_data = VibrationsData(self.n2.copy(), self.h_n2)

        vib_data.hessian = vib_data.hessian[:1, :, :1, :]
        vib_data.atoms = vib_data.atoms[:1]
        vib_data.get_energies_and_modes()

        # --- Modify mask/atoms without corresponding Hessian
        vib_data = VibrationsData(self.n2.copy(), self.h_n2)
        vib_data.mask = [True, False]

        with self.assertRaises(ValueError):
            vib_data.get_energies_and_modes()

    def test_todict(self):
        vib_data = VibrationsData(self.n2.copy(), self.h_n2)
        vib_data_dict = vib_data.todict()

        self.assertEqual(vib_data_dict['ase_objtype'], 'vibrationsdata')
        self.assertEqual(vib_data_dict['indices'], None)
        assert_array_almost_equal(vib_data_dict['atoms']['positions'],
                                  self.n2.positions)
        assert_array_almost_equal(vib_data_dict['hessian'],
                                  self.h_n2)

    def test_dict_roundtrip(self):
        vib_data = VibrationsData(self.n2.copy(), self.h_n2)
        vib_data_dict = vib_data.todict()
        vib_data_roundtrip = VibrationsData.fromdict(vib_data_dict)

        for attr in 'atoms', 'indices':
            self.assertEqual(getattr(vib_data, attr),
                             getattr(vib_data_roundtrip, attr))
        for array_getter in ('get_hessian_2d',):
            assert_array_almost_equal(
                getattr(vib_data, array_getter)(),
                getattr(vib_data_roundtrip, array_getter)())
        for array_attr in ('hessian',):
            assert_array_almost_equal(
                getattr(vib_data, array_attr),
                getattr(vib_data_roundtrip, array_attr))

    def test_bad_hessian(self):
        bad_hessians = (None, 'fish', 1,
                        np.array([1, 2, 3]),
                        np.eye(6),
                        np.array([[[1, 0, 0]],
                                  [[0, 0, 1]]]))

        for bad_hessian in bad_hessians:
            with self.assertRaises(ValueError):
                VibrationsData(self.n2.copy(), bad_hessian)

    def test_bad_hessian2d(self):
        bad_hessians = (None, 'fish', 1,
                        np.array([1, 2, 3]),
                        self.h_n2,
                        np.array([[[1, 0, 0]],
                                  [[0, 0, 1]]]))

        for bad_hessian in bad_hessians:
            with self.assertRaises(ValueError):
                VibrationsData.from_2d(self.n2.copy(), bad_hessian)


class TestSlab(unittest.TestCase):
    "Adsorption of N2 on Ag slab - vibration with frozen molecules"
    def setUp(self):
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

    def test_vibrations_on_surface(self):
        atoms = self.n2_on_ag.copy()
        atoms.calc = EMT()
        vibs = Vibrations(atoms, indices=[-2, -1])
        vibs.run()

        freqs = vibs.get_frequencies()

        vib_data = vibs.get_vibrations()
        assert_array_almost_equal(freqs, vib_data.get_frequencies())


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

# More unittest boilerplate before pytest arrives
def suite():
    import ase.test.vib  # Circular dependency is a bit scary but seems to work
    suite = unittest.defaultTestLoader.loadTestsFromModule(ase.test.vib)
    return suite


# Instead of keeping/displaying unittest results, escalate errors so ASE unit
# test system can handle them. "noqa" tells flake8 that it's ok for these
# functions to have camelCase names (as required by unittest).
class VibrationsTestResults(unittest.TestResult):
    def addFailure(self, test, err):      # noqa: N802
        raise err[1]

    def addError(self, test, err):        # noqa: N802
        raise err[1]


if __name__ in ['__main__', 'test']:
    runner = unittest.TextTestRunner(resultclass=VibrationsTestResults)
    runner.run(suite())
