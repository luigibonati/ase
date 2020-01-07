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
        self.n2 = Atoms('N2',
                        positions=[(0, 0, 0), (0, 0, 1.1)],
                        calculator=EMT())
        QuasiNewton(self.n2).run(fmax=0.01)

    def get_emt_n2(self):
        atoms = self.n2.copy()
        atoms.set_calculator(EMT())
        return atoms

    def tearDown(self):
        for pattern in 'vib.*.pckl', 'vib.*.traj':
            for outfile in glob.glob(pattern):
                os.remove(outfile)

    def test_vibrations(self):
        atoms = self.get_emt_n2()
        vib = Vibrations(atoms)
        vib.run()
        freqs = vib.get_frequencies()
        vib.summary()
        vib.get_mode(-1)
        vib.write_mode(n=None, nimages=20)
        vib_energies = vib.get_energies()

        for image in vib.iterimages():
            self.assertEqual(len(image), 2)

        thermo = IdealGasThermo(vib_energies=vib_energies, geometry='linear',
                                atoms=atoms, symmetrynumber=2, spin=0)
        thermo.get_gibbs_energy(temperature=298.15, pressure=2 * 101325.)

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

class TestVibrationsData(unittest.TestCase):
    def setUp(self):
        self.n2 = Atoms('N2', positions=[[0., 0., 0.05095057],  
                                         [0., 0., 1.04904943]])
        self.h_n2 = np.array([[[[ 4.67554672e-03,  0.0,  0.0],    
                               [-4.67554672e-03,  0.0,  0.0]],   
                                                                 
                              [[ 0.0,  4.67554672e-03,  0.0],    
                               [ 0.0, -4.67554672e-03,  0.0]],   
                                                                 
                              [[ 0.0,  0.0,  3.90392599e+01],    
                               [ 0.0,  0.0, -3.90392599e+01]]],  
                                                                 
                             [[[-4.67554672e-03,  0.0,  0.0],    
                               [ 4.67554672e-03,  0.0,  0.0]],   
                                                                 
                              [[ 0.0, -4.67554672e-03,  0.0],    
                               [ 0.0,  4.67554672e-03,  0.0]],   
                                                                 
                              [[ 0.0,  0.0, -3.90392599e+01],    
                               [ 0.0,  0.0,  3.90392599e+01]]]])

        # Frequencies in cm-1 from the Vibrations() test case
        self.ref_frequencies =  [0.00000000e+00+0.j,
                                 6.06775530e-08+0.j,
                                 3.62010442e-06+0.j,
                                 1.34737571e+01+0.j,
                                 1.34737571e+01+0.j,
                                 1.23118496e+03+0.j]

    def tearDown(self):
        pass

    def test_energies(self):
        vib_data = VibrationsData(self.n2.copy(), self.h_n2)
        energies, modes = vib_data.get_energies_and_modes()
        assert_array_almost_equal(self.ref_frequencies,
                                  energies / units.invcm,
                                  decimal=5)


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
