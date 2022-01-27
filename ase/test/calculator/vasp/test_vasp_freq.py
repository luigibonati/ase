import pytest
import numpy as np
from ase.constraints import FixAtoms


calc = pytest.mark.calculator


@pytest.fixture
def calc_settings():
    """Some simple fast calculation settings"""
    return dict(xc='lda',
                prec='Low',
                algo='Fast',
                setups='minimal',
                ismear=0,
                nelm=1,
                sigma=0.1,
                istart=0,
                ibrion=5,
                nfree=2,
                potim=0.05,
                gamma=True,
                txt="-",
                lwave=False,
                lcharg=False)


@calc('vasp')
def test_vasp_freq(factory, atoms_nh3, calc_settings):
    """
    Run some VASP tests to ensure that the frequency aspects of the
    calculator work. This is conditional on the existence of the
    VASP_COMMAND or VASP_SCRIPT environment variables

    """
    def read_vib_freq_outcar(calc):
        """Helper to read vibrational energies.
        Returns list of real and list of imaginary energies in meV."""
        e = []
        i_e = []
        lines = calc.load_file('OUTCAR')
        for line in lines:
            data = line.split()
            if 'THz' in data:
                if 'f/i=' not in data:
                    e.append(float(data[-2]))
                else:
                    i_e.append(complex(0,float(data[-2])))
        return e, i_e

    def array_almost_equal(a1, a2, tol=np.finfo(type(1.0)).eps):
        """Replacement for old numpy.testing.utils.array_almost_equal."""
        return (np.abs(a1 - a2) < tol).all()

    calc = factory.calc(**calc_settings)
    mol = atoms_nh3
    #one constraint
    c = FixAtoms(indices=[atom.index for atom in mol if atom.symbol == 'N'])
    mol.set_constraint(c)
    mol.calc = calc
    en = mol.get_potential_energy()
    assert isinstance(en, float)

    n_free = 3*(len(mol) - 1) #one constraint

    e, i_e = read_vib_freq_outcar(calc)
    assert len(e) + len(i_e) == n_free
    assert i_e
    outcar_data = i_e[-1::-1] + e[-1::-1]

    vib_obj = calc.get_vibrations()
    vib_data = vib_obj.get_energies()*1000 #to meV
    assert array_almost_equal(vib_data, outcar_data, tol=1e-3), "Difference{:}".format(str(vib_data-outcar_data))
    # Cleanup
    calc.clean()
