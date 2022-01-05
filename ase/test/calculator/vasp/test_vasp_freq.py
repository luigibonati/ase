import pytest
import numpy as np


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
                sigma=1.,
                istart=0,
                ibrion=5,
                nfree=2,
                potim=0.005,
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
    def array_almost_equal(a1, a2, tol=np.finfo(type(1.0)).eps):
        """Replacement for old numpy.testing.utils.array_almost_equal."""
        return (np.abs(a1 - a2) < tol).all()

    calc = factory.calc(**calc_settings)
    mol = atoms_nh3
    mol.calc = calc
    en = mol.get_potential_energy()
    assert isinstance(en, float)

    n_free = 3*len(mol)

    freq, ifreq = mol.calc.read_vib_freq()
    assert len(freq) + len(ifreq) == n_free
    assert ifreq

    modes = mol.calc.read_vib_modes()
    assert len(modes) == n_free
    assert set([x[1].shape for x in modes]) == {(len(mol), 3)}

    steps = 10
    vibs = mol.calc.get_vibrations(index=slice(0, None), steps=steps, factor=0.01)
    assert len(vibs) == n_free
    assert set([len(vib[1]) for vib in vibs]) == {2*steps}
    #first atoms object should be equivalent to the initial molecule
    assert [array_almost_equal(mol.positions, vib[1][0].positions) for vib in vibs]
    #second and last atoms object should be equivalent
    assert [array_almost_equal(vib[1][-1].positions, vib[1][1].positions) for vib in vibs]

    # Cleanup
    calc.clean()
