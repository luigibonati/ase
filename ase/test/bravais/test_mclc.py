import pytest

from ase.lattice import MCLC

# MCLC is one of the most difficult cases.
# Here we put some sanity checks to be sure that
# we get the right MCLC in real cases.


@pytest.mark.parametrize(
    'mclc, ref_alpha', [
        (MCLC(a=13.6155, b=9.02413, c=9.48747, alpha=29.0712), 80.9784),
        (MCLC(a=17.5922, b=18.0523, c=18.8756, alpha=17.6733), 89.3247),
    ])
def test_mclc_reduction(mclc, ref_alpha):
    lat = mclc.tocell().get_bravais_lattice()
    assert lat.name == 'MCLC'
    assert lat.alpha == pytest.approx(ref_alpha, abs=1e-3)
