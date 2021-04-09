"""Tests for XrDebye class"""

from pathlib import Path

import numpy as np
import pytest

from ase.utils.xrdebye import XrDebye, wavelengths
from ase.cluster.cubic import FaceCenteredCubic

tolerance = 1E-5


@pytest.fixture
def xrd():
    # previously calculated values
    # test system -- cluster of 587 silver atoms
    atoms = FaceCenteredCubic('Ag', [(1, 0, 0), (1, 1, 0), (1, 1, 1)],
                              [6, 8, 8], 4.09)
    return XrDebye(atoms=atoms, wavelength=wavelengths['CuKa1'], damping=0.04,
                   method='Iwasa', alpha=1.01, warn=True)


def test_get(xrd):
    # test get()
    expected_get = 116850.37344

    obtained_get = xrd.get(s=0.09)
    assert np.abs((obtained_get - expected_get) / expected_get) < tolerance


def test_xrd(xrd):
    expected_xrd = np.array([18549.274677, 52303.116995, 38502.372027])
    obtained_xrd = xrd.calc_pattern(x=np.array([15, 30, 50]), mode='XRD')
    assert np.allclose(obtained_xrd, expected_xrd, rtol=tolerance)
    xrd.write_pattern('tmp.txt')
    assert Path('tmp.txt').exists()


def test_saxs_and_files(figure, xrd):
    expected_saxs = np.array([372650934.006398, 280252013.563702,
                              488123.103628])
    obtained_saxs = xrd.calc_pattern(x=np.array([0.021, 0.09, 0.53]),
                                     mode='SAXS')
    assert np.allclose(obtained_saxs, expected_saxs, rtol=tolerance)

    xrd.write_pattern('tmp.txt')
    assert Path('tmp.txt').exists()
    ax = figure.add_subplot(111)
    xrd.plot_pattern(ax=ax, filename='pattern.png')
    assert Path('pattern.png').exists()
