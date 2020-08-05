import pytest

from ase.vibrations.resonant_raman import ResonantRamanCalculator
from ase.calculators.h2morse import (H2Morse,
                                     H2MorseExcitedStates,
                                     H2MorseExcitedStatesCalculator)
from ase.vibrations.albrecht import Albrecht


@pytest.fixture
def atoms():
    return H2Morse()


@pytest.fixture
def rrname(tmp_path, atoms):
    """Prepare the Resonant Raman calculation"""
    name = str(tmp_path / 'rrmorse')

    rmc = ResonantRamanCalculator(atoms, H2MorseExcitedStatesCalculator,
                                  overlap=lambda x, y: x.overlap(y),
                                  name=name, txt='-')
    rmc.run()

    return name


def test_one_state(rrname, atoms):
    """One state only"""
    om = 1
    gam = 0.1

    ao = Albrecht(atoms, H2MorseExcitedStates,
                  exkwargs={'nstates': 1},
                  name=rrname, overlap=True,
                  approximation='Albrecht A', txt=None)
    aoi = ao.absolute_intensity(omega=om, gamma=gam)[-1]

    al = Albrecht(atoms, H2MorseExcitedStates,
                  exkwargs={'nstates': 1},
                  name=rrname, approximation='Albrecht A', txt=None)
    ali = al.absolute_intensity(omega=om, gamma=gam)[-1]
    assert ali == pytest.approx(aoi, 1e-9)


def test_all_states(rrname, atoms):
    """Include degenerate states"""
    om = 1
    gam = 0.1

    ao = Albrecht(atoms, H2MorseExcitedStates,
                  name=rrname, overlap=True,
                  approximation='Albrecht A', txt=None)

    aoi = ao.absolute_intensity(omega=om, gamma=gam)[-1]

    al = Albrecht(atoms, H2MorseExcitedStates,
                  name=rrname, approximation='Albrecht A', txt=None)
    ali = al.absolute_intensity(omega=om, gamma=gam)[-1]
    assert ali == pytest.approx(aoi, 1e-5)


def test_multiples(rrname, atoms):
    """Run multiple vibrational excitations"""
    om = 1
    gam = 0.1
    
    ao = Albrecht(atoms, H2MorseExcitedStates,
                  name=rrname, overlap=True, combinations=2,
                  approximation='Albrecht A', txt=None)

    aoi = ao.intensity(omega=om, gamma=gam)
    assert len(aoi) == 27
