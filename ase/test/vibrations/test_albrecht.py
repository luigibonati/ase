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
def rrname(atoms):
    """Prepare the Resonant Raman calculation"""
    name = 'rrmorse'
    rmc = ResonantRamanCalculator(atoms, H2MorseExcitedStatesCalculator,
                                  overlap=lambda x, y: x.overlap(y),
                                  name=name, txt='-')
    rmc.run()
    return name


def test_one_state(rrname, atoms):
    om = 1
    gam = 0.1

    ao = Albrecht(atoms, H2MorseExcitedStates,
                  exkwargs={'nstates': 1},
                  name=rrname, overlap=True,
                  approximation='Albrecht A', txt=None)
    aoi = ao.get_absolute_intensities(omega=om, gamma=gam)[-1]

    al = Albrecht(atoms, H2MorseExcitedStates,
                  exkwargs={'nstates': 1},
                  name=rrname, approximation='Albrecht A', txt=None)
    ali = al.get_absolute_intensities(omega=om, gamma=gam)[-1]
    assert ali == pytest.approx(aoi, 1e-9)


def test_all_states(rrname, atoms):
    """Include degenerate states"""
    om = 1
    gam = 0.1

    ao = Albrecht(atoms, H2MorseExcitedStates,
                  name=rrname, overlap=True,
                  approximation='Albrecht A', txt=None)
    aoi = ao.get_absolute_intensities(omega=om, gamma=gam)[-1]

    al = Albrecht(atoms, H2MorseExcitedStates,
                  name=rrname, approximation='Albrecht A', txt=None)
    ali = al.get_absolute_intensities(omega=om, gamma=gam)[-1]
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


def test_summary(rrname, atoms):
    om = 1
    gam = 0.1
    
    ao = Albrecht(atoms, H2MorseExcitedStates,
                  name=rrname, overlap=True,
                  approximation='Albrecht B', txt=None)
    ao.summary(om, gam)
    
    ao = Albrecht(atoms, H2MorseExcitedStates,
                  name=rrname, overlap=True, combinations=2,
                  approximation='Albrecht A', txt=None)
    ao.extended_summary(om, gam)
