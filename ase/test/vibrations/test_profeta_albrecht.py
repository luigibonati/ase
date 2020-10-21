"""
Test Placzek and Albrecht resonant Raman implementations
"""
import pytest

from ase.vibrations.resonant_raman import ResonantRamanCalculator
from ase.vibrations.placzek import Profeta
from ase.vibrations.albrecht import Albrecht
from ase.calculators.h2morse import (H2Morse,
                                     H2MorseExcitedStates,
                                     H2MorseExcitedStatesCalculator)


def test_compare_placzek_albrecht_intensities():
    atoms = H2Morse()
    name = 'rrmorse'
    rmc = ResonantRamanCalculator(atoms, H2MorseExcitedStatesCalculator,
                                  overlap=lambda x, y: x.overlap(y),
                                  name=name, txt='-')
    rmc.run()

    om = 1
    gam = 0.1
    pri, ali = 0, 0

    """Albrecht A and P-P are approximately equal"""

    pr = Profeta(atoms, H2MorseExcitedStates,
                 name=name, overlap=True,
                 approximation='p-p', txt=None)
    pri = pr.get_absolute_intensities(omega=om, gamma=gam)[-1]
    al = Albrecht(atoms, H2MorseExcitedStates,
                  name=name, overlap=True,
                  approximation='Albrecht A', txt=None)
    ali = al.get_absolute_intensities(omega=om, gamma=gam)[-1]
    print('pri, ali', pri, ali)
    assert pri == pytest.approx(ali, 1e-2)

    """Albrecht B+C and Profeta are approximately equal"""

    pr.approximation = 'Profeta'
    pri = pr.get_absolute_intensities(omega=om, gamma=gam)[-1]
    al.approximation = 'Albrecht BC'
    ali = al.get_absolute_intensities(omega=om, gamma=gam)[-1]
    print('pri, ali', pri, ali)
    assert pri == pytest.approx(ali, 1e-2)

    """Albrecht and Placzek are approximately equal"""

    pr.approximation = 'Placzek'
    pri = pr.get_absolute_intensities(omega=om, gamma=gam)[-1]
    al.approximation = 'Albrecht'
    ali = al.get_absolute_intensities(omega=om, gamma=gam)[-1]
    print('pri, ali', pri, ali)
    assert pri == pytest.approx(ali, 1e-2)
