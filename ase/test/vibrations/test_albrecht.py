import pytest

from ase.vibrations.resonant_raman import ResonantRamanCalculator
from ase.calculators.h2morse import (H2Morse,
                                     H2MorseExcitedStates,
                                     H2MorseExcitedStatesCalculator)
from ase.vibrations.albrecht import Albrecht


def test_overlap():
    name = 'rrmorse'
    atoms = H2Morse()
    om = 1
    gam = 0.1

    rmc = ResonantRamanCalculator(atoms, H2MorseExcitedStatesCalculator,
                                  overlap=lambda x, y: x.overlap(y),
                                  name=name, txt='-')
    rmc.run()

    """One state only"""

    ao = Albrecht(atoms, H2MorseExcitedStates,
                  exkwargs={'nstates': 1},
                  name=name, overlap=True,
                  approximation='Albrecht A', txt=None)
    aoi = ao.absolute_intensity(omega=om, gamma=gam)[-1]

    al = Albrecht(atoms, H2MorseExcitedStates,
                  exkwargs={'nstates': 1},
                  name=name, approximation='Albrecht A', txt=None)
    ali = al.absolute_intensity(omega=om, gamma=gam)[-1]
    assert ali == pytest.approx(aoi, 1e-9)

    """Include degenerate states"""

    ao = Albrecht(atoms, H2MorseExcitedStates,
                  name=name, overlap=True,
                  approximation='Albrecht A', txt=None)
    aoi = ao.absolute_intensity(omega=om, gamma=gam)[-1]

    al = Albrecht(atoms, H2MorseExcitedStates,
                  name=name, approximation='Albrecht A', txt=None)
    ali = al.absolute_intensity(omega=om, gamma=gam)[-1]
    assert ali == pytest.approx(aoi, 1e-5)


def main():
    test_overlap()


if __name__ == '__main__':
    main()
