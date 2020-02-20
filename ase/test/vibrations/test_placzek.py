"""
Test Placzek type resonant Raman implementations
"""
import pytest

from ase.vibrations.resonant_raman import ResonantRamanCalculator
from ase.vibrations.placzek import Placzek, Profeta
from ase.calculators.h2morse import (H2Morse,
                                     H2MorseExcitedStates,
                                     H2MorseExcitedStatesCalculator)


def test_overlap():
    """Test equality with and without overlap"""
    atoms = H2Morse()
    name = 'rrmorse'
    nstates = 3
    rmc = ResonantRamanCalculator(atoms, H2MorseExcitedStatesCalculator,
                                  exkwargs={'nstates': nstates},
                                  overlap=lambda x, y: x.overlap(y),
                                  name=name, txt='-')
    rmc.run()

    om = 1
    gam = 0.1
    po = Profeta(atoms, H2MorseExcitedStates,
                 exkwargs={'nstates': nstates}, approximation='Placzek',
                 overlap=True,
                 gsname=name, exname=name, txt='-')
    poi = po.absolute_intensity(omega=om, gamma=gam)[-1]

    pr = Profeta(atoms, H2MorseExcitedStates,
                 exkwargs={'nstates': nstates}, approximation='Placzek',
                 gsname=name, exname=name,
                 txt=None)
    pri = pr.absolute_intensity(omega=om, gamma=gam)[-1]

    print('overlap', pri, poi, poi / pri)
    assert pri == pytest.approx(poi, 1e-4)


def test_compare_placzek_implementation_intensities():
    """Intensities of different Placzek implementations
    should be similar"""
    atoms = H2Morse()
    name = 'placzek'
    rmc = ResonantRamanCalculator(atoms, H2MorseExcitedStatesCalculator,
                                  overlap=lambda x, y: x.overlap(y),
                                  name=name, txt='-')
    rmc.run()

    om = 1
    gam = 0.1

    pz = Placzek(atoms, H2MorseExcitedStates,
                 gsname=name, exname=name, txt=None)
    pzi = pz.absolute_intensity(omega=om, gamma=gam)[-1]
    print(pzi, 'Placzek')

    # Profeta using frozenset
    pr = Profeta(atoms, H2MorseExcitedStates,
                 approximation='Placzek',
                 gsname=name, exname=name, txt=None)
    pri = pr.absolute_intensity(omega=om, gamma=gam)[-1]
    print(pri, 'Profeta using frozenset')
    assert pzi == pytest.approx(pri, 1e-3)
    
    # Profeta using overlap
    pr = Profeta(atoms, H2MorseExcitedStates,
                 approximation='Placzek', overlap=True,
                 gsname=name, exname=name,
                 txt=None)
    pro = pr.absolute_intensity(omega=om, gamma=gam)[-1]
    print(pro, 'Profeta using overlap')
    assert pro == pytest.approx(pri, 1e-3)


def main():
    #test_overlap()
    test_compare_placzek_implementation_intensities()


if __name__ == '__main__':
    main()
