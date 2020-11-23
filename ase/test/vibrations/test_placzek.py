"""
Test Placzek type resonant Raman implementations
"""
import os
import pytest

from ase.parallel import parprint, world
from ase.vibrations.vibrations import Vibrations
from ase.vibrations.resonant_raman import ResonantRamanCalculator
from ase.vibrations.placzek import Placzek, Profeta
from ase.calculators.h2morse import (H2Morse,
                                     H2MorseExcitedStates,
                                     H2MorseExcitedStatesCalculator)


def test_summary():
    atoms = H2Morse()
    rmc = ResonantRamanCalculator(atoms, H2MorseExcitedStatesCalculator)
    rmc.run()

    pz = Placzek(atoms, H2MorseExcitedStates)
    pz.summary(1.)


def test_names():
    """Test different gs vs excited name. Tests also default names."""
    # do a Vibrations calculation first
    atoms = H2Morse()
    Vibrations(atoms).run()
    assert os.path.isfile('vib.0x-.pckl')

    # do a Resonant Raman calculation
    rmc = ResonantRamanCalculator(atoms, H2MorseExcitedStatesCalculator,
                                  verbose=True)
    rmc.run()
    # remove the corresponding pickle file,
    # then Placzek can not anymore use it for vibrational properties
    assert os.path.isfile('raman.0x-.pckl')
    os.remove('raman.0x-.pckl')  # make sure this is not used
        
    om = 1
    gam = 0.1
    pz = Placzek(atoms, H2MorseExcitedStates,
                 name='vib', exname='raman')
    pzi = pz.get_absolute_intensities(omega=om, gamma=gam)[-1]
    parprint(pzi, 'Placzek')

    # check that work was distributed correctly
    assert len(pz.myindices) <= -(-6 // world.size)
    

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
                 overlap=True, name=name, txt='-')
    poi = po.get_absolute_intensities(omega=om, gamma=gam)[-1]

    pr = Profeta(atoms, H2MorseExcitedStates,
                 exkwargs={'nstates': nstates}, approximation='Placzek',
                 name=name, txt=None)
    pri = pr.get_absolute_intensities(omega=om, gamma=gam)[-1]

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
                 name=name, txt=None)
    pzi = pz.get_absolute_intensities(omega=om, gamma=gam)[-1]
    print(pzi, 'Placzek')

    # Profeta using frozenset
    pr = Profeta(atoms, H2MorseExcitedStates,
                 approximation='Placzek',
                 name=name, txt=None)
    pri = pr.get_absolute_intensities(omega=om, gamma=gam)[-1]
    print(pri, 'Profeta using frozenset')
    assert pzi == pytest.approx(pri, 1e-3)
    
    # Profeta using overlap
    pr = Profeta(atoms, H2MorseExcitedStates,
                 approximation='Placzek', overlap=True,
                 name=name, txt=None)
    pro = pr.get_absolute_intensities(omega=om, gamma=gam)[-1]
    print(pro, 'Profeta using overlap')
    assert pro == pytest.approx(pri, 1e-3)
