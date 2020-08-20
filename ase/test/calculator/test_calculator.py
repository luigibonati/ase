from pathlib import Path
import pytest
from ase.calculators.calculator import Calculator


def test_directory_and_label():
    calc = Calculator()

    assert calc.directory == '.'
    assert calc.label is None

    calc.directory = 'somedir'

    assert calc.directory == 'somedir'
    assert calc.label == 'somedir/'

    # We cannot redundantly specify directory
    with pytest.raises(ValueError):
        calc = Calculator(directory='somedir',
                          label='anotherdir/label')

    # Test only directory in directory
    calc = Calculator(directory='somedir',
                      label='label')

    assert calc.directory == 'somedir'
    assert calc.label == 'somedir/label'

    wdir = '/home/somedir'
    calc = Calculator(directory=wdir,
                      label='label')

    assert calc.directory == wdir
    assert calc.label == wdir + '/' + 'label'

    # Test we can handle pathlib directories
    wdir = Path('/home/somedir')
    calc = Calculator(directory=wdir,
                      label='label')
    assert calc.directory == str(wdir)
    assert calc.label == str(wdir) + '/' + 'label'

    with pytest.raises(ValueError):
        calc = Calculator(directory=wdir,
                          label='somedir/label')

    # Passing in empty directories with directories in label should be OK
    for wdir in ['somedir', '/home/directory']:
        label = wdir + '/label'
        calc = Calculator(directory='', label=label)
        assert calc.label == label
        assert calc.directory == wdir
        calc = Calculator(directory='.', label=label)
        assert calc.label == label
        assert calc.directory == wdir
