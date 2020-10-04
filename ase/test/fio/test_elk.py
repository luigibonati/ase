from ase.build import bulk
from ase.io import write
import os


def test_elk():
    os.environ['ELK_SPECIES_PATH'] = '/home/askhl/src/ase-datafiles/asetest/datafiles/elk'
    atoms = bulk('Si')
    write('elk.in', atoms, format='elk-in')
    with open('elk.in') as fd:
        text = fd.read()

    print(text)
    assert 'avec' in text
