import pytest
import numpy as np
from ase import Atoms


def test_species_index():

    a = Atoms(['H', 'H', 'C', 'C', 'H'])

    assert a.get_index_in_species(3) == 1
    assert a.get_index_in_species(4) == 2

    assert a.get_global_index('C', 0) == 2

    with pytest.raises(RuntimeError, match='combination not found'): 
        a.get_global_index('O', 1)
        a.get_global_index('C', 2)
