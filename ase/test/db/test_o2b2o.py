import pytest
import pickle
from ase.db.core import object_to_bytes, bytes_to_object
from ase.cell import Cell
import numpy as np


@pytest.mark.parametrize('implementation', ['new_format', 'old_format'])
@pytest.mark.parametrize('object1',
                         [1.0,
                          b'1234',
                          {'a': np.zeros((2, 2), np.float32),
                           'b': np.zeros((0, 2), int)},
                          ['a', 42, True, None, np.nan, np.inf, 1j],
                          Cell(np.eye(3)),
                          {'a': {'b': {'c': np.ones(3)}}}])
def test_ase_db_correctly_serialized_objects(implementation, object1):
    """Test that are correctly serialized and deserialized.

    This test tests that different kinds of objects are correctly
    serialized with ase.db.core.object_to_bytes and deserialized with
    ase.db.core.bytes_to_object.

    The "implementaion" parameter denotes different serialization
    schemes and explicitly tests that files written with the old
    format can still be tested.

    """
    pickle1 = pickle.dumps(object1)
    bytes1 = object_to_bytes(object1, implementation=implementation)
    object2 = bytes_to_object(bytes1)
    pickle2 = pickle.dumps(object2)
    assert pickle1 == pickle2, (object1, object2)
