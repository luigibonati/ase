from ase.io.vasp_parsers.kpoints_writer import write_kpoints
from unittest.mock import mock_open, patch

def check_write_kpoints_file(parameters,expected_output):
    mock = mock_open()
    with patch("ase.io.vasp_parsers.kpoints_writer.open", mock):
        write_kpoints(parameters)
        mock.assert_called_once_with("KPOINTS", "w")
        kpoints = mock()
        kpoints.write.assert_called_once_with(expected_output)
 
def test_kpoints_fully_automatic_mode():
    parameters = {"Auto" : 10}
    expected_output = """KPOINTS created by Atomic Simulation Environment
0
Auto
10"""
    check_write_kpoints_file(parameters,expected_output) 


def test_kpoints_fully_automatic_mode():
    parameters = {"Gamma" : [4,4,4]}
    expected_output = """KPOINTS created by Atomic Simulation Environment
0
Gamma
4 4 4"""
    check_write_kpoints_file(parameters,expected_output) 
