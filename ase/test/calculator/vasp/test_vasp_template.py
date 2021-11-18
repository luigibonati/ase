from ase.calculators.vasp.vasp_template import VaspTemplate
from unittest.mock import patch


@patch("ase.io.vasp_parsers.incar_writer.write_incar", autospec=True)
def test_write_input(write_incar):
    template = VaspTemplate()
    template.write_input("directory", "atoms", "parameters", "properties")
    write_incar.assert_called_once_with("parameters")
