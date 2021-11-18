from ase.calculators.vasp.vasp_template import VaspTemplate
from unittest.mock import patch, MagicMock


@patch("ase.io.vasp_parsers.vasp_structure_io.write_vasp_structure", autospec=True)
@patch("ase.io.vasp_parsers.potcar_writer.write_potcar", autospec=True)
@patch("ase.io.vasp_parsers.kpoints_writer.write_kpoints", autospec=True)
@patch("ase.io.vasp_parsers.incar_writer.write_incar", autospec=True)
def test_write_input(write_incar, write_kpoints, write_potcar, write_poscar):
    template = VaspTemplate()
    parameters = {"incar": "foo", "kpoints": "bar", "potcar": "baz"}
    template.write_input("directory", "atoms", parameters, "properties")
    write_incar.assert_called_once_with("directory", parameters["incar"])
    write_kpoints.assert_called_once_with("directory", parameters["kpoints"])
    write_poscar.assert_called_once_with("directory/POSCAR", "atoms")
    write_potcar.assert_called_once_with("directory", parameters["potcar"])


@patch("ase.io.vasp_parsers.vasp_structure_io.write_vasp_structure", autospec=True)
@patch("ase.io.vasp_parsers.potcar_writer.write_potcar", autospec=True)
@patch("ase.io.vasp_parsers.kpoints_writer.write_kpoints", autospec=True)
@patch("ase.io.vasp_parsers.incar_writer.write_incar", autospec=True)
def test_write_input_no_parameters(
    write_incar, write_kpoints, write_potcar, write_poscar
):
    template = VaspTemplate()
    parameters = {}
    template.write_input("directory", "atoms", parameters, "properties")
    write_incar.assert_called_once_with("directory", None)
    write_kpoints.assert_called_once_with("directory", None)
    write_poscar.assert_called_once_with("directory/POSCAR", "atoms")
    write_potcar.assert_called_once_with("directory", None)


def test_execute():
    template = VaspTemplate()
    profile = MagicMock()
    template.execute("directory", profile)
    profile.run.assert_called_once_with("directory", None, "vasp.log")


def test_read_results_from_p4vasp():
    py4vasp = MagicMock()
    with patch.dict("sys.modules", py4vasp=py4vasp):
        template = VaspTemplate()
        template.read_results("directory")
        py4vasp.Calculation.from_path.assert_called_once_with("directory")
